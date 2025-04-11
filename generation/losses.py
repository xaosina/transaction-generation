from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import Batch, PredBatch


@dataclass(frozen=True)
class LossConfig:
    name: Optional[str] = "baseline"
    c_dim: Optional[int] = None
    c_number: Optional[int] = None


def rse_valid(pred, true, valid_mask):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    if true.ndim == 3:
        valid_mask = valid_mask.unsqueeze(-1)
    res = ((pred - true) ** 2) # L, B, [D]
    userwise_res = torch.where(valid_mask, res, torch.nan).nansum(dim=0)  # B, [D]
    userwise_mean = torch.where(valid_mask, true, torch.nan).nanmean(dim=0)  # B, [D]
    tot = ((true - userwise_mean) ** 2) # L, B, [D]
    userwise_tot = torch.where(valid_mask, tot, torch.nan).nansum(dim=0)  # B, [D]
    rse = userwise_res / userwise_tot
    return rse, rse.numel()


class BaselineLoss(Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self._ignore_index = ignore_index

    def _compute_mse(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        if y_pred.time is not None:
            pred_time = y_pred.time  # [L, B]
            true_time = y_true.time  # [L, B]
            loss, count = rse_valid(pred_time[:-1], true_time[1:], valid_mask[1:])
            mse_sum += loss
            mse_count += count

        if y_pred.num_features is not None:
            num_feature_ids = [
                y_true.num_features_names.index(name)
                for name in y_pred.num_features_names
            ]
            pred_num = y_pred.num_features  # [L, B, D]
            true_num = y_true.num_features  # [L, B, totalD]
            loss, count = rse_valid(
                pred_num[:-1], true_num[1:, :, num_feature_ids], valid_mask[1:]
            )

            mse_sum += loss
            mse_count += count

        return mse_sum / mse_count

    def _compute_ce(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)

        total_ce = 0.0
        ce_count = 0
        for key in y_pred.cat_features:
            # [L, B]
            true_cat = y_true[key].clone()
            true_cat[~valid_mask] = self._ignore_index

            true_cat = true_cat.permute(1, 0)  # [B, L]
            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            ce_loss = F.cross_entropy(
                pred_cat[:, :, :-1], true_cat[:, 1:], ignore_index=self._ignore_index
            )
            total_ce += ce_loss
            ce_count += 1

        return total_ce / ce_count

    def _valid_mask(self, y_true) -> torch.Tensor:
        return (
            torch.arange(y_true.lengths.max(), device=y_true.lengths.device)[:, None]
            < y_true.lengths
        )

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        valid_mask = self._valid_mask(y_true)
        mse_loss = self._compute_mse(y_true, y_pred, valid_mask)

        ce_loss = self._compute_ce(y_true, y_pred, valid_mask)

        return mse_loss + ce_loss


class VAELoss(Module):

    def __init__(self, init_beta: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self._ignore_index = ignore_index
        self._beta = init_beta

    def __call__(self, y_true: Batch, data) -> torch.Tensor:
        y_pred, params = data

        valid_mask = self._valid_mask(y_true)

        mse_loss = self._compute_mse(y_true, y_pred, valid_mask)

        ce_loss = self._compute_ce(y_true, y_pred, valid_mask)

        assert (
            "mu_z" in params and "std_z" in params
        ), "Params should contains 'mu_z' and 'std_z'"

        mu_z = params["mu_z"]
        std_z = params["std_z"]

        kld_term = -0.5 * torch.mean(
            (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(-1).mean()
        )

        return mse_loss + ce_loss + self._beta * kld_term

    def _compute_mse(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        if y_pred.time is not None:
            pred_time = y_pred.time  # [L, B]
            true_time = y_true.time  # [L, B]
            mse_time = F.mse_loss(pred_time, true_time, reduction="none")[valid_mask]
            mse_sum += mse_time.sum()
            mse_count += mse_time.numel()

        if y_pred.num_features is not None:
            num_feature_ids = [
                y_true.num_features_names.index(name)
                for name in y_pred.num_features_names
            ]
            pred_num = y_pred.num_features  # [L, B, D]
            true_num = y_true.num_features  # [L, B, totalD]
            mse_num = F.mse_loss(
                pred_num, true_num[:, :, num_feature_ids], reduction="none"
            )[valid_mask.unsqueeze(-1)]

            mse_sum += mse_num.sum()
            mse_count += mse_num.numel()

        return mse_sum / mse_count

    def _compute_ce(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)

        total_ce = 0.0
        ce_count = 0
        for key in y_pred.cat_features:
            # [L, B]
            true_cat = y_true[key].clone()
            true_cat[~valid_mask] = self._ignore_index

            true_cat = true_cat.permute(1, 0)  # [B, L]
            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            ce_loss = F.cross_entropy(
                pred_cat, true_cat, ignore_index=self._ignore_index
            )
            total_ce += ce_loss
            ce_count += 1

        return total_ce / ce_count

    def _valid_mask(self, y_true) -> torch.Tensor:
        return (
            torch.arange(y_true.lengths.max(), device=y_true.lengths.device)[:, None]
            < y_true.lengths
        )

    def update_beta(self, value):
        self._beta = value


def get_loss(config: LossConfig):
    name = config.name
    if name == "baseline":
        return BaselineLoss()
    elif name == "vae":
        return VAELoss(init_beta=1.0)
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")
