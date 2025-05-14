from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import Batch, PredBatch, DataConfig


@dataclass(frozen=True)
class LossConfig:
    name: Optional[str] = "baseline"
    params: dict = field(default_factory=dict)


def rse_valid(pred, true, valid_mask):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    if true.ndim == 3:
        valid_mask = valid_mask.unsqueeze(-1)
    res = (pred - true) ** 2  # L, B, [D]
    userwise_res = torch.where(valid_mask, res, torch.nan).nansum(dim=0)  # B, [D]
    userwise_mean = torch.where(valid_mask, true, torch.nan).nanmean(dim=0)  # B, [D]
    tot = (true - userwise_mean) ** 2  # L, B, [D]
    userwise_tot = torch.where(valid_mask, tot, torch.nan).nansum(dim=0)  # B, [D]
    rse = userwise_res / userwise_tot
    return rse.sum(), rse.numel()


class BaseLoss(Module):
    def __init__(
        self,
        data_conf: DataConfig,
        mse_weight: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        assert 0 <= mse_weight <= 1
        self.data_conf = data_conf
        self.mse_weight = mse_weight
        self._ignore_index = ignore_index

    @property
    def pred_slice(self):
        """Slice applied to prediction tensors (e.g., slice(None, -1))"""
        return slice(None)  # Default: no slicing

    @property
    def true_slice(self):
        """Slice applied to target tensors (e.g., slice(1, None))"""
        return slice(None)  # Default: no slicing

    def _compute_mse(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        data_conf = self.data_conf
        num_names = y_pred.num_features_names or []
        num_names = list(set(data_conf.focus_on) & set(num_names))

        if data_conf.time_name in data_conf.focus_on:
            pred_time = y_pred.time[self.pred_slice]  # [L, B]
            true_time = y_true.time[self.true_slice]  # [L, B]
            current_mask = valid_mask[self.true_slice]
            loss, count = rse_valid(pred_time, true_time, current_mask)
            mse_sum += loss
            mse_count += count

        if num_names:
            true_feature_ids = [
                y_true.num_features_names.index(name) for name in num_names
            ]
            pred_feature_ids = [
                y_pred.num_features_names.index(name) for name in num_names
            ]
            pred_num = y_pred.num_features[self.pred_slice, :, pred_feature_ids]
            true_num = y_true.num_features[self.true_slice, :, true_feature_ids]
            current_mask = valid_mask[self.true_slice]
            loss, count = rse_valid(pred_num, true_num, current_mask)
            mse_sum += loss
            mse_count += count

        return (mse_sum / mse_count) if mse_count != 0 else torch.tensor(0.0)

    def _compute_ce(
        self, y_true: Batch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)
        data_conf = self.data_conf
        cat_names = y_pred.cat_features or {}
        cat_names = list(set(data_conf.focus_on) & set(cat_names))

        total_ce = 0.0
        ce_count = 0
        for key in cat_names:
            true_cat = y_true[key][self.true_slice].clone()
            current_mask = valid_mask[self.true_slice]
            true_cat[~current_mask] = self._ignore_index

            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            pred_cat = pred_cat[:, :, self.pred_slice]

            # Compute loss
            ce_loss = F.cross_entropy(
                pred_cat,
                true_cat.permute(1, 0),  # [B, L']
                ignore_index=self._ignore_index,
            )
            total_ce += ce_loss
            ce_count += 1

        return (total_ce / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def _valid_mask(self, y_true) -> torch.Tensor:
        return (
            torch.arange(y_true.lengths.max(), device=y_true.lengths.device)[:, None]
            < y_true.lengths
        )

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        valid_mask = self._valid_mask(y_true)
        mse_loss = self._compute_mse(y_true, y_pred, valid_mask)
        ce_loss = self._compute_ce(y_true, y_pred, valid_mask)
        return self.combine_losses(mse_loss, ce_loss)

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)


class BaselineLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(None, -1)  # Predictions use all except last time step

    @property
    def true_slice(self):
        return slice(1, None)  # Targets use all except first time step


class VAELoss(BaseLoss):
    def __init__(
        self,
        data_conf: DataConfig,
        mse_weight: float = 0.5,
        init_beta: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__(data_conf, mse_weight, ignore_index)
        self._beta = init_beta

    def __call__(self, y_true: Batch, data) -> torch.Tensor:
        y_pred, params = data
        base_loss = super().__call__(y_true, y_pred)

        mu_z = params["mu_z"]
        std_z = params["std_z"]
        kld_term = -0.5 * torch.mean(
            (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(-1).mean()
        )

        return base_loss + self._beta * kld_term

    def update_beta(self, value):
        self._beta = value


def get_loss(data_conf: DataConfig, config: LossConfig):
    name = config.name
    if name == "baseline":
        return BaselineLoss(data_conf, **config.params)
    elif name == "vae":
        return VAELoss(data_conf, init_beta=1.0, **config.params)
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")
