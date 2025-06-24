from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, PredBatch, LatentDataConfig
from torch_linear_assignment import batch_linear_assignment


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


def rse(pred, true):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    res = (pred - true) ** 2  # L, B, [D]
    userwise_res = res.sum(dim=0)  # B, [D]

    userwise_mean = true.mean(dim=0)  # B, [D]
    tot = (true - userwise_mean) ** 2  # L, B, [D]
    userwise_tot = tot.sum(dim=0)  # B, [D]

    rse = userwise_res / userwise_tot
    return rse.sum(), rse.numel()


class BaseLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
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
        self, y_true: GenBatch, y_pred: PredBatch, valid_mask: torch.Tensor
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
        self, y_true: GenBatch, y_pred: PredBatch, valid_mask: torch.Tensor
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
        return {'loss': self.combine_losses(mse_loss, ce_loss)}

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)


class BaselineLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(None, -1)  # Predictions use all except last time step

    @property
    def true_slice(self):
        return slice(1, None)  # Targets use all except first time step


class NoOrderLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(None, -1)  # Predictions use all except last time step

    @property
    def true_slice(self):
        return slice(1, None)  # Targets use all except first time step

    def bag_mse_loss(self, logits, targets, window=10, ignore_index=-100, eps=1e-8):
        """
        logits  : [B,L,C]
        targets : [B,L]
        Возвращает средний MSE между нормализованными гистограммами p и q.
        """
        B, L, C = logits.shape
        prob = logits.softmax(-1)  # [B,L,C]

        loss, n_windows = 0.0, 0
        for t0 in range(0, L, window):
            t1 = min(L, t0 + window)

            # Распределение предсказанний
            p_counts = prob[:, t0:t1].sum(1)  # [B,C]
            p = p_counts / (p_counts.sum(-1, keepdim=True) + eps)

            # Распределение таргеров
            tgt = targets[:, t0:t1]
            mask = tgt.ne(ignore_index)
            one_hot = F.one_hot(tgt.clamp(min=0), C).float() * mask.unsqueeze(-1)
            q_counts = one_hot.sum(1)  # [B,C]
            q = q_counts / (q_counts.sum(-1, keepdim=True) + eps)

            loss += ((p - q) ** 2).sum(-1).mean() / 2  # MSE
            n_windows += 1

        return loss / max(n_windows, 1)

    def _compute_loss(self, y_true, y_pred, valid_mask):
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)
        data_conf = self.data_conf
        cat_names = y_pred.cat_features or {}
        cat_names = list(set(data_conf.focus_on) & set(cat_names))

        total_loss = 0.0
        ce_count = 0
        assert len(cat_names) == 1

        for key in cat_names:
            true_cat = y_true[key][self.true_slice].clone()
            current_mask = valid_mask[self.true_slice]
            true_cat[~current_mask] = self._ignore_index

            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            pred_cat = pred_cat[:, :, self.pred_slice]

            loss = self.bag_mse_loss(
                pred_cat.permute(0, 2, 1), true_cat.permute(1, 0), window=int(1e10)
            )
            total_loss += loss
            ce_count += 1

        return (total_loss / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        valid_mask = self._valid_mask(y_true)
        loss = self._compute_loss(y_true, y_pred, valid_mask)
        return {'loss': loss}


class TailLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(-1 - self.data_conf.generation_len, -1)  # Take shifted last

    @property
    def true_slice(self):
        return slice(-self.data_conf.generation_len, None)  # Predict only generated_len


class VAELoss(BaseLoss):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        mse_weight: float = 0.5,
        init_beta: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__(data_conf, mse_weight, ignore_index)
        self._beta = init_beta

    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        y_pred, params = data
        base_loss = super().__call__(y_true, y_pred)

        mu_z = params["mu_z"]
        std_z = params["std_z"]
        kld_term = -0.5 * torch.mean(
            (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(-1).mean()
        )

        return {'loss': base_loss + self._beta * kld_term, 'kl_loss': kld_term}

    def update_beta(self, value):
        self._beta = value


class TargetLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        mse_weight: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        assert 0 <= mse_weight <= 1
        self.data_conf = data_conf
        self.mse_weight = mse_weight
        self._ignore_index = ignore_index

    def _compute_mse(self, y_true: GenBatch, y_pred: PredBatch) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        data_conf = self.data_conf
        num_names = y_pred.num_features_names or []
        num_names = list(set(data_conf.focus_on) & set(num_names))
        if data_conf.time_name in data_conf.focus_on:
            pred_time = y_pred.time  # [L, B]
            true_time = y_true.target_time  # [L, B]
            loss, count = rse(pred_time, true_time)
            mse_sum += loss
            mse_count += count

        if num_names:
            true_feature_ids = [
                y_true.num_features_names.index(name) for name in num_names
            ]
            pred_feature_ids = [
                y_pred.num_features_names.index(name) for name in num_names
            ]
            pred_num = y_pred.num_features[:, :, pred_feature_ids]
            true_num = y_true.target_num_features[:, :, true_feature_ids]
            loss, count = rse(pred_num, true_num)
            mse_sum += loss
            mse_count += count

        return (mse_sum / mse_count) if mse_count != 0 else torch.tensor(0.0)

    def _compute_ce(self, y_true: GenBatch, y_pred: PredBatch) -> torch.Tensor:
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=y_true.time.device)
        data_conf = self.data_conf
        cat_names = y_pred.cat_features or {}
        cat_names = list(set(data_conf.focus_on) & set(cat_names))

        total_ce = 0.0
        ce_count = 0
        for key in cat_names:
            true_id = y_true.cat_features_names.index(key)
            true_cat = y_true.target_cat_features[:, :, true_id].clone()

            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            # Compute loss
            ce_loss = F.cross_entropy(
                pred_cat,
                true_cat.permute(1, 0),  # [B, L']
                ignore_index=self._ignore_index,
            )
            total_ce += ce_loss
            ce_count += 1

        return (total_ce / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        mse_loss = self._compute_mse(y_true, y_pred)
        ce_loss = self._compute_ce(y_true, y_pred)
        return self.combine_losses(mse_loss, ce_loss)

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)


class MatchedLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        mse_weight: float = 0.5,
        ignore_index: int = -100,
        max_shift: int = 0,
    ):
        super().__init__()
        assert 0 <= mse_weight <= 1
        self.data_conf = data_conf
        self.mse_weight = mse_weight
        self._ignore_index = ignore_index
        assert -1 <= max_shift <= data_conf.generation_len
        self.max_shift = max_shift

    def __call__(self, y_true: GenBatch, y_pred: PredBatch):
        # Step 1: calculate cost matrix [B, L, L]
        data_conf = self.data_conf
        L, B = y_true.target_time.shape

        cost = torch.zeros(
            L, L, B, device=y_true.target_time.device, dtype=y_true.target_time.dtype
        )
        # 1.1 Calculate R2 score
        assert y_true.num_features_names == y_pred.num_features_names
        num_names = (y_true.num_features_names or []) + [data_conf.time_name]
        num_ids = [
            num_names.index(name) for name in num_names if name in data_conf.focus_on
        ]
        num_pred = y_pred.get_numerical()  # [L, B, Dn+1]
        num_true = y_true.get_target_numerical() # [L, B, Dn+1]
        if num_ids:
            res = (
                num_pred[:, None, :, num_ids] - num_true[None, :, :, num_ids]
            ) ** 2  # [L, L, B, D]

            userwise_mean = num_true.mean(dim=0)  # B, [D]
            tot = (num_true - userwise_mean) ** 2  # L, B, [D]
            userwise_tot = tot.sum(dim=0)  # B, [D]

            res = res / userwise_tot[None, None, :, :]  # Scaled mse, [L, L, B, D]
            cost += res.mean(3)  # [L, L, B] R2 score

        # 1.2 Calculate accuracy
        pred_batch = y_pred.to_batch()
        assert y_true.cat_features_names == pred_batch.cat_features_names
        cat_names = y_true.cat_features_names or []
        cat_ids = [
            cat_names.index(name) for name in cat_names if name in data_conf.focus_on
        ]
        cat_pred, cat_true = pred_batch.cat_features, y_true.target_cat_features
        if cat_ids:
            res = cat_pred[:, None, :, cat_ids] != cat_true[None, :, :, cat_ids]
            cost += res.to(torch.float32).mean(3)  # [L, L, B]
        
        # Step 2: calculate assignment
        cost = cost.permute((2, 0, 1))  # [B, L, L]
        if self.max_shift >= 0:
            i_indices = torch.arange(L, device=cost.device)[:, None] # L, 1
            j_indices = torch.arange(L, device=cost.device)
            distance_from_diagonal = torch.abs(i_indices - j_indices) # L, L
            mask_outside_band = distance_from_diagonal > self.max_shift
            cost.masked_fill_(mask_outside_band, torch.inf)
        breakpoint()
        assignment = batch_linear_assignment(cost).T # L, B
        assignment = batch_linear_assignment(cost.to(device="cpu")).T # L, B
        
        
        assignment = assignment.unsqueeze(-1) # L, B, 1

        # Step 3: calculate loss using new order.
        mse_loss, cat_loss = 0, 0
        if num_ids:
            num_true = num_true.gather(0, assignment.expand(num_true.shape))
            res = (num_pred[:, :, num_ids] - num_true[:, :, num_ids]) ** 2  # L, B, D
            userwise_res = res.sum(dim=0)  # B, D
            rse = userwise_res / userwise_tot # B, D
            mse_loss += rse.mean()
        if cat_ids:
            total_ce = 0.0
            ce_count = 0
            cat_true = cat_true.gather(0, assignment.expand(cat_true.shape))
            cat_names = [name for name in cat_names if name in data_conf.focus_on]
            for key in cat_names:
                true_id = y_true.cat_features_names.index(key)
                true_cat = cat_true[:, :, true_id].clone()
                pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
                # Compute loss
                ce_loss = F.cross_entropy(
                    pred_cat,
                    true_cat.permute(1, 0),  # [B, L']
                    ignore_index=self._ignore_index,
                )
                total_ce += ce_loss
                ce_count += 1
            cat_loss += (total_ce / ce_count)
        return self.combine_losses(mse_loss, cat_loss)
    
    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)


def get_loss(data_conf: LatentDataConfig, config: LossConfig):
    name = config.name
    if name == "baseline":
        return BaselineLoss(data_conf, **config.params)
    elif name == "target":
        return TargetLoss(data_conf, **config.params)
    elif name == "matched":
        return MatchedLoss(data_conf, **config.params)
    elif name == "tail":
        return TailLoss(data_conf, **config.params)
    elif name == "vae":
        return VAELoss(data_conf, init_beta=1.0, **config.params)
    elif name == "no_order_loss":
        return NoOrderLoss(data_conf, **config.params)
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")
