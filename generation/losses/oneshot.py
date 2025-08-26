import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch

from .utils import batch_linear_assignment, rse


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
        return {'loss': self.combine_losses(mse_loss, ce_loss)}

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)



class DistLoss(Module):
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
        EPS = 1e-6
        mse_sum = 0.0
        mse_count = 0

        data_conf = self.data_conf
        num_names = y_pred.num_features_names or []
        num_names = list(set(data_conf.focus_on) & set(num_names))
        
        if data_conf.time_name in data_conf.focus_on:
            pred_time = y_pred.time
            true_time = y_true.target_time
            alpha_raw, beta_raw = pred_time.unbind(dim=1)
            # alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)
            dist = torch.distributions.Normal(alpha_raw, beta)
            mse_sum += -dist.log_prob(true_time + EPS).mean()
            mse_count += 1

        num_names = num_names or []
        for name in num_names:
            # breakpoint()
            id_true = y_true.num_features_names.index(name)
            id_pred = y_pred.num_features_names.index(name)
            true_feature_ids = [id_true, id_true + 1]

            pred_feature_ids = [id_pred, id_pred + 1]
            pred_num = y_pred.num_features[:, pred_feature_ids]
            true_num = y_true.target_num_features[:, true_feature_ids]
            alpha, beta = pred_num.unbind(dim=-1)
            # alpha = torch.nn.functional.softmax(alpha)
            beta = torch.nn.functional.softplus(beta)

            dist = torch.distributions.Normal(alpha, beta)
            mse_sum += -dist.log_prob(true_num + EPS).mean()
            mse_count += 1

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
            # breakpoint()
            true_id = y_true.cat_features_names.index(key)
            true_cat = y_true.target_cat_features[:, :, true_id].clone()

            pred_cat = y_pred.cat_features[key].permute(0, 1)  # [B, C, L]
            # Compute loss
            dist = torch.distributions.Categorical(logits=pred_cat)
            total_ce += -dist.log_prob(true_cat).mean()
            ce_count += 1


        return (total_ce / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        mse_loss = self._compute_mse(y_true, y_pred)
        ce_loss = self._compute_ce(y_true, y_pred)
        return  {'loss': self.combine_losses(mse_loss, ce_loss)}

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)


class GaussianDistLoss(Module):

    EPS = 1e-6
    
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

    def get_numerical_loss(self, y_true: torch.tensor, y_pred:torch.tensor):
        pi_logits, mu_raw, log_sigma = y_pred.unbind(dim=-1)
        pi = torch.nn.functional.softmax(pi_logits, dim=-1)
        sigma = torch.nn.functional.softplus(log_sigma).clamp(1e-3, 5)
        component = torch.distributions.Normal(mu_raw, sigma)
        mixing = torch.distributions.Categorical(probs=pi)
        dist = torch.distributions.MixtureSameFamily(mixing, component)
        return -dist.log_prob(y_true).mean()

    def _compute_mse(self, y_true: GenBatch, y_pred: PredBatch) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        data_conf = self.data_conf
        num_names = y_pred.num_features_names or []
        num_names = list(set(data_conf.focus_on) & set(num_names))
        
        if data_conf.time_name in data_conf.focus_on:
            pred_time = y_pred.time
            true_time = y_true.target_time
            mse_sum += self.get_numerical_loss(true_time, pred_time)
            mse_count += 1

        num_names = num_names or []
        for name in num_names:
            id_true = y_true.num_features_names.index(name)
            id_pred = y_pred.num_features_names.index(name)

            true_num = y_true.target_num_features[..., id_true]
            pred_num = y_pred.num_features[:, id_pred, ...]
            
            mse_sum += self.get_numerical_loss(true_num, pred_num)
            mse_count += 1

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
            # breakpoint()
            true_id = y_true.cat_features_names.index(key)
            true_cat = y_true.target_cat_features[:, :, true_id].clone()

            pred_cat = y_pred.cat_features[key]
            # Compute loss
            dist = torch.distributions.Categorical(logits=pred_cat)
            total_ce += -dist.log_prob(true_cat).mean()
            ce_count += 1


        return (total_ce / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        mse_loss = self._compute_mse(y_true, y_pred)
        ce_loss = self._compute_ce(y_true, y_pred)
        return  {'loss': self.combine_losses(mse_loss, ce_loss)}

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
        cost_device = cost.device
        cost = cost.permute((2, 0, 1))  # [B, L, L]
        if self.max_shift >= 0:
            i_indices = torch.arange(L, device=cost_device)[:, None] # L, 1
            j_indices = torch.arange(L, device=cost_device)
            distance_from_diagonal = torch.abs(i_indices - j_indices) # L, L
            mask_outside_band = distance_from_diagonal > self.max_shift
            cost.masked_fill_(mask_outside_band, torch.inf)

        assignment = batch_linear_assignment(cost.cpu().detach()).T.to(cost_device) # L, B        
        
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
        return {'loss': self.combine_losses(mse_loss, cat_loss)}
    
    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.mse_weight * mse_loss + (1 - self.mse_weight) * ce_loss)