from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    print("Using slow linear assignment implementation")
    from .utils import batch_linear_assignment


class DeTPPLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        loss_subset: float = 0.25,
        matching_weights: dict = None,
    ):
        super().__init__()
        self.matching_weights = matching_weights or {}
        self._k = data_conf.generation_len
        self._loss_subset = loss_subset

        self.data_conf = data_conf

    def forward(self, inputs: GenBatch, outputs: tuple):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (L, B, D).
            outputs: Predicted values with shape (L, B, K, D).

        Returns:
            loss
        """
        outputs, presence_scores = outputs
        target_windows = self.extract_structured_windows(
            inputs
        )  # (L, B, k + 1), where first event is an input for the model.

        # Subset outputs and targets.
        indices, subset_lengths = self.get_loss_indices(inputs)
        target_windows = self.select_subset(
            target_windows, indices, subset_lengths
        )  # (I, B, K + 1).
        outputs = self.select_subset(outputs, indices, subset_lengths)  # (I, B, K, P).
        presence_scores = self.select_subset(presence_scores, indices, subset_lengths)

        return {"loss": self.match_targets(outputs, presence_scores, target_windows)}

    def extract_structured_windows(self, inputs: GenBatch):
        """Extract windows with shape (L, B, k + 1) from inputs with shape (L, B, D)."""
        inputs = deepcopy(inputs)
        L, _ = inputs.shape
        for attr in ["time", "num_features", "cat_features"]:
            tensor = getattr(inputs, attr)
            if tensor is None:
                continue
            parts = [tensor.roll(-i, 0) for i in range(self._k + 1)]
            windows = torch.stack(parts, 2)  # (L, B, k + 1, D)
            setattr(inputs, attr, windows)
        # inputs.lengths -= self._k
        return inputs

    def get_loss_indices(self, inputs: GenBatch):
        """Get positions to evaluate loss at.

        Args:
           inputs: Input features with shape (L, B).

        Returns:
           indices: Batch of indices with shape (I, B) or None if loss must be evaluated at each step.
        """
        l, b = inputs.shape
        k = self._k
        n_indices = min(max(int(round(l * self._loss_subset)), 1), l)
        # Take full windows first.
        mask = (
            torch.arange(l, device=inputs.device)[None] + k < inputs.lengths[:, None]
        )  # (B, L)
        weights = torch.rand(b, l, device=inputs.device) * mask
        indices = weights.topk(n_indices, dim=1)[1].sort(dim=1)[0]  # (B, I).
        subset_lengths = (indices + k < inputs.lengths[:, None]).sum(1)  # B
        return indices.T, subset_lengths

    def select_subset(
        self,
        batch: GenBatch | PredBatch | torch.Tensor,
        indices: torch.Tensor,
        subset_lengths: torch.Tensor,
    ):
        """Select subset of features.

        Args:
            batch: Tensor or PaddedBatch with shape (L, B, *).
            indices: Sorted array of indices in the range [0, L - 1] to select with shape (I, B).

        Returns:
            Subset batch with shape (I, B, *).
        """
        I, B = indices.shape

        if isinstance(batch, torch.Tensor):
            return batch.take_along_dim(indices.reshape(I, B, 1), 0)

        for attr in ["time", "num_features", "cat_features"]:
            field = getattr(batch, attr)
            if isinstance(field, torch.Tensor):
                field = field.take_along_dim(
                    indices.reshape(I, B, *([1] * (field.ndim - 2))), 0
                )
            elif isinstance(field, dict):
                for k, v in field.items():
                    field[k] = v.take_along_dim(
                        indices.reshape(I, B, *([1] * (v.ndim - 2))), 0
                    )
            setattr(batch, attr, field)

        batch.lengths = subset_lengths
        return batch

    def match_targets(
        self, outputs: PredBatch, presence_scores: torch.Tensor, targets: GenBatch
    ):
        """Find closest prediction to each target.
        Args:
            outputs : (I, B, [K], D)
            targets : (I, B, [T + 1], D)
            The first value in each window is the current step, which is ignored during matching.
        Returns:
          - loss
        """
        assert outputs.shape == targets.shape
        assert (outputs.lengths == targets.lengths).all()
        valid_mask = targets.valid_mask
        V = valid_mask.sum()
        assert outputs.time.ndim == targets.time.ndim == 3
        K, T = outputs.time.shape[2], targets.time.shape[2] - 1
        assert presence_scores.shape == (*valid_mask.shape, K)
        losses = {}

        # 1. Compute cat
        for name in self.data_conf.focus_cat:
            pred = outputs.cat_features[name][valid_mask].movedim(-1, 1)  # V, C, K
            true = targets[name][valid_mask][:, None, 1:]  # V, 1, T
            C = pred.shape[1]
            pred = pred.unsqueeze(-1).expand(V, C, K, T)
            true = true.expand(V, K, T)
            ce_loss = F.cross_entropy(pred, true, reduction="none")
            losses[name] = ce_loss

        # 2. Compute num
        for name in self.data_conf.focus_num:
            if name == self.data_conf.time_name:
                pred = outputs.time[valid_mask]
                true = targets.time[valid_mask]
                true = true[:, 1:] - true[:, :1]  # delta from last seen
            else:
                pred = outputs[name][valid_mask]
                true = targets[name][valid_mask][:, 1:]  # V,[T]
            pred, true = pred.reshape(V, K, 1), true.reshape(V, 1, T)
            losses[name] = (pred - true).abs()

        # Final cost = sum[all](-log(1-o)) + sum[selected](-log(o) + log(1-o))
        # So in each cell l_bce = log(1-o) - log(o) = log(1/sigmoid - 1)
        # log(1/sigmoid - 1) = log(1 + exp(-z) - 1) = log(exp(-z)) = -z
        losses["_presence"] = -presence_scores[valid_mask].unsqueeze(-1)

        cost = torch.zeros((V, K, T), device=targets.device)
        for name in losses:
            cost += losses[name] * self.matching_weights.get(name, 1)

        if cost.isnan().any():
            return torch.tensor(float('nan'))
        assignment = batch_linear_assignment(cost).unsqueeze(-1)  # V, K, 1

        losses = list(losses.values())
        loss = sum(losses[1:], start=losses[0]) 
        loss = loss.take_along_dim(assignment.clip(min=0), -1)  # V, K, 1
        loss = loss[assignment >= 0].sum()  # V * T
        # Return sum[all](-log(1-o)) == -sum(logsigmoid(-z))
        leftover = -torch.nn.functional.logsigmoid(-presence_scores[valid_mask]).sum()
        final_loss = (loss + leftover) / V
        return final_loss


# MAKE SURE TO NOT CALCULATE PADDINGS!!! L, B, K --> NEED TO MAKE lengths - K!!!!!!!
# Delta clip(min=0) for generation

# Resort predictions in generation by time (since they generated in arbitrary order)

# For generation:
# # Convert delta time to time.
#       results.payload[self._timestamps_field] += inputs.payload[self._timestamps_field].unsqueeze(2)
