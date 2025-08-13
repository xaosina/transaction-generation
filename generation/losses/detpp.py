import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch

from .utils import rse_valid


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


    def __call__(self, y_true, y_pred) -> dict:
        return {"loss": }


    def forward(self, inputs, outputs):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Predicted values with shape (B, L, P).

        Returns:
            Losses dict and metrics dict.
        """
        indices, matching, losses, matching_metrics = self.get_subset_matching(inputs, outputs)
        # (B, I), (B, I, K), (B, I, K, T), dict.

        # Update statistics.
        if self.training:
            with torch.no_grad():
                b, l = outputs.shape
                reshaped_outputs = PaddedBatch(outputs.payload.reshape(-1, self._k, self._next_item.input_size),
                                               torch.full([b * l], self._k, dtype=torch.long, device=outputs.device))  # (BL, K, P).
                with module_mode(self, training=False):
                    presence_logits = self._next_item.predict_next(
                        reshaped_outputs, None,
                        fields=set(),
                        logits_fields_mapping={PRESENCE: "_presence_logits"}
                    ).payload["_presence_logits"]  # (BL, K, 1).
                presence_logits = presence_logits.reshape(b, l, self._k)  # (B, L, K).
                if self._drop_partial_windows in {True, "calibration"}:
                    full_matching = PaddedBatch(matching.payload, indices.payload["full_mask"].sum(1))
                    presence_logits = PaddedBatch(presence_logits, (outputs.seq_lens - self._prefetch_k).clip(min=0))
                else:
                    full_matching = matching
                    presence_logits = PaddedBatch(presence_logits, outputs.seq_lens)
                self.update_calibration_statistics(full_matching, presence_logits)

        # Compute matching losses.
        if (matching.payload < 0).all():
            losses = {name: outputs.payload.mean() * 0 for name in self.fields}
            return losses, matching_metrics

        index_mask = indices.seq_len_mask  # (B, I).
        matching_mask = matching.payload >= 0  # (B, I, K).
        matching = matching.payload.clip(min=0)  # (B, I, K).

        losses = {k: v.take_along_dim(matching.unsqueeze(3), 3).squeeze(3)
                  for k, v in losses.items()}  # (B, I, K).

        pos_presence = losses.pop("_presence")
        neg_presence = -losses.pop("_presence_neg")
        presence_losses = torch.where(matching_mask, pos_presence, neg_presence)

        losses = {k: v[matching_mask].mean() for k, v in losses.items()}
        losses["_presence"] = presence_losses[index_mask].mean()

        
        return losses, matching_metrics

    def get_subset_matching(self, inputs, outputs):
        """Apply stride and compute matching.

        Args:
            inputs: Model input features with shape (B, L).
            outputs: Model outputs model output features with shape (B, L, D).

        Returns:
            A tuple of:
                - indices of last seen event with shape (B, I).
                - relative matching with shape (B, I, K).
                - losses with shape (B, I, K, T).
                - metrics dictionary.
        """
        l, b = inputs.shape
        target_windows = self.extract_structured_windows(inputs)  # (B, L, k + 1), where first event is an input for the model.
        assert target_windows.shape == (b, l)

        # Reshape outputs.
        outputs = PaddedBatch(outputs.payload.reshape(b, l, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (B, L, K, P).
        assert (target_windows.seq_lens == outputs.seq_lens).all()

        # Subset outputs and targets.
        indices = self.get_loss_indices(inputs)
        target_windows = self.select_subset(target_windows, indices)  # (B, I, K + 1).
        outputs = self.select_subset(outputs, indices)  # (B, I, K, P).

        # Compute matching and return.
        l = outputs.shape[1]
        n_targets = target_windows.payload[target_windows.seq_names[0]].shape[2] - 1  # K.
        if (l == 0) or (n_targets == 0):
            matching = PaddedBatch(torch.full([b, l, self._k], -1, dtype=torch.long, device=inputs.device),
                                   target_windows.seq_lens)
            return indices, matching, {}, {}

        matching, losses, metrics = self.match_targets(
            outputs, target_windows
        ) # (B, I, K) with indices in the range [-1, L - 1].

        return indices, matching, losses, metrics
    
    def extract_structured_windows(self, inputs):
        """Extract windows with shape (B, L, k + 1) from inputs with shape (B, L)."""
        # Join targets before windowing.
        l, b = inputs.shape
        device = inputs.device
        fields = list(sorted(set(self.fields) - {"_presence"}))
        inputs, lengths, length_mask = dict(inputs.payload), inputs.seq_lens, inputs.seq_len_mask

        # Pad events with out-of-horizon.
        inf_ts = inputs.time[length_mask].max().item() + self._horizon + 1
        inputs[self._timestamps_field].masked_fill_(~length_mask, inf_ts)

        # Extract windows.
        k = self._prefetch_k
        joined = torch.stack([inputs[name] for name in fields], -1)  # (B, L, N).
        d = joined.shape[-1]
        parts = [joined.roll(-i, 1) for i in range(k + 1)]
        joined_windows = torch.stack(parts, 2)  # (B, L, k + 1, N).
        assert joined_windows.shape[:3] == (b, l, k + 1)

        # Split.
        windows = {}
        for i, name in enumerate(fields):
            windows[name] = joined_windows[..., i].to(inputs[name].dtype)  # (B, L, k + 1).

        # Pad partial windows with out-of-horizon.
        mask = torch.arange(l, device=device)[:, None] + torch.arange(k + 1, device=device) >= l  # (L, k + 1)
        windows[self._timestamps_field].masked_fill_(mask[None], inf_ts)

        return PaddedBatch(windows, lengths)