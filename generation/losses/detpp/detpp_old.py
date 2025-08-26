from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch

from .padded_batch import PaddedBatch
from .utils import ScaleGradient, module_mode

PRESENCE = "_presence"
PRESENCE_PROB = "_presence_prob"
LABELS_LOGITS = "_labels_logits"


class DeTPPLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        prefetch_factor: float = 1,
        loss_subset: float = 0.25,
        drop_partial_windows="calibration",

    ):
        super().__init__()
        self._k = data_conf.generation_len
        self._prefetch_k = int(round(self._k * prefetch_factor))
        self._loss_subset = loss_subset
        self._drop_partial_windows = drop_partial_windows

        self.data_conf = data_conf


    def update_calibration_statistics(self, matching, presence_logits):
        """Update calibration statistics.

        The method uses exponential smoothing to track head matching frequencies.
        These frequencies are used to choose the optimal presence threshold during inference.

        Args:
            matching: Loss matching with shape (B, L1, K).
            presence_logits: Predicted presence logits with shape (B, L2, K).
        """
        # (B, L1, K), (B, L2, K).

        matching = matching.payload[matching.seq_len_mask]  # (V, K).
        if len(matching) > 0:
            means = (matching >= 0).float().mean(0)  # (K).
            matching_priors = (
                self._matching_priors * (1 - self._momentum) + means * self._momentum
            )
        else:
            matching_priors = self._matching_priors.clone()

        presence_logits = presence_logits.payload[
            presence_logits.seq_len_mask
        ]  # (V, K).
        if len(presence_logits) > 0:
            presence_logits = torch.sort(presence_logits, dim=0)[0]  # (V, K).
            indices = (1 - self._matching_priors) * len(presence_logits)
            bottom_indices = (
                indices.floor().long().clip(max=len(presence_logits) - 1)
            )  # (K).
            up_indices = (
                indices.ceil().long().clip(max=len(presence_logits) - 1)
            )  # (K).
            bottom_quantiles = presence_logits.take_along_dim(
                bottom_indices[None], 0
            ).squeeze(
                0
            )  # (K).
            up_quantiles = presence_logits.take_along_dim(up_indices[None], 0).squeeze(
                0
            )  # (K).
            quantiles = 0.5 * (bottom_quantiles + up_quantiles)
            matching_thresholds = (
                self._matching_thresholds * (1 - self._momentum)
                + quantiles * self._momentum
            )
        else:
            matching_thresholds = self._matching_thresholds.clone()

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(
                matching_priors, torch.distributed.ReduceOp.SUM
            )
            matching_priors /= world_size
            assert (matching_priors <= 1).all(), "Distributed reduction failed."
            torch.distributed.all_reduce(
                matching_thresholds, torch.distributed.ReduceOp.SUM
            )
            matching_thresholds /= world_size

        self._matching_priors.copy_(matching_priors)
        self._matching_thresholds.copy_(matching_thresholds)

    @property
    def num_events(self):
        return self._k

    def forward(self, inputs: GenBatch, outputs: PredBatch, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Predicted values with shape (B, L, P).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        indices, matching, losses, matching_metrics = self.get_subset_matching(
            inputs, outputs
        )
        # (B, I), (B, I, K), (B, I, K, T), dict.

        # Update statistics.
        if self.training:
            with torch.no_grad():
                b, l = outputs.shape
                reshaped_outputs = PaddedBatch(
                    outputs.payload.reshape(-1, self._k, self._next_item.input_size),
                    torch.full(
                        [b * l], self._k, dtype=torch.long, device=outputs.device
                    ),
                )  # (BL, K, P).
                reshaped_states = (
                    states.flatten(1, 2).unsqueeze(2) if states is not None else None
                )  # (N, BL, 1, D).
                with module_mode(self, training=False):
                    presence_logits = self._next_item.predict_next(
                        reshaped_outputs,
                        reshaped_states,
                        fields=set(),
                        logits_fields_mapping={PRESENCE: "_presence_logits"},
                    ).payload[
                        "_presence_logits"
                    ]  # (BL, K, 1).
                presence_logits = presence_logits.reshape(b, l, self._k)  # (B, L, K).
                if self._drop_partial_windows in {True, "calibration"}:
                    full_matching = PaddedBatch(
                        matching.payload, indices.payload["full_mask"].sum(1)
                    )
                    presence_logits = PaddedBatch(
                        presence_logits,
                        (outputs.seq_lens - self._prefetch_k).clip(min=0),
                    )
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

        losses = {
            k: v.take_along_dim(matching.unsqueeze(3), 3).squeeze(3)
            for k, v in losses.items()
        }  # (B, I, K).

        pos_presence = losses.pop("_presence")
        neg_presence = -losses.pop("_presence_neg")
        presence_losses = torch.where(matching_mask, pos_presence, neg_presence)

        losses = {k: v[matching_mask].mean() for k, v in losses.items()}
        losses["_presence"] = presence_losses[index_mask].mean()

        # Compute next-item loss.
        if any(weight > 0 for weight in self._next_item_loss_weight.values()):
            predictions = self.predict_next(
                outputs,
                states,
                fields=self.data_fields,
                logits_fields_mapping={
                    k: f"_{k}_logits" for k in self._categorical_fields
                },
            )  # (B, L).
            # A workaround for "start" time delta scheme in the next-item loss function.
            fixed_predictions = {}
            for field in self.data_fields:
                if field in self._categorical_fields:
                    fixed_predictions[field] = (
                        predictions.payload[f"_{field}_logits"][:, :-1]
                        .flatten(0, 1)
                        .unsqueeze(1)
                    )  # (BL, 1, C).
                else:
                    fixed_predictions[field] = predictions.payload[field][
                        :, :-1
                    ].flatten()[
                        :, None, None
                    ]  # (BL, 1, 1).
            b, l = inputs.shape
            fixed_times = inputs.payload[self._timestamps_field]  # (B, L).
            fixed_times = torch.stack(
                [fixed_times[:, :-1], fixed_times[:, 1:]], 2
            ).flatten(
                0, 1
            )  # (BL, 2).
            fixed_inputs = {}
            for field in self.data_fields:
                if field == self._timestamps_field:
                    fixed_inputs[field] = fixed_times
                else:
                    fixed_inputs[field] = (
                        inputs.payload[field][:, 1:]
                        .flatten(0, 1)
                        .unsqueeze(1)
                        .repeat(1, 2)
                    )
            fixed_inputs = PaddedBatch(
                fixed_inputs, torch.full([b * (l - 1)], 2, device=inputs.device)
            )  # (BL, 2).
            fixed_states = (
                states[:, :, :-1].flatten(1, 2).unsqueeze(2)
                if states is not None
                else None
            )  # (N, BL, 1, D).

            next_item_losses, _ = self._next_item(
                fixed_inputs, fixed_predictions, fixed_states, reduction="none"
            )  # (BL, 1).
            mask = inputs.seq_len_mask[:, 1:].flatten()  # (BL).
            next_item_losses = {k: v[mask].mean() for k, v in next_item_losses.items()}
            for field in self.data_fields:
                losses[f"next_item_{field}"] = ScaleGradient.apply(
                    next_item_losses[field], self._next_item_loss_weight[field]
                )
        return losses, matching_metrics
