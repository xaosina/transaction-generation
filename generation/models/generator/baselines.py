from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from ...data.data_types import GenBatch, LatentDataConfig, gather
from . import BaseGenerator, ModelConfig
from typing import Any

class GroundTruthGenerator(BaseGenerator):
    """To check that all preprocessing is fine. Get perfect baseline."""

    def forward(self, x: GenBatch):
        raise "No need to train a GroundTruthGenerator."

    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs: Any
    ) -> GenBatch:
        assert hist.target_time.shape[0] == gen_len
        gen_batch = deepcopy(hist)
        gen_batch.append(gen_batch.get_target_batch())

        gen_batch.target_time = None
        gen_batch.target_num_features = None
        gen_batch.target_cat_features = None

        if with_hist:
            return gen_batch  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return gen_batch.tail(gen_len)


class ModeGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs: Any) -> GenBatch:
        def get_mode(tensor, lengths, discrete=True):
            if tensor is None:
                return None
            shape = list(tensor.shape)
            shape[0] = 1
            B = shape[1]
            res = torch.empty(*shape, device=tensor.device, dtype=tensor.dtype)
            for b in range(B):
                valid_length = lengths[b]
                valid_tensor = tensor[:valid_length, b]
                if discrete:
                    mode_value, _ = torch.mode(valid_tensor, dim=0)
                    res[0, b] = mode_value
                else:
                    mean_value = torch.mean(valid_tensor, dim=0)
                    res[0, b] = mean_value
            new_shape = [1] * len(res.shape)
            new_shape[0] = gen_len
            return res.repeat(new_shape)  # [gen_len, B, D]

        assert isinstance(hist.time, torch.Tensor)
        assert not hist.monotonic_time, "Vpadlu delat dlya monotonic"
        hist = deepcopy(hist)
        gen_batch = replace(
            hist,
            lengths=torch.ones_like(hist.lengths) * gen_len,
            time=get_mode(hist.time, hist.lengths, False),
            num_features=get_mode(hist.num_features, hist.lengths, False),
            cat_features=get_mode(hist.cat_features, hist.lengths),
            cat_mask=get_mode(hist.cat_mask, hist.lengths),
            num_mask=get_mode(hist.num_mask, hist.lengths),
        )
        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch


class BaselineRepeater(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        self.shift = (model_config.params or {}).get("shift", 0)
        self.shuffle_batch = (model_config.params or {}).get("shuffle_batch", False)
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs: Any) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)
        hist = deepcopy(hist)
        gen_batch = hist.tail(gen_len, self.shift)
        if self.shuffle_batch:
            B = gen_batch.time.shape[1]
            shuffled_indices = torch.randperm(B)
            gen_batch.time = gen_batch.time[:, shuffled_indices]
            if gen_batch.num_features is not None:
                gen_batch.num_features = gen_batch.num_features[:, shuffled_indices]
            if gen_batch.cat_features is not None:
                gen_batch.cat_features = gen_batch.cat_features[:, shuffled_indices]

        if hist.monotonic_time:  # Time is monotonic.
            corr = torch.cat((torch.zeros_like(hist.time[:1]), hist.time))
            corr = corr[hist.lengths - gen_len, torch.arange(hist.time.shape[1])]
            gen_batch.time = gen_batch.time + gen_batch.time[-1] - corr
            # This complicated correction assures same behavior as with timediff
        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch


class PerfectRepeater(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        self.data_conf = data_conf
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs: Any) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)
        assert not hist.monotonic_time
        hist = deepcopy(hist)
        L, B = hist.time.shape
        data_conf = self.data_conf

        # Step 1: Calculate cost
        cost = torch.zeros(
            (L - gen_len + 1, B), device=hist.time.device, dtype=hist.time.dtype
        )
        # 1.1 Calculate R2 score
        num_names = (hist.num_features_names or []) + [data_conf.time_name]
        num_ids = [
            num_names.index(name) for name in num_names if name in data_conf.focus_on
        ]
        num_hist = hist.get_numerical()  # [L, B, Dn+1]
        num_true = hist.target_time.unsqueeze(-1)  # [L, B, 1]
        if num_ids:
            num_true = torch.cat(
                (hist.target_num_features, num_true), dim=2
            )  # [gen_len, B, D]

            res = num_hist.unfold(0, gen_len, 1)  # [L - gen_len + 1, B, D, gen_len]
            res = res.permute(0, 3, 1, 2)  # [L - gen_len + 1, gen_len, B, D]

            res = ((res[:, :, :, num_ids] - num_true[None, :, :, num_ids]) ** 2).sum(
                1
            )  # [L - gen_len + 1, B, D]

            userwise_mean = num_true.mean(dim=0)  # B, [D]
            tot = (num_true - userwise_mean) ** 2  # L, B, [D]
            userwise_tot = tot.sum(dim=0)  # B, [D]

            res = res / userwise_tot[None, :, :]  # Scaled mse, [L - gen_len + 1, B, D]
            cost += res.mean(2)  # [L - gen_len + 1, B] R2 score

        # 1.2 Calculate accuracy
        cat_names = hist.cat_features_names or []
        cat_ids = [
            cat_names.index(name) for name in cat_names if name in data_conf.focus_on
        ]
        cat_hist, cat_true = hist.cat_features, hist.target_cat_features
        if cat_ids:
            res = cat_hist.unfold(0, gen_len, 1)  # [L - gen_len + 1, B, D, gen_len]
            res = res.permute(0, 3, 1, 2)  # [L - gen_len + 1, gen_len, B, D]

            res = res[:, :, :, cat_ids] != cat_true[None, :, :, cat_ids]
            cost += res.to(torch.float32).mean(1).mean(-1)  # [L - gen_len + 1, B]

        # Step 2: gather best slices
        best_start = cost.argmin(0)  # [B]
        samples = (
            torch.arange(gen_len, device=cost.device)[:, None] + best_start
        )  # [gen_len, B]

        gen_batch = replace(
            hist,
            lengths=torch.ones_like(hist.lengths) * gen_len,
            time=gather(hist.time, samples),
            num_features=gather(hist.num_features, samples),
            cat_features=gather(hist.cat_features, samples),
            cat_mask=gather(hist.cat_mask, samples),
            num_mask=gather(hist.num_mask, samples),
        )

        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch


# Should be fixed res = ...
# class PerfectRepeaterF1(BaseGenerator):
#     def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
#         from torcheval.metrics.functional import multiclass_f1_score
#         self.data_conf = data_conf
#         super().__init__()

#     def forward(self, x: GenBatch):
#         raise "No need to train a repeator."

#     def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
#         assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
#         assert isinstance(hist.time, torch.Tensor)
#         assert not hist.monotonic_time
#         hist = deepcopy(hist)
#         L, B = hist.time.shape
#         data_conf = self.data_conf

#         # Step 1: Calculate cost
#         cost = torch.zeros(
#             (L - gen_len + 1, B), device=hist.time.device, dtype=hist.time.dtype
#         )

#         # 1.2 Calculate F1
#         cat_names = hist.cat_features_names or []
#         cat_ids = [
#             cat_names.index(name) for name in cat_names if name in data_conf.focus_on
#         ]
#         assert len(cat_ids) == 1
#         cat_hist, cat_true = hist.cat_features, hist.target_cat_features
#         if cat_ids:
#             res = cat_hist.unfold(0, gen_len, 1)  # [L - gen_len + 1, B, D, gen_len]
#             res = res.permute(0, 3, 1, 2)  # [L - gen_len + 1, gen_len, B, D]

#             for label in range(1, cat_true[:, :, cat_ids].max() + 1):


#             res = res[:, :, :, cat_ids] != cat_true[None, :, :, cat_ids]
#             cost += res.to(torch.float32).mean(1).mean(-1) # [L - gen_len + 1, B]

#         # Step 2: gather best slices
#         best_start = cost.argmin(0) # [B]
#         samples = torch.arange(gen_len, device=cost.device)[:, None] + best_start # [gen_len, B]

#         gen_batch = replace(
#             hist,
#             lengths=torch.ones_like(hist.lengths) * gen_len,
#             time=gather(hist.time, samples),
#             num_features=gather(hist.num_features, samples),
#             cat_features=gather(hist.cat_features, samples),
#             cat_mask=gather(hist.cat_mask, samples),
#             num_mask=gather(hist.num_mask, samples),
#         )

#         if with_hist:
#             hist.append(gen_batch)
#             return hist
#         else:
#             return gen_batch


class BaselineHistSampler(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs: Any) -> GenBatch:
        # assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)

        hist = deepcopy(hist)
        samples = torch.tensor(
            np.array(
                [
                    np.sort(np.random.choice(length, size=gen_len, replace=True))
                    for length in hist.lengths.numpy(force=True)
                ]
            ),
            device=hist.lengths.device,
        ).T  # [gen_len, B]
        gen_batch = replace(
            hist,
            lengths=torch.ones_like(hist.lengths) * gen_len,
            time=gather(hist.time, samples),
            num_features=gather(hist.num_features, samples),
            cat_features=gather(hist.cat_features, samples),
            cat_mask=gather(hist.cat_mask, samples),
            num_mask=gather(hist.num_mask, samples),
        )
        if hist.monotonic_time:  # Time is monotonic.
            corr = torch.cat((torch.zeros_like(hist.time[:1]), hist.time))
            pred_first_time = corr[samples[0], torch.arange(hist.time.shape[1])]
            last_time = hist.time[hist.lengths - 1, torch.arange(hist.time.shape[1])]
            gen_batch.time = gen_batch.time - pred_first_time + last_time
            # This complicated correction assures same behavior as with timediff
        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch
