from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch
from . import BaseGenerator
from ...data.data_types import GenBatch, gather


class GroundTruthGenerator(BaseGenerator):
    """To check that all preprocessing is fine. Get perfect baseline."""

    def forward(self, x: GenBatch):
        raise "No need to train a GroundTruthGenerator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
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

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
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
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)
        hist = deepcopy(hist)
        gen_batch = hist.tail(gen_len)
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


class BaselineHistSampler(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
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
