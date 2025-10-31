"""GenBatch transforms for data loading pipelines."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import pickle

try:
    from torch_linear_assignment import batch_linear_assignment
    CPU_ASSIGNMENT = False
except Exception:
    print("Using slow linear assignment implementation")
    CPU_ASSIGNMENT = True
    from ..losses.utils import batch_linear_assignment

import numpy as np
import torch

from generation.data.preprocess.quantile_transformer import (
    QuantileTransformerTorch,
)

from ..utils import RateLimitFilter
from .data_types import GenBatch

logger = logging.getLogger(__name__)
logger.addFilter(RateLimitFilter(60))

MISSING_CAT_VAL = 0


class BatchTransform(ABC):
    """Base class for all batch transforms.

    The BatchTransform is a Callable object that modifies GenBatch in-place.
    """

    @abstractmethod
    def __call__(self, batch: GenBatch):
        """Apply transform to the batch."""
        ...

    @abstractmethod
    def reverse(self, batch: GenBatch): ...


@dataclass
class NewFeatureTransform(BatchTransform):
    """Does not modify old features. Only adds new ones. Always adds on the right (torch.cat(tensor, new_feature, dim=-1))"""

    @property
    def num_names(self) -> list[str] | None:
        return []

    @property
    def cat_cardinalities(self) -> list[str] | None:
        return {}

    @property
    def num_names_removed(self) -> list[str] | None:
        return []

    @property
    def cat_names_removed(self) -> list[str] | None:
        return []

    def new_focus_on(self, focus_on) -> list[str]:
        return focus_on


class Identity(NewFeatureTransform):
    def __call__(self, batch: GenBatch):
        pass

    def reverse(self, batch: GenBatch):
        pass


@dataclass
class ShuffleBatch(BatchTransform):
    """Randomly permutes events in a sequence batch, optionally preserving the last `keep_last_n` events.
    Attributes:
        keep_last_n (int): Number of events at the end of each sequence to preserve in place.
        -1 means no shuffle
    """

    untouched_slice: list[int | None] = field(default_factory=lambda: [None, None])

    def __post_init__(self):
        assert len(self.untouched_slice) == 2
        assert isinstance(self.untouched_slice, list)
        a, b = self.untouched_slice
        assert None in self.untouched_slice, "At least one element must be None"
        assert (b is None) or (b < 0)
        assert (a is None) or (a <= 0)

    def __call__(self, batch: GenBatch):
        if self.untouched_slice == [None, None]:
            return
        max_len = batch.time.shape[0]
        bs = len(batch)
        i_len = torch.arange(max_len, device=batch.lengths.device)[:, None]  # [L, 1]
        i_batch = torch.arange(bs)  # [B]

        up_len, bot_len = batch.lengths, torch.zeros_like(batch.lengths)

        if self.untouched_slice[1] is None:
            up_len = batch.lengths + self.untouched_slice[0]
            bot_len = torch.zeros_like(batch.lengths)
            assert (up_len > 0).all(), up_len
        else:
            up_len = batch.lengths
            bot_len = batch.lengths + self.untouched_slice[1]
            assert (bot_len >= 0).all(), bot_len

        valid = (i_len >= bot_len) & (i_len < up_len)  # [L, B]
        permutation_within_len = torch.multinomial(valid.float().T, max_len).T  # [L, B]
        t_values = i_len.expand(-1, bs).clone()
        if self.untouched_slice[1] is None:
            permutation_within_len = torch.where(
                valid, permutation_within_len, t_values
            )
        else:
            t_values.T[valid.T] = permutation_within_len[
                : -self.untouched_slice[1]
            ].T.ravel()
            permutation_within_len = t_values

        if batch.cat_features is not None:
            batch.cat_features = batch.cat_features[permutation_within_len, i_batch]

        if batch.num_features is not None:
            batch.num_features = batch.num_features[permutation_within_len, i_batch]

        if batch.cat_mask is not None:
            batch.cat_mask = batch.cat_mask[permutation_within_len, i_batch]

        if batch.num_mask is not None:
            batch.num_mask = batch.num_mask[permutation_within_len, i_batch]

        if not batch.monotonic_time:
            batch.time = batch.time[permutation_within_len, i_batch]

    def reverse(self, batch: GenBatch):
        pass


@dataclass
class LocalShuffle(BatchTransform):
    """Randomly permutes events in a sequence batch, allows up to max_shift from initial position
    Attributes:
        max_shift (int): allowed shift from initial position
    """

    max_shift: int = 0
    independent_tail: int = 0

    def __call__(self, batch: GenBatch):
        if self.max_shift == 0:
            return
        L, B = batch.time.shape
        # Step 1: reinforce max_shift
        cost = torch.randn(B, L, L, device=batch.time.device)  # [B, L, L]
        if self.max_shift >= 0:
            i_indices = torch.arange(L, device=cost.device)[:, None]  # L, 1
            j_indices = torch.arange(L, device=cost.device)
            distance_from_diagonal = torch.abs(i_indices - j_indices)  # L, L
            mask_outside_band = distance_from_diagonal > self.max_shift
            cost.masked_fill_(mask_outside_band, torch.inf)

        # Step 2: Dont use padding tokens
        mask = torch.arange(L, device=batch.time.device) >= batch.lengths[:, None]
        mask = mask[:, None] | mask[:, :, None]
        mask[:, torch.arange(L), torch.arange(L)] = False
        cost[mask] = torch.inf
        
        # Step 3: Make independent shuffle for tail
        if self.independent_tail > 0:
            hist = torch.arange(L, device=batch.time.device) < (
                batch.lengths[:, None] - self.independent_tail
            ) # B, L
            target = torch.arange(L, device=batch.time.device) >= (
                batch.lengths[:, None] - self.independent_tail
            ) # B, L
            intersection = hist[:, None] & target[:, :, None]
            intersection = intersection | intersection.transpose(-1, -2)
            cost[intersection] = torch.inf
        
        # Step 3: permute everything
        if CPU_ASSIGNMENT:
            cost = cost.detach().cpu()
        assignment = batch_linear_assignment(cost).T  # L, B

        def reorder_tensor(tensor, order):
            if tensor.ndim == 3:
                order = order.unsqueeze(-1)
            return tensor.gather(0, order.expand(tensor.shape))

        if batch.cat_features is not None:
            batch.cat_features = reorder_tensor(batch.cat_features, assignment)

        if batch.num_features is not None:
            batch.num_features = reorder_tensor(batch.num_features, assignment)

        if batch.cat_mask is not None:
            batch.cat_mask = reorder_tensor(batch.cat_mask, assignment)

        if batch.num_mask is not None:
            batch.num_mask = reorder_tensor(batch.num_mask, assignment)

        if not batch.monotonic_time:
            batch.time = reorder_tensor(batch.time, assignment)

    def reverse(self, batch: GenBatch):
        pass


@dataclass
class CutTargetSequence(BatchTransform):
    """Creates generation targets for each batch."""

    target_len: int
    """Len of target sequence."""

    def __call__(self, batch: GenBatch):
        assert (
            batch.lengths.min() > self.target_len
        ), f"target_len is too big for this batch. Current min length = {batch.lengths.min()}"
        if self.target_len == 0:
            return
        target_batch = batch.tail(self.target_len)
        batch.target_time = target_batch.time
        batch.target_num_features = target_batch.num_features
        batch.target_cat_features = target_batch.cat_features

        batch.lengths = batch.lengths - self.target_len
        mask = torch.arange(batch.time.shape[0])[:, None] < batch.lengths  # [L, B]
        new_max_len = batch.lengths.max()

        batch.time = (batch.time * mask)[:new_max_len]
        if batch.num_features is not None:
            batch.num_features = (batch.num_features * mask.unsqueeze(-1))[:new_max_len]
        if batch.cat_features is not None:
            batch.cat_features = (batch.cat_features * mask.unsqueeze(-1))[:new_max_len]

    def reverse(self, batch: GenBatch):
        if self.target_len == 0:
            return
        batch.append(batch.get_target_batch())

        batch.target_time = None
        batch.target_num_features = None
        batch.target_cat_features = None


@dataclass
class RescaleTime(BatchTransform):
    """Rescale time: subtract location and divide by scale."""

    loc: float
    """Location to subtract from time."""
    scale: float
    """Scale to divide time by."""
    disable: False
    def __call__(self, batch: GenBatch):
        if self.disable:
            return
        assert isinstance(batch.time, torch.Tensor)
        batch.time = batch.time.float()
        batch.time.sub_(self.loc).div_(self.scale)

    def reverse(self, batch: GenBatch):
        if self.disable:
            return
        batch.time.mul_(self.scale).add_(self.loc)


@dataclass
class VariableRangeDecimal(NewFeatureTransform):
    """From https://www.arxiv.org/abs/2504.07566"""

    name: str
    n: int = 4
    smallest_magnitude: int = -5
    biggest_magnitude: int = 13

    @property
    def cat_cardinalities(self):
        assert self.biggest_magnitude > self.smallest_magnitude
        new_cats = {}
        new_cats[f"{self.name}_mag"] = (
            self.biggest_magnitude - self.smallest_magnitude + 1
        )
        for i in range(1, self.n + 1):
            new_cats[f"{self.name}_{i}"] = 10
        return new_cats

    @property
    def num_names_removed(self) -> list[str] | None:
        return [self.name]

    def __call__(self, batch: GenBatch):
        assert self.name in batch.num_features_names
        feature_id = batch.num_features_names.index(self.name)
        feature = batch.num_features[:, :, feature_id]  # L, B
        assert (feature >= 0).all(), "Dont support negative values"
        original_magnitude = (
            torch.floor(torch.log10(feature.abs() + 1e-12))
            .clamp(self.smallest_magnitude, self.biggest_magnitude)
            .to(torch.long)
        )
        adjusted_magnitude = original_magnitude - self.smallest_magnitude

        x_normalized = feature / (10.0 ** original_magnitude.float())
        # Extract digits
        digits = []
        remainder = x_normalized
        for _ in range(self.n):
            digit = remainder.floor().to(torch.long)
            digits.append(digit)
            remainder = (remainder - digit) * 10

        # Prepare categorical features
        mag_feature = adjusted_magnitude.unsqueeze(-1)
        digit_features = [d.unsqueeze(-1) for d in digits]
        new_cat_features = torch.cat([mag_feature] + digit_features, dim=-1)

        # Update batch.cat_features
        if batch.cat_features is None:
            batch.cat_features = new_cat_features
            batch.cat_features_names = list(self.cat_cardinalities)
        else:
            batch.cat_features = torch.cat(
                [batch.cat_features, new_cat_features], dim=-1
            )
            batch.cat_features_names += list(self.cat_cardinalities)

        # Remove numerical feature
        if len(batch.num_features_names) == 1:
            batch.num_features_names = None
            batch.num_features = None
        else:
            remaining_ids = [
                i for i in range(len(batch.num_features_names)) if i != feature_id
            ]
            batch.num_features = batch.num_features[remaining_ids]
            batch.num_features_names = [
                n for n in batch.num_features_names if n != self.name
            ]

    def reverse(self, batch: GenBatch):
        # Get the names of the added categorical features
        all_names = list(self.cat_cardinalities)

        # Check if these names are present in the batch's cat features
        if batch.cat_features_names is None or not set(all_names).issubset(
            batch.cat_features_names
        ):
            raise ValueError("Categorical features to reverse not found in batch")

        # Get indices of the names to remove
        new_indices = [batch.cat_features_names.index(name) for name in all_names]
        cat_features = batch.cat_features[..., new_indices]

        # Remove these features from batch
        remaining_names = [
            name for name in batch.cat_features_names if name not in all_names
        ]
        remaining_indices = [
            i for i in range(len(batch.cat_features_names)) if i not in new_indices
        ]
        remaining_cat_features = batch.cat_features[..., remaining_indices]

        # Update batch's cat features and names
        batch.cat_features = remaining_cat_features
        batch.cat_features_names = remaining_names
        if len(remaining_names) == 0:
            batch.cat_features = None
            batch.cat_features_names = None

        # Reconstruct the original numerical feature
        adjusted_magnitude = cat_features[..., 0]
        digits = [cat_features[..., i] for i in range(1, self.n + 1)]

        # Convert adjusted_magnitude back to original_magnitude
        original_magnitude = adjusted_magnitude + self.smallest_magnitude

        # Reconstruct x_normalized
        x_normalized = torch.zeros_like(adjusted_magnitude, dtype=torch.float32)
        for i in range(self.n):
            x_normalized += digits[i].float() * (10.0 ** (-i))

        # Compute original feature
        original_feature = x_normalized * (10.0 ** original_magnitude.float())

        # Add back to num_features
        if batch.num_features is None:
            batch.num_features = original_feature.unsqueeze(-1)
            batch.num_features_names = [self.name]
        else:
            batch.num_features = torch.cat(
                [batch.num_features, original_feature.unsqueeze(-1)], dim=-1
            )
            batch.num_features_names.append(self.name)


@dataclass
class TimeToDiff(BatchTransform):
    """Applies diff transform to time."""

    disable: bool = False

    def __call__(self, batch: GenBatch):
        if self.disable:
            return
        assert isinstance(batch.time, torch.Tensor)
        _, B = batch.time.shape
        batch.time = batch.time.diff(
            dim=0, prepend=torch.zeros(1, B, device=batch.time.device)
        )
        batch.monotonic_time = False

    def reverse(self, batch: GenBatch):
        if self.disable:
            return
        if (batch.time < 0).any():
            logger.warning("Incorrect diffed time. Result will be non monotonic.")
        batch.time = batch.time.cumsum(0)
        batch.monotonic_time = True


@dataclass
class DatetimeToFloat(BatchTransform):
    """Cast time from np.datetime64 to float by rescale.
    scale:
    """

    loc: str | np.datetime64
    """
    Location to subtract. If string is passed, it is converted to ``np.datetime64``
    beforehand.
    """
    scale: tuple[int, str] | np.timedelta64
    """
    Scale to divide time by. If tuple is passed, it is passed to the ``np.timedelta64``
    function. The first item is a value and the second is a unit.
    """

    def __post_init__(self):
        if isinstance(self.loc, str):
            self.loc = np.datetime64(self.loc)
        if isinstance(self.scale, Sequence):
            self.scale = np.timedelta64(*self.scale)

    def __call__(self, batch: GenBatch):
        assert isinstance(batch.time, np.ndarray)
        assert isinstance(self.loc, np.datetime64)
        assert isinstance(self.scale, np.timedelta64)
        batch.time = torch.asarray(
            (batch.time - self.loc) / self.scale, dtype=torch.float32
        )


@dataclass
class Logarithm(BatchTransform):
    """Apply natural logarithm to specific feature."""

    names: list[str]
    """Feature names to transform by taking the logarithm."""

    def __call__(self, batch: GenBatch):
        num_names = batch.num_features_names or []
        for name in self.names:
            if name in num_names:
                batch[name] = torch.log1p(torch.abs(batch[name])) * torch.sign(
                    batch[name]
                )

    def reverse(self, batch: GenBatch):
        num_names = batch.num_features_names or []
        for name in self.names:
            if name in num_names:
                x = batch[name].clamp(-88, 88)  # To prevent overflow
                batch[name] = torch.expm1(torch.abs(x)) * torch.sign(x)


@dataclass
class Rescale(BatchTransform):
    """Rescale feature: subtract location and divide by scale."""
   
    name: str
    """Feature name."""
    loc: Any
    """Value to subtract from the feature values."""
    scale: Any
    """Value to divide by the feature values."""
    def __call__(self, batch: GenBatch):
        batch[self.name].sub_(self.loc).div_(self.scale)

    def reverse(self, batch: GenBatch):
        batch[self.name].mul_(self.scale).add_(self.loc)


@dataclass
class ForwardFillNans(BatchTransform):
    """Fill NaN values by propagating forwad lase non-nan values.

    The algoritm starts from the second step. If some values are NaNs, the values from
    the prevoius step are used to fill them. If the first time step contains NaNs, some
    NaNs will not be filled after the forward pass. To handle it ``backward=True`` might
    be specified to fill remaining NaN values from last to first after the forwad pass.
    But even after a backward pass the batch may contain NaNs, if some feature has all
    NaN values. To fill it use ``FillNans`` transform.
    """

    backward: bool = False
    """Wether to do backward fill after the forwad fill (see the class description)."""

    def __call__(self, batch: GenBatch):
        if batch.num_features is None:
            return
        if batch.num_features.shape[0] == 1:
            return

        for i in range(1, batch.num_features.shape[0]):
            batch.num_features[i] = torch.where(
                torch.isnan(batch.num_features[i]),
                batch.num_features[i - 1],
                batch.num_features[i],
            )

        if not self.backward:
            return

        for i in range(batch.num_features.shape[0] - 2, -1, -1):
            batch.num_features[i] = torch.where(
                torch.isnan(batch.num_features[i]),
                batch.num_features[i + 1],
                batch.num_features[i],
            )

    def reverse(self, batch):
        pass


@dataclass
class FillNans(BatchTransform):
    """Fill NaNs with specified values."""

    fill_value: Mapping[str, float] | float
    """
    If float, all NaNs in all numerical features will be replaced with the
    ``fill_value``. Mapping sets feature-specific replacement values.
    """

    def __call__(self, batch: GenBatch):
        if batch.num_features is None:
            return

        if isinstance(self.fill_value, float | int):
            batch.num_features.nan_to_num_(nan=self.fill_value)
            return

        for name, val in self.fill_value.items():
            batch[name].nan_to_num_(nan=val)

    def reverse(self, batch):
        pass


class ContrastiveTarget(BatchTransform):
    """Set target for contrastive losses.

    New target is LongTensor such that items with different indices have different
    target labels.
    """

    def __call__(self, batch: GenBatch):
        if batch.index is None:
            raise ValueError("GenBatch must contain index")

        index = (
            batch.index
            if isinstance(batch.index, np.ndarray)
            else batch.index.cpu().numpy()
        )
        idx_map = {idx: i for i, idx in enumerate(np.unique(index))}
        batch.target = torch.tensor([idx_map[idx] for idx in index])


class TargetToLong(BatchTransform):
    """
    Cast target to LongTensor
    """

    def __call__(self, batch: GenBatch):
        if batch.target is not None:
            batch.target = batch.target.long()


@dataclass
class RandomSlices(BatchTransform):
    """Sample random slices from input sequences.

    The transform is taken from https://github.com/dllllb/coles-paper. It samples random
    slices from initial sequences. The batch size after this transform will be
    ``split_count`` times larger.
    """

    split_count: int
    """How many sample slices to draw for each input sequence."""
    cnt_min: int
    """Minimal sample sequence length."""
    cnt_max: int
    """Maximal sample sequence length."""
    short_seq_crop_rate: float = 1.0
    """
    Must be from (0, 1]. If ``short_seq_crop_rate`` < 1, and if a
        sequence of length less than cnt_min is encountered, the mininum sample
        length for this sequence is set as a ``short_seq_crop_rate`` time the actual
        sequence length.
    """
    seed: int | None = None
    """Value to seed the random generator."""

    def __post_init__(self):
        self._gen = np.random.default_rng(self.seed)

    def __call__(self, batch: GenBatch):

        lens = []
        times = []
        nums = []
        cats = []
        inds = []
        targets = []
        max_len = 0

        def add_slice(i, start, length):
            assert length > 0
            end = start + length
            lens.append(length)
            times.append(batch.time[start:end, i])
            inds.append(batch.index[i])
            if batch.num_features is not None:
                nums.append(batch.num_features[start:end, i])
            if batch.cat_features is not None:
                cats.append(batch.cat_features[start:end, i])
            if batch.target is not None:
                targets.append(batch.target[i])

        for i in range(len(batch)):
            c_len = batch.lengths[i].item()
            assert isinstance(c_len, int)
            if c_len < self.cnt_min and self.short_seq_crop_rate >= 1.0:
                for _ in range(self.split_count):
                    add_slice(i, 0, c_len)
                continue

            cnt_max = min(self.cnt_max, c_len)
            cnt_min = self.cnt_min
            if (
                int(c_len * self.short_seq_crop_rate) <= self.cnt_min
                and self.short_seq_crop_rate < 1.0
            ):
                cnt_min = max(int(c_len * self.short_seq_crop_rate), 1)

            if cnt_max > cnt_min:
                new_len = self._gen.integers(cnt_min, cnt_max, size=self.split_count)
            else:
                new_len = np.full(self.split_count, cnt_min)

            max_len = max(max_len, *new_len)
            available_start_pos = (c_len - new_len).clip(0, None)
            start_pos = (
                self._gen.uniform(size=self.split_count)
                * (available_start_pos + 1 - 1e-9)
            ).astype(int)

            for sp, ln in zip(start_pos, new_len):
                add_slice(i, sp, ln)

        def cat_pad(tensors, dtype):
            t0 = tensors[0]
            res = torch.zeros(max_len, len(tensors), *t0.shape[1:], dtype=dtype)
            for i, ten in enumerate(tensors):
                res[: ten.shape[0], i] = ten
                res[ten.shape[0] :, i] = ten[-1]
            return res

        batch.lengths = torch.tensor(lens)
        if batch.target is not None:
            batch.target = torch.tensor(targets, dtype=batch.target.dtype)
        if isinstance(batch.index, torch.Tensor):
            batch.index = torch.tensor(inds, dtype=batch.index.dtype)
        else:  # np.ndarray
            batch.index = np.array(inds, dtype=batch.index.dtype)

        batch.time = cat_pad(times, batch.time.dtype)
        if batch.cat_features is not None:
            batch.cat_features = cat_pad(cats, batch.cat_features.dtype)
        if batch.num_features is not None:
            batch.num_features = cat_pad(nums, batch.num_features.dtype)


class MaskValid(BatchTransform):
    """Add mask indicating valid values to batch.

    Mask has shape (max_seq_len, batch_size, n_features) and has True values where there
    are non-NaN values (nonzero category) and where the data is not padded.
    """

    def __call__(self, batch: GenBatch):
        max_len = batch.lengths.amax().item()
        assert isinstance(max_len, int)
        len_mask = (torch.arange(max_len)[:, None] < batch.lengths)[..., None]

        if batch.num_features is not None:
            batch.num_mask = len_mask & ~torch.isnan(batch.num_features)

        if batch.cat_features is not None:
            batch.cat_mask = len_mask & (batch.cat_features != 0)


@dataclass
class QuantileTransform(BatchTransform):
    """Add quantile transform for the feature"""

    model_path: str
    feature_name: str

    def __post_init__(self):
        self.qt_model = QuantileTransformerTorch()
        self.qt_model.load(self.model_path)

    def __call__(self, batch: GenBatch):
        batch[self.feature_name] = self.qt_model.transform(batch[self.feature_name])

    def reverse(self, batch):
        batch[self.feature_name] = self.qt_model.inverse_transform(
            batch[self.feature_name]
        )


@dataclass
class NGramTransform(NewFeatureTransform):
    """Add quantile transform for the feature"""

    model_path: str
    feature_name: str
    feature_counts: int = 54
    ngram_counts: int = -1
    max_l: int = -1
    min_hist_len: int = 32
    gen_len: int = 32
    disable: bool = False

    @staticmethod
    def merge_ngrams(
        seq: np.ndarray, time: np.ndarray, ngram_map: dict[tuple[int, ...], int], n: int
    ) -> np.ndarray:
        out_seq, out_time, i = [], [], 0
        while i < len(seq):
            if i <= len(seq) - n and tuple(seq[i : i + n]) in ngram_map:
                out_seq.append(ngram_map[tuple(seq[i : i + n])])
                out_time.append(np.mean(time[i : i + n]))
                i += n
            else:
                out_seq.append(seq[i])
                out_time.append(time[i])
                i += 1
        return (np.asarray(out_seq, dtype=int), np.asarray(out_time, dtype=float))

    def decode_merged_sequence(
        self,
        merged: np.ndarray,
        averaged_time: np.ndarray,
        ngram_map: dict[tuple[int, ...], int],
        n_order: list[int],
    ) -> np.ndarray:
        restored_time = averaged_time
        restored_seq = merged

        for n in n_order:
            restored_seq, restored_time = self.demerge(
                restored_seq, restored_time, ngram_map, n
            )

        return np.asarray(restored_seq, dtype=int), np.asarray(
            restored_time, dtype=float
        )

    def demerge(
        self,
        merged: np.ndarray,
        averaged_time: np.ndarray,
        ngram_map: dict[tuple[int, ...], int],
        n: int,
    ):
        out_seq = []
        out_times = []
        inv = {v: k for k, v in ngram_map[f"{n}-grams"].items()}
        for i, t in enumerate(merged):
            if t in inv:
                out_seq.extend(inv[t])
                out_times.extend([averaged_time[i]] * len(inv[t]))
            else:
                out_seq.append(t)
                out_times.append(averaged_time[i])

        return out_seq, out_times

    def encode_sequence(
        self,
        seq: list[int],
        time: list[float],
        mapping_dict: dict[int, dict[tuple[int, ...], int]],
        n_order: list[int],
    ) -> np.ndarray:
        out_seq = np.asarray(seq, dtype=int)
        out_time = np.asarray(time, dtype=float)

        for n in n_order:
            out = self.merge_ngrams(out_seq, out_time, mapping_dict[f"{n}-grams"], n)
        return out

    def __post_init__(self):
        if not self.disable:
            self.min_len = self.min_hist_len + self.gen_len
            self.mapping = None
            with open(self.model_path, "rb") as file:
                self.mapping = pickle.load(file)

            self.ngram_counts = sum([len(el) for el in self.mapping.values()])

    @property
    def cat_cardinalities(self) -> list[str] | None:
        if self.disable:
            return {}

        assert self.ngram_counts != -1
        new_cats = {}
        new_cats[f"{self.feature_name}_merged"] = (
            self.feature_counts + self.ngram_counts + 1
        )
        return new_cats

    @property
    def cat_names_removed(self) -> list[str] | None:
        if self.disable:
            return []
        else:
            return [self.feature_name]

    def new_focus_on(self, focus_on):
        if self.disable:
            return focus_on

        new_focus_on = [i for i in focus_on if i != self.feature_name]
        new_focus_on.append(self.feature_name + "_merged")
        return new_focus_on

    def __call__(self, batch: GenBatch):
        if self.disable:
            return
        L, B = batch[self.feature_name].shape
        self.max_l = L
        feature_id = batch.cat_features_names.index(self.feature_name)
        new_cat_features = np.copy(batch[self.feature_name].cpu().numpy())
        new_times = np.copy(batch.time.cpu().numpy())
        for i in range(B):
            seq = batch.cat_features[:, i, feature_id].cpu().numpy()
            time = batch.time[:, i].cpu().numpy()
            seq_len = batch.lengths[i]
            seq = seq[:seq_len]
            time = time[:seq_len]

            coded_seq, averaged_time = self.encode_sequence(
                seq, time, mapping_dict=self.mapping, n_order=[3, 2]
            )
            assert time.shape == seq.shape
            new_seq_len = coded_seq.shape[0]
            new_seq = np.zeros((L,))
            new_time = np.zeros((L,))

            new_seq[:new_seq_len] = coded_seq
            new_time[:new_seq_len] = averaged_time

            new_cat_features[:, i] = new_seq
            new_times[:, i] = new_time
            batch.lengths[i] = new_seq_len

        max_len = batch.lengths.max()
        # remove old feature
        mask = torch.arange(batch.cat_features.shape[-1]) != feature_id
        batch.cat_features = batch.cat_features[:, :, mask]
        batch.cat_features_names.remove(self.feature_name)

        batch.cat_features = torch.cat(
            (batch.cat_features, torch.tensor(new_cat_features)[..., None]), dim=-1
        )
        batch.cat_features_names.append(f"{self.feature_name}_merged")

        batch.time = torch.tensor(new_times)[:max_len]
        if batch.num_features is not None:
            batch.num_features = batch.num_features[:max_len]
        batch.cat_features = batch.cat_features[:max_len]

        # mask = batch.lengths > self.min_len
        # batch.lengths = batch.lengths[mask]
        # batch.cat_features = batch.cat_features[:, mask, :]
        # batch.time = batch.time[:, mask]
        # batch.index = batch.index[mask]

    def reverse(self, batch):
        if self.disable:
            return
        new_feature_name = self.feature_name + "_merged"
        assert new_feature_name in self.cat_cardinalities
        _, B = batch[new_feature_name].shape
        L = self.max_l
        feature_id = batch.cat_features_names.index(new_feature_name)
        new_cat_features = np.zeros((L, B))
        new_times = np.zeros((L, B))

        target_cat_features = np.copy(batch.target_cat_features)
        target_times = np.copy(batch.target_time)
        for i in range(B):
            seq = batch.cat_features[:, i, feature_id].cpu().numpy()
            time = batch.time[:, i].cpu().numpy()

            target_seq = batch.target_cat_features[:, i, feature_id].cpu().numpy()
            target_time = batch.target_time[:, i].cpu().numpy()

            seq_len = batch.lengths[i]
            seq = seq[:seq_len]
            time = time[:seq_len]
            decoded_seq, decoded_time = self.decode_merged_sequence(
                seq, time, ngram_map=self.mapping, n_order=[2, 3]
            )
            decoded_target, decoded_target_time = self.decode_merged_sequence(
                target_seq, target_time, ngram_map=self.mapping, n_order=[2, 3]
            )
            assert time.shape == seq.shape
            new_seq_len = decoded_seq.shape[0]

            new_cat_features[:new_seq_len, i] = decoded_seq
            new_times[:new_seq_len, i] = decoded_time

            target_cat_features[:, i, feature_id] = decoded_target[: self.gen_len]
            target_times[:, i] = decoded_target_time[: self.gen_len]

            batch.lengths[i] = new_seq_len

        cat_feature_to_remove = self.feature_name + "_merged"

        batch.cat_features_names.index(cat_feature_to_remove)
        remaining_names = [
            name
            for name in batch.cat_features_names
            if name != self.feature_name + "_merged"
        ]
        new_indices = [
            batch.cat_features_names.index(name)
            for name in batch.cat_features_names
            if name == self.feature_name + "_merged"
        ]
        remaining_indices = [
            i for i in range(len(batch.cat_features_names)) if i not in new_indices
        ]
        remaining_cat_features = batch.cat_features[..., remaining_indices]

        # Update batch's cat features and names
        batch.cat_features = remaining_cat_features
        batch.cat_features_names = remaining_names
        if len(remaining_names) == 0:
            batch.cat_features = None
            batch.cat_features_names = None
        if batch.cat_features is None:
            batch.cat_features = torch.tensor(new_cat_features, dtype=float)[..., None]
            batch.time = torch.tensor(new_times, dtype=float)
            batch.cat_features_names = [f"{self.feature_name}"]
            batch.target_cat_features = torch.tensor(target_cat_features, dtype=float)
            batch.target_time = torch.tensor(target_times, dtype=float)

        else:
            if batch.num_features is not None:
                new_num_features = torch.zeros((L, B, len(batch.num_features_names)))
                new_num_features[: batch.num_features.shape[0], ...] = (
                    batch.num_features
                )
                batch.num_features = new_num_features
            batch.cat_features_names.append(f"{self.feature_name}")
            new_cat_features_tensor = torch.zeros(
                (L, B, len(batch.cat_features_names)), dtype=float
            )
            new_cat_features_tensor[: batch.cat_features.shape[0], :, :-1] = (
                batch.cat_features
            )
            new_cat_features_tensor[:, :, feature_id] = torch.tensor(
                new_cat_features, dtype=float
            )
            batch.cat_features = new_cat_features_tensor
            batch.time = torch.tensor(new_times, dtype=float)
            batch.target_cat_features = torch.tensor(target_cat_features, dtype=float)
            batch.target_time = torch.tensor(target_times, dtype=float)


@dataclass
class ShuffleUsers(BatchTransform):

    shuffle: bool = False

    def __call__(self, batch: GenBatch):
        return

    def reverse(self, batch: GenBatch):
        if self.shuffle:
            B = batch.index.size
            perm = torch.randperm(B)
            if batch.target_cat_features is not None:
                batch.target_cat_features = batch.target_cat_features[:, perm, :]
            if batch.target_num_features is not None:
                batch.target_num_features = batch.target_num_features[:, perm, :]
            if batch.target_time is not None:
                batch.target_time = batch.target_time[:, perm]


@dataclass
class HideFeaturesFromTrain(NewFeatureTransform):

    cat_features: Optional[list[str]] = None
    num_features: Optional[list[str]] = None

    @property
    def num_names_removed(self) -> list[str] | None:
        return self.num_features

    @property
    def cat_names_removed(self) -> list[str] | None:
        return self.cat_features

    def new_focus_on(self, focus_on) -> list[str]:
        return focus_on

    def __post_init__(
        self,
    ):
        self.source_batch = None

    def __call__(self, batch: GenBatch):
        if self.cat_features:
            ff_ids = [
                index
                for index, name in enumerate(batch.cat_features_names)
                if name not in self.cat_features
            ]
            batch.cat_features_names = [batch.cat_features_names[id] for id in ff_ids]
            batch.cat_features = batch.cat_features[:, :, ff_ids]
            if batch.cat_features_names.__len__() == 0:
                batch.cat_features = None
                batch.cat_features_names = None

        if self.num_features:
            ff_ids = [
                index
                for index, name in enumerate(batch.num_features_names)
                if name not in self.num_features
            ]
            batch.num_features_names = [batch.num_features_names[id] for id in ff_ids]
            batch.num_features = batch.num_features[:, :, ff_ids]
            if batch.num_features_names.__len__() == 0:
                batch.num_features = None
                batch.num_features_names = None

    def reverse(self, batch):
        pass
