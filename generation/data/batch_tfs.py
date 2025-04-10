"""GenBatch transforms for data loading pipelines."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from generation.data.preprocess.quantile_transformer import (
    QuantileTransformerTorch,
)

from .data_types import GenBatch

logger = logging.getLogger(__name__)

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


@dataclass
class CutTargetSequence(BatchTransform):
    """Creates generation targets for each batch."""

    target_len: int
    """Len of target sequence."""

    def __call__(self, batch: GenBatch):
        assert (
            batch.lengths.min() > self.target_len
        ), "target_len is too big for this batch"
        target_batch = batch.tail(self.target_len)
        batch.target_time = target_batch.time
        batch.target_num_features = target_batch.num_features
        batch.target_cat_features = target_batch.cat_features

        batch.lengths = batch.lengths - self.target_len
        mask = torch.arange(batch.time.shape[0])[:, None] < batch.lengths  # [L, B]
        new_max_len = batch.lengths.max()

        batch.time = (batch.time * mask)[:new_max_len]
        if batch.num_features:
            batch.num_features = (batch.num_features * mask.unsqueeze(-1))[:new_max_len]
        if batch.cat_features:
            batch.cat_features = (batch.cat_features * mask.unsqueeze(-1))[:new_max_len]

    def reverse(self, batch: GenBatch):
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

    def __call__(self, batch: GenBatch):
        assert isinstance(batch.time, torch.Tensor)
        batch.time = batch.time.float()
        batch.time.sub_(self.loc).div_(self.scale)

    def reverse(self, batch: GenBatch):
        batch.time.mul_(self.scale).add_(self.loc)


@dataclass
class TimeToDiff(BatchTransform):
    """Applies diff transform to time."""

    def __call__(self, batch: GenBatch):
        assert isinstance(batch.time, torch.Tensor)
        _, B = batch.time.shape
        batch.time = batch.time.diff(dim=0, prepend=torch.zeros(1, B))
        batch.monotonic_time = False

    def reverse(self, batch: GenBatch):
        if (batch.time < 0).any():
            logger.warning("Incorrect diffed time. Result will be non monotonic.")
        batch.time = batch.time.cumsum(0)
        batch.monotonic_time = True


@dataclass
class TimeToFeatures(NewFeatureTransform):
    """Add time to numerical features.

    To apply this transform first cast time to Tensor.
    Has to be applied BEFORE mask creation. And AFTER DatetoTime
    """

    process_type: Literal["cat", "diff", "none"] = "none"
    """
    How to add time to features. The options are:

    - ``"cat"`` --- add absolute time to other numerical features,
    - ``"diff"`` --- add time intervals between sequential events. In this case the
      first interval in a sequence equals zero.
    - ``"none"`` --- do not add time to features. This option is added for the ease of
      optuna usage.
    """
    time_name: str = "time"
    """Name of new feature with time, default ``"time"``."""

    @property
    def num_names(self):
        return [self.time_name]

    def __call__(self, batch: GenBatch):
        assert self.process_type in [
            "diff",
            "cat",
            "none",
        ], "time_process may only be cat|diff|none"
        assert isinstance(batch.time, torch.Tensor)
        if self.process_type == "none":
            return
        t = batch.time[..., None].clone()
        if self.process_type == "diff":
            assert batch.monotonic_time
            t = t.diff(dim=0, prepend=t[[0]])

        if batch.num_features_names is None:
            batch.num_features_names = [self.time_name]
            assert batch.num_features is None
            batch.num_features = t
            return

        assert batch.num_features is not None
        batch.num_features_names.append(self.time_name)
        batch.num_features = torch.cat((batch.num_features, t), dim=2)

    def reverse(self, batch: GenBatch):
        assert isinstance(batch.time, torch.Tensor)
        # Don't do anything if no numerical features or if no time feature.
        if (not batch.num_features_names) or (
            self.time_name not in batch.num_features_names
        ):
            return
        # If only time feature present, set None.
        if len(batch.num_features_names) == 1:
            batch.num_features = None
            batch.num_features_names = None
            return
        # If other feature also present, remove time feature.
        time_index = batch.num_features_names.index(self.time_name)
        batch.num_features = batch.num_features[
            :, :, torch.arange(batch.num_features.size(2)) != time_index
        ]
        batch.num_features_names.pop(time_index)


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
        for name in self.names:
            batch[name] = torch.log1p(torch.abs(batch[name])) * torch.sign(batch[name])


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
