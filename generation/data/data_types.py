import logging
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np
import torch
from ebes.types import Batch

from ..utils import RateLimitFilter

logger = logging.getLogger(__name__)  # noqa: F821
logger.addFilter(RateLimitFilter(60))


@dataclass(frozen=True)
class DataConfig:
    train_path: str = ""
    test_path: str = ""
    batch_size: int = 128
    num_workers: int = 4
    val_ratio: float = 0.15
    train_resamples: int = 1
    #
    max_seq_len: int = 0
    min_history_len: int = 16
    generation_len: int = 16
    train_random_end: str = "none"  # Literal["index", "time", "none"] = "none"
    val_random_end: str = "none"  # Literal["index", "time", "none"] = "none"
    #
    time_name: str = "Time"
    cat_cardinalities: Mapping[str, int] | None = None
    num_names: Optional[list[str]] = None
    index_name: Optional[str] = None
    train_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    val_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    padding_value: float = 0

    @property
    def seq_cols(self):
        seq_cols = [self.time_name]
        if self.cat_cardinalities is not None:
            seq_cols += list(self.cat_cardinalities)
        if self.num_names is not None:
            seq_cols += self.num_names
        return seq_cols


@dataclass(kw_only=True)
class GenBatch(Batch):
    target_cat_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_num_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_time: np.ndarray | torch.Tensor | None = None  # (target_len, batch)
    monotonic_time: bool = True

    def get_target_batch(self):
        assert self.target_time is not None
        return deepcopy(
            replace(
                self,
                lengths=torch.full_like(self.lengths, self.target_time.shape[0]),
                time=self.target_time,
                num_features=self.target_num_features,
                cat_features=self.target_cat_features,
                target_cat_features=None,
                target_num_features=None,
                target_time=None,
                cat_mask=None,
                num_mask=None,
            )
        )

    def append(self, other):
        def append_tensor(tensor, other_tensor):
            if tensor is None:
                return None
            new_tensor = torch.cat((tensor, torch.zeros_like(other_tensor)))
            if tensor.ndim == 3:
                other_tensor = other_tensor.reshape(-1, tensor.shape[2])
            elif tensor.ndim == 2:
                other_tensor = other_tensor.reshape(-1)
            new_tensor[seq_indices.flatten(), batch_indices.flatten()] = other_tensor
            return new_tensor

        assert isinstance(other, self.__class__)
        assert self.lengths.shape[0] == other.lengths.shape[0]
        assert (other.lengths == other.time.shape[0]).all()
        if self.monotonic_time and (other.time.amin(0) < self.time.amax(0)).any():
            logger.warning("Incorrect appended time. Result will be non monotonic.")
        target_len, B = other.time.shape[0], other.lengths.shape[0]
        seq_indices = (
            torch.arange(target_len, device=self.lengths.device)[:, None] + self.lengths
        )  # [target_len, B]
        batch_indices = (
            torch.arange(B).unsqueeze(0).expand(target_len, B)
        )  # [target_len, B]

        self.lengths += other.time.shape[0]
        self.time = append_tensor(self.time, other.time)
        self.num_features = append_tensor(self.num_features, other.num_features)
        self.cat_features = append_tensor(self.cat_features, other.cat_features)
        self.cat_mask = append_tensor(self.cat_mask, other.cat_mask)
        self.num_mask = append_tensor(self.num_mask, other.num_mask)

    def tail(self, tail_len: int):
        """Returns a new batch containing only last tail_len elements of each sequence."""
        assert self.lengths.min() >= tail_len, "tail_len is too big"

        start_index = self.lengths - tail_len  # [1, B]
        target_ids = (
            torch.arange(tail_len, device=start_index.device)[:, None] + start_index
        )  # [target_len, B]

        return replace(
            self,
            lengths=torch.ones_like(self.lengths) * tail_len,
            time=gather(self.time, target_ids),
            index=copy(self.index),
            num_features=gather(self.num_features, target_ids),
            cat_features=gather(self.cat_features, target_ids),
            cat_features_names=copy(self.cat_features_names),
            num_features_names=copy(self.num_features_names),
            cat_mask=gather(self.cat_mask, target_ids),
            num_mask=gather(self.num_mask, target_ids),
        )


@dataclass(kw_only=True)
class PredBatch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    num_features_names: list[str] | None = None
    cat_features: dict[str, torch.Tensor] | None = None  # {"name": (len, batch, C)}

    def get_numerical(self):
        return torch.cat((self.time.unsqueeze(-1), self.num_features), dim=2)

    def to_batch(self):
        cat_features = None
        cat_feature_names = (
            None if self.cat_features is None else list(self.cat_features.keys())
        )
        if self.cat_features:
            cat_features = []
            for cat_name, cat_tensor in self.cat_features.items():
                cat_features.append(cat_tensor.argmax(dim=2))
            cat_features = torch.stack(cat_features, dim=2)

        return GenBatch(
            lengths=self.lengths,
            time=self.time,
            index=None,
            num_features=self.num_features,
            cat_features=cat_features,
            cat_features_names=cat_feature_names,
            num_features_names=self.num_features_names,
        )


def gather(tensor, target_ids):
    if tensor is None:
        return None
    if 2 <= tensor.ndim <= 3:  # time
        if tensor.ndim == 3:  # cat|num
            target_ids = target_ids[:, :, None].expand(-1, -1, tensor.shape[2])
        return torch.gather(tensor, 0, target_ids)  # [target_len, B, D]
    else:
        raise ValueError
