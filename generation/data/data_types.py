from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

import numpy as np
import torch


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    batch_size: int = 128
    num_workers: int = 4
    val_ratio: int = 0.15
    train_resamples: int = 1
    #
    max_seq_len: int = 0
    min_history_len: int = 16
    generation_len: int = 16
    train_random_end: Literal["index", "time", "none"] = "none"
    val_random_end: Literal["index", "time", "none"] = "none"
    #
    time_name: str = "Time"
    cat_cardinalities: Mapping[str, int] | None = None
    num_names: list[str] | None = None
    index_name: str | None = None
    train_transforms: list[Mapping[str, Any] | str] | None = None
    val_transforms: list[Mapping[str, Any] | str] | None = None
    padding_value: float = 0

    @property
    def seq_cols(self):
        seq_cols = [self.time_name]
        if self.cat_cardinalities is not None:
            seq_cols += list(self.cat_cardinalities.keys())
        if self.num_names is not None:
            seq_cols += self.num_names
        return seq_cols


@dataclass(kw_only=True)
class Batch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    index: torch.Tensor | np.ndarray  # (batch,)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features_names: list[str] | None = None
    num_features_names: list[str] | None = None

    target_cat_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_num_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_time: np.ndarray | torch.Tensor | None = None  # (target_len, batch)

    def to(self, device: str):
        for field in fields(self):
            f = getattr(self, field.name)
            if isinstance(f, torch.Tensor):
                setattr(self, field.name, f.to(device))

        return self

    def __len__(self) -> int:
        """Batch size."""

        return len(self.lengths)


@dataclass(kw_only=True)
class Seq:
    tokens: torch.Tensor  # of shape (len, batch, features)
    lengths: torch.Tensor  # of shape (batch,)
    time: torch.Tensor  # of shape (len, batch)
    masks: torch.Tensor | None = None  # of shape (len, batch, features)

    def to(self, device):
        self.tokens = self.tokens.to(device)
        self.lengths = self.lengths.to(device)
        self.time = self.time.to(device)
        self.masks = self.masks.to(device) if self.masks else None
        return self

    def __len__(self):
        return len(self.lengths)
