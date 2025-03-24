from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch


@dataclass
class DataConfig:
    train_path: str = ""
    test_path: str = ""
    batch_size: int = 128
    num_workers: int = 4
    val_ratio: int = 0.15
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
    train_transforms: Optional[list[Mapping[str, Any] | str]] = None
    val_transforms: Optional[list[Mapping[str, Any] | str]] = None
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
    index: torch.Tensor | np.ndarray | None = None  # (batch,)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features_names: list[str] | None = None
    num_features_names: list[str] | None = None
    cat_mask: torch.Tensor | None = None  # (len, batch, features)
    num_mask: torch.Tensor | None = None  # (len, batch, features)

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

    def _find_tensor_and_idx(self, feature_name):
        if self.cat_features is not None and self.cat_features_names is not None:
            try:
                cat_idx = self.cat_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.cat_features, cat_idx

        if self.num_features is not None and self.num_features_names is not None:
            try:
                num_idx = self.num_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.num_features, num_idx

        raise ValueError(
            f"Cannot access feature by name {feature_name}."
            f" Known cat names are: {self.cat_features_names}."
            f" Known num names are: {self.num_features_names}."
        )

    def __setitem__(self, feature_name: str, value: Any):
        tensor, idx = self._find_tensor_and_idx(feature_name)
        tensor[:, :, idx] = value

    def __getitem__(
        self,
        feature_name: str,
    ) -> torch.Tensor:  # of shape (len, batch)
        tensor, idx = self._find_tensor_and_idx(feature_name)
        return tensor[:, :, idx]

    def __eq__(self, other):
        assert isinstance(other, Batch)
        equal = True
        for field in fields(self):
            my_field, other_field = getattr(self, field.name), getattr(
                other, field.name
            )
            if isinstance(my_field, torch.Tensor | np.ndarray):
                equal = equal and (my_field == other_field).all()
        return equal


@dataclass(kw_only=True)
class PredBatch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    num_features_names: list[str] | None = None
    cat_features: dict[str, torch.Tensor] | None = None  # {"name": (len, batch, C)}

    def to(self, device: str):
        for field in fields(self):
            f = getattr(self, field.name)
            if isinstance(f, torch.Tensor):
                setattr(self, field.name, f.to(device))

        return self

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

        return Batch(
            lengths=self.lengths,
            time=self.time,
            index=None,
            num_features=self.num_features,
            cat_features=cat_features,
            cat_features_names=cat_feature_names,
            num_features_names=self.num_features_names,
        )
