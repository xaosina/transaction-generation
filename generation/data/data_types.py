from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional
from ebes.types import Batch
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
class GenBatch(Batch):
    target_cat_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_num_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_time: np.ndarray | torch.Tensor | None = None  # (target_len, batch)

def get_gen_target(self):
    assert self.target_time is not None
    return replace(
        self,
        lengths=self.target_time[0].repeat(self.target_time.shape[1]),
        time=self.target_time,
        num_features=self.target_num_features,
        cat_features=self.target_cat_features,
        target_cat_features=None,
        target_num_features=None,
        target_time=None,
        cat_mask=None,
        num_mask=None
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
