from dataclasses import dataclass, fields

import numpy as np
import torch


@dataclass(kw_only=True)
class Batch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    index: torch.Tensor | np.ndarray  # (batch,)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features: torch.Tensor | None = None  # (len, batch, features)
    target: torch.Tensor | None = None  # (batch,), (len, batch) or (batch, n_targets)
    cat_features_names: list[str] | None = None
    num_features_names: list[str] | None = None
    cat_mask: torch.Tensor | None = None  # (len, batch, features)
    num_mask: torch.Tensor | None = None  # (len, batch, features)

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
