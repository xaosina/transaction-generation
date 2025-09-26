import logging
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ebes.types import Batch, Seq

from ..utils import RateLimitFilter

logger = logging.getLogger(__name__)  # noqa: F821
logger.addFilter(RateLimitFilter(60))


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "shakespeare"
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
    cat_cardinalities: Optional[Union[list[list[Any]], Mapping[str, Any]]] = None
    num_names: Optional[list[str]] = None
    index_name: Optional[str] = None
    loader_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    train_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    val_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    padding_value: float = 0
    # List of features to focus on in loss and metrics. If None->focus on all
    focus_on: Optional[list[str]] = None
    drop_nonfocus: bool = False
    target_token: str = "none"

    @property
    def seq_cols(self):
        seq_cols = [self.time_name]
        if self.cat_cardinalities is not None:
            seq_cols += list(self.cat_cardinalities)
        if self.num_names is not None:
            seq_cols += self.num_names
        return seq_cols

    @property
    def focus_num(self):
        all_num = [self.time_name]
        all_num += self.num_names or []
        return [col for col in all_num if col in self.focus_on]

    @property
    def focus_cat(self):
        cat_d = self.cat_cardinalities or {}
        return [col for col in cat_d if col in self.focus_on]

    def __post_init__(self):
        time_name = self.time_name
        num_names = set(self.num_names or [])
        if isinstance(self.cat_cardinalities, list):
            cat_dict = {c[0]: c[1] for c in self.cat_cardinalities}
            object.__setattr__(self, "cat_cardinalities", cat_dict)
        cat_names = set(self.cat_cardinalities or {})

        if (time_name in cat_names | num_names) or (cat_names & num_names):
            raise ValueError("Conflict. time_name, num_names and cat_names intersect.")

        focus = self.focus_on
        if focus:
            new_focus = [
                f if f != "<target_token>" else self.target_token for f in focus
            ]
            object.__setattr__(self, "focus_on", new_focus)
            if not set(new_focus).issubset(self.seq_cols):
                raise ValueError("focus_on must be a subset of seq_cols")
            if self.drop_nonfocus:
                new_cat, new_num = self.cat_cardinalities or {}, self.num_names or []
                new_cat = {k: v for k, v in new_cat.items() if k in new_focus}
                new_num = [n for n in new_num if n in new_focus]
                object.__setattr__(self, "cat_cardinalities", new_cat or None)
                object.__setattr__(self, "num_names", new_num or None)
        elif focus is None:
            object.__setattr__(self, "focus_on", self.seq_cols)


@dataclass(frozen=True)
class LatentDataConfig:
    focus_on: list[str]
    time_name: str
    generation_len: int
    cat_cardinalities: Mapping[str, int] | None = None
    num_names: Optional[list[str]] = None

    @property
    def focus_cat(self):
        return [name for name in self.cat_cardinalities or [] if name in self.focus_on]

    @property
    def focus_num(self):
        names = self.num_names or [] + [self.time_name]
        return [name for name in names if name in self.focus_on]

    def check_focus_on(self, use_time):
        """Checks if focus_on is compatable with autoregressive approach."""
        if self.time_name not in self.focus_on:
            assert (
                not use_time
            ), "Time not in focus_on, but will be fed into network during generation"
        seq_cols = set(list(self.cat_cardinalities or {}) + (self.num_names or []))
        set_focus = set(self.focus_on)
        assert (
            seq_cols <= set_focus
        ), f"Can't use this features autoregressivly and NOT focus on them: {seq_cols - set_focus}"


@dataclass(kw_only=True)
class GenBatch(Batch):
    target_cat_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_num_features: torch.Tensor | None = None  # (target_len, batch, features)
    target_time: np.ndarray | torch.Tensor | None = None  # (target_len, batch)
    monotonic_time: bool = True

    @property
    def shape(self):
        if self.num_features is not None:
            return self.num_features.shape[:2]
        if self.cat_features is not None:
            return self.cat_features.shape[:2]
        assert self.time is not None, "All tensors are None!"
        return self.time.shape

    @property
    def device(self):
        if self.num_features is not None:
            return self.num_features.device
        if self.cat_features is not None:
            return self.cat_features.device
        assert isinstance(self.time, torch.tensor), "No tensors in batch!"
        return self.time.device

    @property
    def valid_mask(self):
        L = self.lengths.max()
        valid_mask = (
            torch.arange(L, device=self.lengths.device)[:, None] < self.lengths
        )  # L, B
        return valid_mask

    def __getitem__(self, name: str):
        if self.num_features_names and (name in self.num_features_names):
            return self.num_features[..., self.num_features_names.index(name)]
        elif self.cat_features_names and (name in self.cat_features_names):
            return self.cat_features[..., self.cat_features_names.index(name)]
        else:
            raise KeyError(f"No feature named '{name}'")

    def get_numerical(self):
        tensors = [self.time.unsqueeze(-1)]
        if self.num_features is not None:
            tensors = [self.num_features] + tensors
        return torch.cat(tensors, dim=2)

    def get_target_numerical(self):
        assert self.target_time is not None
        tensors = [self.target_time.unsqueeze(-1)]
        if self.target_num_features is not None:
            tensors = [self.target_num_features] + tensors
        return torch.cat(tensors, dim=2)

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

    def tail(self, tail_len: int, shift=0):
        """Returns a new batch containing only last tail_len elements of each sequence."""
        assert self.lengths.min() >= tail_len, "tail_len is too big"
        start_index = (self.lengths - tail_len - shift).clip(0)  # [1, B]
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
    
    def head(self, head_len: int):
        """Return a new batch containing only the first head_len elements of each sequence."""
        if self.lengths.max() <= head_len:
            return deepcopy(self)
        
        def _cutoff_tail(tensor):
            if tensor is None:
                return None
            return tensor[:head_len]
        
        return replace(
            self,
            lengths = torch.where(self.lengths <= head_len, self.lengths, head_len),
            time = _cutoff_tail(self.time),
            index = copy(self.index),
            num_features = _cutoff_tail(self.num_features),
            cat_features = _cutoff_tail(self.cat_features),
            cat_features_names = copy(self.cat_features_names),
            num_features_names = copy(self.num_features_names),
            cat_mask = _cutoff_tail(self.cat_mask),
            num_mask = _cutoff_tail(self.num_mask),
        )
        



@dataclass(kw_only=True)
class PredBatch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch, [K])
    num_features: torch.Tensor | None = None  # (len, batch, [K], features)
    num_features_names: list[str] | None = None
    cat_features: dict[str, torch.Tensor] | None = (
        None  # {"name": (len, batch, [K], C)}
    )

    def __getitem__(self, name: str):
        if name in self.num_features_names:
            return self.num_features[..., self.num_features_names.index(name)]
        elif name in self.cat_features:
            return self.cat_features[name]
        else:
            raise KeyError(f"No feature named '{name}'")

    @property
    def shape(self):
        if self.num_features is not None:
            return self.num_features.shape[:2]
        if self.cat_features is not None:
            return list(self.cat_features.items())[0][1].shape[:2]
        assert self.time is not None, "All tensors are None!"
        return self.time.shape

    @property
    def device(self):
        if self.num_features is not None:
            return self.num_features.device
        if self.cat_features is not None:
            return list(self.cat_features.items())[0].device
        assert isinstance(self.time, torch.tensor), "No tensors in batch!"
        return self.time.device

    @property
    def valid_mask(self):
        L = self.lengths.max()
        valid_mask = (
            torch.arange(L, device=self.lengths.device)[:, None] < self.lengths
        )  # L, B
        return valid_mask

    def k_reshape(self, k):
        assert self.time.ndim == 2
        L, BK = self.time.shape
        assert BK % k == 0
        B = BK // k
        self.lengths = self.lengths[::k]
        self.time = self.time.reshape(L, B, k)
        if self.num_features is not None:
            self.num_features = self.num_features.reshape(L, B, k, -1)
        if self.cat_features is not None:
            for name in self.cat_features:
                self.cat_features[name] = self.cat_features[name].reshape(L, B, k, -1)
        return self

    def get_numerical(self):
        tensors = [self.time.unsqueeze(-1)]
        if self.num_features is not None:
            tensors = [self.num_features] + tensors
        return torch.cat(tensors, dim=2)

    def to_batch(self, topk=1, temperature=1.0):
        cat_features = None
        cat_feature_names = (
            None if self.cat_features is None else list(self.cat_features.keys())
        )
        if self.cat_features:
            cat_features = []
            for cat_name, cat_tensor in self.cat_features.items():
                assert topk != 0, "Incorrect topk"
                if topk == 1:
                    samples = cat_tensor.argmax(dim=-1)
                else:
                    shape = cat_tensor.shape
                    logits = (cat_tensor / temperature).view(-1, shape[-1])
                    if topk > 1:
                        v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                        logits[logits < v[..., [-1]]] = -float("Inf")
                    probs = F.softmax(logits, dim=-1)
                    samples = torch.multinomial(probs, num_samples=1).squeeze(1)
                    samples = samples.view(shape[:-1])                    
                cat_features.append(samples)

            cat_features = torch.stack(cat_features, dim=-1)

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
    ndim_diff = tensor.ndim - target_ids.ndim
    target_expanded = target_ids.view(*target_ids.shape, *(1,) * ndim_diff)
    target_expanded = target_expanded.expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, 0, target_expanded)  # [L, B, ...]


def get_seq_tail(seq: Seq, tail_len: int):
    assert seq.lengths.min() >= tail_len, "tail_len is too big"
    start_index = seq.lengths - tail_len  # [1, B]
    target_ids = (
        torch.arange(tail_len, device=start_index.device)[:, None] + start_index
    )  # [target_len, B]

    return Seq(
        tokens=gather(seq.tokens, target_ids),
        lengths=torch.ones_like(seq.lengths) * tail_len,
        time=gather(seq.time, target_ids),
        masks=gather(seq.masks, target_ids),
    )


def append_padded(
        a : torch.Tensor | None, 
        la : torch.Tensor, 
        b : torch.Tensor | None, 
        lb : torch.Tensor
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """
    a: [L1, B, *X]  (padded sequences)
    la: [B]         (lengths of each seq in a)
    b: [L2, B, *X]  (padded sequences)
    lb: [B]         (lengths of each seq in b)

    Returns:
        out: [Lout, B, *X] padded concatenation
        lout: [B] new lengths
    """
    if (a is None) or (b is None):
        return None, la + lb
    
    
    L1, B = a.shape[0], a.shape[1]
    L2 = b.shape[0]

    lout = la + lb
    Lout = lout.max().item()

    # allocate output
    out = a.new_zeros((Lout, B) + a.shape[2:])

    # --- copy a ---
    mask_a = torch.arange(L1, device=a.device)[:, None] < la[None]   # [L1, B]
    out[:L1] = torch.where(mask_a[(...,) + (None,) * (a.ndim - 2)], a, out[:L1])

    # --- copy b ---
    start = la[None]  # [1, B]
    idx = torch.arange(L2, device=b.device)[:, None] + start  # [L2, B]
    batch_idx = torch.arange(B, device=b.device).expand(L2, B) # [L2, B]
    mask_b = torch.arange(L2, device=b.device)[:, None] < lb[None]  # [L2, B]

    out[idx[mask_b], batch_idx[mask_b]] = b[mask_b]

    return out[:Lout], lout


def seq_append(seq: Seq, other_seq: Seq) -> Seq:

    assert seq.lengths is not None, "seq lengths should be specified!"
    assert other_seq.lengths is not None, "other_seq lengths should be specified!"
    assert len(seq) == len(other_seq), "seq lenghts should be equal!"

    return Seq(
        tokens=append_padded(seq.tokens, seq.lengths, other_seq.tokens, other_seq.lengths)[0],
        lengths=seq.lengths + other_seq.lengths,
        time=append_padded(seq.time, seq.lengths, other_seq.time, other_seq.lengths)[0],
        masks=append_padded(seq.masks, seq.lengths, other_seq.masks, other_seq.lengths)[0],
    )


def valid_mask(x: Seq | GenBatch):
    L = x.lengths.max()
    valid_mask = torch.arange(L, device=x.lengths.device)[:, None] < x.lengths  # L, B
    return valid_mask
