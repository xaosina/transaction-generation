from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data_types import GenBatch


@dataclass(kw_only=True, frozen=True)
class SequenceCollator:
    time_name: str
    cat_cardinalities: Mapping[str, int] | None = None
    num_names: list[str] | None = None
    index_name: str | None = None
    max_seq_len: int = 0
    batch_transforms: list[Callable[[GenBatch], None]] | None = None
    return_orig: bool = False
    padding_side: str = "end"
    padding_value: float = 0

    def __call__(self, seqs: pd.DataFrame) -> GenBatch:
        assert (
            self.padding_side == "end" and self.padding_value == 0
        ), "CutTargetSequence and some masking will fail"

        if self.return_orig:
            orig_seqs = deepcopy(seqs)

        ml = min(seqs["_seq_len"].max(), self.max_seq_len)  # type: ignore
        bs = len(seqs)

        num_features = None
        num_names = deepcopy(self.num_names)
        if num_names:
            num_features = torch.full(
                (ml, bs, len(num_names)), self.padding_value, dtype=torch.float32
            )

        cat_features = None
        cat_cardinalities = {}
        cat_names = None
        if self.cat_cardinalities is not None:
            cat_cardinalities = self.cat_cardinalities
            cat_names = list(self.cat_cardinalities.keys())
        if self.cat_cardinalities:
            cat_features = torch.full(
                (ml, bs, len(self.cat_cardinalities)),
                self.padding_value,
                dtype=torch.long,
            )

        indices = None
        if self.index_name:
            indices = []

        seq_time_dtype: np.dtype = seqs.iloc[0][self.time_name].dtype  # type: ignore
        times = np.full((ml, bs), self.padding_value, dtype=seq_time_dtype)

        seq_lens = torch.empty(bs, dtype=torch.long)

        for b in range(len(seqs)):
            s = seqs.iloc[b]
            sl = min(s["_seq_len"], ml)  # type: ignore
            seq_lens[b] = sl
            # If pad end, then data are filled at the beggining
            slice_idx = slice(0, sl) if self.padding_side == "end" else slice(-sl, None)

            if num_names is not None:
                for i, name in enumerate(num_names):
                    assert num_features is not None
                    num_features[slice_idx, b, i] = torch.tensor(s[name][-sl:])

            for i, (name, card) in enumerate(cat_cardinalities.items()):
                assert cat_features is not None
                cat_features[slice_idx, b, i] = torch.tensor(s[name][-sl:]).clamp_(
                    max=card - 1
                )

            if indices is not None:
                indices.append(s[self.index_name])

            times[slice_idx, b] = s[self.time_name][-sl:]

        index = np.asanyarray(indices)

        try:
            times = torch.asarray(times)
        except TypeError:
            # keep numpy
            pass

        batch = GenBatch(
            num_features=num_features,
            cat_features=cat_features,
            index=index,
            time=times,
            lengths=seq_lens,
            cat_features_names=cat_names,
            num_features_names=num_names,
        )

        if self.batch_transforms is not None:
            for tf in self.batch_transforms:
                tf(batch)

        return (batch, orig_seqs) if self.return_orig else batch

    def reverse(self, batch: GenBatch) -> pd.DataFrame:
        def to_numpy(x):
            return x.numpy(force=True) if isinstance(x, torch.Tensor) else x

        def collect(arr, mask):
            arr = to_numpy(arr)
            _, _, D = arr.shape
            arr = arr.swapaxes(0, 1).reshape(-1, D)  # B*L, D
            arr = arr[mask]  # sum(seq_lens, D)
            return [np.split(arr[:, d], np.cumsum(seq_lens[:-1])) for d in range(D)]

        batch = deepcopy(batch)
        if self.batch_transforms:
            for tf in reversed(self.batch_transforms):
                tf.reverse(batch)

        cat_names = batch.cat_features_names or []
        num_names = batch.num_features_names or []
        seq_lens = to_numpy(batch.lengths)

        L, B = batch.time.shape
        valid_mask = (np.arange(L) < seq_lens[:, None]).ravel()
        # Combine features into (L, B, D) array
        features = collect(batch.time[:, :, None], valid_mask)
        if num_names:
            features += collect(batch.num_features, valid_mask)
        if cat_names:
            features += collect(batch.cat_features, valid_mask)

        data = np.empty((B, len(features)), dtype=object)
        for d in range(len(features)):
            data[:, d] = features[d]

        # Assemble DataFrame
        df = pd.DataFrame(data, columns=[self.time_name] + num_names + cat_names)
        df.insert(0, self.index_name, batch.index)
        df.insert(1, "_seq_len", seq_lens)

        return df
