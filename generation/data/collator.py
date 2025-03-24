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
    padding_side: str = "start"
    padding_value: float = 0

    def __call__(self, seqs: Sequence[pd.Series]) -> GenBatch:
        assert (
            self.padding_side == "start"
        ), "Don't support yet. CutTargetSequence will fail"
        ml = min(max(s["_seq_len"] for s in seqs), self.max_seq_len)  # type: ignore
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

        seq_time_dtype: np.dtype = seqs[0][self.time_name].dtype  # type: ignore
        times = np.full((ml, bs), self.padding_value, dtype=seq_time_dtype)

        seq_lens = torch.empty(bs, dtype=torch.long)

        for b, s in enumerate(seqs):
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

        return batch

    def reverse(self, batch: GenBatch) -> Sequence[pd.Series]:
        batch = deepcopy(batch)
        def numpy_if_possible(arr):
            if isinstance(arr, torch.Tensor):
                return arr.numpy(force=True)
            return arr

        if self.batch_transforms is not None:
            for tf in self.batch_transforms[::-1]:
                tf.reverse(batch)

        num_features = numpy_if_possible(batch.num_features)
        cat_features = numpy_if_possible(batch.cat_features)
        index = batch.index
        times = numpy_if_possible(batch.time)
        seq_lens = batch.lengths.numpy(force=True)
        cat_names = batch.cat_features_names
        num_names = batch.num_features_names

        seqs = []
        for b in range(len(batch)):
            s = pd.Series()
            sl = seq_lens[b]
            s["_seq_len"] = sl
            # If pad start, then data are filled at the end
            slice_idx = (
                slice(-sl, None) if self.padding_side == "start" else slice(0, sl)
            )
            if num_names is not None:
                for i, name in enumerate(num_names):
                    assert num_features is not None
                    s[name] = num_features[slice_idx, b, i]
            for i, name in enumerate(cat_names):
                assert cat_features is not None
                s[name] = cat_features[slice_idx, b, i]
            s[self.time_name] = times[slice_idx, b]
            s[self.index_name] = index[b]
            seqs.append(s)

        return seqs
