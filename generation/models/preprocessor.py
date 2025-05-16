from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch.nn as nn
from ebes.model.preprocess import Batch2Seq, SeqBatchNorm
from ebes.types import Seq

from ..data.data_types import LatentDataConfig, GenBatch
from ..data.batch_tfs import NewFeatureTransform
from ..utils import create_instances_from_module
from ..data import batch_tfs

class Batch2TransformedSeq(Batch2Seq):
    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        num_features: Sequence[str] | None = None,
        cat_emb_dim: int | None = None,
        num_emb_dim: int | None = None,
        num_norm: bool = False,
        batch_transforms: Mapping[str, Mapping[str, Any] | str] | None = None,
    ):
        super(Batch2Seq, self).__init__()
        # Establish initial features
        cat_cardinalities = cat_cardinalities if cat_cardinalities is not None else {}
        if num_features is not None:
            num_count = len(num_features)
        else:
            num_count = 0
        # Init batch_transforms. Update initial features.
        self.batch_transforms = create_instances_from_module(batch_tfs, batch_transforms)
        if self.batch_transforms:
            for tfs in self.batch_transforms:
                assert isinstance(tfs, NewFeatureTransform)
                for _ in tfs.num_names:
                    num_count += 1
                for cat_name, card in tfs.cat_cardinalities.items():
                    cat_cardinalities[cat_name] = card
                for _ in tfs.num_names_removed:
                    num_count -= 1
                cat_cardinalities = {
                    k: v
                    for k, v in cat_cardinalities.items()
                    if k not in tfs.cat_names_removed
                }

        self._out_dim = 0
        self._cat_embs = nn.ModuleDict()
        cat_dims = []
        for name, card in cat_cardinalities.items():
            if cat_emb_dim is None:
                dim = int(min(600, round(1.6 * card**0.56)))
            elif isinstance(cat_emb_dim, int):
                dim = cat_emb_dim
            else:
                raise TypeError

            self._out_dim += dim
            cat_dims.append(dim)
            self._cat_embs[name] = nn.Embedding(card, dim)

        if num_emb_dim is None:
            if not cat_dims:
                raise ValueError(
                    "Auto dim choice for num embeddings does not work with no cat "
                    "features"
                )
            num_emb_dim = int(sum(cat_dims) / len(cat_dims))

        if num_count:
            self.batch_norm = SeqBatchNorm(num_count) if num_norm else None
            self._num_emb = nn.Conv1d(
                in_channels=num_count,
                out_channels=num_emb_dim * num_count,
                kernel_size=1,
                groups=num_count,
            )
        self._out_dim += num_emb_dim * num_count

    def forward(
        self, batch: GenBatch, copy=True
    ) -> Seq:  # of shape (len, batch_size, )
        if copy:
            batch = deepcopy(batch)

        if self.batch_transforms is not None:
            for tf in self.batch_transforms:
                tf(batch)
        return super().forward(batch, copy=False)


@dataclass(frozen=True)
class PreprocessorConfig:
    cat_emb_dim: Optional[int] = None
    num_emb_dim: Optional[int] = None
    num_norm: bool = (True,)
    batch_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None


def create_preprocessor(data_conf: LatentDataConfig, pre_conf: PreprocessorConfig):
    return Batch2TransformedSeq(
        cat_cardinalities=data_conf.cat_cardinalities,
        num_features=data_conf.num_names,
        cat_emb_dim=pre_conf.cat_emb_dim,
        num_emb_dim=pre_conf.num_emb_dim,
        num_norm=pre_conf.num_norm,
        batch_transforms=pre_conf.batch_transforms,
    )
