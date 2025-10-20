from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from ebes.model import BaseModel
from ebes.model.preprocess import SeqBatchNorm
from ebes.model.seq2seq import Projection
from ebes.types import Seq

from ...data import batch_tfs
from ...data.batch_tfs import NewFeatureTransform
from ...data.data_types import GenBatch, PredBatch
from ...utils import create_instances_from_module
from .utils import get_features_after_transform


@dataclass(frozen=True)
class AEConfig:
    name: str = ""
    params: Optional[dict[str, Any]] = None
    pretrain: bool = False
    frozen: bool = False
    checkpoint: Optional[str] = None
    batch_transforms: Optional[
        list[Mapping[str, Any] | str] | Mapping[str, Mapping[str, Any] | str]
    ] = None


class BaseAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch) -> PredBatch: ...

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch: ...


class BaselineAE(BaseAE):
    def __init__(self, data_conf, ae_config):
        super().__init__()
        ae_config: AEConfig = ae_config.autoencoder
        batch_transforms = create_instances_from_module(
            batch_tfs, ae_config.batch_transforms
        )
        num_names, cat_cardinalities = get_features_after_transform(
            data_conf, batch_transforms, ae_config
        )

        if batch_transforms is not None:
            assert ae_config.frozen, "Transformes are designed for pretrained models!"
        self.encoder = Batch2TransformedSeq(
            cat_cardinalities=cat_cardinalities,
            num_features=num_names,
            cat_emb_dim=ae_config.params["cat_emb_dim"],
            num_emb_dim=ae_config.params["num_emb_dim"],
            num_norm=ae_config.params["num_norm"],
            use_time=ae_config.params["use_time"],
            batch_transforms=batch_transforms,
        )
        self.model_config = ae_config

        self.decoder = ReconstructorBase(
            self.encoder.output_dim,
            cat_cardinalities=cat_cardinalities,
            num_features=num_names,
            batch_transforms=batch_transforms,
        )

    def forward(self, x: GenBatch) -> PredBatch:
        raise "We don't pretrain AE"

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        raise "We don't use AE yet"


class Batch2TransformedSeq(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        num_features: Sequence[str] | None = None,
        cat_emb_dim: int | None = None,
        num_emb_dim: int | None = None,
        num_norm: bool = False,
        use_time: bool = True,
        batch_transforms: list | None = None,
    ):
        super().__init__()
        # Establish initial features
        cat_cardinalities = cat_cardinalities if cat_cardinalities is not None else {}
        if num_features is not None:
            num_count = len(num_features)
        else:
            num_count = 0
        # Init batch_transforms. Update initial features.
        self.batch_transforms = batch_transforms
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
        self.use_time = use_time
        if use_time:
            num_count += 1

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

    @property
    def output_dim(self):
        return self._out_dim

    def forward(
        self, batch: GenBatch, copy=True
    ) -> Seq:  # of shape (len, batch_size, )
        if copy:
            batch = deepcopy(batch)

        if self.batch_transforms is not None:
            for tf in self.batch_transforms:
                tf(batch)

        if not isinstance(batch.time, torch.Tensor):
            raise ValueError(
                "`time` field in batch must be a Tensor. "
                "Consider proper time preprocessing"
            )

        embs = []
        masks = []
        if batch.cat_features_names:
            for i, cf in enumerate(batch.cat_features_names):
                embs.append(self._cat_embs[cf](batch[cf]))
                if batch.cat_mask is not None:
                    mask = batch.cat_mask[:, :, i].unsqueeze(2)
                    mask = torch.repeat_interleave(
                        mask, self._cat_embs[cf].embedding_dim, 2
                    )
                    masks.append(mask)

        x = []
        if batch.num_features is not None:
            x += [batch.num_features]
        if self.use_time:
            x += [batch.time[..., None]]
        if x:
            x = torch.cat(x, dim=2)
            if self.batch_norm:
                x = self.batch_norm(x, batch.lengths)
            x = x.permute(1, 2, 0)  # batch, features, len
            x = self._num_emb(x)
            embs.append(x.permute(2, 0, 1))
            if batch.num_mask is not None:
                masks.append(
                    torch.repeat_interleave(
                        batch.num_mask,
                        self._num_emb.out_channels // self._num_emb.in_channels,
                        dim=2,
                    )
                )

        tokens = torch.cat(embs, dim=2)
        masks = torch.cat(masks, dim=2) if len(masks) > 0 else None
        return Seq(tokens=tokens, lengths=batch.lengths, time=batch.time, masks=masks)


class ReconstructorBase(BaseModel):
    def __init__(
        self,
        in_features: int,
        cat_cardinalities: Mapping[str, int],
        num_features: Sequence[str] | None = None,
        batch_transforms: list | None = None,
    ):
        super().__init__()

        self.projector = Projection(in_features, 2 * in_features)
        self.batch_transforms = batch_transforms
        self.num_names = num_features
        self.cat_cardinalities = cat_cardinalities
        out_dim = 1  # Time
        if self.num_names:
            out_dim += len(self.num_names)
        if self.cat_cardinalities:
            out_dim += sum(self.cat_cardinalities.values())

        self.head = Projection(2 * in_features, out_dim)

    def forward(self, x: Seq) -> PredBatch:
        x = self.projector(x)

        out = self.head(x).tokens
        time = out[..., 0]
        start_id = 1

        num_features = None
        if self.num_names:
            num_features = out[..., 1 : len(self.num_names) + 1]
            start_id += len(self.num_names)
        cat_features = None
        if self.cat_cardinalities:
            cat_features = {}
            for cat_name, cat_dim in self.cat_cardinalities.items():
                cat_features[cat_name] = out[..., start_id : start_id + cat_dim]
                start_id += cat_dim
        assert start_id == out.shape[-1], f"{start_id}, {out.shape}"
        return PredBatch(
            lengths=x.lengths,
            time=time,
            num_features=num_features,
            num_features_names=self.num_names,
            cat_features=cat_features,
        )

    def generate(self, x: Seq, topk=1, temperature=1.0) -> GenBatch:
        # TODO add orig_hist just like in VAE
        batch = self.forward(x).to_batch(topk, temperature)
        if self.batch_transforms is not None:
            for tf in reversed(self.batch_transforms):
                tf.reverse(batch)
        return batch


@dataclass(frozen=True)
class PreprocessorConfig:
    cat_emb_dim: Optional[int] = None
    num_emb_dim: Optional[int] = None
    num_norm: bool = (True,)
    batch_transforms: Optional[Mapping[str, Mapping[str, Any] | str]] = None
