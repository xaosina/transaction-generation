from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

# from .transformer.ar import AR
from ebes.model import BaseModel
from ebes.model.seq2seq import BaseSeq2Seq
from ebes.types import Seq
from torch import nn

# from .transformer.ar import AR


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    params: Optional[Mapping[str, Any]] = None


class ARTransformer(BaseSeq2Seq):
    """Autoregressive transformer"""

    def __init__(self, cfg):
        super().__init__()
        params = cfg.params

        self.projector = nn.Linear(params.input_dim, params.d_model)

        self.transformer = AR(
            params.num_tokens,
            d_model=params.d_model,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
        )

    def forward(self, seq: Seq):
        x = seq.tokens  # B, L, D
        padding_mask = torch.arange(x.shape[0], device=x.device) >= x.lengths[:, None]

        x = self.projector(x)
        x = self.transformer(x, padding_mask)

        return x

    def generate(self, seq: Seq, max_step: int) -> Seq:
        x = seq.tokens  # B, L, D
        padding_mask = torch.arange(x.shape[0], device=x.device) >= x.lengths[:, None]

        x = self.projector(x)

        mask_row = torch.ones(
            (padding_mask.size(0), 1),
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        generated = []
        for _ in range(max_step):
            ret = self.transformer(x, padding_mask)  # B, L, D

            x = torch.cat([x, ret[:, :-1, :]], dim=1)
            padding_mask = torch.cat([padding_mask, mask_row], dim=1)

            generated.append(ret)
        generated = torch.cat(generated, dim=1)

        return generated


class AutoregressiveEncoder(BaseSeq2Seq):
    def __init__(self, name: str, params: dict = None):
        super().__init__()
        params = params or {}
        self.model = BaseModel.get_model(name, **params)

    @property
    def output_dim(self):
        return self.model.output_dim

    def forward(self, seq: Seq) -> Seq:
        return self.model(seq)

    def generate(self, seq: Seq) -> Seq:  # returns a sequence of shape [1, B, D]
        seq = self.forward(seq)
        last_valid = seq.tokens[
            seq.lengths - 1, torch.arange(seq.tokens.shape[1])
        ].unsqueeze(0)
        return Seq(
            tokens=last_valid, lengths=torch.ones(last_valid.shape[1]), time=None
        )
