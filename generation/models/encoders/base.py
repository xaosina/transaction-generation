from dataclasses import dataclass
from typing import Optional, Mapping, Any
import torch

# from .transformer.ar import AR
from ebes.model import BaseModel, Projection
from ebes.model.seq2seq import BaseSeq2Seq
from ebes.types import Seq
from torch import nn


@dataclass(frozen=True)
class LatentEncConfig:
    name: str
    params: Optional[Mapping[str, Any]] = None


class AutoregressiveEncoder(BaseSeq2Seq):
    def __init__(self, name: str, params: dict = None):
        super().__init__()
        params = params or {}
        self.model = BaseModel.get_model(name, **params)
        self.projector = Projection(self.model.output_dim, params['input_size'])

    @property
    def output_dim(self):
        return self.projector.output_dim

    def forward(self, seq: Seq) -> Seq:
        seq = self.model(seq)
        return self.projector(seq)

    def generate(self, seq: Seq) -> Seq:  # returns a sequence of shape [1, B, D]
        seq = self.forward(seq)
        last_valid = seq.tokens[
            seq.lengths - 1, torch.arange(seq.tokens.shape[1])
        ].unsqueeze(0)
        return Seq(
            tokens=last_valid, lengths=torch.ones(last_valid.shape[1]), time=None
        )
