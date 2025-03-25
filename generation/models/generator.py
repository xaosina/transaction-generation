from copy import deepcopy
from dataclasses import dataclass

from ebes.model import BaseModel
from ebes.model.seq2seq import GRU, Projection
from ebes.types import Seq
import torch

from ..data.data_types import GenBatch, DataConfig, PredBatch
from .preprocessor import create_preprocessor
from .reconstructors import ReconstructorBase

# # TODO: Think
# @dataclass
# class Generator:
#     preprocess: PreprocessConfig = field(default=None)
#     encoder: PreprocessConfig = field(default=None)
#     projector: ProjectorConfig = field(default=None)
#     reconstructor: PreprocessConfig = field(default=None)


@dataclass
class GeneratorConfig:

    forward_reconstructed: bool = False


class Generator(BaseModel):  # TODO work
    def __init__(self, data_conf: DataConfig):
        super().__init__()
        self.preprocess = create_preprocessor(data_conf, 4, 4, "diff", True)

        self.encoder = GRU(self.preprocess.output_dim, 128, 1)

        self.projector = Projection(self.encoder.output_dim, self.encoder.output_dim)

        self.reconstructor = ReconstructorBase(data_conf, self.projector.output_dim)

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = self.preprocess(x)  # Sequence of [L, B, D]
        x = self.encoder(x)
        x = self.projector(x)
        x = self.reconstructor(x)
        return x

    def generate(self, hist: GenBatch, gen_len: int, with_hist=True) -> GenBatch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        self.eval()
        hist = deepcopy(hist)

        with torch.no_grad():
            for _ in range(gen_len):
                x = self.preprocess(hist)
                x = self.encoder.generate(x) # Sequence of shape [1, B, D]
                x = self.projector(x)
                x = self.reconstructor.generate(x) # GenBatch with sizes [1, B, D] for cat, num
                hist = hist.append(x) # Append GenBatch, result is [L+1, B, D]
        if with_hist:
            return hist # Return GenBatch of size [L + gen_len, B, D]
        else:
            return hist.tail(gen_len) # Return GenBatch of size [gen_len, B, D]
