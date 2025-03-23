from dataclasses import dataclass

from ebes.model import BaseModel
from ebes.model.seq2seq import GRU, Projection
from ebes.types import Seq

from ..data.data_types import Batch, DataConfig
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

    def forward(self, x: Batch) -> dict:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (Batch): Input sequence [L, B, D]

        """
        x = self.preprocess(x)  # Sequence of [L, B, D]
        x = self.encoder(x)
        x = self.projector(x)
        x = self.reconstructor(x)
        return x

    def generate(self, x: Seq) -> Batch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        x = self.preprocess(x)
        x = self.encoder.generate(x)
        x = self.projector(x)
        x = self.reconstructor.generate(x)
        return x
