from copy import deepcopy
from dataclasses import dataclass, field

import torch
from ebes.model import BaseModel
from ebes.model.seq2seq import Projection

from ..data.data_types import DataConfig, GenBatch, PredBatch
from .encoders import GenGRU
from .preprocessor import PreprocessorConfig, create_preprocessor
from .reconstructors import ReconstructorBase


@dataclass
class ModelConfig:
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)


class BaselineRepeater(BaseModel):
    def __init__(self, data_conf: DataConfig):
        super().__init__()
        self.data_conf = data_conf

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)
        hist = deepcopy(hist)
        gen_batch = hist.tail(gen_len)
        if hist.monotonic_time:  # Time is monotonic.
            corr = torch.cat((torch.zeros_like(hist.time[:1]), hist.time))
            corr = corr[hist.lengths - gen_len, torch.arange(hist.time.shape[1])]
            gen_batch.time = gen_batch.time + gen_batch.time[-1] - corr
            # This complicated correction assures same behavior as with timediff
        if with_hist:
            return hist.append(gen_batch)
        else:
            return gen_batch


class Generator(BaseModel):  # TODO work
    def __init__(self, data_conf: DataConfig, model_config: ModelConfig):
        super().__init__()
        self.preprocess = create_preprocessor(data_conf, model_config.preprocessor)

        self.encoder = GenGRU(self.preprocess.output_dim, 128, 1)

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

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        hist = deepcopy(hist)

        with torch.no_grad():
            for _ in range(gen_len):
                x = self.preprocess(hist)
                x = self.encoder.generate(x)  # Sequence of shape [1, B, D]
                x = self.projector(x)
                x = self.reconstructor.generate(
                    x
                )  # GenBatch with sizes [1, B, D] for cat, num
                hist.append(x)  # Append GenBatch, result is [L+1, B, D]
        if with_hist:
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return hist.tail(gen_len)  # Return GenBatch of size [gen_len, B, D]
