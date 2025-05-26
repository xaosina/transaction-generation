from copy import deepcopy
from dataclasses import dataclass, field

import torch
from ebes.model import BaseModel
from ebes.model.seq2seq import Projection

from generation.models.autoencoders.vae import Decoder as VAE_Decoder
from generation.models.autoencoders.vae import Encoder as VAE_Encoder
from generation.models.autoencoders.vae import VaeConfig

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from ..encoders import AutoregressiveEncoder, EncoderConfig
from ..preprocessor import PreprocessorConfig, create_preprocessor
from ..reconstructors import ReconstructorBase


@dataclass(frozen=True)
class TPPConfig:
    feature_name: str = ""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    tpp: TPPConfig = field(default_factory=TPPConfig)


class BaseGenerator(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch) -> PredBatch: ...

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch: ...


class Generator(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.preprocess = create_preprocessor(data_conf, model_config.preprocessor)

        encoder_params = model_config.encoder.params or {}
        encoder_params["input_size"] = self.preprocess.output_dim
        self.encoder = AutoregressiveEncoder(model_config.encoder.name, encoder_params)

        self.projector = Projection(
            self.encoder.output_dim, 2 * self.encoder.output_dim
        )

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


class VAE(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.encoder = VAE_Encoder(
            model_config.vae,
            cat_cardinalities=data_conf.cat_cardinalities,
            num_names=data_conf.num_names,
            batch_transforms=model_config.preprocessor.batch_transforms,
        )

        self.decoder = VAE_Decoder(
            model_config.vae,
            cat_cardinalities=data_conf.cat_cardinalities,
            num_names=data_conf.num_names,
        )

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Variational AutoEncoder
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """

        assert not self.encoder.pretrained
        x, params = self.encoder(x)
        x = self.decoder(x)
        return x, params

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        hist = deepcopy(hist)
        assert hist.target_time.shape[0] == gen_len, hist.target_time.shape
        x = self.encoder(hist.get_target_batch())
        if not self.encoder.pretrained:
            x = x[0]
        x = self.decoder.generate(x)
        if with_hist:
            hist.append(x)
            return hist
        else:
            return x
