from copy import deepcopy
from dataclasses import dataclass, field

import torch
from ebes.model import BaseModel
from ebes.model.seq2seq import Projection

from generation.models.autoencoders.vae import Decoder as VAE_Decoder
from generation.models.autoencoders.vae import Encoder as VAE_Encoder
from generation.models.autoencoders.vae import VaeConfig

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from ..encoders import AutoregressiveEncoder, LatentEncConfig
from generation.models.autoencoders.base import AEConfig, BaseAE
from generation.models import autoencoders
from utils import freeze_module

@dataclass(frozen=True)
class TPPConfig:
    feature_name: str = ""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    # preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    latent_encoder: LatentEncConfig = field(default_factory=LatentEncConfig)
    # vae: VaeConfig = field(default_factory=VaeConfig)
    tpp: TPPConfig = field(default_factory=TPPConfig)
    autoencoder: AEConfig = field(default_factory=AEConfig)


class BaseGenerator(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch) -> PredBatch: ...

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch: ...


class Generator(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config.autoencoder
        )

        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location=self.autoencoder.device)
            msg = self.autoencoder.load_state_dict(ckpt["model"]["autoencoder"], strict=False)

        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)

        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.preprocess.output_dim
        
        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = self.autoencoder.encoder(x)  # Sequence of [L, B, D]
        x = self.encoder(x)
        x = self.autoencoder.decoder(x)
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
                x = self.autoencoder.encoder(hist)
                x = self.encoder.generate(x)  # Sequence of shape [1, B, D]
                x = self.autoencoder.decoder.generate(
                    x
                )  # GenBatch with sizes [1, B, D] for cat, num
                hist.append(x)  # Append GenBatch, result is [L+1, B, D]
        if with_hist:
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return hist.tail(gen_len)  # Return GenBatch of size [gen_len, B, D]
