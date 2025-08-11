from copy import deepcopy
from dataclasses import replace

import torch
from ebes.model import BaseSeq2Seq
from ebes.types import Seq

from generation.models import autoencoders
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch, get_valid_mask
from ..encoders import AutoregressiveEncoder
from . import BaseGenerator, ModelConfig


class ConditionalHead(BaseSeq2Seq):
    """FC head for the sequence encoder

    Args:
        input_size: Embedding size.
        k: The number of output tokens.
    """

    def __init__(self, input_size, k):
        super().__init__()
        self.input_size = input_size
        self.proj = torch.nn.Linear(input_size * 2, input_size)
        self.relu = torch.nn.ReLU()

        self.queries = torch.nn.Parameter(torch.randn(k, input_size))  # (K, D).
        self.k = k

    @property
    def output_dim(self):
        return self.input_size * self.k

    def forward_impl(self, ctx):
        b, d = ctx.shape
        x = self.queries[None].repeat(b, 1, 1)  # (B, K, D_x).
        x = torch.cat([ctx.unsqueeze(1).repeat(1, self.k, 1), x], -1).flatten(
            0, 1
        )  # (BK, D)
        x = self.proj(x)  # (BK, O).
        return x.reshape(b, self.output_dim)  # (B, KO).

    def forward(self, seq: Seq):
        mask = get_valid_mask(seq)
        x = seq.tokens
        assert x.ndim > 2  # (L, B, D).
        shape = list(x.shape)
        x_masked = x[mask]  # (V, D).
        v = len(x_masked)
        x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(
            *([v] + shape[2:-1] + [self.output_size])
        )  # (V, *, D).
        x_new = torch.zeros(
            *[shape[:-1] + [self.output_dim]],
            dtype=x_mapped.dtype,
            device=x_mapped.device
        )  # (L, B, *, D).
        x_new[mask] = x_mapped
        return replace(seq, tokens=x_new)


class DeTPP(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config.autoencoder
        )

        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(
                ckpt["model"]["autoencoder"], strict=False
            )

        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)

        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.autoencoder.encoder.output_dim

        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )

        self.next_k_head = ConditionalHead(
            self.autoencoder.encoder.output_dim, model_config.params["k"]
        )

        data_conf.check_focus_on(self.autoencoder.encoder.use_time)

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = deepcopy(x)
        deltas = x.time
        deltas[:, 1:] -= deltas[:, :-1]
        deltas[:, 0] = 0
        # deltas.clip_(min=0, max=self._max_time_delta)
        x.time = deltas
        x = self.autoencoder.encoder(x, copy=False)  # Sequence of [L, B, D]
        x = self.encoder(x)
        x = self.next_k_head(x) # L, B, D*k
        x = self.autoencoder.decoder(x)
        return x

    def generate(
        self,
        hist: GenBatch,
        gen_len: int,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
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
                    x, topk=topk, temperature=temperature
                )  # GenBatch with sizes [1, B, D] for cat, num
                hist.append(x)  # Append GenBatch, result is [L+1, B, D]
        if with_hist:
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return hist.tail(gen_len)  # Return GenBatch of size [gen_len, B, D]
