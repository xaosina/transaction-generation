import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from ebes.model import BaseModel, TakeLastHidden, ValidHiddenMean
from ebes.model.seq2seq import Projection
from ebes.types import Seq

from generation.models import autoencoders
from generation.models.autoencoders.base import AEConfig
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from ..encoders import AutoregressiveEncoder, LatentEncConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    latent_encoder: Optional[LatentEncConfig] = None
    # tpp: TPPConfig = field(default_factory=TPPConfig)
    autoencoder: Optional[AEConfig] = None
    pooler: str = "last"
    params: Optional[dict[str, Any]] = None


class BaseGenerator(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: GenBatch) -> PredBatch: ...

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch: ...

    def get_embeddings(self, hist: GenBatch) -> torch.Tensor: ...


class AutoregressiveGenerator(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )

        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(ckpt["model"], strict=False)

        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)

        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.autoencoder.encoder.output_dim

        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )

        data_conf.check_focus_on(self.autoencoder.encoder.use_time)

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

    def get_embeddings(self, hist: GenBatch):
        hist = deepcopy(hist)
        x = self.autoencoder.encoder(hist)
        x = self.encoder.generate(x)
        assert x.tokens.shape[0] == 1
        return x.tokens[0]
        

class Reshaper(BaseModel):
    def __init__(self, gen_len: int):
        super().__init__()
        self.gen_len = gen_len

    def forward(self, seq: Seq) -> Seq:
        tensor = seq.tokens
        assert (
            tensor.shape[1] % self.gen_len == 0
        ), f"hidden_size doesnt divide by {self.gen_len}"
        B, D = tensor.shape
        return Seq(
            tokens=tensor.view(B, self.gen_len, D // self.gen_len)
            .permute(1, 0, 2)
            .contiguous(),
            lengths=torch.ones((B,), dtype=torch.long, device=tensor.device)
            * self.gen_len,
            time=None,
        )


class OneShotGenerator(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(ckpt["model"], strict=False)

        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)

        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.autoencoder.encoder.output_dim

        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )
        self.poller = (
            TakeLastHidden() if model_config.pooler == "last" else ValidHiddenMean()
        )

        self.projector = Projection(
            self.encoder.output_dim,
            self.encoder.output_dim * data_conf.generation_len,
        )

        self.reshaper = Reshaper(data_conf.generation_len)

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = self.autoencoder.encoder(x)  # Sequence of [L, B, D]
        x = self.encoder(x)  # [L, B, D]
        x = self.poller(x)  # [B, D]
        x = Seq(tokens=x, lengths=None, time=None)
        x = self.projector(x)  # [B, D * gen_len]
        x = self.reshaper(x)  # [gen_len, B, D]
        x = self.autoencoder.decoder(x)
        return x

    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, topk=1, temperature=1.0
    ) -> GenBatch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        assert (
            gen_len == self.reshaper.gen_len
        ), f"Can't generate other than {self.reshaper.gen_len}"
        hist = deepcopy(hist)

        with torch.no_grad():
            pred = self.forward(hist).to_batch(topk, temperature)
        if with_hist:
            hist.append(pred)
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return pred  # Return GenBatch of size [gen_len, B, D]

    def get_embeddings(self, hist: GenBatch):
        hist = deepcopy(hist)
        x = self.autoencoder.encoder(hist)
        x = self.encoder.generate(x)
        assert x.tokens.shape[0] == 1
        assert isinstance(self.poller, TakeLastHidden)
        return x.tokens[0]

    def collect(
        self,
        hist: GenBatch,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
        """
        Auto-regressive embedding collecting

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        hist = deepcopy(hist)
        x = self.autoencoder.encoder(hist)
        x = self.encoder(x)  # [L, B, D]
        x = self.poller(x)  # [B, D]
        return x


class OneShotDistributionGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(ckpt["model"], strict=False)

        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)

        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.autoencoder.encoder.output_dim

        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )
        self.poller = (
            TakeLastHidden() if model_config.pooler == "last" else ValidHiddenMean()
        )

        self.gen_len = data_conf.generation_len
        if data_conf.num_names is not None:
            self.num_projection = torch.nn.Linear(
                len(data_conf.num_names), len(data_conf.num_names) * 2
            )
        self.time_projection = torch.nn.Linear(1, 2)

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = self.autoencoder.encoder(x)  # Sequence of [L, B, D]
        x = self.encoder(x)  # [L, B, D]
        x = self.poller(x)  # [B, D]
        x = Seq(
            tokens=x,
            lengths=torch.ones_like(x, dtype=torch.long),
            time=None,
        )
        x = self.autoencoder.decoder(x)
        if x.time is not None:
            x.time = self.time_projection(x.time[:, None])
        if x.num_features is not None:
            x.num_features = self.num_projection(x.num_features)
        return x

    def get_embeddings(self, hist: GenBatch):
        hist = deepcopy(hist)
        x = self.autoencoder.encoder(hist)
        x = self.encoder.generate(x)
        assert x.tokens.shape[0] == 1
        assert isinstance(self.poller, TakeLastHidden)
        return x.tokens[0]

    def scale_to_gen_len(self, probs: torch.tensor, gen_len: int):
        scaled = probs * gen_len
        integer_parts = torch.floor(scaled).int()
        fractional_parts = scaled - integer_parts
        remaining = gen_len - torch.sum(integer_parts, dim=1, keepdim=True)

        _, sorted_indices = fractional_parts.sort(dim=1, descending=True)

        arange = torch.arange(probs.size(1), device=probs.device).unsqueeze(0)
        mask_sorted = arange < remaining

        mask_original = torch.zeros_like(mask_sorted)
        mask_original.scatter_(1, sorted_indices, mask_sorted)

        return integer_parts + mask_original.long()

    def counts_to_indices(self, counts: torch.Tensor) -> torch.Tensor:
        """
        counts : (B, K)  — целые, сумма каждой строки одинаковая (gen_len)
        return : (B, gen_len) — развёрнутый список категорий для каждой строки
        """
        device = counts.device
        arange = torch.arange(counts.size(1), device=device)

        idx_rows = [
            torch.repeat_interleave(arange, row)[torch.randperm(self.gen_len)]
            for row in counts
        ]
        return torch.stack(idx_rows)

    def sample(self, tensor: PredBatch, gen_len: int) -> GenBatch:
        num_features = None
        cat_features = None

        cat_features_names = list(tensor.cat_features.keys()) or []
        if cat_features_names:
            cat_features = []
            for cat_name in cat_features_names:
                params = tensor.cat_features[cat_name]
                probs = torch.nn.functional.softmax(params, dim=-1)
                scaled = self.scale_to_gen_len(probs, gen_len)

                assert all(scaled.sum(dim=1) == gen_len)

                cat_features.append(self.counts_to_indices(scaled).T)

            cat_features = torch.stack(cat_features, dim=2)

        num_names = tensor.num_features_names or []

        if len(num_names) > 0:
            num_features = []
            for name in num_names:
                idx = 2 * num_names.index(name)
                idxs = [idx, idx + 1]
                alpha_raw, beta_raw = tensor.num_features[:, idxs].unbind(dim=1)
                # alpha = torch.nn.functional.softmax(alpha_raw)
                beta = torch.nn.functional.softplus(beta_raw)

                dist = torch.distributions.Normal(alpha_raw, beta)
                num_features.append(dist.sample((gen_len,)))
            num_features = torch.stack(num_features, dim=2)
        alpha_raw, beta_raw = tensor.time.unbind(dim=1)
        # alpha = torch.nn.functional.softplus(alpha_raw)
        beta = torch.nn.functional.softplus(beta_raw)

        dist = torch.distributions.Normal(alpha_raw, beta)
        tensor.time = dist.sample((gen_len,))

        assert (tensor.lengths == 1).all()
        tensor.lengths *= gen_len
        return GenBatch(
            lengths=tensor.lengths,
            time=tensor.time,
            index=None,
            num_features=num_features,
            num_features_names=num_names,
            cat_features=cat_features,
            cat_features_names=cat_features_names,
        )

    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        hist = deepcopy(hist)

        with torch.no_grad():
            pred = self.sample(self.forward(hist), gen_len)
        if with_hist:
            hist.append(pred)
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return pred  # Return GenBatch of size [gen_len, B, D]
