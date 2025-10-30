import math
from copy import deepcopy
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from ebes.types import Seq

from generation.data import batch_tfs
from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch
from generation.data.utils import create_instances_from_module
from generation.models.autoencoders.base import AEConfig, BaseAE

from .modules import Tokenizer, Transformer, Reconstructor
from .utils import get_features_after_transform


class AR_VAE(BaseAE):
    def __init__(self, data_conf: LatentDataConfig, model_config):
        super().__init__()
        model_config: AEConfig = model_config.autoencoder
        vae_params = {
            "num_layers": model_config.params.get("num_layers", 2),
            "d_token": model_config.params.get("d_token", 6),
            "n_head": model_config.params.get("n_head", 2),
            "factor": model_config.params.get("factor", 64),
        }
        batch_transforms = create_instances_from_module(
            batch_tfs, model_config.batch_transforms
        )
        num_names, cat_cardinalities = get_features_after_transform(
            data_conf, batch_transforms, model_config
        )

        self.encoder = Encoder(
            **vae_params,
            use_time=model_config.params["use_time"],
            pretrain=model_config.pretrain,
            frozen=model_config.frozen,
            cat_cardinalities=cat_cardinalities,
            num_features=num_names,
            batch_transforms=batch_transforms,
        )
        self.model_config = model_config

        self.decoder = Decoder(
            **vae_params,
            cat_cardinalities=cat_cardinalities,
            num_names=num_names,
            batch_transforms=batch_transforms,
        )

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Variational AutoEncoder
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """

        assert not self.encoder.pretrained
        x, params = self.encoder(x, reparametrize=True)
        x = self.decoder(x)
        return x, params

    def generate(
        self,
        hist: GenBatch,
        gen_len: int,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
        hist = deepcopy(hist)
        assert hist.target_time.shape[0] == gen_len, hist.target_time.shape
        x = self.encoder(hist.get_target_batch())
        x = self.decoder.generate(x, topk=topk, temperature=temperature)
        if with_hist:
            hist.append(x)
            return hist
        else:
            return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_token: int = 6,
        n_head: int = 2,
        factor: int = 64,
        use_time: bool = True,
        pretrain: bool = False,
        frozen: bool = False,
        cat_cardinalities: Mapping[str, int] | None = None,
        num_features: Sequence[str] | None = None,
        batch_transforms: list | None = None,
        bias=True,
    ):
        super(Encoder, self).__init__()
        self.d_token = d_token
        self.pretrained = not pretrain
        self.frozen = frozen

        if num_features is not None:
            num_count = len(num_features)
        else:
            num_count = 0

        self.use_time = use_time
        # if use_time:
        num_count += 1
        self.batch_transforms = batch_transforms

        self.tokenizer = Tokenizer(
            d_numerical=num_count,
            categories=list(cat_cardinalities.values()) if cat_cardinalities else None,
            d_token=d_token,
            bias=bias,
        )
        self._out_dim = (num_count + len(cat_cardinalities)) * self.d_token

        self.encoder_mu = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.encoder_std = None
        if not self.pretrained:
            self.encoder_std = Transformer(num_layers, d_token, n_head, d_token, factor)

    @property
    def output_dim(self):
        return self._out_dim

    def reparametrize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: GenBatch, copy=True, reparametrize=False) -> Seq:
        if copy:
            batch = deepcopy(batch)

        if self.batch_transforms:
            for tf in self.batch_transforms:
                tf(batch)

        x_num, x_cat = batch.num_features, batch.cat_features

        x_num = [] if x_num is None else [x_num]
        if self.use_time:
            time = batch.time
        else:
            time = torch.zeros_like(batch.time)
        x_num += [time.unsqueeze(-1)]  # L, B, 1
        x_num = torch.cat(x_num, dim=2)

        D_num = x_num.size(-1) if x_num is not None else 0
        D_cat = x_cat.size(-1) if x_cat is not None else 0
        time = batch.time
        L, B = time.shape
        D_vae = (D_num + D_cat) * self.d_token
        assert L > 0, "No features for VAE training"
        valid_mask = (
            torch.arange(L, device=time.device)[:, None] < batch.lengths
        ).ravel()  # L * B

        if x_num is not None:
            x_num = x_num.view(-1, D_num)[valid_mask]  # L*B, D_num
        if x_cat is not None:
            x_cat = x_cat.view(-1, D_cat)[valid_mask]  # L*B, D_cat

        x = self.tokenizer(x_num, x_cat)  # (L*B)', D_num + D_cat, d_token
        mu_z = self.encoder_mu(x)  # (L*B)', D_num + D_cat, d_token

        # Handle VAE output based on training mode
        if self.pretrained:
            output = mu_z
        else:
            std_z = self.encoder_std(x)
            output = self.reparametrize(mu_z, std_z)

        with_pad = torch.zeros(L * B, D_vae, device=output.device)
        with_pad[valid_mask] = output.view(-1, D_vae)
        with_pad = with_pad.view(L, B, D_vae)

        # Prepare return values
        seq = Seq(tokens=with_pad, lengths=batch.lengths, time=time)
        if self.frozen or self.pretrained:
            return seq
        return (seq, {"mu_z": mu_z, "std_z": std_z})


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_token: int = 6,
        n_head: int = 2,
        factor: int = 64,
        num_names: Sequence[str] | None = None,
        cat_cardinalities: Mapping[str, int] = None,
        batch_transforms: list | None = None,
    ):
        super(Decoder, self).__init__()
        self.batch_transforms = batch_transforms
        self.d_token = d_token
        self.num_names = num_names
        self.cat_cardinalities = cat_cardinalities
        num_counts = 1  # Time
        if self.num_names:
            num_counts += len(self.num_names)

        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.reconstructor = Reconstructor(
            num_counts, self.cat_cardinalities or {}, d_token=d_token
        )

    def forward(self, seq: Seq) -> PredBatch:
        x = seq.tokens
        L, B, D_vae = x.shape
        D = D_vae // self.d_token
        assert D * self.d_token == D_vae, "Invalid token dimensions"
        # Process sequence mask and reshape inputs
        valid_mask = (
            torch.arange(L, device=x.device)[:, None] < seq.lengths
        ).ravel()  # L, B
        x = x.view(L * B, D, self.d_token)[valid_mask]

        # Decode and reconstruct
        h = self.decoder(x)
        recon_num, recon_cat = self.reconstructor(h)

        # Prepare numerical features
        num_features = None
        with_pad = torch.zeros(L * B, recon_num.shape[1], device=x.device)
        with_pad[valid_mask] = recon_num
        recon_num = with_pad.view(L, B, -1)
        time = recon_num[:, :, 0]
        if self.num_names:
            D_num = len(self.num_names)
            num_features = recon_num[:, :, 1 : D_num + 1]

        # Prepare categorical features
        cat_features = {}
        if self.cat_cardinalities:
            for name, cat in recon_cat.items():
                D_cat = cat.shape[-1]
                with_pad = torch.zeros(L * B, D_cat, device=x.device)
                with_pad[valid_mask] = cat
                cat_features[name] = with_pad.view(L, B, D_cat)
        return PredBatch(
            lengths=seq.lengths,
            time=time,
            num_features=num_features,
            num_features_names=self.num_names,
            cat_features=cat_features if cat_features else None,
        )

    def generate(
        self, seq: Seq, topk=1, temperature=1.0, orig_hist: GenBatch = None
    ) -> GenBatch:
        batch = self.forward(seq).to_batch(topk, temperature)
        if self.batch_transforms is not None:
            if orig_hist is not None:
                batch_len = batch.shape[0]
                hist = deepcopy(orig_hist)
                for tf in self.batch_transforms:
                    tf(hist)
                hist.append(batch)
                batch = hist
            for tf in reversed(self.batch_transforms):
                tf.reverse(batch)
            if orig_hist is not None:
                batch = batch.tail(batch_len)
        return batch
