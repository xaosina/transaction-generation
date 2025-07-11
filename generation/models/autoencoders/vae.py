from copy import deepcopy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from ebes.types import Seq

from generation.data import batch_tfs
from generation.data.batch_tfs import NewFeatureTransform
from generation.data.data_types import GenBatch, PredBatch, LatentDataConfig
from generation.data.utils import create_instances_from_module
from generation.models.autoencoders.base import BaseAE, AEConfig


@dataclass
class VaeConfig:
    num_layers: int = 2
    d_token: int = 6
    n_head: int = 2
    factor: int = 64
    pretrained: bool = False


class VAE(BaseAE):
    def __init__(self, data_conf: LatentDataConfig, model_config: AEConfig):
        super().__init__()
        self.encoder = Encoder(
            model_config.vae,
            cat_cardinalities=data_conf.cat_cardinalities,
            num_features=data_conf.num_names,
            batch_transforms=model_config.preprocessor.batch_transforms,
        )

        self.decoder = Decoder(
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


class Encoder(nn.Module):
    def __init__(
        self,
        vae_conf: VaeConfig,
        cat_cardinalities: Mapping[str, int] | None = None,
        num_features: Sequence[str] | None = None,
        batch_transforms: Mapping[str, Mapping[str, Any] | str] | None = None,
        bias=True,
    ):
        super(Encoder, self).__init__()
        self.d_token = vae_conf.d_token
        self.pretrained = vae_conf.pretrained

        if num_features is not None:
            num_count = len(num_features)
        else:
            num_count = 0
        self.batch_transforms = create_instances_from_module(
            batch_tfs, batch_transforms
        )
        if self.batch_transforms:
            for tfs in self.batch_transforms:
                assert isinstance(tfs, NewFeatureTransform)
                for _ in tfs.num_names:
                    num_count += 1
                for cat_name, card in tfs.cat_cardinalities.items():
                    cat_cardinalities[cat_name] = card
                for _ in tfs.num_names_removed:
                    num_count -= 1
                cat_cardinalities = {
                    k: v
                    for k, v in cat_cardinalities.items()
                    if k not in tfs.cat_names_removed
                }
        self.tokenizer = Tokenizer(
            d_numerical=num_count,
            categories=list(cat_cardinalities.values()) if cat_cardinalities else None,
            d_token=vae_conf.d_token,
            bias=bias,
        )

        self.encoder_mu = Transformer(
            vae_conf.num_layers,
            vae_conf.d_token,
            vae_conf.n_head,
            vae_conf.d_token,
            vae_conf.factor,
        )
        self.encoder_std = None
        if not self.pretrained:
            self.encoder_std = Transformer(
                vae_conf.num_layers,
                vae_conf.d_token,
                vae_conf.n_head,
                vae_conf.d_token,
                vae_conf.factor,
            )

    def reparametrize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: GenBatch) -> Seq:
        if self.batch_transforms:
            for tf in self.batch_transforms:
                tf(batch)

        x_num, x_cat = batch.num_features, batch.cat_features
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
        return seq if self.pretrained else (seq, {"mu_z": mu_z, "std_z": std_z})


class Decoder(nn.Module):
    def __init__(
        self,
        vae_conf: VaeConfig,
        num_names: Sequence[str] | None = None,
        cat_cardinalities: Mapping[str, int] = None,
    ):
        super(Decoder, self).__init__()
        self.d_token = vae_conf.d_token
        self.num_names = num_names
        self.cat_cardinalities = cat_cardinalities
        num_counts = 1  # Time
        if self.num_names:
            num_counts += len(self.num_names)

        self.decoder = Transformer(
            vae_conf.num_layers,
            vae_conf.d_token,
            vae_conf.n_head,
            vae_conf.d_token,
            vae_conf.factor,
        )
        self.reconstructor = Reconstructor(
            num_counts,
            self.cat_cardinalities or {},
            d_token=vae_conf.d_token,
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

    def generate(self, seq: Seq) -> GenBatch:
        return self.forward(seq).to_batch()


class Tokenizer(nn.Module):

    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = self.bias
            x = x + bias[None]

        return x


class Reconstructor(nn.Module):
    def __init__(self, d_numerical, cat_cardinalities, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token

        self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleDict()
        for cat_name, cat_dim in self.cat_cardinalities.items():
            recon = nn.Linear(d_token, cat_dim)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons[cat_name] = recon

    def forward(self, h):
        h_num = h[:, : self.d_numerical]
        h_cat = h[:, self.d_numerical :]
        assert (h_num.shape[-2] + h_cat.shape[-2]) == (
            self.d_numerical + len(self.cat_cardinalities.values())
        ), f"{h_num.shape[-2], h_cat.shape[-2], self.d_numerical, self.cat_cardinalities.values()}"

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = {}
        for i, cat_name in enumerate(self.cat_cardinalities.keys()):
            recon_x_cat[cat_name] = self.cat_recons[cat_name](h_cat[:, i])

        return recon_x_num, recon_x_cat


class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        activation="relu",
        prenormalization=True,
        initialization="kaiming",
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    "linear0": nn.Linear(d_token, d_hidden),
                    "linear1": nn.Linear(d_hidden, d_token),
                    "norm1": make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer["linear0"](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer["linear1"](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization="kaiming"):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ["xavier", "kaiming"]

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == "xavier" and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):

        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
