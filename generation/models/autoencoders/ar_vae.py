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

from .modules import (
    SimpleTokenizer,
    Transformer,
    TransformerDecoderBlock,
)
from .utils import get_features_after_transform


class AR_VAE(BaseAE):
    def __init__(self, data_conf: LatentDataConfig, model_config):
        super().__init__()
        model_config: AEConfig = model_config.autoencoder
        vae_params = {
            "num_layers": model_config.params.get("num_layers", 2),
            "d_token": model_config.params.get("d_token", 6),
            "n_head": model_config.params.get("n_head", 2),
            "factor": model_config.params.get("factor", 32),
        }
        batch_transforms = create_instances_from_module(
            batch_tfs, model_config.batch_transforms
        )
        num_names, cat_cardinalities = get_features_after_transform(
            data_conf, batch_transforms, model_config
        )
        assert not num_names, f"Encode numerical into categorical! {num_names}"

        self.encoder = Encoder(
            **vae_params,
            pretrain=model_config.pretrain,
            cat_cardinalities=cat_cardinalities,
            batch_transforms=batch_transforms,
        )
        self.model_config = model_config

        self.decoder = Decoder(
            **vae_params,
            cat_cardinalities=cat_cardinalities,
            batch_transforms=batch_transforms,
        )

    def forward(self, batch: GenBatch) -> PredBatch:
        """
        Forward pass of the Variational AutoEncoder
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        assert not self.encoder.pretrained
        hidden, params = self.encoder(batch, reparametrize=True)
        x = self.decoder(hidden, batch)
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
        factor: int = 32,
        pretrain: bool = False,
        cat_cardinalities: Mapping[str, int] | None = None,
        batch_transforms: list | None = None,
    ):
        super(Encoder, self).__init__()
        self.d_token = d_token
        self.pretrained = not pretrain

        self.batch_transforms = batch_transforms

        self.tokenizer = SimpleTokenizer(
            categories=list(cat_cardinalities.values()), d_token=d_token
        )
        self._out_dim = len(cat_cardinalities) * self.d_token

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

        x_cat = batch.cat_features
        D_cat, (L, B) = x_cat.size(-1), batch.shape
        D_vae = (D_cat) * self.d_token
        assert L > 0, "No features for VAE training"
        valid_mask = batch.valid_mask.ravel()  # L * B
        x_cat = x_cat.view(-1, D_cat)[valid_mask]  # L*B, D_cat

        x = self.tokenizer(x_cat)  # (L*B)', D_cat, d_token
        mu_z = self.encoder_mu(x)  # (L*B)', D_cat, d_token

        if reparametrize:
            std_z = self.encoder_std(x)
            output = self.reparametrize(mu_z, std_z)
        else:
            output = mu_z

        with_pad = torch.zeros(L * B, D_vae, device=output.device)
        with_pad[valid_mask] = output.view(-1, D_vae)
        with_pad = with_pad.view(L, B, D_vae)

        # Prepare return values
        seq = Seq(tokens=with_pad, lengths=batch.lengths, time=batch.time)
        if reparametrize:
            return (seq, {"mu_z": mu_z, "std_z": std_z})
        else:
            return seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_token: int = 6,
        n_head: int = 2,
        factor: int = 32,
        cat_cardinalities: Mapping[str, int] = None,
        batch_transforms: list | None = None,
    ):
        super(Decoder, self).__init__()
        self.batch_transforms = batch_transforms
        self.d_token = d_token
        self.cat_cardinalities = cat_cardinalities
        # Tokenizers
        self.pos_emb = nn.Parameter(torch.zeros(1, len(cat_cardinalities) + 1, d_token))
        self.start_token = nn.Parameter(torch.randn((1, 1, d_token)))
        self.embeddings = SimpleTokenizer(list(cat_cardinalities.values()), d_token)
        # Transformer
        self.transformer = nn.ModuleList([])
        for _ in range(num_layers):
            new_block = TransformerDecoderBlock(
                d_token, n_head, dim_feedforward=factor * d_token, dropout=0.1
            )
            self.transformer.append(new_block)
        # Reconstructor
        self.reconstructors = nn.ModuleDict(
            {
                cat_name: nn.Linear(d_token, cat_dim)
                for cat_name, cat_dim in cat_cardinalities.items()
            }
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.embeddings.category_embeddings.weight, std=0.02)
        nn.init.normal_(self.start_token, std=0.02)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.float32), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _forward(self, z_seq: torch.Tensor, tgt_tokens: torch.Tensor = None):
        B, L, _ = z_seq.shape
        x = self.start_token.expand((B, -1, -1))
        if tgt_tokens is not None:
            x = torch.cat([x, self.embeddings(tgt_tokens)], dim=1)
        assert x.shape[1] <= L + 1
        x = x[:, :L]
        T = x.shape[1]

        x = x + self.pos_emb[:, :T, :]

        tgt_mask = self._generate_square_subsequent_mask(T).to(
            x.device, non_blocking=True
        )
        for layer in self.transformer:
            x = layer(x, z_seq, tgt_mask=tgt_mask)

        recon_x_cat = {}
        for i, cat_name in enumerate(list(self.cat_cardinalities)[:T]):
            recon_x_cat[cat_name] = self.reconstructors[cat_name](x[:, i])
        return recon_x_cat

    def _generate(self, z_seq: torch.Tensor):
        B, L, _ = z_seq.shape
        generated = None

        for i in range(L):
            logits = self._forward(z_seq, generated)  # list[(B, vocab_size)]
            last_logit = logits[list(self.cat_cardinalities)[i]]
            next_token = last_logit.argmax(dim=-1, keepdim=True)  # B, 1
            if generated is None:
                generated = next_token
            else:
                generated = torch.cat([generated, next_token], dim=1)
        return logits

    def _prepare_tensors(self, seq: Seq, batch: GenBatch = None):
        z_seq = seq.tokens
        L, B, D_vae = z_seq.shape
        D = D_vae // self.d_token
        assert D * self.d_token == D_vae, "Invalid token dimensions"
        # Process sequence mask and reshape inputs
        valid_mask = (
            torch.arange(L, device=z_seq.device)[:, None] < seq.lengths
        ).ravel()  # L * B
        z_seq = z_seq.view(L * B, D, self.d_token)[valid_mask]

        if batch is not None:
            return z_seq, batch.cat_features.view(L * B, D)[valid_mask]
        return z_seq

    def _dict_to_pred(self, logits: dict, seq: Seq):
        L, B, _ = seq.tokens.shape
        valid_mask = (
            torch.arange(L, device=seq.tokens.device)[:, None] < seq.lengths
        ).ravel()
        cat_features = {}
        for name, cat in logits.items():
            D_cat = cat.shape[-1]
            with_pad = torch.zeros(L * B, D_cat, device=seq.tokens.device)
            with_pad[valid_mask] = cat
            cat_features[name] = with_pad.view(L, B, D_cat)
        return PredBatch(
            lengths=seq.lengths,
            time=torch.zeros((L, B), device=valid_mask.device),
            cat_features=cat_features,
        )

    def forward(self, seq: Seq, batch: GenBatch) -> PredBatch:
        z_seq, tgt_tokens = self._prepare_tensors(seq, batch)
        logits = self._forward(z_seq, tgt_tokens)
        return self._dict_to_pred(logits, seq)

    def generate(
        self, seq: Seq, topk=1, temperature=1.0, orig_hist: GenBatch = None
    ) -> GenBatch:
        z_seq = self._prepare_tensors(seq)
        logits = self._generate(z_seq)
        batch = self._dict_to_pred(logits, seq).to_batch(topk, temperature)
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
