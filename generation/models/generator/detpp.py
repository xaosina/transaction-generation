from copy import deepcopy
from dataclasses import replace

import torch
import torch.nn as nn
from ebes.model import BaseSeq2Seq
from ebes.types import Seq

from generation.models import autoencoders
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch, valid_mask
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
        x = self.relu(x)
        return x.reshape(b, self.output_dim)  # (B, KO).

    def forward(self, seq: Seq):
        mask = valid_mask(seq)
        x = seq.tokens
        assert x.ndim > 2  # (L, B, D).
        shape = list(x.shape)
        x_masked = x[mask]  # (V, D).
        v = len(x_masked)
        x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(
            *([v] + shape[2:-1] + [self.output_dim])
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

        k_factor = model_config.params["k_factor"]
        assert k_factor >= 1
        self.k_output = int(k_factor * data_conf.generation_len)
        self.k_gen = model_config.params.get("k_gen") or data_conf.generation_len
        # elif (k_output is not None) and (k_gen is not None):
        #     self.k_output = model_config.params.get("k_output")
        #     self.k_gen = model_config.params.get("k_gen")
        # else:
        #     raise ValueError("Specify either k_factor or (k_output and k_gen)")
        self.next_k_head = ConditionalHead(
            self.autoencoder.encoder.output_dim, self.k_output
        )
        self.presence_head = nn.Linear(self.autoencoder.encoder.output_dim, 1)

    def _apply_delta(self, x: GenBatch):
        x = deepcopy(x)
        deltas = x.time
        deltas[:, 1:] -= deltas[:, :-1]
        deltas[:, 0] = 0
        # deltas.clip_(min=0, max=self._max_time_delta)
        x.time = deltas
        return x

    def _sort_time_and_revert_delta(self, hist, pred):
        # Sort by time.
        order = pred.time.argsort(dim=0)  # (L, B).
        for attr in ["time", "num_features", "cat_features"]:
            tensor = getattr(pred, attr)
            if tensor is None:
                continue
            shaped_order = order.reshape(
                *(list(order.shape) + [1] * (tensor.ndim - order.ndim))
            )
            tensor = tensor.take_along_dim(shaped_order, dim=0)
            setattr(pred, attr, tensor)
        # Revert delta from hist
        pred.time += hist.time[hist.lengths - 1, torch.arange(hist.shape[1])]
        return pred

    def forward(self, x: GenBatch) -> PredBatch:
        L, B = x.shape
        x = self._apply_delta(x)
        x = self.autoencoder.encoder(x, copy=False)  # Sequence of [L, B, D]
        x = self.encoder(x)  # [L, B, D]
        x = self.next_k_head(x)  # L, B, K * D
        x = Seq(
            tokens=x.tokens.reshape(L, B * self.k_output, -1),
            lengths=x.lengths.repeat_interleave(self.k_output, 0),
            time=None,
        )
        presence_scores = self.presence_head(x.tokens).reshape(L, B, -1)  # [L, B, K]
        x = self.autoencoder.decoder(x)  # [L, B * K, preds]
        x = x.k_reshape(self.k_output)  # [L, B, K, pred]

        return (x, presence_scores)

    def get_embeddings(self, hist: GenBatch):
        hist = deepcopy(hist)
        x = self.autoencoder.encoder(hist)
        x = self.encoder.generate(x)
        assert x.tokens.shape[0] == 1
        return x.tokens[0]

    def generate(
        self,
        hist: GenBatch,
        gen_len: int,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
        orig_hist = deepcopy(hist)
        hist = deepcopy(hist)
        already_generated = 0
        with torch.no_grad():
            for _ in range(0, gen_len, self.k_gen):
                # 1. Generate k_gen
                L, B = hist.shape
                x = self._apply_delta(hist)
                x = self.autoencoder.encoder(x, copy=False)
                x = self.encoder.generate(x)  # Sequence of shape [1, B, D]
                x = self.next_k_head(x)  # 1, B, K * D
                # Filter events
                x = x.tokens.reshape(B, self.k_output, -1).transpose(0, 1)  # K, B, D
                presence_scores = self.presence_head(x).reshape(self.k_output, B)
                topk_indices = torch.topk(presence_scores, self.k_gen, dim=0)[1]
                x = torch.take_along_dim(x, topk_indices.unsqueeze(-1), dim=0)
                x = Seq(
                    tokens=x,
                    lengths=torch.full((B,), self.k_gen, device=hist.device),
                    time=None,
                )
                # Reconstruct
                x = self.autoencoder.decoder.generate(x, topk=topk, temperature=temperature)
                x.time = x.time.clip(min=0)
                x = self._sort_time_and_revert_delta(hist, x)
                # 2. Save gen and append to history
                already_generated += self.k_gen
                hist.append(x)  # Append GenBatch, result is [L+K, B, D]
        
        # Cut excess predictions:
        pred_batch = hist.tail(already_generated).head(gen_len)
        
        if with_hist:
            orig_hist.append(pred_batch)
            return orig_hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return pred_batch  # Return GenBatch of size [gen_len, B, D]