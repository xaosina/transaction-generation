import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from ebes.model import BaseModel, TakeLastHidden, ValidHiddenMean
from ebes.model.seq2seq import Projection
from ebes.types import Seq

from generation.models import autoencoders
from generation.models.autoencoders.base import AEConfig
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from ..encoders import AutoregressiveEncoder, LatentEncConfig
from .base import BaseGenerator, ModelConfig


class ARMatchingGenerator(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        self.data_conf = data_conf
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
        assert self.encoder.model.net.__class__.__name__ == "GRU"

        data_conf.check_focus_on(self.autoencoder.encoder.use_time)

    def forward(self, x: GenBatch) -> torch.tensor:
        """
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        gru = self.encoder.model.net
        (L, B), device = x.shape, x.device
        target = deepcopy(x)
        x: Seq = self.autoencoder.encoder(x)  # Sequence of [L, B, D]
        D = x.tokens.shape[2]

        available = target.valid_mask  # L, B
        current_ids = torch.zeros(B, dtype=torch.long, device=device)
        preds = []
        matched_indices = []
        hx = self.encoder.model.hx
        if hx is not None:
            hx = hx.expand(-1, B, -1).contiguous()
        for t in range(L - 1):
            available[current_ids, torch.arange(B)] = False
            input_event = x.tokens.take_along_dim(current_ids[None, :, None], 0)
            assert input_event.shape == (1, B, D)

            out, hx = gru(input_event, hx)
            out = Seq(tokens=out, lengths=None, time=None)  # 1, B, H
            out = self.encoder.projector(out)  # 1, B, D
            out = self.autoencoder.decoder(out)  # PredBatch
            preds.append(out)
            with torch.no_grad():
                matched_id = self.match_best_target(
                    out=out, target=target, current_ids=current_ids, available=available
                )
            current_ids = matched_id
            matched_indices.append(matched_id)
        matched_indices = torch.stack(matched_indices, dim=0)  # L-1, B
        # Now compute loss
        loss = 0
        valid_mask = target.valid_mask[1:]
        for name in self.data_conf.focus_num:
            if name == self.data_conf.time_name:
                pred = torch.cat([pred.time for pred in preds], 0)  # L-1, B
                true = target.time.take_along_dim(matched_indices, 0) # L-1, B
            else:
                pred = torch.cat([pred[name] for pred in preds], 0)  # L-1, B
                true = target[name].take_along_dim(matched_indices, 0) # L-1, B
            loss += (pred - true).abs()[valid_mask].mean()
        for name in self.data_conf.focus_cat:
            pred = torch.cat([pred[name] for pred in preds], 0).movedim(0, -1)  # B C, L-1
            true = target[name].take_along_dim(matched_indices, 0).T # B, L-1
            loss += F.cross_entropy(pred, true, reduction="none")[valid_mask.T].mean()
        return loss

    def match_best_target(
        self,
        out: PredBatch,
        target: GenBatch,
        current_ids: torch.Tensor,
        available: torch.Tensor,
    ):
        L, B = target.shape
        cost = torch.zeros((L, B), device=target.device)
        for name in self.data_conf.focus_cat:
            pred = out.cat_features[name][0]  # B, C
            true = target[name].movedim(0, -1)  # B, L
            C = pred.shape[1]
            pred = pred.unsqueeze(-1).expand(B, C, L)
            ce_loss = F.cross_entropy(pred, true, reduction="none").T  # L, B
            assert ce_loss.shape == (L, B)
            cost += ce_loss
        for name in self.data_conf.focus_num:
            if name == self.data_conf.time_name:
                pred = out.time  # 1, B
                true = target.time  # L, B
            else:
                pred = out[name]  # 1, B
                true = target[name]  # L, B
            mae = (pred - true).abs()
            assert mae.shape == (L, B)
            cost += mae
        # Also punish matching for not predicting closest event
        next_in_line = available.to(torch.int8).argmax(0)[None]  # 1, B
        # next_in_line = target.time.take_along_dim(next_in_line, 0)
        # cost += (target.time - next_in_line) ** 2
        mask = deepcopy(available)
        mask[torch.arange(L, device=target.device)[:, None] > (next_in_line + 5)] = False
        #
        cost[~mask] = torch.inf
        res = cost.argmin(0)
        return res

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
