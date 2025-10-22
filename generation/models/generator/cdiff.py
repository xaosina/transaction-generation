import logging

import torch
from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from dacite import Config, from_dict
from ebes.model import BaseModel, TakeLastHidden

from . import BaseGenerator, ModelConfig
from .cdiffu.tabular_diffusion_model import DiffusionTabularModel
from copy import deepcopy
from generation.utils import freeze_module
from torch import nn

logger = logging.getLogger()


class CrossDiffusionModel(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.data_conf = data_conf
        self.model_config = model_config

        self.history_encoder, _ = self._init_history_encoder()
        self.repeat_samples = 1
        self.history_len = model_config.params["history_len"]
        self.generation_len = model_config.params["generation_len"]
        self.model = DiffusionTabularModel(
            data_conf=self.data_conf, model_config=self.model_config
        )

    def _init_history_encoder(self) -> BaseGenerator:
        # initializing history encoder
        params = self.model_config.params.get("history_encoder")
        history_encoder = None  # default history encoder
        if params is None:
            print("no history encoder!")
            return None, None

        checkpoint = params.pop("checkpoint", None)
        # Support old encoders
        if params["name"] == "GRU":
            params = {
                "input_size": self.autoencoder.encoder.output_dim,
                "num_layers": 1,
                "hidden_size": 256,
            }
            hist_enc_dim = 256
            history_encoder = nn.Sequential(
                BaseModel.get_model("GRU", **params), TakeLastHidden()
            )
        else:
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            history_encoder = BaseGenerator.get_model(
                params["name"], self.data_conf, cfg
            )
            hist_enc_dim = history_encoder.encoder.output_dim
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = history_encoder.load_state_dict(ckpt["model"], strict=True)
            logger.info("History encoder: " + str(msg))
            history_encoder = freeze_module(history_encoder)
        return history_encoder, hist_enc_dim

    def forward(self, x: GenBatch) -> torch.Tensor:

        ## hist_num contains time and others numerical features
        ## hist_cat contains categorical features
        ## also apply to target_cat and target_num
        hist_batch = x.tail(self.history_len)
        target_batch = x.get_target_batch()

        if hist_batch.num_features_names:
            tgt_num = torch.cat(
                [target_batch.time.unsqueeze(-1), target_batch.num_features], dim=-1
            )
            hist_num = torch.cat(
                [hist_batch.time.unsqueeze(-1), hist_batch.num_features], dim=-1
            )
        else:
            hist_num = hist_batch.time.unsqueeze(-1)
            tgt_num = target_batch.time.unsqueeze(-1)

        ### input from framework is (L,B,F), then transpose all into (B,L,F)
        ### all share the same shape (B,L,F), where F is the feature count in numerical and categorical
        hist_num, hist_cat = hist_num.permute(1, 0, 2), hist_batch.cat_features.permute(
            1, 0, 2
        )
        tgt_num, tgt_cat = tgt_num.permute(1, 0, 2), target_batch.cat_features.permute(
            1, 0, 2
        )

        cat_order = hist_batch.cat_features_names
        loss = self.model.compute_loss(tgt_num, tgt_cat, hist_num, hist_cat, cat_order)

        return loss

    def get_history_emb(self, hist, latent_hist):
        if self.history_encoder is None:
            return None
        hist = deepcopy(hist)
        if isinstance(self.history_encoder, BaseGenerator):
            return self.history_encoder.get_embeddings(hist)  # B, D
        else:
            encoded_hist = latent_hist
            return self.history_encoder(encoded_hist)  # B, D

    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        ### generate multiple samples and take average on
        ### numerical features and take mode on cate features
        # breakpoint()
        hist_batch = hist.tail(self.history_len)

        event_types = None
        if self.history_encoder is not None:

            event_types = self.history_encoder.generate(
                hist, self.generation_len, topk=-1
            ).cat_features.permute(1, 0, 2)

        if hist_batch.num_features_names:
            hist_num = torch.cat(
                [hist_batch.time.unsqueeze(-1), hist_batch.num_features], dim=-1
            )
        else:
            hist_num = hist_batch.time.unsqueeze(-1)

        features_name = [hist_batch.cat_features_names, hist_batch.num_features_names]
        cat_list = hist_batch.cat_features_names
        # breakpoint(

        hist_num, hist_cat = hist_num.permute(1, 0, 2), hist_batch.cat_features.permute(
            1, 0, 2
        )

        pred_cat = torch.empty(
            hist_cat.size(0),
            gen_len,
            hist_cat.size(-1),
            self.repeat_samples,
            dtype=torch.long,
        ).to("cuda")
        pred_num = torch.empty(
            hist_num.size(0),
            gen_len,
            hist_num.size(-1),
            self.repeat_samples,
            dtype=hist_num.dtype,
        ).to("cuda")
        for i in range(self.repeat_samples):
            pred_c, pred_n = self.model.sample(
                hist_num, hist_cat, gen_len, cat_list, tgt_e=event_types
            )

            pred_cat[..., i] = pred_c
            pred_num[..., i] = pred_n

        time = pred_num[..., 0, :].mean(
            dim=-1
        )  # Take time from zero index and mean through all repeat_samples - more stable
        num = pred_num[
            ..., 1:, 0
        ]  # Take other features - amount shouldn't be averaged - because it's too unstable thing.
        pred_cat = pred_cat[
            ..., 0
        ].long()  # Make them long. No taking mode from 10 repeats thing to preserve natural distribution and save cardinality

        sampled_batch = self.toGenBatch(pred_cat, num, time, features_name)
        return sampled_batch

    def toGenBatch(self, cat, num, time, features_name):
        s_length = torch.ones(cat.size(0)) * cat.size(1)
        cat = cat.permute(1, 0, 2)
        num = num.permute(1, 0, 2)
        time = time.permute(1, 0)
        return GenBatch(
            lengths=s_length,
            time=time,
            index=None,
            num_features=num if num.size(-1) > 0 else None,
            cat_features=cat,
            cat_features_names=features_name[0],
            num_features_names=features_name[1],
        )
