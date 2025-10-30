import logging
from typing import List, Optional

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

        self.help_net, _ = self._init_helpnet()
        self.history_encoder, hist_dim = self._init_history_encoder()

        self.repeat_samples: int = int(model_config.params.get("repeat_samples", 1))
        logger.info(
            f"Generation is reapeating {self.repeat_samples} times to smooth time feature."
        )

        self.history_len: int = int(model_config.params["history_len"])
        prefix_dim = model_config.params.get("prefix_dim", None)
        self.generation_len: int = int(model_config.params["generation_len"])
        self.model = DiffusionTabularModel(
            data_conf=self.data_conf, 
            model_config=self.model_config,
            outer_history_encoder_dim=hist_dim,
            prefix_dim=prefix_dim
        )

        self.fix_features: Optional[set] = set(
            self.model_config.params.get("fix_features", [])
        )
        self.diff_features: Optional[set] = set(
            self.model_config.params.get("diff_features", [])
        )
        self.focus: set = self.fix_features | self.diff_features

    @staticmethod
    def _to_blf(t: torch.Tensor) -> torch.Tensor:
        # (L, B, F) -> (B, L, F)  или (L, B) -> (B, L)
        return t.permute(1, 0, *range(2, t.ndim))  # шок

    def _init_helpnet(self) -> BaseGenerator:
        # initializing history encoder
        params = self.model_config.params.get("help_net")
        if not params:
            print("No HelpNet provided.")
            return None, None

        params = deepcopy(params)
        checkpoint = params.pop("checkpoint", None)
        name = params.get("name")

        if name == "GroundTruthGenerator":
            enc = BaseModel.get_model(name)
            hist_dim = None
        elif name == "GRU":
            input_size = params.get("input_size", 256)
            gru_cfg = dict(input_size=input_size, num_layers=1, hidden_size=256)
            hist_dim = 256
            enc = nn.Sequential(BaseModel.get_model("GRU", **gru_cfg), TakeLastHidden())
        else:
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            enc = BaseGenerator.get_model(name, self.data_conf, cfg)
            hist_dim = getattr(getattr(enc, "encoder", None), "output_dim", None)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = enc.load_state_dict(ckpt["model"], strict=True)
            logger.info("History encoder: " + str(msg))
            enc = freeze_module(enc)

        return enc, hist_dim

    def _init_history_encoder(self) -> BaseGenerator:

        params = self.model_config.params.get("history_encoder")
        if not params:
            print("No history encoder provided.")
            return None, None

        params = deepcopy(params)
        checkpoint = params.pop("checkpoint", None)
        name = params.get("name")

        if name == "AutoregressiveGenerator":
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            enc = BaseGenerator.get_model(name, self.data_conf, cfg)
            hist_dim = getattr(getattr(enc, "encoder", None), "output_dim", None)
        else:
            raise ValueError("Not implemented any methods yet. Only GRU.")
        
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = enc.load_state_dict(ckpt["model"], strict=True)
            logger.info("History encoder: " + str(msg))
            enc = freeze_module(enc)

        return enc, hist_dim


    def _select_indices(self, names_all: List[str]) -> List[int]:
        if not names_all:
            return []
        if self.focus is None:
            return list(range(len(names_all)))
        return [i for i, n in enumerate(names_all) if n in self.focus]

    def feature_preprocess(self, hist, tgt=None):
        tgt_num, tgt_cat = None, None

        num_names_all = hist.num_features_names or []
        cat_names_all = hist.cat_features_names or []

        num_idxs = self._select_indices(num_names_all)
        cat_idxs = self._select_indices(cat_names_all)

        if not cat_idxs:
            raise ValueError(
                "At least one categorical feature is required for this model."
            )

        # numerical preprocessing
        hist_num_features = []
        if self.data_conf.time_name in self.focus:
            hist_num_features.append(hist.time.unsqueeze(-1))
        if num_idxs:
            hist_num_features.append(hist.num_features[..., num_idxs])

        hist_num = torch.cat(hist_num_features, dim=-1)
        # hist_num = (
        #     torch.cat(
        #         [hist.time.unsqueeze(-1), hist.num_features[..., num_idxs]], dim=-1
        #     )
        #     if num_idxs
        #     else hist.time.unsqueeze(-1)
        # )
        
        tgt_num = None
        tgt_num_features = []
        if tgt is not None:
            if self.data_conf.time_name in self.focus:
                tgt_num_features.append(tgt.time.unsqueeze(-1))
            if num_idxs:
                tgt_num_features.append(tgt.num_features[..., num_idxs])

            tgt_num = torch.cat(tgt_num_features, dim=-1)
        
        # if tgt is not None:
        #     tgt_num = (
        #         torch.cat(
        #             [tgt.time.unsqueeze(-1), tgt.num_features[..., num_idxs]], dim=-1
        #         )
        #         if num_idxs
        #         else tgt.time.unsqueeze(-1)
        #     )

        # categorical
        hist_cat = hist.cat_features[..., cat_idxs]
        tgt_cat = None
        if tgt is not None:
            tgt_cat = tgt.cat_features[..., cat_idxs]

        # (L,B,*) -> (B,L,*)
        hist_num = self._to_blf(hist_num)
        hist_cat = self._to_blf(hist_cat)
        if tgt_num is not None:
            tgt_num = self._to_blf(tgt_num)
        if tgt_cat is not None:
            tgt_cat = self._to_blf(tgt_cat)

        cat_order = [cat_names_all[i] for i in cat_idxs]
        return tgt_num, tgt_cat, hist_num, hist_cat, cat_order

    def forward(self, x: GenBatch) -> torch.Tensor:
        x.time = x.time.float()
        x.target_time = x.target_time.float()

        embeddings = None
        if self.history_encoder is not None:
            embeddings = self.history_encoder.get_embeddings(x)
        self.set_numerical_diff_mask(x)

        hist = x.tail(self.history_len)
        tgt = x.get_target_batch()

        data = self.feature_preprocess(hist, tgt)
        loss = self.model.compute_loss(*data, embeddings)
        return loss

    def set_numerical_diff_mask(self, data):
        diff_idx = []
        if self.data_conf.time_name in self.diff_features:
            diff_idx.append(0)
        for feature in self.diff_features:
            if feature in self.data_conf.num_names:
                diff_idx.append((1 * (self.data_conf.time_name in self.focus)) + data.num_features_names.index(feature)) # 1 because we have time
        self.model.time_diff_.set_diffusion_mask(sorted(diff_idx), 1 + len(self.data_conf.num_names))
        

    def generate(
        self, x: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:

        hist = x.tail(self.history_len)
        tgt = x.get_target_batch()

        embeddings = None
        if self.history_encoder is not None:
            embeddings = self.history_encoder.get_embeddings(x)

        _, _, hist_num, hist_cat, cat_order = self.feature_preprocess(
            hist, tgt
        )

        self.set_numerical_diff_mask(x)

        # Provide categorical features inside sampling proc.
        # Check that numerical sampling with masking is working.
        provided_num_features = []
        provided_cat_features = []
        num_features_used = []
        cat_features_used = []
        if self.help_net is not None:
            pred = self.help_net.generate(x, self.generation_len, topk=-1)
            num_features_number = 0
            if self.data_conf.time_name in self.focus:
                num_features_number += 1
                provided_num_features.append(pred.time.unsqueeze(-1))
            for num_feature in self.data_conf.num_names:
                if num_feature in self.focus:
                    idx = pred.num_features_names.index(num_feature)
                    provided_num_features.append(pred.num_features[..., [idx]])
                    num_features_used.append(idx)

            for cat_feature in self.data_conf.cat_cardinalities:
                if cat_feature in self.focus:
                    idx = pred.cat_features_names.index(cat_feature)
                    provided_cat_features.append(pred.cat_features[..., [idx]])
                    cat_features_used.append(idx)


            tgt_num = self._to_blf(torch.cat(provided_num_features, dim=-1).float())
            tgt_cat = self._to_blf(torch.cat(provided_cat_features, dim=-1).long())

        # # self.model.time_diff_.set_diffusion_mask(diff_idx, num_features_dim=hist_num.size(-1))
        # breakpoint()
        # # Optional - get features from history_encoder
        # provided_num_feature = None
        # if self.history_encoder is not None:
        #     breakpoint()
        #     pred = self.history_encoder.generate(x, self.generation_len, topk=-1)
        #     if self.fix_features:
        #         if self.data_conf.time_name in self.fix_features:
        #             provided_time_feature = self._to_blf(pred.time).unsqueeze(-1)
        #         for name in self.data_conf.num_names:
        #             provided_num_feature = []
        #             if idx in self.fix_features:
        #                 idx = pred.num_features_index.index(name)
        #                 provided_num_feature.append(self._to_blf(pred.num_features[..., [idx]]))
                
        #         for name in pred.cat_features_names:
        #             provided_cat_feature = []
        #             if name in self.fix_features:
        #                 idx = pred.cat_features_names.index(name)
        #                 provided_cat_feature.append(self._to_blf(pred.cat_features[..., [idx]]))

        B, _, F_cat = hist_cat.shape
        _, _, F_num = hist_num.shape

        pred_cat = torch.empty(
            B,
            gen_len,
            F_cat,
            self.repeat_samples,
            dtype=torch.long,
        ).to("cuda")

        pred_num = torch.empty(
            B,
            gen_len,
            F_num,
            self.repeat_samples,
            dtype=torch.float,
        ).to("cuda")
        # breakpoint()
        for i in range(self.repeat_samples):
            pc, pn = self.model.sample(
                hist_num,
                hist_cat,
                gen_len,
                cat_order,
                tgt_e=tgt_cat if bool(set(self.data_conf.cat_cardinalities.keys()) & self.fix_features) else None,
                tgt_x=tgt_num,
                hist_emb=embeddings,
            )

            pred_cat[..., i] = pc
            pred_num[..., i] = pn
        # Take time from zero index and mean through all repeat_samples - more stable
        # Take other features - amount shouldn't be averaged - because it's too unstable thing.
        # Make them long. No taking mode from 10 repeats thing to preserve natural distribution and save cardinality


        cat = x.target_cat_features.clone()
        num = x.target_num_features.clone()
        time = x.target_time.clone()
        
        generated_cats = set(self.data_conf.cat_cardinalities.keys()) & self.diff_features
        if generated_cats:
            for feature in generated_cats:
                new_vals = pred_cat[..., 0, 0]
                idx = x.cat_features_names.index(feature)
                cat[..., idx] = new_vals.long().permute(1, 0)

        # Numerical features process
        if self.data_conf.time_name in self.focus:
            pred_time = pred_num[..., [0], :]
            pred_num = pred_num[..., 1:, :]
        
        if self.data_conf.time_name in self.diff_features:
            time = pred_time[..., 0, :].mean(dim=-1).permute(1, 0)
        
        generated_nums = set(self.data_conf.num_names) & self.diff_features
        for i, feature in enumerate(generated_nums):
            new_vals = pred_num[..., i, 0] # 0 - takes first. If you need some aggregating preprocess based on several samplings for your num feature - welcome
            idx = x.num_features_names.index(feature)
            num[..., idx] = new_vals.float().permute(1, 0)


        # if has_provided:
        #     cat = x.target_cat_features.clone()
        #     num = x.target_num_features.clone()
        #     time = x.target_time.clone()

        #     if (
        #         provided_cat_feature is not None
        #         and self.diff_features in x.cat_features_names
        #     ):
        #         assert (
        #             pred_cat.shape[-2] == 1
        #         ), "Expected exactly one provided categorical feature."
        #         new_vals = pred_cat[..., 0, 0]
        #         cat_idx = x.cat_features_names.index(self.diff_features)

        #         cat[..., cat_idx] = new_vals.long().permute(1, 0)
        #     elif (
        #         provided_num_feature is not None
        #         and self.diff_features in x.num_features_names
        #     ):
        #         assert pred_num.shape[-2] == 2, "Expected time and one numeric feature."
        #         new_vals = pred_num[..., 1, 0]
        #         num_idx = x.num_features_names.index(self.diff_features)
        #         num[..., num_idx] = new_vals.permute(1, 0)
        #     else:
        #         time = pred_num[..., 0, :].mean(dim=-1).permute(1, 0)
        # else:
        #     time = pred_num[..., 0, :].mean(dim=-1).permute(1, 0)
        #     num = pred_num[..., 1:, 0].permute(1, 0, 2)
        #     cat = pred_cat[..., 0].permute(1, 0, 2).long()

        sampled_batch = self.toGenBatch(
            cat, num, time, x.num_features_names, x.cat_features_names
        )
        return sampled_batch

    def toGenBatch(self, cat, num, time, num_features_names, cat_features_names):
        s_length = torch.ones(cat.size(0)) * cat.size(1)
        assert time.ndim == 2
        return GenBatch(
            lengths=s_length,
            time=time,
            index=None,
            num_features=num,
            cat_features=cat,
            cat_features_names=cat_features_names,
            num_features_names=num_features_names,
        )
