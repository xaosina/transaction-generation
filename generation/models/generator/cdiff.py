import torch
from ...data.data_types import GenBatch, LatentDataConfig, PredBatch

from . import BaseGenerator, ModelConfig
from .cdiffu.tabular_diffusion_model import DiffusionTabularModel


class CrossDiffusionModel(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        self.repeat_samples = 10
        self.history_len = model_config.params["history_len"]
        self.generation_len = model_config.params["generation_len"]
        self.model = DiffusionTabularModel(data_conf, model_config)
        self.params = [data_conf, model_config]

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

    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        ### generate multiple samples and take average on
        ### numerical features and take mode on cate features
        # breakpoint()
        hist_batch = hist.tail(self.history_len)

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

        pred_cat = torch.empty(hist_cat.size(0), gen_len, 2, 0).to("cuda")
        pred_num = torch.empty(hist_num.size(0), gen_len, 2, 0).to("cuda")
        for i in range(self.repeat_samples):
            pred_c, pred_n = self.model.sample(hist_num, hist_cat, gen_len, cat_list)

            pred_cat = torch.cat([pred_cat, pred_c.unsqueeze(dim=-1)], dim=-1)
            pred_num = torch.cat([pred_num, pred_n.unsqueeze(dim=-1)], dim=-1)
        pred_num = pred_num.mean(dim=-1).squeeze(-1)
        pred_cat = torch.mode(pred_cat, dim=-1).values.long()
        sampled_batch = self.toGenBatch(pred_cat, pred_num, features_name)

        return sampled_batch

    def toGenBatch(self, pred_cat, pred_num, features_name):

        s_length = torch.ones(pred_cat.size(0)) * pred_cat.size(1)
        pred_cat = pred_cat.permute(1, 0, 2)
        pred_num = pred_num.permute(1, 0, 2)
        return GenBatch(
            lengths=s_length,
            time=pred_num[:, :, 0],
            index=None,
            num_features=pred_num[:, :, 1:],
            cat_features=pred_cat,
            cat_features_names=features_name[0],
            num_features_names=features_name[1],
        )
