from typing import Optional
import torch
import torch.nn.functional as F

from .utils import index_to_log_onehot, log_onehot_to_index
from .utils import log_sample_categorical
from .hist_enc1 import HistoryEncoder
from .type_denoising_ml import TypeDenoisingModule
from .time_denoising_ml import TimeDenoisingModule
from .time_diffusion_model import DiffusionTimeModel
from .type_diffusion_model import DiffusionTypeModel
from .utils import log_sample_categorical_multi_task
from .utils import log_onehot_to_index_multi_task
from copy import deepcopy

class DiffusionTabularModel(torch.nn.Module):
    def __init__(self, data_conf, model_config, outer_history_encoder_dim=None, prefix_dim=None):
        super(DiffusionTabularModel, self).__init__()

        device = "cuda"
        self.device = device

        emb_feature_dim = 1 << model_config.params["emb_dim_features_exp"]
        transformer_dim = 1 << model_config.params["dim_exp"]
        transformer_heads = 1 << model_config.params["heads_exp"]
        num_encoder_layers = model_config.params["encoder_layer"]
        num_decoder_layers = model_config.params["decoder_layer"]
        dim_feedforward = 1 << model_config.params["hidden_scale_exp"]
        self.n_steps = model_config.params["diffusion_steps"]
        num_timesteps = model_config.params["diffusion_steps"]
        self.data_config = data_conf
        self.model_config = model_config
        self.loss_names = self.model_config.params["losses"]
        
        self.fix_features: Optional[set] = set(self.model_config.params.get("fix_features", []))
        self.diff_features: Optional[set] = set(self.model_config.params.get("diff_features", []))

        self.diff_focus = self.fix_features | self.diff_features

        if self.diff_focus is not None:
            self.num_cat_dict = {key: val for key, val in data_conf.cat_cardinalities.items() if key in self.diff_focus}
        else:
            self.num_cat_dict = data_conf.cat_cardinalities

        self.num_classes_list = []

        assert data_conf.time_name is not None, 'It seems you forget provide a time feature.'
        ## because of time interv must in the num features, so we add 1
        len_num_features = 1
        if self.diff_focus is not None:
            len_num_features += len([v for v in data_conf.num_names if v in self.diff_focus])
        else:
            len_num_features += len(data_conf.num_names)

        self.hist_enc_func_ = HistoryEncoder(
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            num_classes=self.num_cat_dict,
            device=device,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            len_numerical_features=len_num_features,
            feature_dim=emb_feature_dim,
        )

        self.denoise_fn_type = TypeDenoisingModule(
            transformer_dim=transformer_dim,
            num_classes=self.num_cat_dict,
            n_steps=num_timesteps,
            transformer_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers,
            device=device,
            len_numerical_features=len_num_features,
            feature_dim=emb_feature_dim,
            outer_history_encoder_dim=outer_history_encoder_dim,
            prefix_dim=prefix_dim,

        )

        self.denoise_fn_dt = TimeDenoisingModule(
            transformer_dim=transformer_dim,
            num_classes=self.num_cat_dict,
            n_steps=num_timesteps,
            transformer_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers,
            device=device,
            feature_dim=emb_feature_dim,
            len_numerical_features=len_num_features,
            outer_history_encoder_dim=outer_history_encoder_dim,
            prefix_dim=prefix_dim,
        )

        self.type_diff_ = DiffusionTypeModel(
            n_steps=self.n_steps,
            denoise_fn=self.denoise_fn_type,
            num_classes=self.num_cat_dict,
        )

        self.time_diff_ = DiffusionTimeModel(
            n_steps=self.n_steps, denoise_func=self.denoise_fn_dt
        )

    def compute_loss(self, tgt_x, tgt_e, hist_x, hist_e, cat_order, hist_emb=None):
        ## tgt_x,tgt_e : B,gen_len,F_{1};B,gen_len,F_{2}
        ## hist_x,hist_e: B,hist_len,F_{1};B,hist_len,F_{2}
        #
        hist = self.get_hist(hist_x, hist_e)
        ## t,pt are (B,)
        t, pt = self.sample_time(tgt_e.size(0), device=self.device)

        losses = {}
        if 'type_loss' in self.loss_names:
            losses['type_loss'] = self.type_diff_.compute_loss(tgt_e, tgt_x, hist, cat_order, t, pt, hist_emb=hist_emb)
        if 'time_loss' in self.loss_names:
            losses['time_loss'] = self.time_diff_.compute_loss(tgt_x, tgt_e, hist, cat_order, t, hist_emb=hist_emb)
        
        selected_losses = [losses[id] for id in self.loss_names]
        return sum(selected_losses).mean()

    def get_hist(self, hist_x, hist_e):
        ## hist: (B,hist_len,transformer_dim)

        hist = self.hist_enc_func_(hist_x=hist_x, hist_e=hist_e)
        return hist

    def sample_time(self, b, device, method="uniform"):
        t = torch.randint(0, self.n_steps, (b,), device=device).long()

        pt = torch.ones_like(t).float() / self.n_steps
        return t, pt

    def sample(self, hist_x, hist_e, tgt_len, cat_list, tgt_e=None, tgt_x=None, hist_emb=None):
        self.num_classes_list = [self.num_cat_dict[i] for i in cat_list]

        e, x = self.sample_chain(hist_x, hist_e, tgt_len, cat_list, tgt_x=tgt_x, tgt_e=tgt_e, hist_emb=hist_emb)

        return log_onehot_to_index_multi_task(e[-1], self.num_classes_list) if tgt_e is None else tgt_e, x[-1]

    def sample_chain(self, hist_x, hist_e, tgt_len, cat_list, tgt_x=None, tgt_e=None, hist_emb=None):
        hist = self.get_hist(hist_x, hist_e)
        # shape = [hist.size(0), tgt_len]
        shape = [hist.size(0), tgt_len, hist_x.size(-1)]

        fix_mask = (~self.time_diff_._diff_mask(tgt_x).bool())
        full_noise = torch.randn(shape).to(self.device)

        if tgt_x is None:
            init_x = full_noise
        else:
            init_x = deepcopy(tgt_x)
            init_x[~fix_mask] = full_noise[~fix_mask]
            
        # x_t_list = [init_x.unsqueeze(0)]
        x_t_list = [init_x]

        x_t = init_x

        shape = (tgt_len,)
        b = hist.size(0)
        # uniform_logits = torch.zeros(
        #     (b, self.num_classes,) + shape, device=self.device)
        if tgt_e is None:
            uniform_logits = torch.zeros(
                (
                    b,
                    sum(self.num_classes_list),
                )
                + shape,
                device=self.device,
            )
            # e_t = log_sample_categorical(uniform_logits,self.num_classes)
            #
            e_t = log_sample_categorical_multi_task(uniform_logits, self.num_classes_list)

        else:
            e_t = tgt_e

        e_t_list = [e_t]
        for i in reversed(range(0, self.n_steps)):
            # e_t_index = log_onehot_to_index(e_t)
            
            e_t_index = log_onehot_to_index_multi_task(e_t, self.num_classes_list) if tgt_e is None else e_t
            x_seq = self.time_diff_._one_diffusion_rev_step(
                self.time_diff_.denoise_func_, x_t, e_t_index, i, hist, cat_list, hist_emb=hist_emb
            )
            assert (x_seq[fix_mask] == tgt_x[fix_mask]).all()
            x_t_list.append(x_seq)
            # e_t, t, x_t, hist, non_padding_mask
            t_type = torch.full((b,), i, device=self.device, dtype=torch.long)
            e_seq = self.type_diff_.p_sample(e_t, t_type, x_t, hist, cat_list, hist_emb=hist_emb) if tgt_e is None else tgt_e
            e_t_list.append(e_seq)
            x_t = x_seq
            e_t = e_seq
        return e_t_list, x_t_list

    def _build_num_diff_idx(self, num_features_names):
        """
        num_features_names: список имён числовых фич в порядке hist_num[:, :, 1:], т.е. без time.
        Возвращает список индексов по осям hist_num[..., F_num], где 0 — всегда time.
        """
        diff_idx = []
        # time — если указан в diff_features
        if (self.data_config.time_name in self.diff_features) or (len(self.diff_features) == 0):
            diff_idx.append(0)
        # остальные числовые
        if num_features_names:
            for j, name in enumerate(num_features_names, start=1):  # j=1..F_num-1
                if name in self.diff_features:
                    diff_idx.append(j)
        return sorted(set(diff_idx))
