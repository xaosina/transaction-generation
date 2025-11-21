from typing import Optional
import torch
import torch.nn.functional as F

from .utils import index_to_log_onehot, log_onehot_to_index
from .utils import log_sample_categorical
from .hist_encoder import HistoryEncoder
from .type_denoising_ml import TypeDenoisingModule
from .time_denoising_ml import TimeDenoisingModule
from .time_diffusion_model import DiffusionTimeModel
from .type_diffusion_model import DiffusionTypeModel
from .utils import log_sample_categorical_multi_task
from .utils import log_onehot_to_index_multi_task
from copy import deepcopy


class DiffusionTabularModel(torch.nn.Module):
    def __init__(
        self, data_conf, model_config, outer_history_encoder_dim=None, prefix_dim=None
    ):
        super(DiffusionTabularModel, self).__init__()

        device = "cuda"
        self.device = device

        time_emb_feature_dim = 1 << model_config.params["time_emb_dim_features_exp"]
        num_emb_feature_dim = 1 << model_config.params["num_emb_dim_features_exp"]
        cat_emb_feature_dim = 1 << model_config.params["cat_emb_dim_features_exp"]
        transformer_dim = 1 << model_config.params["dim_exp"]
        transformer_heads = 1 << model_config.params["heads_exp"]
        dim_feedforward = 1 << model_config.params["hidden_scale_exp"]
        num_encoder_layers = model_config.params["encoder_layer"]
        num_decoder_layers = model_config.params["decoder_layer"]
        self.cfg_p_uncond = model_config.params.get("cfg_p", 0.0)
        self.order_invariant_mode = bool(model_config.params.get("order_invariant_mode", False))

        if (dim_feedforward % transformer_heads) != 0:
            raise ValueError("(dim_feedforward % transformer_heads) != 0")

        prefix_dim = 1 << model_config.params.get("prefix_dim_exp", None)

        self.n_steps = model_config.params["diffusion_steps"]
        num_timesteps = model_config.params["diffusion_steps"]
        history_encoder_causal_mask = model_config.params["history_encoder_causal_mask"]
        outer_he_use_post_norm = model_config.params["use_post_norm"]
        use_rezero = model_config.params["use_rezero"]
        use_simple_t_project = model_config.params.get("use_simple_t_project")
        num_diffusion_t_type = model_config.params["num_diffusion_t_type"]
        cat_diffusion_t_type = model_config.params["cat_diffusion_t_type"]
        self.data_config = data_conf
        self.model_config = model_config
        self.loss_names = self.model_config.params["losses"]
        self.fix_features: Optional[set] = set(
            self.model_config.params.get("fix_features", [])
        )
        self.diff_features: Optional[set] = set(
            self.model_config.params.get("diff_features", [])
        )

        self.diff_focus = self.fix_features | self.diff_features

        if self.diff_focus is not None:
            self.num_cat_dict = {
                key: val
                for key, val in data_conf.cat_cardinalities.items()
                if key in self.diff_focus
            }
        else:
            self.num_cat_dict = data_conf.cat_cardinalities

        self.num_classes_list = []

        assert (
            data_conf.time_name is not None
        ), "It seems you forget provide a time feature."
        ## because of time interv must in the num features, so we add 1
        len_num_features = 1
        if self.diff_focus is not None:
            len_num_features += len(
                [v for v in data_conf.num_names if v in self.diff_focus]
            )
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
            num_feature_dim=num_emb_feature_dim,
            cat_feature_dim=cat_emb_feature_dim,
            time_feature_dim=time_emb_feature_dim,
            causal_mask=history_encoder_causal_mask,
            use_simple_time_proj=use_simple_t_project,
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
            num_feature_dim=num_emb_feature_dim,
            cat_feature_dim=cat_emb_feature_dim,
            time_feature_dim=time_emb_feature_dim,
            outer_history_encoder_dim=outer_history_encoder_dim,
            prefix_dim=prefix_dim,
            use_rezero=use_rezero,
            use_post_norm=outer_he_use_post_norm,
            diffusion_t_type=cat_diffusion_t_type,
            use_simple_time_proj=use_simple_t_project,
            order_invariant_mode=self.order_invariant_mode
        )

        self.denoise_fn_dt = TimeDenoisingModule(
            transformer_dim=transformer_dim,
            num_classes=self.num_cat_dict,
            n_steps=num_timesteps,
            transformer_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers,
            device=device,
            num_feature_dim=num_emb_feature_dim,
            cat_feature_dim=cat_emb_feature_dim,
            time_feature_dim=time_emb_feature_dim,
            len_numerical_features=len_num_features,
            outer_history_encoder_dim=outer_history_encoder_dim,
            prefix_dim=prefix_dim,
            use_post_norm=outer_he_use_post_norm,
            diffusion_t_type=num_diffusion_t_type,
            use_simple_time_proj=use_simple_t_project,
            order_invariant_mode=self.order_invariant_mode
        )

        self.type_diff_ = DiffusionTypeModel(
            n_steps=self.n_steps,
            denoise_fn=self.denoise_fn_type,
            num_classes=self.num_cat_dict,
        )

        self.time_diff_ = DiffusionTimeModel(
            n_steps=self.n_steps, denoise_func=self.denoise_fn_dt
        )
        self.plug_embedding = None
        self.plug_hist = None
        if self.cfg_p_uncond > 0:
            self.plug_hist = torch.nn.Parameter(
                torch.zeros(1, self.data_config.generation_len, transformer_dim),
                requires_grad=True,
            )
            if outer_history_encoder_dim is not None:
                self.plug_embedding = torch.nn.Parameter(
                    torch.zeros(1, outer_history_encoder_dim), requires_grad=True
                )

    def _permute_target(self, tgt_x, tgt_e):
        if tgt_x is None or tgt_e is None:
            return tgt_x, tgt_e

        B, L = tgt_x.size(0), tgt_x.size(1)
        device = tgt_x.device

        # отдельная перестановка для каждого объекта в батче
        perms = torch.stack(
            [torch.randperm(L, device=device) for _ in range(B)],
            dim=0
        )  # (B, L)

        # расширяем под фичи
        idx_x = perms.unsqueeze(-1).expand(-1, -1, tgt_x.size(-1))
        idx_e = perms.unsqueeze(-1).expand(-1, -1, tgt_e.size(-1))

        tgt_x_perm = torch.gather(tgt_x, 1, idx_x)
        tgt_e_perm = torch.gather(tgt_e, 1, idx_e)
        return tgt_x_perm, tgt_e_perm


    def compute_loss(self, tgt_x, tgt_e, hist_x, hist_e, cat_order, hist_emb=None):
        B = hist_x.size(0)
        if torch.rand((1,)).item() < self.cfg_p_uncond:
            assert self.cfg_p_uncond != 0.0 # Stupid check
            hist = self.plug_hist.repeat(B, 1, 1)
            hist_emb = self.plug_embedding.repeat(B, 1)
        else:
            hist = self.get_hist(hist_x, hist_e)

        if self.order_invariant_mode:
            tgt_x, tgt_e = self._permute_target(tgt_x, tgt_e)

        t, pt = self.sample_time(tgt_e.size(0), device=self.device)

        losses = {}
        if "type_loss" in self.loss_names:
            losses["type_loss"] = self.type_diff_.compute_loss(
                tgt_e, tgt_x, hist, cat_order, t, pt, hist_emb=hist_emb
            )
        if "time_loss" in self.loss_names:
            losses["time_loss"] = self.time_diff_.compute_loss(
                tgt_x, tgt_e, hist, cat_order, t, hist_emb=hist_emb
            )
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

    def sample(
        self,
        hist_x,
        hist_e,
        tgt_len,
        cat_list,
        tgt_e=None,
        tgt_x=None,
        hist_emb=None,
        return_chain: bool = False,  # не хранить цепочку по умолчанию
        step_stride: int = 1,  # подвыборка шагов: 1 == как было
        cfg_w=1.0,
        dtype=None,  # torch.float16 / torch.bfloat16 / None
    ):
        self.num_classes_list = [self.num_cat_dict[i] for i in cat_list]

        e_last, x_last = self.sample_chain(
            hist_x,
            hist_e,
            tgt_len,
            cat_list,
            tgt_x=tgt_x,
            tgt_e=tgt_e,
            hist_emb=hist_emb,
            return_chain=return_chain,
            step_stride=step_stride,
            dtype=dtype,
            cfg_w=cfg_w,
        )

        if tgt_e is None:
            e_last = log_onehot_to_index_multi_task(e_last, self.num_classes_list)
        else:
            e_last = tgt_e
        return e_last, x_last

    @torch.no_grad()
    def sample_chain(
        self,
        hist_x,
        hist_e,
        tgt_len,
        cat_list,
        tgt_x=None,
        tgt_e=None,
        hist_emb=None,
        return_chain: bool = False,
        step_stride: int = 1,
        dtype=None,
        cfg_w=1.0,
    ):
        hist = self.get_hist(hist_x, hist_e)
        B, F_num = hist.size(0), hist_x.size(-1)
        T = tgt_len

        shape = (B, T, F_num)
        fix_mask = None

        device = self.device

        if tgt_x is None:
            x_t = torch.randn(shape).to(device)
        else:
            diff_mask = self.time_diff_._diff_mask(tgt_x).bool()
            fix_mask = ~diff_mask
            x_t = tgt_x.clone()
            noise = torch.randn(shape).to(device)
            x_t = torch.where(diff_mask, noise, x_t)

        if return_chain:
            x_chain_last = [x_t]

        if tgt_e is None:
            total_C = sum(self.num_classes_list)
            uniform_logits = x_t.new_zeros((B, total_C, T))
            e_t = log_sample_categorical_multi_task(
                uniform_logits, self.num_classes_list
            )
        else:
            e_t = tgt_e

        if return_chain:
            e_chain_last = [e_t]
        # ----- timesteps (можно редуцировать шаги) -----
        timesteps = range(self.n_steps - 1, -1, -step_stride)

        t_type = torch.empty((B,), device=device, dtype=torch.long)

        if tgt_e is not None:
            e_t_index = e_t
        else:
            e_t_index = log_onehot_to_index_multi_task(e_t, self.num_classes_list)

        plug_emb = None
        plug_hist = None
        if cfg_w != 1.0:
            plug_emb = self.plug_embedding.repeat(B, 1)
            plug_hist = self.plug_hist.repeat(B, 1, 1)

        for i in timesteps:
            x_t = self.sample_numerical(
                x_t,
                e_t_index,
                i,
                hist,
                cat_list,
                hist_emb,
                tgt_x,
                fix_mask=fix_mask,
                cfg_w=cfg_w,
                plug_hist=plug_hist,
                plug_emb=plug_emb,
            )

            if return_chain:
                x_chain_last.append(x_t)

            if tgt_e is None:
                t_type.fill_(i)
                e_t = self.type_diff_.p_sample(
                    e_t,
                    t_type,
                    x_t,
                    hist,
                    cat_list,
                    hist_emb=hist_emb,
                    cfg_w=cfg_w,
                    plug_hist=plug_hist,
                    plug_emb=plug_emb,
                )
                e_t_index = log_onehot_to_index_multi_task(e_t, self.num_classes_list)
                if return_chain:
                    e_chain_last.append(e_t)
        e_out = e_t if not return_chain else e_chain_last
        x_out = x_t if not return_chain else x_chain_last
        return e_out, x_out

    def sample_numerical(
        self,
        x_t,
        e_t_index,
        i,
        hist,
        cat_list,
        hist_emb,
        tgt_x,
        fix_mask=None,
        cfg_w=1.0,
        plug_hist=None,
        plug_emb=None,
    ):

        x_t = self.time_diff_._one_diffusion_rev_step(
            self.time_diff_.denoise_func_,
            x_t,
            e_t_index,
            i,
            hist,
            cat_list,
            hist_emb=hist_emb,
            cfg_w=cfg_w,
            plug_hist=plug_hist,
            plug_emb=plug_emb,
        )

        if tgt_x is not None:
            x_t = torch.where(fix_mask, tgt_x, x_t)

        return x_t
    

    def _build_num_diff_idx(self, num_features_names):
        """
        num_features_names: список имён числовых фич в порядке hist_num[:, :, 1:], т.е. без time.
        Возвращает список индексов по осям hist_num[..., F_num], где 0 — всегда time.
        """
        diff_idx = []
        # time — если указан в diff_features
        if (self.data_config.time_name in self.diff_features) or (
            len(self.diff_features) == 0
        ):
            diff_idx.append(0)
        # остальные числовые
        if num_features_names:
            for j, name in enumerate(num_features_names, start=1):  # j=1..F_num-1
                if name in self.diff_features:
                    diff_idx.append(j)
        return sorted(set(diff_idx))
