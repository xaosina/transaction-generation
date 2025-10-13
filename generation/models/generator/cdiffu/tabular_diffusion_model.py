import torch
import torch.nn.functional as F 

from .utils import index_to_log_onehot,log_onehot_to_index
from .utils import log_sample_categorical
from .hist_enc1 import HistoryEncoder
from .type_denoising_ml import TypeDenoisingModule
from .time_denoising_ml import TimeDenoisingModule
from .time_diffusion_model import DiffusionTimeModel
from .type_diffusion_model import DiffusionTypeModel
from .utils import log_sample_categorical_multi_task
from .utils import log_onehot_to_index_multi_task


class DiffusionTabularModel(torch.nn.Module):
    def __init__(self, data_conf,model_config):
        super(DiffusionTabularModel, self).__init__()

        device = "cuda"
        self.device = device
        
        num_classes =  data_conf.cat_cardinalities['small_group']
        emb_feature_dim = model_config.params['emb_dim_features']
        transformer_dim =  model_config.params['dim']      
        transformer_heads = model_config.params['heads']
        num_encoder_layers = model_config.params['encoder_layer']
        num_decoder_layers = model_config.params['decoder_layer']
        dim_feedforward = model_config.params['feed_forward']
        self.n_steps = model_config.params['diffusion_steps']
        num_timesteps = model_config.params['diffusion_steps']
        self.d_conf = data_conf
        self.m_conf = model_config

        self.num_cat_dict = data_conf.cat_cardinalities
        self.num_classes_list = []
        ## because of time interv must in the num features, so we add 1
        len_num_features = len(data_conf.num_names) + 1


        self.hist_enc_func_ = HistoryEncoder(
            transformer_dim=transformer_dim, transformer_heads=transformer_heads, num_classes=self.num_cat_dict,
            device=device, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            len_numerical_features =len_num_features,feature_dim = emb_feature_dim
        )


        self.denoise_fn_type = TypeDenoisingModule(
            transformer_dim=transformer_dim, num_classes=self.num_cat_dict , n_steps=num_timesteps,
            transformer_heads=transformer_heads, dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers, device=device,len_numerical_features =len_num_features,
            feature_dim = emb_feature_dim)


        self.denoise_fn_dt = TimeDenoisingModule(
            transformer_dim=transformer_dim, num_classes=self.num_cat_dict , n_steps=num_timesteps,
            transformer_heads=transformer_heads, dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers, device=device,feature_dim = emb_feature_dim,
            len_numerical_features =len_num_features
        )

        self.type_diff_ = DiffusionTypeModel(n_steps=self.n_steps, denoise_fn=self.denoise_fn_type,
                                                       num_classes=self.num_cat_dict)

        self.time_diff_ = DiffusionTimeModel(n_steps=self.n_steps, denoise_func=self.denoise_fn_dt)


    def compute_loss(self, tgt_x, tgt_e, hist_x, hist_e,cat_order):
        ## tgt_x,tgt_e : B,gen_len,F_{1};B,gen_len,F_{2}
        ## hist_x,hist_e: B,hist_len,F_{1};B,hist_len,F_{2}
        #
        hist = self.get_hist(hist_x, hist_e)
        ## t,pt are (B,)
        t, pt = self.sample_time(tgt_e.size(0), device=self.device)
        type_loss = self.type_diff_.compute_loss(tgt_e, tgt_x, hist,cat_order, t, pt)
        time_loss = self.time_diff_.compute_loss(tgt_x, tgt_e, hist,cat_order, t)
        return (time_loss + type_loss).sum(-1).mean()

    def get_hist(self, hist_x, hist_e):
        ## hist: (B,hist_len,transformer_dim)
        hist = self.hist_enc_func_(hist_x=hist_x, hist_e=hist_e)
        return hist

    def sample_time(self, b, device, method='uniform'):
        t = torch.randint(0, self.n_steps, (b,), device=device).long()

        pt = torch.ones_like(t).float() / self.n_steps
        return t, pt

    
    def sample(self, hist_x, hist_e, tgt_len,cat_list):
        self.num_classes_list = [self.num_cat_dict[i] for i in cat_list]
        e, x = self.sample_chain(hist_x, hist_e, tgt_len,cat_list)

        return log_onehot_to_index_multi_task(e[-1],self.num_classes_list), x[-1]
    
    def sample_chain(self, hist_x, hist_e,  tgt_len,cat_list):
        hist = self.get_hist(hist_x, hist_e)

        #shape = [hist.size(0), tgt_len]
        shape = [hist.size(0),tgt_len,2]
        init_x = torch.randn(shape).to(self.device)
        # x_t_list = [init_x.unsqueeze(0)]
        x_t_list = [init_x]

        x_t = init_x

        shape = (tgt_len,)
        b = hist.size(0)
        # uniform_logits = torch.zeros(
        #     (b, self.num_classes,) + shape, device=self.device)
        uniform_logits = torch.zeros(
            (b, sum(self.num_classes_list),) + shape, device=self.device)
        #e_t = log_sample_categorical(uniform_logits,self.num_classes)
        #
        e_t = log_sample_categorical_multi_task(uniform_logits,self.num_classes_list)

        e_t_list = [e_t]
        for i in reversed(range(0, self.n_steps)):
            #e_t_index = log_onehot_to_index(e_t)
            e_t_index = log_onehot_to_index_multi_task(e_t,self.num_classes_list)
            x_seq = self.time_diff_._one_diffusion_rev_step(self.time_diff_.denoise_func_, x_t, e_t_index, i, hist,cat_list)
            x_t_list.append(x_seq)
            # e_t, t, x_t, hist, non_padding_mask
            t_type = torch.full((b,), i, device=self.device, dtype=torch.long)
            e_seq = self.type_diff_.p_sample(e_t, t_type, x_t, hist,cat_list)
            e_t_list.append(e_seq)
            x_t = x_seq
            e_t = e_seq
        return e_t_list, x_t_list

