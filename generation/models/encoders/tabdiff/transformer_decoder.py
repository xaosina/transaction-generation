from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
#from .diffusion_utils import EDMLoss
import pdb
import torch.nn as nn
import torch.nn.init as nn_init
import math
from .transformer import Transformer
from .history_encoder import HistoryEncoder

class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x
    
def num_module_params(moduledict: Dict, verbose: bool = True) -> Dict:
    numparamsdict = dict()
    for name, model in moduledict.items():
        assert isinstance(model, torch.nn.Module)
        num_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"Module {name}: #params = {num_params}")
        numparamsdict[name] = num_params
    return numparamsdict

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class Transformer_denoise(nn.Module):

    def __init__(
            self, 
            d_token=4, 
            dim_t=512, 
            length=2,
            base_factor=32,
            num_classes=1,
            rawhist_length=0,
            d_numerical =2,
            categories = [1,1]
    ):
        super().__init__()

        self.device = "cuda:0"
        self.length = length
        self.time_emb_dim = 4
        self.d_token = d_token
        # mapping time
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), 
            nn.SiLU(), 
            nn.Linear(dim_t, self.time_emb_dim)
        )


        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = True)

        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

        transformer_dim = 8
        self.hist_enc = HistoryEncoder(token_fn=self.tokenizer)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=2,
            dim_feedforward=64, dropout=0.1,
            batch_first=True
        )

        decoder_norm = nn.LayerNorm(transformer_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=2,
                                             norm=decoder_norm)
        order_dim = 8
        self.order_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) /int(order_dim)) for i in range((int(order_dim)))],
            device=torch.device(self.device))
        self.output_layer = nn.Linear(transformer_dim,4)

    def forward(self, x_num, x_cat,t,hist):

        device = "cuda:0"
        ## 1. time embeding, t shape: (bs*ls,)
        bs,ls = hist["cat"].size(0),self.length
        ## emb:(bs*ls,dim_t)
        emb = self.map_noise(t)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        ## emb:(bs*ls,dim_time_emb)
        emb = self.time_embed(emb)


        ## 3. tokenizer for x_num and x_cat, their shape is (bs*ls,F,d_token)
        current_token = self.tokenizer(x_num,x_cat)[:,1:,:]
        F_token = current_token.size(1)
        L_final = self.length * F_token
        #history_token = self.tokenizer(hist_num,hist_cat)[:,1:,:]

        ## 4. reshape tokens
        curr_token = current_token.reshape(bs,L_final,self.d_token)

        t_emb_expanded = emb.unsqueeze(1).repeat(1, F_token, 1)
        
        # b) 重塑和融合: (bs*ls, F_token, C) -> (bs, L_final, C)
        t_emb_final = t_emb_expanded.reshape(bs, L_final, self.d_token)

        order = torch.arange(L_final, device=device).unsqueeze(0).repeat(bs, 1)
        order_emb = self.order_enc(order) # order_emb 形状: (bs, L_final, C)

        ## 5. concate time embeding and tokens,
        memory = self.hist_enc(hist)
        tgt = torch.cat([curr_token,t_emb_final],dim=-1) + order_emb
        #tgt = curr_token + t_emb_final + order_emb
        ## 6. get the mask for subsequent event
        tgt_mask = self.generate_square_subsequent_mask(L_final).to(device)
        ## 7. input into transformer encoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        ## 8. get the predicted output from transformer
        out = self.output_layer(output)

        x = out.reshape(bs*ls,-1,self.d_token) ##(Bs*len,F,d_token)
        x_num_pred, x_cat_pred = self.detokenizer(x)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred,x_cat_pred
    
    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def order_enc(self, dt_seq):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = dt_seq.unsqueeze(-1) / self.order_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,              # Expected standard deviation of the training data.
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t,hist, sigma):
        
        x_num = x_num.to(torch.float32)

        sigma = sigma.to(torch.float32)
        assert sigma.ndim == 2
        if sigma.dim() > 1: # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
            sigma_cond = (0.002 ** (1/7) + t * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma 
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4

        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, c_noise.flatten(),hist)
        elif self.net_conditioning == "t":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, t,hist)

        assert F_x.dtype == dtype
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        
        return D_x, x_cat_pred


class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t,hist, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t,hist, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t,hist)


class Tokenizer(nn.Module):

    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + list(categories[:-1])).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.cat_weight = nn.Parameter(Tensor(sum(categories), d_token))
            nn.init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))
        self.d_token = d_token
        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
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
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
    
        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            for start, end in zip(self.category_offsets, torch.cat([self.category_offsets[1:], torch.tensor([x_cat.shape[1]], device=x_cat.device)])):
                if start < end:
                    x = torch.cat(
                        [x, x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None]],
                        dim=1,
                    )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )

            x = x + bias[None]
    

        return x
    
class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
      
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat

