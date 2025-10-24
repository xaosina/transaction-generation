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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2, ceil_mode=True) if down else nn.Identity(),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True) if up else nn.Identity()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffX = torch.tensor([x2.size()[2] - x1.size()[2]])
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CondINorm(nn.Module):
    def __init__(self, in_channels, z_channels, eps=1e-5):
        super(CondINorm, self).__init__()
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv1d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv1d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )
        self.instance_norm = nn.InstanceNorm1d(in_channels)

    def forward(self, x, z):
        shift = self.shift_conv.forward(z)
        scale = self.scale_conv.forward(z)
        # size = x.size()
        # x_reshaped = x.view(size[0], size[1], size[2])
        # mean = x_reshaped.mean(1, keepdim=True)
        # var = x_reshaped.var(1, keepdim=True)
        # std =  torch.rsqrt(var + self.eps)
        norm_x = self.instance_norm(x)
        # pdb.set_trace()
        # assert scale.shape[:2] == x.shape[:2]
        assert len(scale.shape) == 3
        assert len(x.shape) == 3
        #assert scale.shape[2] == 1
        # norm_features = ((x_reshaped - mean) * std).view(*size)
        # output = norm_features * scale + shift
        return norm_x * scale + shift


class CondDoubleConv(nn.Module):
    """(convolution => [CIN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, z_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = CondINorm(mid_channels, z_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CondINorm(out_channels, z_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, z):
        x = self.conv1(x)
        x = self.norm1(x, z)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, z)
        x = self.relu2(x)
        return x


class CondUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, z_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = CondDoubleConv(in_channels, out_channels, z_channels, in_channels // 2)

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, z)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CUNet(nn.Module):

    def __init__(
            self, 
            n_channels, 
            n_classes, 
            z_channels, 
            base_factor=32, 
            reduce_profile = [False, True, False, True]
    ):
        super(CUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.z_channels = z_channels
        self.base_factor = base_factor
        self.reduce_profile = reduce_profile
        r1, r2, r3, r4 = reduce_profile[0], reduce_profile[1], reduce_profile[2], reduce_profile[3]

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor, down=r1)
        self.down2 = Down(2 * base_factor, 4 * base_factor, down=r2)
        self.down3 = Down(4 * base_factor, 8 * base_factor, down=r3)
        factor = 2
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor, down=r4)
        self.adain1 = CondINorm(16 * base_factor // factor, z_channels)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor, up=r4)
        self.adain2 = CondINorm(8 * base_factor // factor, z_channels)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, up=r3)
        self.adain3 = CondINorm(4 * base_factor // factor, z_channels)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, up=r2)
        self.adain4 = CondINorm(2 * base_factor // factor, z_channels)
        self.up4 = Up(2 * base_factor, base_factor, up=r1)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x, z):
        # pdb.set_trace()
        x1 = self.inc(x)
        x2 = self.down1(x1, )
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.adain1(x5, z)
        x = self.up1(x, x4)
        x = self.adain2(x, z)
        x = self.up2(x, x3)
        x = self.adain3(x, z)
        x = self.up3(x, x2)
        x = self.adain4(x, z)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class Unet1DDiffusion(nn.Module):

    def __init__(
            self, 
            d_token=4, 
            dim_t=512, 
            length=2,
            base_factor=32,
            num_classes=1,
            rawhist_length=0,
            d_numerical =2,
            categories = [1,1],
            with_transformer = True
    ):
        super().__init__()
        self.d_token = d_token
        self.dim_t = dim_t
        self.length = length
        self.num_classes = num_classes
        self.rawhist_length = rawhist_length
        self.unet_length = length + rawhist_length
        #self.one_lat_dim = latent_dim
        self.one_lat_dim = (len(categories) + d_numerical)*d_token
        
        def _check_in_borders(val: int, lb: int, rb: int):
            return val >= lb and val < rb
        
        if _check_in_borders(self.unet_length, 2, 6):
            reduce_profile = [False, False, False, False]
        elif _check_in_borders(self.unet_length, 6, 12):
            assert self.unet_length % 2 == 0
            reduce_profile = [False, True, False, False]
        elif _check_in_borders(self.unet_length, 12, 24):
            assert self.unet_length % 4 == 0
            reduce_profile = [False, True, False, True]
        elif _check_in_borders(self.unet_length, 24, 48):
            assert self.unet_length %  8 == 0
            reduce_profile = [False, True, True, True]
        elif _check_in_borders(self.unet_length, 48, 128):
            assert self.unet_length %  16 == 0
            reduce_profile = [True, True, True, True]
        else:
            raise Exception(f'The overal length, {self.unet_length}, is too high or too low')
        
        cunet_chans = self.one_lat_dim
        basic_emb_dim = cunet_chans

        # setting z_channels (input dimension of AdaIn)
        z_channels = 0

        TIME_Z_CHAN_FACTOR = 1

        self.time_emb_dim = TIME_Z_CHAN_FACTOR * basic_emb_dim


        z_channels += self.time_emb_dim


        self.cunet = CUNet(
            cunet_chans, 
            cunet_chans, 
            z_channels, 
            base_factor=base_factor, 
            reduce_profile=reduce_profile
        )
        
        self.with_transformer = with_transformer
        # mapping time
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), 
            nn.SiLU(), 
            nn.Linear(dim_t, self.time_emb_dim)
        )

        name2model = {
            'cunet': self.cunet,
            'map_noise': self.map_noise,
            'time_embed': self.time_embed,
        }
        num_module_params(name2model)

        #d_numerical = 1
        #categories = [6,205]
        #d_token = 4
        #bias = True
        num_layers = 4
        n_head = 2
        factor = 32
        #use_mlp= True
        #d_token =  self.one_lat_dim
        d_token_in = self.one_lat_dim
        d_token_out = self.one_lat_dim
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = True)
        
        if with_transformer:

            self.encoder = Transformer(num_layers, d_token_in, n_head, d_token_out, factor)
            self.decoder = Transformer(num_layers, d_token_in, n_head, d_token_out, factor)
        # d_in = d_token * (d_numerical + len(categories))
        #self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
       
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat,t,rawhist):

        
        bs,ls = rawhist["num"].size(0),rawhist["num"].size(1)

        hist_num,hist_cat = rawhist["num"].reshape(bs*ls,-1),rawhist["cat"].reshape(bs*ls,-1)
        current_token = self.tokenizer(x_num,x_cat)[:,1:,:]
        history_token = self.tokenizer(hist_num,hist_cat)[:,1:,:]


        
        t = t.view(bs,ls)[:,0]
        #
        #t = t.view(bs,ls)
        emb = self.map_noise(t)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        emb = self.time_embed(emb)[:, :, None]    

        x = current_token.view(bs,ls,-1)
        hist = history_token.view(bs,ls,-1)
        x = torch.cat([hist,x],dim=1)
        if self.with_transformer:
            x = self.encoder(x)

        x = x.transpose(1, 2)  # (bs, one_lat_dim, len)
        x = self.cunet(x, emb)
        x = x.transpose(1, 2) #(bs,len,one_lat_dim)
        if self.with_transformer:
            x = self.decoder(x)
        
        
        x = x[:,-self.length :].reshape(bs*ls,-1,self.d_token) ##(Bs*len,F,d_token)
        x_num_pred, x_cat_pred = self.detokenizer(x)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)



        return x_num_pred,x_cat_pred


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

