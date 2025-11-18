import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from .base import LDMDenoiser
from typing import Dict, Union, Callable


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


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2, ceil_mode=True) if down else nn.Identity(),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        self.up = (
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            if up
            else nn.Identity()
        )
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
        assert scale.shape[2] == 1
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
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = CondDoubleConv(
            in_channels, out_channels, z_channels, in_channels // 2
        )

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
            ],
        )
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
        reduce_profile=[False, True, False, True],
    ):
        super(CUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.z_channels = z_channels
        self.base_factor = base_factor
        self.reduce_profile = reduce_profile
        r1, r2, r3, r4 = (
            reduce_profile[0],
            reduce_profile[1],
            reduce_profile[2],
            reduce_profile[3],
        )

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
        x2 = self.down1(
            x1,
        )
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


class MLPDiffusion(nn.Module):

    def __init__(
            self, 
            d_in, 
            dim_t=512,
            num_classes=1,
            dim_classes_emb = 64,
            history_embedding_condition: bool = False,
            history_embedding_dim = 24,
            dim_reference = 0,
        ):
        super().__init__()
        self.dim_t = dim_t
        self.is_history_embedding_conditional = history_embedding_condition
        self.is_reference_conditional = dim_reference > 0
        self.dim_reference = dim_reference

        dim_proj_in = d_in 
        if self.is_history_embedding_conditional:
            dim_proj_in += history_embedding_dim
        if self.is_reference_conditional:
            dim_proj_in += dim_reference

        self.proj = nn.Linear(dim_proj_in, dim_t)

        self.num_classes = num_classes
        self.is_class_conditional = num_classes > 1
        self.dim_classes_emb = dim_classes_emb
        self.map_classes = nn.Identity()
        if self.is_class_conditional:
            self.map_classes = nn.Embedding(num_classes, dim_classes_emb)

        self.dim_mlp_input = dim_t
        if self.is_class_conditional:
            self.dim_mlp_input += dim_classes_emb

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_mlp_input, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

        name2model = {
            'proj': self.proj,
            'mlp': self.mlp,
            'map_noise': self.map_noise,
            'time_embed': self.time_embed,
            'map_classes': self.map_classes,
            # 'map_history_embedding': self.map_history_embedding,
        }
        num_module_params(name2model)

    def forward(
            self, 
            x, 
            noise_labels, 
            class_labels = None, 
            history_embedding = None,
            reference = None,
        ):
        if self.is_class_conditional:
            assert class_labels is not None
        if self.is_history_embedding_conditional:
            assert history_embedding is not None
        if self.is_reference_conditional:
            assert reference is not None
            assert isinstance(reference, torch.Tensor)
            assert reference.ndim == 2
            assert reference.size(1) == self.dim_reference
        
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        emb = self.time_embed(emb)

        if self.is_reference_conditional:
            x = torch.cat([reference, x], dim=1)
        if self.is_history_embedding_conditional:
            x = torch.cat([x, history_embedding], dim=1)

        x = self.proj(x) + emb
        if self.is_class_conditional:
            # pdb.set_trace()
            class_emb = self.map_classes(class_labels).view(
                class_labels.size(0), self.dim_classes_emb
            )
            x = torch.cat([x, class_emb], dim=1)

        return self.mlp(x)


class Unet1D(LDMDenoiser):

    def __init__(
        self, 
        latent_size,
        generation_len = 32,
        reference_len = 32,
        history_embedding_dim = 0,
        trans_time_conditional=False,
        num_classes=1,
        dim_t=512,
        base_factor=32,
    ):
        super().__init__()

        self.dim_t = dim_t
        self.length = generation_len
        self.num_classes = num_classes
        self.reference_len = reference_len
        self.unet_length = generation_len + reference_len
        self.one_lat_dim = latent_size

        self.is_class_conditional = num_classes > 1
        self.is_trans_time_conditional = trans_time_conditional
        self.is_history_embedding_conditional = history_embedding_dim > 0
        self.is_reference_conditional = reference_len > 0
        
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
            assert self.unet_length % 8 == 0
            reduce_profile = [False, True, True, True]
        elif _check_in_borders(self.unet_length, 48, 128):
            assert self.unet_length % 16 == 0
            reduce_profile = [True, True, True, True]
        else:
            raise Exception(
                f"The overal length, {self.unet_length}, is too high or too low"
            )

        cunet_chans = self.one_lat_dim
        basic_emb_dim = cunet_chans

        # setting z_channels (input dimension of AdaIn)
        z_channels = 0

        TIME_Z_CHAN_FACTOR = 1
        CLASS_CONDI_Z_CHAN_FACTOR = 1
        history_embedding_CONDI_Z_CHAN_FACTOR = 2

        self.time_emb_dim = TIME_Z_CHAN_FACTOR * basic_emb_dim
        self.classes_emb_dim = CLASS_CONDI_Z_CHAN_FACTOR * basic_emb_dim
        self.history_embedding_emb_dim = history_embedding_CONDI_Z_CHAN_FACTOR * basic_emb_dim

        z_channels += self.time_emb_dim
        if self.is_class_conditional:
            z_channels += self.classes_emb_dim
        if self.is_history_embedding_conditional:
            z_channels += self.history_embedding_emb_dim

        self.cunet = CUNet(
            cunet_chans + (1 if self.is_trans_time_conditional else 0),
            cunet_chans,
            z_channels,
            base_factor=base_factor,
            reduce_profile=reduce_profile,
        )

        # mapping time
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, self.time_emb_dim)
        )

        # mapping classes
        self.map_classes = nn.Identity()
        if self.is_class_conditional:
            self.map_classes = nn.Embedding(num_classes, self.classes_emb_dim)
        
        # mapping history_embedding
        self.map_history_embedding = nn.Identity()
        if self.is_history_embedding_conditional:
            self.map_history_embedding = nn.Sequential(
                nn.Linear(history_embedding_dim, self.history_embedding_emb_dim), 
                nn.SiLU(), 
                nn.Linear(self.history_embedding_emb_dim, self.history_embedding_emb_dim)
            )

        # for cfg
        self.plug_history_embedding = nn.Parameter(
            torch.zeros(1, self.history_embedding_emb_dim),
            requires_grad=True,
        )
        self.plug_reference_seq = nn.Parameter(
            torch.zeros(1, self.reference_len, self.one_lat_dim), 
            requires_grad=True,
        )
        self.plug_class_emb = nn.Parameter(
            torch.zeros(1, self.classes_emb_dim),
            requires_grad=True,
        )

        name2model = {
            'cunet': self.cunet,
            'map_noise': self.map_noise,
            'time_embed': self.time_embed,
            'map_classes': self.map_classes,
            'map_history_embedding': self.map_history_embedding
        }
        num_module_params(name2model)

    def forward(
        self, 
        x : torch.Tensor, 
        noise_labels : torch.Tensor, 
        class_labels : torch.Tensor | None = None, 
        time_deltas: torch.Tensor | None = None,
        history_embedding : torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        **kwargs,):
        if (noise_labels.ndim > 1) and (noise_labels.size(1) > 1):
            raise Exception('Unet does not support per-event time scheduling')
        
        bs = x.size(0)

        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        emb = self.time_embed(emb)[:, :, None]

        # adding class condition
        if self.is_class_conditional:

            if class_labels is None:
                class_emb = self.plug_class_emb.unsqueeze(-1).repeat(bs, 1, 1)
            else:
                assert bs == class_labels.size(0)
                class_emb = self.map_classes(class_labels).view(bs, self.classes_emb_dim, 1)
            emb = torch.cat([emb, class_emb], dim=1)
        
        # adding history_embedding condition
        if self.is_history_embedding_conditional:
            if history_embedding is None:
                history_embedding_emb = self.plug_history_embedding.unsqueeze(-1).repeat(bs, 1, 1)
            else:
                assert bs == history_embedding.size(0)
                history_embedding_emb = self.map_history_embedding(history_embedding).view(
                    history_embedding.size(0), 
                    self.history_embedding_emb_dim, 
                    1
                )
            emb = torch.cat([emb, history_embedding_emb], dim=1)
        
        assert x.shape == (bs, self.length, self.one_lat_dim)

        # adding reference condition
        if self.is_reference_conditional:
            if reference is None:
                reference = self.plug_reference_seq.repeat(bs, 1, 1)
            assert reference.shape == (bs, self.reference_len, self.one_lat_dim)
            x = torch.cat([reference, x], dim=1)
            assert x.size(1) == self.unet_length

        if self.is_trans_time_conditional:
            assert x.shape[:2] == time_deltas.shape[:2]
            x = torch.cat([x, time_deltas.unsqueeze(-1)], dim=-1)

        x = x.transpose(1, 2)  # (bs, one_lat_dim, len)
        x = self.cunet(x, emb)
        x = x.transpose(1, 2)

        # removing leading history dims (if presented)
        return x[:, -self.length:]
