# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################
# Code is based on the DiT (https://arxiv.org/pdf/2212.09748) implementation
# from https://github.com/facebookresearch/DiT by Meta which is licensed under CC-BY-NC 4.0.
# You may obtain a copy of the License at
#
# https://creativecommons.org/licenses/by-nc/4.0/deed.en
#
##################################################################################################

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None,:]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class CausalAttention(Attention):
    """
    Causal Attention with mask applied.
    """
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, C // num_heads

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and causal masking.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CausalAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, causal_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        y = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(y, mask=causal_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, latent_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, latent_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        num_rows=32,
        latent_size=2,
        hidden_size=1152,
        depth=7,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.num_rows = num_rows
        self.learn_sigma = learn_sigma
        self.in_channels = 1
        self.out_channels = 1
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert hidden_size % num_heads == 0 
        self.repeat_num = hidden_size // latent_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_rows, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, latent_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed_by_row(self.hidden_size, self.num_rows)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)

        # Zero-out output layers:
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].bias, std=0.02)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.normal_(self.final_layer.linear.bias, std=0.02)

    def forward(self, x, A_t, attn_mask=None):
        """
        Forward pass of DiT.
        x: (B, N, L) tensor of spatial inputs (images or latent representations of images)
        A_t: (B, N) tensor of asynchronous diffusion timesteps
        mask_len: Tensor. Masks all events after the prediction length.
        """
        x = x.repeat(1,1,self.repeat_num) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        A_t = self.t_embedder(A_t)                          # (N, D)
        c = A_t                                             # (N, D)

        if attn_mask is None:
            # Create a causal mask
            causal_mask = torch.ones(1,1,x.shape[1],x.shape[1], dtype=x.dtype, device=x.device)
            causal_mask[:,:,:,self.num_rows:] = 0
        else:
            causal_mask = attn_mask

        for block in self.blocks:
            x = block(x, c, causal_mask=causal_mask) # (N, T, D)

        x = self.final_layer(x, c)
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_by_row(embed_dim, num_rows):
    """
    num_rows: int of the number of rows (height)
    return:
    pos_embed: [num_rows, embed_dim] or [1 + num_rows, embed_dim] (with or without cls_token)
    """
    grid_h = np.arange(num_rows, dtype=np.float32)  # only varies by row (y-axis)

    # Get the 1D positional embedding for the rows (y-axis only)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_h)
    pos_embed = pos_embed

    return pos_embed
