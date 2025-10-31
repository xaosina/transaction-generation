import torch
import torch.nn as nn
import math

def get_time_pos_emb(emb_type, dim, n_steps):
    if emb_type == "discrete":
        return DiscreteSinusoidalPosEmb(int(dim), n_steps)
    elif emb_type == "continuous":
        return ContinuousSinusoidalPosEmb(int(dim), n_steps)
    else:
        raise ValueError(f"Unknown embedding type: {emb_type}")

class ContinuousSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps=1000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiscreteSinusoidalPosEmb(nn.Module):
    # why need rescale?
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, t):
        t = t / self.num_steps * self.rescale_steps
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
