from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from .diffusion_utils import EDMLoss
from ..torch_util import append_dims


class Precond(nn.Module):

    def __init__(
        self,
        denoise_fn: nn.Module,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(
            self, x, sigma, 
            class_labels : torch.Tensor | None = None, 
            time_deltas: torch.Tensor | None = None,
            history_embedding : torch.Tensor | None = None,
            reference: torch.Tensor | None = None,
        ):
        # pdb.set_trace()
        dtype = torch.float32

        x = x.to(dtype)

        sigma = append_dims(sigma.to(dtype), x.ndim)
        assert sigma.ndim == x.ndim

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F(
            (x_in).to(dtype), 
            c_noise.flatten(), 
            time_deltas=time_deltas,
            class_labels=class_labels, 
            history_embedding=history_embedding, 
            reference=reference,
        )

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDM(nn.Module):
    def __init__(
        self,
        denoise_fn : nn.Module,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        gamma=5,
    ):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn)
        self.loss_fn = EDMLoss(
            P_mean, P_std, sigma_data, gamma=gamma
        )

    def forward(
            self, 
            x: torch.Tensor, 
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None,
            time_deltas: torch.Tensor | None = None,
            reference: torch.Tensor | None = None,
            match_emb_size: int = None
        ):

        loss = self.loss_fn(self.denoise_fn_D, x, class_labels, time_deltas, history_embedding, reference, match_emb_size)
        return loss.mean()
