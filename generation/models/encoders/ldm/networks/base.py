import torch
import torch.nn as nn


class LDMDenoiser(nn.Module):
    '''
    Basic class for latent diffusion models denoisers
    '''

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        x : torch.Tensor,
        noise_labels: torch.Tensor, # times/sigmas etc
        class_labels : torch.Tensor | None = None, 
        history_embedding : torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        raise Exception('Not implemented error')