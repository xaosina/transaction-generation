import torch
from torch import nn

from ebes.model.seq2seq import BaseSeq2Seq
from ..data.types import Seq
from .transformer.ar import AR


class ARTransformer(BaseSeq2Seq):

    def __init__(self, 
                 cfg):
        super().__init__()
        params = cfg.params

        self.projector = nn.Linear(params.input_dim, params.d_model)

        self.transformer = AR(
            params.num_tokens,
            d_model=params.d_model,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
        )

    def forward(self, x: Seq):
        x = Seq.tokens.permute(1, 0, 2)
        m = Seq.masks.permute(1, 0, 2)
        
        x = self.projector(x)
        x = self.transformer(x, m)
        
        return x

    def generate(self, x: torch.Tensor, max_step: int) -> Seq:
        x = Seq.tokens.permute(1, 0, 2)
        m = Seq.tokens.permute(1, 0, 2)
        
        x = self.projector(x)
        
        mask_row = torch.ones(
            (m.size(0), 1),
            dtype=m.dtype,
            device=m.device
        )

        generated = []
        for _ in range(max_step):
            ret = self.transformer(x, m)

            x = torch.cat(x, ret, dim=1)
            m = torch.cat(m, mask_row, dim=1)
            
            generated.append(ret)
        
        return torch.cat(generated)

