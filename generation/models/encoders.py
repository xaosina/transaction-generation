import torch
from torch import nn

from ebes.model.seq2seq import BaseSeq2Seq
from ..data.types import Seq
from .transformer.ar import AR


class ARTransformer(BaseSeq2Seq):
    """Autoregressive transformer"""

    def __init__(self, cfg):
        super().__init__()
        params = cfg.params

        self.projector = nn.Linear(params.input_dim, params.d_model)

        self.transformer = AR(
            params.num_tokens,
            d_model=params.d_model,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
        )

    def forward(self, seq: Seq):
        x = seq.tokens  # B, L, D
        padding_mask = torch.arange(x.shape[0], device=x.device) >= x.lengths[:, None]

        x = self.projector(x)
        x = self.transformer(x, padding_mask)

        return x

    def generate(self, seq: Seq, max_step: int) -> Seq:
        x = seq.tokens  # B, L, D
        padding_mask = torch.arange(x.shape[0], device=x.device) >= x.lengths[:, None]

        x = self.projector(x)

        mask_row = torch.ones(
            (padding_mask.size(0), 1),
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        generated = []
        for _ in range(max_step):
            ret = self.transformer(x, padding_mask)  # B, L, D

            x = torch.cat([x, ret[:, :-1, :]], dim=1)
            padding_mask = torch.cat([padding_mask, mask_row], dim=1)

            generated.append(ret)
        generated = torch.cat(generated, dim=1)
        
        return generated
