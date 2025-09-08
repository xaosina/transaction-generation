import torch
import torch.nn as nn
from .model import EDM
from .model import Unet1DDiffusion
from .diffusion_utils import sample
from ebes.model.seq2seq import BaseSeq2Seq
# from ...generator import Reshaper

from ebes.types import Seq

# it is the copy of ..generator.base.Reshaper, to prevent circular import
class Reshaper(nn.Module):
    def __init__(self, gen_len: int):
        super().__init__()
        self.gen_len = gen_len

    def forward(self, seq: Seq) -> Seq:
        tensor = seq.tokens
        assert (
            tensor.shape[1] % self.gen_len == 0
        ), f"hidden_size doesnt divide by {self.gen_len}"
        B, D = tensor.shape
        return Seq(
            tokens=tensor.view(B, self.gen_len, D // self.gen_len).permute(1, 0, 2).contiguous(),
            lengths=torch.ones((B,), dtype=torch.long, device=tensor.device) * self.gen_len,
            time=None,
        )

class ConditionalDiffusionEncoder(BaseSeq2Seq):

    @staticmethod
    def seq2tensor(s: Seq | None) -> torch.Tensor | None:
        '''
        transforms Seq [L, B, D] with fixed L to [B, L, D]
        '''
        if s is None:
            return None
        assert s.lengths.min() == s.lengths.max()
        return s.tokens.transpose(0, 1)


    def __init__(self, name: str, params: dict = None):

        super().__init__()
        params = params or {}

        hstate_condition = params['history_encoder_dim'] > 0
        self.denoise_fn = Unet1DDiffusion(
            latent_dim=params['input_size'],
            dim_t=params['dim_t'],
            length=params['generation_len'],
            base_factor=params['base_factor'],
            num_classes=params['num_classes'],
            hstate_condition=hstate_condition,
            hstate_dim=params['history_encoder_dim'],
            rawhist_length=params['history_len']
        )

        self.model = EDM(
            denoise_fn=self.denoise_fn
        )
        self.gen_reshaper = Reshaper(params['generation_len'])

        self.num_model_params = sum(p.numel() for p in self.denoise_fn.parameters())
        self.generation_len = params['generation_len']
        self.history_len = params['history_len']
        self.latent_dim = params['input_size']
        self.history_encoder_dim = params['history_encoder_dim']
    
    @property
    def output_dim(self):
        return self.latent_dim

    def forward(
            self, 
            target_seq: Seq,
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None,
            history_seq: Seq | None = None
        ) -> torch.Tensor :

        target_seq = self.seq2tensor(target_seq) # (B, L_gen, D)
        history_seq = self.seq2tensor(history_seq) # (B, L_hist, D)

        # asserts dimensions
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)
        target_seq = torch.flatten(target_seq, start_dim=1)
        if history_seq is not None:
            assert history_seq.shape == (bs, self.history_len, self.latent_dim)
            history_seq = torch.flatten(history_seq, start_dim=1)
        if history_embedding is not None:
            assert history_embedding.shape == (bs, self.history_encoder_dim)

        return self.model(
            target_seq,
            class_labels,
            history_embedding,
            history_seq,
        ) # returns loss
    
    def generate(
            self, 
            n_seqs: int, 
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None, 
            history_seq: Seq | None = None
        ) -> Seq :

        self.model.eval()

        history_seq = self.seq2tensor(history_seq)
        if history_seq is not None:
            assert history_seq.shape == (n_seqs, self.history_len, self.latent_dim)
            history_seq = torch.flatten(history_seq, start_dim=1)

        if history_embedding is not None:
            assert history_embedding.shape == (n_seqs, self.history_encoder_dim)

        _samp = sample(
            self.model.denoise_fn_D, 
            n_seqs, 
            self.latent_dim * self.generation_len, 
            class_labels=class_labels, 
            hstate=history_embedding, 
            rawhist=history_seq
        ) # [B, L*D]

        return self.gen_reshaper(
            Seq(
                tokens=_samp, 
                lengths=torch.ones((n_seqs,)).to(_samp.device) * self.generation_len, 
                time=None
            )
        ) # [L, B, D]
