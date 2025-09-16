import torch
import torch.nn as nn

from .tabsyn.model import Unet1DDiffusion

# diffusion
from .tabsyn.model import EDM
from .tabsyn.diffusion_utils import sample

# bridge
from .tabsyn.ddbm.karras_diffusion import (
    KarrasDenoiser,
    karras_sample
)
from .tabsyn.ddbm.resample import (
    create_named_schedule_sampler,
    LossAwareSampler
)

from ebes.model.seq2seq import BaseSeq2Seq
# from ...generator import Reshaper

from ebes.types import Seq

def seq2tensor(s: Seq | None) -> torch.Tensor | None:
        '''
        transforms Seq [L, B, D] with fixed L to [B, L, D]
        '''
        if s is None:
            return None
        assert s.lengths.min() == s.lengths.max()
        return s.tokens.transpose(0, 1)


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
    def input_history_len(self):
        return self.history_len
    
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

        target_seq = seq2tensor(target_seq) # (B, L_gen, D)
        history_seq = seq2tensor(history_seq) # (B, L_hist, D)

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

        history_seq = seq2tensor(history_seq)
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


class ConditionalBridgeEncoder(BaseSeq2Seq):

    def __init__(self, name: str, params: dict = None):

        super().__init__()
        params = params or {}

        hstate_condition = params['history_encoder_dim'] > 0
        if params['history_len'] > 0:
            assert params['history_len'] == params['generation_len'], 'if history condition is used, then it must coincide with generation len'

        self.denoise_fn = Unet1DDiffusion(
            latent_dim=params['input_size'],
            dim_t=params['dim_t'],
            length=params['generation_len'],
            base_factor=params['base_factor'],
            num_classes=params['num_classes'],
            hstate_condition=hstate_condition,
            hstate_dim=params['history_encoder_dim'],
            rawhist_length=params['history_len'],
        )

        self.diffusion = KarrasDenoiser(
            pred_mode=params['pred_mode'], 
            weight_schedule=params["weight_schedule"],
        ) #TODO: adapt parametersm, understand what they mean!
        self.schedule_sampler = create_named_schedule_sampler(
            params['schedule_sampler'], self.diffusion)

        self.gen_reshaper = Reshaper(params['generation_len'])

        self.num_model_params = sum(p.numel() for p in self.denoise_fn.parameters())
        self.generation_len = params['generation_len']
        self.history_condition_len = params['history_len']
        self.latent_dim = params['input_size']
        self.history_encoder_dim = params['history_encoder_dim']
    
    @property
    def input_history_len(self):
        return self.generation_len
    
    @property
    def output_dim(self):
        return self.latent_dim

    def forward(
            self, 
            target_seq: Seq,
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None,
            history_seq: Seq | None = None,
        ) -> torch.Tensor :

        assert history_seq is not None

        target_seq = seq2tensor(target_seq) # (B, L_gen, D)
        history_seq = seq2tensor(history_seq) # (B, L_hist == L_gen, D)

        # asserts dimensions
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)
        target_seq = torch.flatten(target_seq, start_dim=1)
        assert history_seq.shape == (bs, self.input_history_len, self.latent_dim)
        history_seq = torch.flatten(history_seq, start_dim=1)

        if history_embedding is not None:
            assert history_embedding.shape == (bs, self.history_encoder_dim)

        t, weights = self.schedule_sampler.sample(target_seq.shape[0], target_seq.device)
        
        losses = self.diffusion.training_bridge_losses(
            self.denoise_fn,
            target_seq,
            t,
            model_kwargs=dict(
                hstate=history_embedding,
                rawhist=history_seq if self.history_condition_len > 0 else None,
                class_labels=class_labels,
                xT=history_seq,
            )
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        
        loss = (losses["loss"] * weights).mean()

        return loss
    
    def generate(
            self, 
            n_seqs: int, 
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None, 
            history_seq: Seq | None = None
        ) -> Seq :

        assert history_seq is not None

        history_seq = seq2tensor(history_seq)
        assert history_seq.shape == (n_seqs, self.input_history_len, self.latent_dim)
        history_seq = torch.flatten(history_seq, start_dim=1)

        if history_embedding is not None:
            assert history_embedding.shape == (n_seqs, self.history_encoder_dim)
        
        _samp, path, nfe = karras_sample(
            self.diffusion,
            self.denoise_fn,
            history_seq,
            None,
            steps=40, #TODO: to config!
            model_kwargs=dict(
                hstate=history_embedding,
                rawhist=history_seq if self.history_condition_len > 0 else None,
                class_labels=class_labels,
                xT=history_seq,
            ),
            device=history_seq.device,
            sampler="heun",
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            churn_step_ratio=0., #TODO: to config
            rho=7.0, #TODO: to config
            guidance=1., #TODO: what is it? to config!
        )

        return self.gen_reshaper(
            Seq(
                tokens=_samp, 
                lengths=torch.ones((n_seqs,)).to(_samp.device) * self.generation_len, 
                time=None
            )
        ) # [L, B, D]
