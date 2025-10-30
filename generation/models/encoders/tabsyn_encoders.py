import torch
import torch.nn as nn
import logging
from typing import Tuple, List, Dict, Any
from copy import deepcopy

logger = logging.getLogger(__name__)

from .tabsyn.model import Unet1DDiffusion

# diffusion
from .tabsyn.model import EDM
from .tabsyn.diffusion_utils import sample

# # bridge
# from .tabsyn.ddbm.karras_diffusion import (
#     KarrasDenoiser,
#     karras_sample
# )
# from .tabsyn.ddbm.resample import (
#     create_named_schedule_sampler,
#     LossAwareSampler
# )

# bridge
from .tabsyn.dbim.train_util import (
    get_diffusion,
    get_sampling_params
)
from .tabsyn.dbim.karras_diffusion import karras_sample
from .tabsyn.dbim.resample import create_named_schedule_sampler

# FM (adiff4tpp)
from .adiff4tpp.DiT_models import DiT
from .adiff4tpp.train_util import get_loss_func as get_adifftpp_loss_func
from .adiff4tpp.train_util import sample_t as sample_t_adifftpp
from .adiff4tpp.async_lib import obtain_noise_schedule as obtain_adifftpp_noise_schedule
from torchdiffeq import odeint

from ...data.data_types import seq_append

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
        if params.get("matching", False):
            self.match_emb_size = self.generation_len
        else:
            self.match_emb_size = None
    
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
            match_emb_size=self.match_emb_size,
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

        assert not set(['noise_schedule', 'gen_sampler', 'sampling_nfe']) - set(params.keys())

        self.sampling_params = get_sampling_params(params)
        logger.info(f'Bridge diffusion: used sampling params: {self.sampling_params}')
        self.diffusion = get_diffusion(params)
        self.schedule_sampler = create_named_schedule_sampler(
            "real-uniform", self.diffusion)

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
        
        loss = (losses["loss"] * weights).mean()

        return loss
    
    def generate(
            self, 
            n_seqs: int, 
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None, 
            history_seq: Seq | None = None,
            return_path : bool = False,
            return_x0_pred : bool = False,
        ) -> Seq | Tuple[Seq, Dict[str, Any]]:

        assert history_seq is not None

        history_seq = seq2tensor(history_seq)
        assert history_seq.shape == (n_seqs, self.input_history_len, self.latent_dim)
        history_seq = torch.flatten(history_seq, start_dim=1)

        if history_embedding is not None:
            assert history_embedding.shape == (n_seqs, self.history_encoder_dim)
        
        _samp, _path, _, _pred_x0, _, _ = karras_sample(
            self.diffusion,
            self.denoise_fn,
            history_seq,
            None,
            steps=self.sampling_params['steps'],
            mask=None,
            model_kwargs=dict(
                hstate=history_embedding,
                rawhist=history_seq if self.history_condition_len > 0 else None,
                class_labels=class_labels,
                xT=history_seq,
            ),
            device=history_seq.device,
            rho = self.sampling_params['rho'],
            sampler=self.sampling_params['sampler'],
            churn_step_ratio=self.sampling_params['churn_step_ratio'],
            eta=self.sampling_params['eta'],
            order=self.sampling_params['order'],
        )

        samp = self.gen_reshaper(
                Seq(
                    tokens=_samp, 
                    lengths=torch.ones((n_seqs,)).to(_samp.device) * self.generation_len, 
                    time=None
                )
            ) # [L, B, D]

        if (not return_path) and (not return_x0_pred):
            return samp
        else:
            bridge_traj = dict()
            if return_path:
                bridge_traj['path'] = [
                            self.gen_reshaper(
                                Seq(
                                    tokens=path_inst, 
                                    lengths=torch.ones((n_seqs,)).to(path_inst.device) * self.generation_len, 
                                    time=None
                                )
                        ) for path_inst in _path
                    ]
            if return_x0_pred:
                bridge_traj['x0_pred'] = [
                            self.gen_reshaper(
                                Seq(
                                    tokens=pred_x0_inst, 
                                    lengths=torch.ones((n_seqs,)).to(pred_x0_inst.device) * self.generation_len, 
                                    time=None
                                )
                        ) for pred_x0_inst in _pred_x0
                    ]
            return samp, bridge_traj


class AsynDiffEncoder(BaseSeq2Seq):

    def __init__(self, name: str, params: dict = None):

        super().__init__()

        self.params = params
        self.model = DiT(
            generation_len=params['generation_len'],
            history_len=params['history_len'],
            latent_size=params['input_size'],
            hidden_size=params['hidden_size'],
            depth=params['depth'],
            num_heads=params['num_heads'],
            mlp_ratio=params['mlp_ratio'],
            hstate_dim=params['history_encoder_dim'],
            learn_sigma=params['learn_sigma']
        )
        self.Aschedule = params['schedule']
        self.data_init = params['data_init']  
        # self.use_history_mask = params['history_mask']

        self.loss_func = get_adifftpp_loss_func(params['loss_type'])
        self.gen_reshaper = Reshaper(params['generation_len'])
        self.mask = params['mask']
        self.generation_len = params['generation_len']
        self.history_len = params['history_len']
        self.latent_dim = params['input_size']
        # self.data_init_noisesigma = params.get('data_init_noisesigma', 0.0)
        self.data_init_noisesigma = params['data_init_noisesigma']
    
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
            history_seq: Seq | None = None,
        ) -> torch.Tensor :

        target_seq = seq2tensor(target_seq) # (B, L_gen, D)
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)

        if history_seq is not None:
            history_seq = seq2tensor(history_seq) # (B, L_hist, D)
            assert history_seq.shape == (bs, self.input_history_len, self.latent_dim)

        # hist_target_seq = torch.cat([history_seq, target_seq], dim=1)
        # assert hist_target_seq.shape == (bs, self.generation_len + self.input_history_len, self.latent_dim)

        # ht_len = hist_target_seq.size(1)
        # ht_lens_batch = torch.ones((target_seq.size(0),), dtype=torch.long, device=target_seq.device) * ht_len
        
        A = obtain_adifftpp_noise_schedule(self.Aschedule)(self.generation_len)

        # col_indices = torch.arange(ht_len).unsqueeze(0).to(target_seq.device)
        # history_mask = col_indices < (ht_lens_batch - self.generation_len).unsqueeze(1)

        noise_fixed = torch.randn_like(target_seq)

        if self.data_init:
            assert self.input_history_len == self.generation_len
            assert history_seq is not None
            noise_fixed = deepcopy(history_seq)
            noise_fixed += torch.randn_like(noise_fixed) * self.data_init_noisesigma

        # Sample t, zt
        t = sample_t_adifftpp(bs).view(-1,1) # (bs,)
        A_t = A(t).to(target_seq.device) # (bs, L_gen)
        A_t_dot = A.derivative(t).unsqueeze(-1).to(target_seq.device) # (bs, L_gen, 1)
        zt = A_t.unsqueeze(-1)*target_seq + (1-A_t.unsqueeze(-1))*noise_fixed
        target = target_seq - noise_fixed
        
        pred = self.model(
            zt, 
            A_t, # noise labels (times)
            class_labels=class_labels, 
            hstate=history_embedding, 
            rawhist=history_seq
        )

        # Compute loss
        # if self.use_history_mask:
        # pred = pred[~history_mask]
        # pred[history_mask] = 0.
        # target[history_mask] = 0.
        # breakpoint()

        assert pred.shape == target.shape

        loss = self.loss_func(pred, target, A_t_dot)
        loss = loss.mean()
        return loss

    def generate(
            self, 
            n_seqs: int, 
            class_labels: torch.Tensor | None = None, 
            history_embedding: torch.Tensor | None = None, 
            history_seq: Seq | None = None,
        ) -> Seq :

        if history_seq is not None:
            history_seq = seq2tensor(history_seq)
            assert history_seq.shape == (n_seqs, self.input_history_len, self.latent_dim)

        dtype, device = next(iter(self.parameters())).dtype, next(iter(self.parameters())).device
        noise_fixed = torch.randn(
            (n_seqs, self.generation_len, self.latent_dim), 
            dtype=dtype, 
            device=device,
        )

        # hist_target_seq = torch.cat([history_seq, target_init_seq], dim=1)
        # assert hist_target_seq.shape == (n_seqs, self.generation_len + self.input_history_len, self.latent_dim)

        # ht_len = hist_target_seq.size(1)
        # ht_lens_batch = torch.ones((n_seqs,), dtype=torch.long, device=history_seq.device) * ht_len
        # A = obtain_adifftpp_noise_schedule(self.Aschedule)(ht_lens_batch, ht_len).to(history_seq.device)

        A = obtain_adifftpp_noise_schedule(self.Aschedule)(self.generation_len).to(device)

        # col_indices = torch.arange(ht_len).unsqueeze(0).to(history_seq.device)
        # history_mask = col_indices < (ht_lens_batch - self.generation_len).unsqueeze(1)

        # noise_fixed = deepcopy(hist_target_seq)

        if self.data_init:
            assert self.generation_len == self.history_len
            assert history_seq is not None
            noise_fixed = deepcopy(history_seq)
            noise_fixed += torch.randn_like(noise_fixed) * self.data_init_noisesigma
        
        # Define the ODE function for solving the reverse flow
        def ode_func(t, x):
            t = t.view(-1,1).repeat(n_seqs, 1)
            assert t.size(0) == n_seqs
            A_t = A(t)
            A_t_dot = A.derivative(t).unsqueeze(-1)
            # Compute vector field: x_0 - epsilon
            v = self.model(
                x, 
                A_t, 
                class_labels=class_labels, 
                hstate=history_embedding, 
                rawhist=history_seq
            )
            assert len(A_t_dot.shape) == len(v.shape)
            return A_t_dot*v

        # Sample t, zt
        solution = odeint(ode_func, noise_fixed, A.times, rtol=1e-5, atol=1e-5, method=self.params["int_ode"])
        # Extract the result at t=0
        
        pred = solution[-1].view(n_seqs,-1)
        # pred = x_restored[:,self.input_history_len:,:].view(n_seqs,-1)
        return self.gen_reshaper(
            Seq(
                tokens=pred, 
                lengths=torch.ones((n_seqs,)).to(pred.device) * self.generation_len, 
                time=None
            )
        )
