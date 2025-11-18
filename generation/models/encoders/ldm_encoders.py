import torch
import torch.nn as nn
import logging
from typing import Tuple, List, Dict, Any
from copy import deepcopy

logger = logging.getLogger(__name__)

# nn backbones
from .ldm.networks import load_ldm_denoiser

# TabSyn++
from .ldm import tabsyn

# TabBridge
from .ldm import dbim

# ADiff4tpp (FM)
from .ldm import adiff4tpp
from torchdiffeq import odeint

from ebes.model.seq2seq import BaseSeq2Seq

# from ...generator import Reshaper

from ebes.types import Seq


def seq2tensor(s: Seq | None) -> torch.Tensor | None:
    """
    transforms Seq [L, B, D] with fixed L to [B, L, D]
    """
    if s is None:
        return None
    assert s.lengths.min() == s.lengths.max()
    return s.tokens.transpose(0, 1)


class SeqReshaper(nn.Module):

    def __init__(self, gen_len: int):
        super().__init__()
        self.gen_len = gen_len

    def forward(self, seq: Seq) -> Seq:
        tensor = seq.tokens
        assert tensor.ndim == 3  # (bs, len, D)
        assert (
            tensor.size(1) == self.gen_len
        ), f"wrong length, got {tensor.size(1)} while expected {self.gen_len}"

        B = tensor.size(0)
        return Seq(
            tokens=tensor.permute(1, 0, 2).contiguous(),
            lengths=torch.ones((B,), dtype=torch.long, device=tensor.device)
            * self.gen_len,
            time=None,
        )


class ConditionalDiffusionEncoder(BaseSeq2Seq):

    def __init__(self, name: str, params: dict = None):

        super().__init__()
        params = params or {}

        denoiser_name = params["denoiser_name"]
        self.denoise_fn = load_ldm_denoiser(denoiser_name, params)

        self.model = tabsyn.EDM(denoise_fn=self.denoise_fn)
        self.gen_reshaper = SeqReshaper(params["generation_len"])

        self.num_model_params = sum(p.numel() for p in self.denoise_fn.parameters())
        self.generation_len = params["generation_len"]
        self.reference_len = params["reference_len"]
        self.latent_dim = params["input_size"]
        self.history_encoder_dim = params["history_encoder_dim"]
        self.trans_time_conditional=params['time_deltas']

        if params.get("matching", False):
            self.match_emb_size = self.generation_len
        else:
            self.match_emb_size = None

        # cfg
        self.cfg_p_uncond = params.get("cfg_p_uncond", 0.0)
        self.cfg_w = params.get("cfg_w", 1.0)
        if self.cfg_p_uncond > 0.0:
            logger.info("model will be trained with cfg support!")
        else:
            assert self.cfg_w == 1.0

    @property
    def output_dim(self):
        return self.latent_dim

    def forward(
        self,
        target_seq: Seq,
        class_labels: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
        time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:

        target_seq = seq2tensor(target_seq)  # (B, L_gen, D)
        reference_seq = seq2tensor(reference_seq)  # (B, L_ref, D)

        # cfg
        if torch.rand((1,)).item() < self.cfg_p_uncond:
            class_labels = None
            history_embedding = None
            reference_seq = None

        # asserts dimensions
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)

        if reference_seq is not None:
            assert reference_seq.shape == (bs, self.reference_len, self.latent_dim)

        if time_deltas is not None:
            assert time_deltas.shape == (bs, self.history_len + self.generation_len)

        if history_embedding is not None:
            assert history_embedding.shape == (bs, self.history_encoder_dim)

        return self.model(
            target_seq,
            class_labels,
            time_deltas,
            history_embedding,
            reference_seq,
            match_emb_size=self.match_emb_size,
        )  # returns loss

    def generate(
        self,
        n_seqs: int,
        class_labels: torch.Tensor | None = None,
        time_deltas: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
    ) -> Seq:

        self.model.eval()

        reference_seq = seq2tensor(reference_seq)
        if reference_seq is not None:
            assert reference_seq.shape == (n_seqs, self.reference_len, self.latent_dim)

        if history_embedding is not None:
            assert history_embedding.shape == (n_seqs, self.history_encoder_dim)

        if time_deltas is not None:
            assert time_deltas.shape == (n_seqs, self.history_len + self.generation_len)

        _samp = tabsyn.sample(
            self.model.denoise_fn_D,
            n_seqs,
            dims=(self.generation_len, self.latent_dim),
            class_labels=class_labels,
            time_deltas=time_deltas,
            history_embedding=history_embedding,
            reference=reference_seq,
            cfg_w=self.cfg_w,
        )  # [B, L, D]

        return self.gen_reshaper(
            Seq(
                tokens=_samp,
                lengths=None,
                time=None,
            )
        )  # [L, B, D]


class ConditionalBridgeEncoder(BaseSeq2Seq):

    def __init__(self, name: str, params: dict = None):

        super().__init__()
        params = params or {}

        if not hasattr(params, "reference_len"):
            params["reference_len"] = params["generation_len"]

        denoiser_name = params["denoiser_name"]
        self.denoise_fn = load_ldm_denoiser(denoiser_name, params)

        assert not set(["noise_schedule", "gen_sampler", "sampling_nfe"]) - set(
            params.keys()
        )

        self.sampling_params = dbim.get_sampling_params(params)
        logger.info(f"Bridge diffusion: used sampling params: {self.sampling_params}")
        self.diffusion = dbim.get_diffusion(params)
        self.schedule_sampler = dbim.create_named_schedule_sampler(
            "real-uniform", self.diffusion
        )

        self.gen_reshaper = SeqReshaper(params["generation_len"])

        self.num_model_params = sum(p.numel() for p in self.denoise_fn.parameters())
        self.generation_len = params["generation_len"]
        self.latent_dim = params["input_size"]
        self.history_encoder_dim = params["history_encoder_dim"]

    @property
    def reference_len(self):
        return self.generation_len

    @property
    def output_dim(self):
        return self.latent_dim

    def forward(
        self,
        target_seq: Seq,
        class_labels: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
    ) -> torch.Tensor:

        assert reference_seq is not None

        target_seq = seq2tensor(target_seq)  # (B, L_gen, D)
        reference_seq = seq2tensor(reference_seq)  # (B, L_hist == L_gen, D)

        # asserts dimensions
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)
        assert reference_seq.shape == (bs, self.reference_len, self.latent_dim)

        if history_embedding is not None:
            assert history_embedding.shape == (bs, self.history_encoder_dim)

        t, weights = self.schedule_sampler.sample(
            target_seq.shape[0], target_seq.device
        )

        losses = self.diffusion.training_bridge_losses(
            self.denoise_fn,
            target_seq,
            t,
            model_kwargs=dict(
                history_embedding=history_embedding,
                reference=reference_seq,
                class_labels=class_labels,
                xT=reference_seq,
            ),
        )

        loss = (losses["loss"] * weights).mean()

        return loss

    def generate(
        self,
        n_seqs: int,
        class_labels: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
        return_path: bool = False,
        return_x0_pred: bool = False,
    ) -> Seq | Tuple[Seq, Dict[str, Any]]:

        assert reference_seq is not None

        reference_seq = seq2tensor(reference_seq)
        assert reference_seq.shape == (n_seqs, self.reference_len, self.latent_dim)

        if history_embedding is not None:
            assert history_embedding.shape == (n_seqs, self.history_encoder_dim)

        _samp, _path, _, _pred_x0, _, _ = dbim.karras_sample(
            self.diffusion,
            self.denoise_fn,
            reference_seq,
            None,
            steps=self.sampling_params["steps"],
            mask=None,
            model_kwargs=dict(
                history_embedding=history_embedding,
                reference=reference_seq,
                class_labels=class_labels,
                xT=reference_seq,
            ),
            device=reference_seq.device,
            rho=self.sampling_params["rho"],
            sampler=self.sampling_params["sampler"],
            churn_step_ratio=self.sampling_params["churn_step_ratio"],
            eta=self.sampling_params["eta"],
            order=self.sampling_params["order"],
        )

        samp = self.gen_reshaper(
            Seq(tokens=_samp, lengths=None, time=None)
        )  # [L, B, D]

        if (not return_path) and (not return_x0_pred):
            return samp
        else:
            bridge_traj = dict()
            if return_path:
                bridge_traj["path"] = [
                    self.gen_reshaper(Seq(tokens=path_inst, lengths=None, time=None))
                    for path_inst in _path
                ]
            if return_x0_pred:
                bridge_traj["x0_pred"] = [
                    self.gen_reshaper(Seq(tokens=pred_x0_inst, lengths=None, time=None))
                    for pred_x0_inst in _pred_x0
                ]
            return samp, bridge_traj


class AsyncDiffEncoder(BaseSeq2Seq):

    def __init__(self, name: str, params: dict = None):

        super().__init__()

        self.params = params or {}

        denoiser_name = params["denoiser_name"]
        self.denoise_fn = load_ldm_denoiser(denoiser_name, params)

        self.async_t_schedule_name = params["schedule"]
        self.data_init = params["data_init"]
        # self.use_history_mask = params['history_mask']

        self.loss_func = adiff4tpp.get_loss_func(params["loss_type"])
        self.gen_reshaper = SeqReshaper(params["generation_len"])
        # self.mask = params['mask']
        self.generation_len = params["generation_len"]
        self.reference_len = params["reference_len"]
        self.latent_dim = params["input_size"]
        # self.data_init_noisesigma = params.get('data_init_noisesigma', 0.0)
        self.data_init_noisesigma = params["data_init_noisesigma"]

    @property
    def output_dim(self):
        return self.latent_dim

    def forward(
        self,
        target_seq: Seq,
        class_labels: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
    ) -> torch.Tensor:

        target_seq = seq2tensor(target_seq)  # (B, L_gen, D)
        bs = target_seq.size(0)
        assert target_seq.shape == (bs, self.generation_len, self.latent_dim)

        if reference_seq is not None:
            reference_seq = seq2tensor(reference_seq)  # (B, L_hist, D)
            assert reference_seq.shape == (bs, self.reference_len, self.latent_dim)

        async_t_shedule = adiff4tpp.obtain_noise_schedule(self.async_t_schedule_name)(
            self.generation_len
        )

        noise_fixed = torch.randn_like(target_seq)

        if self.data_init:
            assert self.reference_len == self.generation_len
            assert reference_seq is not None
            noise_fixed = deepcopy(reference_seq)
            noise_fixed += torch.randn_like(noise_fixed) * self.data_init_noisesigma

        # Sample t, zt
        t = adiff4tpp.sample_t(bs).view(-1, 1)  # (bs, 1)
        a_t = async_t_shedule(t).to(target_seq.device)  # (bs, L_gen)
        a_t_dot = (
            async_t_shedule.derivative(t).unsqueeze(-1).to(target_seq.device)
        )  # (bs, L_gen, 1)
        zt = a_t.unsqueeze(-1) * target_seq + (1 - a_t.unsqueeze(-1)) * noise_fixed
        target = target_seq - noise_fixed

        pred = self.denoise_fn(
            zt,
            a_t,  # noise labels (times)
            class_labels=class_labels,
            denoise_fn=history_embedding,
            reference=reference_seq,
        )

        # Compute loss
        assert pred.shape == target.shape

        loss = self.loss_func(pred, target, a_t_dot)
        loss = loss.mean()
        return loss

    def generate(
        self,
        n_seqs: int,
        class_labels: torch.Tensor | None = None,
        history_embedding: torch.Tensor | None = None,
        reference_seq: Seq | None = None,
    ) -> Seq:

        if reference_seq is not None:
            reference_seq = seq2tensor(reference_seq)
            assert reference_seq.shape == (n_seqs, self.reference_len, self.latent_dim)

        dtype, device = (
            next(iter(self.parameters())).dtype,
            next(iter(self.parameters())).device,
        )
        noise_fixed = torch.randn(
            (n_seqs, self.generation_len, self.latent_dim),
            dtype=dtype,
            device=device,
        )

        async_t_shedule = adiff4tpp.obtain_noise_schedule(self.async_t_schedule_name)(
            self.generation_len
        ).to(device)

        if self.data_init:
            assert self.generation_len == self.reference_len
            assert reference_seq is not None
            noise_fixed = deepcopy(reference_seq)
            noise_fixed += torch.randn_like(noise_fixed) * self.data_init_noisesigma

        # Define the ODE function for solving the reverse flow
        def ode_func(t, x):
            t = t.view(-1, 1).repeat(n_seqs, 1)
            assert t.size(0) == n_seqs
            a_t = async_t_shedule(t)
            a_t_dot = async_t_shedule.derivative(t).unsqueeze(-1)

            # Compute vector field: x_0 - epsilon
            v = self.denoise_fn(
                x,
                a_t,
                class_labels=class_labels,
                history_embedding=history_embedding,
                reference=reference_seq,
            )
            assert len(a_t_dot.shape) == len(v.shape)
            return a_t_dot * v

        # Sample t, zt
        solution = odeint(
            ode_func,
            noise_fixed,
            async_t_shedule.times,
            rtol=1e-5,
            atol=1e-5,
            method=self.params["int_ode"],
        )

        # Extract the result at t=0
        return self.gen_reshaper(Seq(tokens=solution[-1], lengths=None, time=None))
