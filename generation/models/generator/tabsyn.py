import logging
from copy import deepcopy
from typing import Any, Dict, List

import torch
from dacite import Config, from_dict
from ebes.model import BaseModel, TakeLastHidden
from torch import nn

# from generation.models.encoders import ConditionalDiffusionEncoder
from generation.models import autoencoders, encoders
from generation.utils import freeze_module

from ...data.data_types import (
    GenBatch,
    LatentDataConfig,
    get_seq_tail,
    seq_append,
    seq_to_device,
)
from . import BaseGenerator, ModelConfig
from .utils import match_seqs, post_process_generation

logger = logging.getLogger()


class LatentDiffusionGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.data_conf = data_conf
        self.model_config = model_config

        self.autoencoder = self._init_autoencoder()
        self.history_encoder, hist_enc_dim = self._init_history_encoder()

        # initializing encoder
        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.autoencoder.encoder.output_dim
        encoder_params["history_encoder_dim"] = hist_enc_dim
        self.encoder = getattr(encoders, model_config.latent_encoder.name)(
            model_config.latent_encoder.name, encoder_params
        )

        self.generation_len = encoder_params["generation_len"]
        self.history_len = self.encoder.input_history_len

        self.repeat_matching = model_config.params.get("repeat_matching", False)

    def _init_autoencoder(self):
        # initializing autoencoder
        model_config, data_conf = self.model_config, self.data_conf
        autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = autoencoder.load_state_dict(ckpt["model"], strict=True)

        assert model_config.autoencoder.frozen, "Not implemented unfrozen auto"
        if model_config.autoencoder.frozen:
            autoencoder = freeze_module(autoencoder)
        logger.info(f"Tabsyn latent dimension is {autoencoder.encoder.output_dim}")
        return autoencoder

    def _init_history_encoder(self):
        # initializing history encoder
        params = self.model_config.params.get("history_encoder")
        history_encoder = None  # default history encoder
        if params is None:
            print("no history encoder!")
            return

        checkpoint = params.pop("checkpoint")
        # Support old encoders
        if params["name"] == "GRU":
            params = {
                "input_size": self.autoencoder.encoder.output_dim,
                "num_layers": 1,
                "hidden_size": 256,
            }
            hist_enc_dim = 256
            history_encoder = nn.Sequential(
                [BaseModel.get_model("GRU", **params), TakeLastHidden()]
            )
        else:
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            history_encoder = BaseGenerator.get_model(params["name"])(
                self.data_conf, cfg
            )
            hist_enc_dim = history_encoder.encoder.output_dim
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = history_encoder.load_state_dict(ckpt["model"], strict=True)
            history_encoder = freeze_module(history_encoder)
        return history_encoder, hist_enc_dim

    def get_history_emb(self, hist):
        if self.history_encoder is None:
            return None
        hist = deepcopy(hist)
        if isinstance(self.history_encoder, BaseGenerator):
            return self.history_encoder.get_embeddings(hist)  # B, D
        else:
            encoded_hist = self.autoencoder.encoder(hist)
            return self.history_encoder(encoded_hist)  # B, D

    def get_history_seq(self, hist):
        if self.history_len > 0:
            return self.autoencoder.encoder(
                hist.tail(self.history_len)
            )  # TODO: use Seq.tail directly
        else:
            return None

    def forward(self, x: GenBatch) -> torch.Tensor:
        """
        Forward pass of Latent diffusion model with inherent loss compute
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        target_seq = x.get_target_batch()
        if self.repeat_matching:
            repeat = x.tail(self.history_len)
            target_seq = match_seqs(repeat, target_seq, self.data_conf)

        target_seq = self.autoencoder.encoder(target_seq)
        history_embedding = self.get_history_emb(x)
        history_seq = self.get_history_seq(x)
        # TODO: static condition
        return self.encoder(target_seq, None, history_embedding, history_seq)

    @torch.no_grad()
    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        """
        Diffusion generation

        Args:
            hist (GenBatch): history batch

        """
        assert gen_len % self.generation_len == 0
        B = len(hist)
        hist = deepcopy(hist)
        for _ in range(0, gen_len, self.generation_len):
            history_embedding = self.get_history_emb(hist)
            history_seq = self.get_history_seq(hist)
            sampled_seq = self.encoder.generate(
                B, None, history_embedding, history_seq
            )  # Seq [L, B, D]
            sampled_batch = self.autoencoder.decoder.generate(sampled_seq)
            if self.repeat_matching or self.model_config.latent_encoder.params.get(
                "matching", False
            ):
                sampled_batch = post_process_generation(sampled_batch, hist)
            hist.append(sampled_batch)

        if with_hist:
            return hist
        return hist.tail(gen_len)

    @torch.no_grad()
    def generate_traj(
        self,
        hist: GenBatch,
        *args,
        path_idss: List[int] | None = None,
        with_hist: bool = False,
        with_path: bool = False,
        with_x0_pred: bool = False,
        **kwargs,
    ) -> GenBatch:
        """
        Diffusion generation

        Args:
            hist (GenBatch): history batch

        """
        gen_len = self.generation_len

        if (not with_path) and (not with_x0_pred):
            logger.warning("generate_traj reduces to generate!")
            return self.generate(hist, gen_len, with_hist=with_hist)

        B = len(hist)
        hist = deepcopy(hist)

        history_embedding = self.get_history_emb(hist)
        history_seq = self.get_history_seq(hist)
        try:
            sampled_seq, traj = self.encoder.generate(
                B,
                None,
                history_embedding,
                history_seq,
                return_path=with_path,
                return_x0_pred=with_x0_pred,
            )  # Seq [L, B, D]
        except:
            raise Exception(
                "Latent diffusion encoder does not supports path trace yet!"
            )

        assert isinstance(traj, dict)
        assert len(traj) > 0

        def _process_traj_list(traj_list):

            if path_idss is not None:
                traj_len = len(traj_list)
                assert max(path_idss) < traj_len, (
                    f"Too large path index {max(path_idss)} is requested;"
                    f" number of generation steps is {traj_len}."
                )

                traj_list = [traj_list[idx] for idx in path_idss]

            traj_batch = None

            for traj_inst in traj_list:

                _temp = self.autoencoder.decoder.generate(
                    seq_to_device(traj_inst, sampled_seq.tokens.device)
                )
                if traj_batch is None:
                    traj_batch = _temp
                else:
                    traj_batch.append(_temp)

            return traj_batch

        sampled_batch = self.autoencoder.decoder.generate(sampled_seq)

        if with_path:
            sampled_batch.append(_process_traj_list(traj["path"]))
        if with_x0_pred:
            sampled_batch.append(_process_traj_list(traj["x0_pred"]))

        if with_hist:
            hist.append(sampled_batch)
            return hist
        return sampled_batch


class LatentForeseeGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        # initializing autoencoder
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(ckpt["model"], strict=False)
        else:
            raise Exception(
                f"A checkpoint of pretrained autoencoder should be provided!"
            )

        self.autoencoder = freeze_module(self.autoencoder)
        if not model_config.autoencoder.frozen:
            logger.warning(
                f"The autoencoder is frozen, although the `frozen` flag is not set to True"
            )

        logger.info(f"Tabsyn latent dimension is {self.autoencoder.encoder.output_dim}")

        self.generation_len = data_conf.generation_len

    def forward(self, x: GenBatch) -> torch.Tensor:
        """
        Forward pass of Latent diffusion model with inherent loss compute
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        raise "No need to train a latent foresee generator!"

    @torch.no_grad()
    def generate(
        self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        """
        Diffusion generation

        Args:
            hist (GenBatch): history batch

        """
        assert hist.target_time.shape[0] == gen_len

        gen_batch = deepcopy(hist)
        target_batch = gen_batch.get_target_batch()
        latent_seq = self.autoencoder.encoder(target_batch)
        vae_target_batch = self.autoencoder.decoder.generate(latent_seq)

        gen_batch.append(vae_target_batch)

        gen_batch.target_time = None
        gen_batch.target_num_features = None
        gen_batch.target_cat_features = None

        if with_hist:
            return gen_batch  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return gen_batch.tail(gen_len)
