from copy import deepcopy

import torch
from ebes.model import BaseModel, TakeLastHidden

from ...data.data_types import GenBatch, LatentDataConfig
from ...data.data_types import seq_append, get_seq_tail
from ..encoders import ConditionalDiffusionEncoder
from generation.models import autoencoders
from generation.utils import freeze_module
from typing import Any, Dict

from . import BaseGenerator, ModelConfig
import logging
logger = logging.getLogger()

def _set_or_check_match(_dict: Dict, field: str, value: Any):
    if _dict.get(field) is None:
        _dict[field] = value
    assert _dict[field] == value, (
        f"The value {_dict[field]} of field '{field}' in the dictionary "
        f"does not match the provided value {value}"
    )


class LatentDiffusionGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        # initializing autoencoder
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(
                ckpt["model"], strict=False
            )
        
        #TODO: I think it should be frozen by default
        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)
        logger.info(f"Tabsyn latent dimension is {self.autoencoder.encoder.output_dim}")

        # initializing encoder
        encoder_params = model_config.latent_encoder.params or {}
        _set_or_check_match(encoder_params, "input_size", self.autoencoder.encoder.output_dim)

        self.encoder = ConditionalDiffusionEncoder(
            model_config.latent_encoder.name, encoder_params)
        
        self.generation_len = encoder_params['generation_len']
        self.history_len = encoder_params['history_len']

        # initializing history encoder
        history_encoder_data = model_config.params.get('history_encoder')
        self.history_encoder = None # default history encoder
        if history_encoder_data is not None:

            history_encoder_params = history_encoder_data['params'] or {}
            _set_or_check_match(history_encoder_params, "input_size", self.autoencoder.encoder.output_dim)

            self.history_encoder = BaseModel.get_model(
                history_encoder_data['name'], 
                **history_encoder_data['params']
            )

            if history_encoder_data['checkpoint']:
                ckpt = torch.load(history_encoder_data['checkpoint'], map_location="cpu")
                msg = self.autoencoder.load_state_dict(
                    ckpt["state_dict"], strict=False #TODO: I do not know how history_encoder will be stored
                )
            
            if history_encoder_data['frozen']:
                self.history_encoder = freeze_module(self.history_encoder)
            
            self.history_pooler = TakeLastHidden()

        # self.poller = (
        #     TakeLastHidden() if model_config.pooler == "last" else ValidHiddenMean()
        # )
        # self.reshaper = Reshaper(data_conf.generation_len)

        # self.projector = Projection(
        #     self.encoder.output_dim // data_conf.generation_len,
        #     self.encoder.output_dim,
        # )

    def forward(self, x: GenBatch) -> torch.Tensor:
        """
        Forward pass of Latent diffusion model with inherent loss compute
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        target_seq = self.autoencoder.encoder(x.get_target_batch())

        history_embedding = None
        if self.history_encoder:
            full_history_seq = self.autoencoder.encoder(x)
            history_embedding = self.history_pooler(
                self.history_encoder(
                    full_history_seq
                )
            )
        
        history_seq = None
        if self.history_len > 0:
            history_seq = self.autoencoder.encoder(x.tail(self.history_len)) #TODO: use Seq.tail directly

        #TODO: static condition

        loss = self.encoder(target_seq, None, history_embedding, history_seq)
        # x = self.poller(x)  # [B, D] # не нужен
        # x = self.reshaper(x)  # [gen_len, B, D // gen_len] # не нужен
        # x = self.projector(x) # не нужен
        # x = self.autoencoder.decoder(x)
        return loss
    
    @torch.no_grad()
    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs) -> GenBatch:
        """
        Diffusion generation

        Args:
            hist (GenBatch): history batch

        """
        breakpoint()
        n_seqs = len(hist)
        hist = deepcopy(hist)
        full_history_seq = self.autoencoder.encoder(hist)

        history_embedding = None
        history_seq = None
        pred_batch = None

        for _ in range(0, gen_len, self.generation_len):

            if self.history_encoder:
                history_embedding = self.history_pooler(
                    self.history_encoder(
                        full_history_seq
                    )
                )
            
            if self.history_len > 0:
                history_seq = get_seq_tail(full_history_seq, self.history_len)

            #TODO: static condition

            sampled_seq = self.encoder.generate(n_seqs, None, history_embedding, history_seq) # Seq [L, B, D]
            sampled_batch = self.autoencoder.decoder(sampled_seq).to_batch() #TODO: check it is correct

            if pred_batch is None:
                pred_batch = sampled_batch
            else:
                pred_batch.append(sampled_batch)

            full_history_seq = seq_append(full_history_seq, sampled_seq)
        
        if pred_batch.shape[0] > gen_len:
            pred_batch = pred_batch.head(gen_len)
        
        if with_hist:
            hist.append(pred_batch)
            return hist
        return pred_batch
