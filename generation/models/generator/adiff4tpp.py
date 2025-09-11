
import torch
from .async_lib import obtain_noise_schedule

from ...data.data_types import GenBatch, LatentDataConfig
from ..encoders import AsynDiffEncoder
from generation.models import autoencoders
from generation.utils import freeze_module

from . import BaseGenerator, ModelConfig
import logging
logger = logging.getLogger()


class AsynDiffGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

      
        # initilized autoencoder
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )

        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(
                ckpt["model"], strict=False
            )
        
        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)
        logger.info(f"Tabsyn latent dimension is {self.autoencoder.encoder.output_dim}")


        # initilization of diffusion encoder
        self.encoder_params =  model_config.latent_encoder.params   
        self.history_len = self.encoder_params['history_len']
        self.generation_len = self.encoder_params ['generation_len']  
        self.Aschedule = self.encoder_params['schedule']   
        self.encoder = AsynDiffEncoder(model_config.latent_encoder.name, self.encoder_params)


    def forward(self, x: GenBatch) -> torch.Tensor:
        
        # x.append(x.get_target_batch())

        # batch_len = x.lengths
        # max_len = x.time.shape[0]
        # A = obtain_noise_schedule(self.Aschedule)(batch_len,max_len)
        
        # target_seq = self.autoencoder.encoder(x)
        
        # loss = self.encoder(target_seq,A)
        target_batch,history_batch = x.get_target_batch(),x.tail(self.history_len)
        history_batch.append(target_batch)
        input_seq =  self.autoencoder.encoder(history_batch)

        batch_len = history_batch.lengths
        max_len = self.history_len+ self.generation_len
        A = obtain_noise_schedule(self.Aschedule)(batch_len,max_len)
        loss = self.encoder(input_seq,A)
        
        return loss

    @torch.no_grad()
    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs) -> GenBatch:

        # hist.append(hist.get_target_batch())

        # batch_len= hist.lengths
        # max_len = batch_len.max()
        # A = obtain_noise_schedule(self.Aschedule)(torch.ones_like(batch_len)*max_len,max_len).to("cuda")

        # z = self.autoencoder.encoder(hist)
        # pred = self.encoder.generate(z,gen_len,A,batch_len)
        # # Decode latent event
        # pred_batch = self.autoencoder.decoder(pred).to_batch()

        # if with_hist:
        #     hist.append(pred_batch)
        #     return hist
        # return pred_batch
        target_batch,history_batch = hist.get_target_batch(),hist.tail(self.history_len)
        history_batch.append(target_batch)
        batch_len = history_batch.lengths
        max_len = self.history_len+ self.generation_len
        A = obtain_noise_schedule(self.Aschedule)(torch.ones_like(batch_len)*max_len,max_len).to("cuda")

        z = self.autoencoder.encoder(history_batch)
        pred = self.encoder.generate(z,gen_len,A,batch_len)
        # Decode latent event
        pred_batch = self.autoencoder.decoder(pred).to_batch()

        if with_hist:
            hist.append(pred_batch)
            return hist
        return pred_batch



