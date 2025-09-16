
import torch
from .async_lib import obtain_noise_schedule

from ...data.data_types import GenBatch, LatentDataConfig,PredBatch

from ..encoders import ADiffEncoder
from generation.models import autoencoders
from generation.utils import freeze_module

from . import BaseGenerator, ModelConfig
import logging
logger = logging.getLogger()


class ADiffGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

      
        # initilized autoencoder
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        #breakpoint()
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(
                ckpt, strict=False
            )
        
        if model_config.autoencoder.frozen:
            self.autoencoder = freeze_module(self.autoencoder)
        #logger.info(f"Adiff4tpp latent dimension is {self.autoencoder.encoder.output_dim}")


        # initilization of diffusion encoder
        self.encoder_params =  model_config.latent_encoder.params   
        self.history_len = self.encoder_params['history_len']
        self.generation_len = self.encoder_params ['generation_len']  
        self.Aschedule = self.encoder_params['schedule']   
        self.encoder = ADiffEncoder(model_config.latent_encoder.name, self.encoder_params)


    def forward(self, x: GenBatch) -> torch.Tensor:

        target_batch,history_batch = x.get_target_batch(),x.tail(self.history_len)
        history_batch.append(target_batch) ## xx_features: T*B*num_features

        batch_cat,batch_num = history_batch.cat_features,history_batch.num_features
        
        z_token = self.autoencoder.VAE.Tokenizer(batch_num.view(-1,1),batch_cat.view(-1,1))
        z = self.autoencoder.VAE.get_embedding(z_token).view(batch_num.shape[1],batch_num.shape[0],3*32)

        batch_len = history_batch.lengths ## valid seq lengths in each batch
        max_len = self.history_len+ self.generation_len ## num of row for each input
        A = obtain_noise_schedule(self.Aschedule)(batch_len,max_len)
        loss = self.encoder(z,A)
        
        return loss

    @torch.no_grad()
    def generate(self, hist: GenBatch, gen_len: int, with_hist=False, **kwargs) -> GenBatch:

        #breakpoint()
        target_batch,history_batch = hist.get_target_batch(),hist.tail(self.history_len)
        history_batch.append(target_batch) ## xx_features: T*B*num_features
        batch_cat,batch_num = history_batch.cat_features,history_batch.num_features
        z_token = self.autoencoder.VAE.Tokenizer(batch_num.view(-1,1),batch_cat.view(-1,1))
        z = self.autoencoder.VAE.get_embedding(z_token).view(batch_num.shape[1],batch_num.shape[0],3*32)


        batch_len = history_batch.lengths

        max_len = self.history_len+ self.generation_len
        A = obtain_noise_schedule(self.Aschedule)(torch.ones_like(batch_len)*max_len,max_len).to("cuda")
        #breakpoint()
        pred = self.encoder.generate(z,gen_len,A,batch_len) ## tokens.shape: T*B*(32*features)
        # Decode latent event
        ## pred_num: (B*T)*1
        pred_num, pred_cat = self.autoencoder.get_decoding(pred.tokens.reshape(-1,3,pred.tokens.shape[-1] // 3))
        pred_num = pred_num.view(len(batch_len),self.generation_len,-1)
        pred_cat = pred_cat[0].view(len(batch_len),self.generation_len,-1)
        pred_cat = pred_cat.argmax(dim = -1).unsqueeze(-1)
        #breakpoint()
        pred_batch = PredBatch(lengths=torch.ones(len(batch_len))*self.generation_len,
                               cat_features=pred_cat.permute(1, 0, 2).contiguous(),
                               num_features=pred_num.permute(1, 0, 2).contiguous(),
                               time=hist.target_time) 
        
        if with_hist:
            hist.append(pred_batch)
            return hist
        return pred_batch



