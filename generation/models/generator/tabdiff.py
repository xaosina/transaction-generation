import torch
from . import BaseGenerator,ModelConfig
from ...data.data_types import GenBatch, LatentDataConfig
#from generation.models import encoders
from generation.models.encoders.tabdiff.tabdiff_encoder import ConditionalContinuousDiscreteEncoder



class ContinuousDiscreteDiffusionGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        ## UnifiedCtimeDiffusion is encoder
        encoder_params = {}
        # self.encoder = getattr(encoders, model_config.encoder.name)(
        #      model_config.encoder.name, encoder_params
        # )

        self.history_len =  model_config.params['history_len']
        self.generation_len = model_config.params['generation_len']
        self.encoder = ConditionalContinuousDiscreteEncoder(data_conf,model_config)


    def forward(self, x: GenBatch) -> torch.Tensor:
        
        history_batch = x.tail(self.history_len)
        target_batch = x.get_target_batch()

        hist_cat,hist_num = history_batch.cat_features,history_batch.num_features
        tgt_cat,tgt_num = target_batch.cat_features,target_batch.num_features

        tgt_data = {"cat":tgt_cat,"num":tgt_num}
        hist_data = {"cat":hist_cat,"num":hist_num}

        loss = self.encoder(hist_data,tgt_data)

        return loss


    def generate(self, hist: GenBatch, gen_len: int, with_hist=False,**kwargs) -> GenBatch:
        
        pass