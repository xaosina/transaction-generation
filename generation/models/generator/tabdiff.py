import torch
from . import BaseGenerator,ModelConfig
from ...data.data_types import GenBatch, LatentDataConfig
#from generation.models import encoders
from generation.models.encoders.tabdiff.tabdiff_encoder import ConditionalContinuousDiscreteEncoder


class ContinuousDiscreteDiffusionGenerator(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.history_len =  model_config.params['history_len']
        self.generation_len = model_config.params['generation_len']
        self.encoder = ConditionalContinuousDiscreteEncoder(data_conf,model_config)

        self.data_conf = data_conf
        self.model_config = model_config

    def forward(self, x: GenBatch) -> torch.Tensor:
        
        
        history_batch = x.tail(self.history_len)
        target_batch = x.get_target_batch()

        if history_batch.num_features_names:
            tgt_num = torch.cat([target_batch.time.unsqueeze(-1),target_batch.num_features],dim = -1)
            hist_num = torch.cat([history_batch.time.unsqueeze(-1),history_batch.num_features],dim = -1)
        else:
            hist_num = history_batch.time.unsqueeze(-1)
            tgt_num = target_batch.time.unsqueeze(-1)

        # hist_cat,hist_num = history_batch.cat_features,history_batch.num_features
        # tgt_cat,tgt_num = target_batch.cat_features,target_batch.num_features
        ### all input have shape of (L,B,C_i)
        hist_cat = history_batch.cat_features
        tgt_cat = target_batch.cat_features
        #breakpoint()
        tgt_data = {"cat":tgt_cat,"num":tgt_num}
        hist_data = {"cat":hist_cat,"num":hist_num}

        loss = self.encoder(hist_data,tgt_data)

        return loss


    def generate(self, hist: GenBatch, gen_len: int, with_hist=False,**kwargs) -> GenBatch:
        history_batch = hist.tail(self.history_len)
        tgt = hist.get_target_batch()
        if history_batch.num_features_names:
            hist_num = torch.cat([history_batch.time.unsqueeze(-1),history_batch.num_features],dim = -1)
        else:
            hist_num = history_batch.time.unsqueeze(-1)

        # hist_cat,hist_num = history_batch.cat_features,history_batch.num_features
        # tgt_cat,tgt_num = target_batch.cat_features,target_batch.num_features
        ### all input have shape of (L,B,C_i)
        hist_cat = history_batch.cat_features

        #hist_cat,hist_num = history_batch.cat_features,history_batch.num_features
        hist_data = {"cat":hist_cat,"num":hist_num}

        

        pred_num,pred_cat = self.encoder.sample(hist_data,gen_len)
        print(pred_num[:10],tgt.time.permute(1,0)[:10])
        pred_cat = pred_cat.view(-1,gen_len,pred_cat.shape[1])
        pred_num = pred_num.view(-1,gen_len,pred_num.shape[1])
        features_name = [history_batch.cat_features_names,history_batch.num_features_names]
        sampled_batch = self.wrap_to_PredBatch(pred_cat,pred_num,features_name)

        return sampled_batch

    def wrap_to_PredBatch(self,pred_cat,pred_num,features_name,topk=1, temperature=1.0):

        s_length = torch.ones(pred_cat.size(0))*pred_cat.size(1)
        pred_cat = pred_cat.permute(1,0,2)
        pred_num = pred_num.permute(1,0,2)
        return GenBatch(
            lengths=s_length,
            time=pred_num[:,:,0],
            index=None,
            num_features=pred_num[:,:,1:],
            cat_features=pred_cat,
            cat_features_names=features_name[0],
            num_features_names=features_name[1],
        )
