import torch
from . import BaseGenerator,ModelConfig
from ...data.data_types import GenBatch, LatentDataConfig
#from generation.models import encoders
from generation.models.encoders.tabdiff.tabdiff_encoder import ConditionalContinuousDiscreteEncoder


class ContinuousDiscreteDiffusionGenerator1(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.history_len =  model_config.params['history_len']
        self.generation_len = model_config.params['generation_len']

        ## this should init a conditional generator for tgt
        self.cond_generator = None

        self.data_conf = data_conf
        self.model_config = model_config
        self.cond_col = model_config.params["cond_col"]

        if data_conf.time_name:
            num_names_list = [data_conf.time_name] + data_conf.num_names 
        else:
            num_names_list = data_conf.num_names

        self.features_name_dict = {"cat":list(data_conf.cat_cardinalities.keys()),
                                   "num": num_names_list,
                                   }

        self.cond_corrupt_idx = self.split_cond_corrupt_idx()

        self.encoder = ConditionalContinuousDiscreteEncoder(data_conf,model_config,self.features_name_dict)

        

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


        tgt_data = {"cat":tgt_cat,"num":tgt_num}
        hist_data = {"cat":hist_cat,"num":hist_num}

        cond_dict,corrupt_dict = self.cond_corrupt_idx
        ## later we will need the output from DeTPP as the tgt_data in cond_data,
        ## because we need it as a condition, and in corrupt_data, we still use the data from gt.
        cond_data = self.get_data(cond_dict,tgt_data)
        corrupt_data = self.get_data(corrupt_dict,tgt_data)

        h_cond_data = self.get_data(cond_dict,hist_data)
        h_corrupt_data = self.get_data(corrupt_dict,hist_data)

        loss = self.encoder(h_cond_data,h_corrupt_data,cond_data,corrupt_data)

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

        
        cond_data = None
        pred_num,pred_cat = self.encoder.sample(hist_data,gen_len,cond_data)
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
    
    def split_cond_corrupt_idx(self):
        cond_col_idx = {"cat":[],"num":[]}
        corrupt_col_idx = {"cat":[],"num":[]}

        for col_type in ["cat","num"]:
            for idx,col in enumerate(self.features_name_dict[col_type]):
                if col in self.cond_col[col_type]:
                    cond_col_idx[col_type].append(idx)
                else:
                    corrupt_col_idx[col_type].append(idx)

        return cond_col_idx, corrupt_col_idx
            

    def get_data(self,idx_dict,seq):
        
        tensor_dict = {}
        for col_type,idx_list in idx_dict.items():

            tensor_dict[col_type] = seq[col_type][:,:,idx_list]

        return tensor_dict



