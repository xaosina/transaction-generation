import torch
from torch import nn
from . import BaseGenerator,ModelConfig
from ...data.data_types import GenBatch, LatentDataConfig
from ebes.model import BaseModel,TakeLastHidden
from generation.utils import freeze_module
#from generation.models import encoders
from generation.models.encoders.tabdiff.tabdiff_encoder1 import ConditionalContinuousDiscreteEncoder1

from ...models import generator as gen_models
from dacite import Config, from_dict

from generation.models.autoencoders.base import AEConfig
from ..encoders import LatentEncConfig
from collections import defaultdict


class ContinuousDiscreteDiffusionGenerator1(BaseGenerator):

    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.history_len =  model_config.params['history_len']
        self.generation_len = model_config.params['generation_len']

        self.data_conf = data_conf
        self.model_config = model_config
        self.cond_col = model_config.params["cond_col"]

        if data_conf.time_name:
            num_names_list = [data_conf.time_name] + data_conf.num_names 
        else:
            num_names_list = data_conf.num_names

        self.features_name_dict = {"cat":list(data_conf.cat_cardinalities.keys()),
                                   "num": num_names_list}

        #self.cond_corrupt_idx = self.split_cond_corrupt_idx()
        cond_idx,cond_name,corrupt_idx,corrupt_name = self.split_cond_corrupt_idx()

        col_name_dict = (cond_name,corrupt_name)
        self.cond_corrupt_idx = (cond_idx,corrupt_idx)
        self.encoder = ConditionalContinuousDiscreteEncoder1(data_conf,model_config,col_name_dict)
        if "target_condition_encoder" in model_config.params:
            self.target_condition_encoder, _ = self._init_history_encoder()


    def _init_history_encoder(self):
        # initializing history encoder
        params = self.model_config.params.get("target_condition_encoder")
        history_encoder = None  # default history encoder
        if params is None:
            print("no history encoder!")
            return

        checkpoint = params.pop("checkpoint", None)
        # Support old encoders
        if params["name"] == "GRU":
            params = {
                "input_size": self.autoencoder.encoder.output_dim,
                "num_layers": 1,
                "hidden_size": 256,
            }
            hist_enc_dim = 256
            history_encoder = nn.Sequential(
                BaseModel.get_model("GRU", **params), TakeLastHidden()
            )
        else:
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            history_encoder = BaseGenerator.get_model(
                params["name"], self.data_conf, cfg
            )
            hist_enc_dim = history_encoder.encoder.output_dim
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = history_encoder.load_state_dict(ckpt["model"], strict=True)
            history_encoder = freeze_module(history_encoder)
        return history_encoder, hist_enc_dim

    def forward(self, x: GenBatch) -> torch.Tensor:
        gen_len = 32
        history_batch = x.tail(self.history_len)
        target_batch = x.get_target_batch()
        

        
        
        hist_data = self.Seq2DictTensor(history_batch)
        tgt_data = self.Seq2DictTensor(target_batch)
        # tgt_data = {"cat":tgt_cat,"num":tgt_num}
        # hist_data = {"cat":hist_cat,"num":hist_num}

        cond_dict,corrupt_dict = self.cond_corrupt_idx

        if "target_condition_encoder" in self.model_config.params:
            out_precondition = self.target_condition_encoder.generate(x,gen_len)
            pred_cond_data = self.Seq2DictTensor(out_precondition)
            cond_data = self.get_data(cond_dict,pred_cond_data)
        else:
            cond_data = self.get_data(cond_dict,tgt_data)
        ## later we will need the output from DeTPP as the tgt_data in cond_data,
        ## because we need it as a condition, and in corrupt_data, we still use the data from gt.
        #cond_data = self.get_data(cond_dict,tgt_data)
        
        corrupt_data = self.get_data(corrupt_dict,tgt_data)

        h_cond_data = self.get_data(cond_dict,hist_data)
        h_corrupt_data = self.get_data(corrupt_dict,hist_data)

        loss = self.encoder(h_cond_data,h_corrupt_data,cond_data,corrupt_data)

        return loss


    def generate(self, hist: GenBatch, gen_len: int, with_hist=False,**kwargs) -> GenBatch:
        
        history_batch = hist.tail(self.history_len)
        target_batch = hist.get_target_batch()
        bs = len(history_batch.lengths)

        hist_data = self.Seq2DictTensor(history_batch)
        tgt_data = self.Seq2DictTensor(target_batch)
        cond_dict,corrupt_dict = self.cond_corrupt_idx

        h_cond_data = self.get_data(cond_dict,hist_data)
        h_corrupt_data = self.get_data(corrupt_dict,hist_data)

        cond_data = self.get_data(cond_dict,tgt_data)
        pred_num,pred_cat = self.encoder.sample(h_cond_data,h_corrupt_data,gen_len,cond_data)
        #print(pred_num[:10],target_batch.time.permute(1,0)[:10])
        pred_cat = pred_cat.reshape(bs,gen_len,pred_cat.shape[1])
        pred_num = pred_num.reshape(bs,gen_len,pred_num.shape[1])

        features_name = [history_batch.cat_features_names,history_batch.num_features_names]
        
        aggregate_batch = self.replace_data(corrupt_dict,tgt_data,pred_num,pred_cat)
        sampled_batch = self.wrap_to_PredBatch(aggregate_batch['cat'],aggregate_batch['num'],features_name)
        return sampled_batch


    def replace_data(self,corrupt_idx_dict,tgt_data,pred_num,pred_cat):
        pred_num = pred_num.permute(1,0,2)
        pred_cat = pred_cat.permute(1,0,2)
        for keys,values in corrupt_idx_dict.items():
            if keys == "cat":
                for t_idx,idx in enumerate(values):
                    tgt_data["cat"][:,:,idx] = pred_cat[:,:,t_idx]
            elif keys == "num":
                for t_idx,idx in enumerate(values):
                    tgt_data["num"][:,:,idx] = pred_num[:,:,t_idx]
        return tgt_data

    def wrap_to_PredBatch(self,pred_cat,pred_num,features_name,topk=1, temperature=1.0):

        s_length = torch.ones(pred_cat.size(1))*pred_cat.size(0)

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

        cond_cols = self.cond_col if self.cond_col is not None else {}
        
        # Initialize standard dictionaries
        cond_col_idx = {"cat": [], "num": []}
        corrupt_col_idx = {"cat": [], "num": []}
        cond_col_name = {"cat": [], "num": []}
        corrupt_col_name = {"cat": [], "num": []}

        for col_type in ["cat", "num"]:
            # Ensure the column type exists in the features_name_dict before iterating
            if col_type in self.features_name_dict:
                
                # Get the list of conditional columns for this type, defaulting to an empty list
                # if the column type isn't defined in cond_cols (e.g., only 'cat' is conditional)
                cond_names_for_type = cond_cols.get(col_type, [])
                
                for idx, col in enumerate(self.features_name_dict[col_type]):
                    
                    # Check if the column name is in the list of conditional names
                    if col in cond_names_for_type:
                        cond_col_idx[col_type].append(idx)
                        cond_col_name[col_type].append(col)
                    else:
                        corrupt_col_idx[col_type].append(idx)
                        corrupt_col_name[col_type].append(col)

        return cond_col_idx,cond_col_name, corrupt_col_idx,corrupt_col_name
            

    def get_data(self,idx_dict,seq):
        
        tensor_dict = defaultdict(lambda: None)
        for col_type,idx_list in idx_dict.items():

            tensor_dict[col_type] = seq[col_type][:,:,idx_list]

        return tensor_dict

    def Seq2DictTensor(self,seq):

        if seq.num_features_names:
            seq_num = torch.cat([seq.time.unsqueeze(-1),seq.num_features],dim = -1)
        else:
            seq_num = seq.time.unsqueeze(-1)

        seq_cat = seq.cat_features.detach().clone()

        seq_dict = {"cat":seq_cat,"num":seq_num}

        return seq_dict