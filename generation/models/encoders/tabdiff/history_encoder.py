import torch
import torch.nn as nn
import math

class HistoryEncoder(nn.Module):
    def __init__(self,
                 token_fn,
                 d_token = 4,
                 transformer_dim=20,
                 transformer_heads=2,
                 dim_feedforward=64,
                 dropout=0.1,
                 num_encoder_layers=3,
                 num_features = 4
                 ): 
        
        super(HistoryEncoder, self).__init__()

        self.tokenizer = token_fn
        self.total_d_model = d_token * num_features
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.total_d_model, nhead=transformer_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.total_d_model)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_encoder_layers,
                                             norm=encoder_norm)
        
        self.output_layer = nn.Linear(self.total_d_model,transformer_dim)


    def forward(self, hist):

        
        device = "cuda:0"
        ## 1. tokenizer for hist_x,hist_e
        bs,ls = hist["num"].size(0),hist["num"].size(1)
        x_num, x_cat = hist["num"].reshape(-1,hist["num"].shape[2]),hist["cat"].reshape(-1,hist["cat"].shape[2])
        hist_token =  self.tokenizer(x_num,x_cat)[:,1:,:]

        #num_features = 4
        #src = hist_token.reshape(bs,ls*num_features,-1)
        src = hist_token.reshape(bs,ls,-1)
        # Transformer 编码
        #src_mask = self.generate_square_subsequent_mask(4*ls).to(device)
        src_mask = self.generate_square_subsequent_mask(ls).to(device) 
        src_mask = src_mask != 0
        
        
        memory = self.encoder(src, mask=src_mask)
        
        out = self.output_layer(memory)

        return out


    def temporal_enc(self, dt_seq):
        """对时间间隔 dt 进行正弦/余弦编码"""
        if dt_seq.dim() == 1:
            dt_seq = dt_seq.unsqueeze(-1)
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask