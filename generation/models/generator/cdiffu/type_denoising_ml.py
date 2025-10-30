import torch
import torch.nn as nn
from .temporal_pos_enc import DiscreteSinusoidalPosEmb
import math


class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x


class TypeDenoisingModule(nn.Module):
    def __init__(self, transformer_dim=32, num_classes=10, n_steps=100, transformer_heads=2,
                 dim_feedforward=64, dropout=0.1, n_decoder_layers=1, device='cuda', batch_first=True,
                 len_numerical_features=1,feature_dim=8):
        """

        :param transformer_dim:
        :param num_classes:
        :param n_steps:
        :param transformer_heads:
        :param dim_feedforward:
        :param dropout:
        :param n_decoder_layers:
        :param device:
        :param batch_first:
        """
        super(TypeDenoisingModule, self).__init__()


        self.num_classes_dict = num_classes
        self.feature_dim = feature_dim

        # Diffusion time embedding
        self.time_pos_emb = DiscreteSinusoidalPosEmb(int(feature_dim), n_steps)

        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, feature_dim)
        )

        dynamic_feature_dim_sum = 2*feature_dim

        self.cat_emb = nn.ModuleDict() 
        for key,value in self.num_classes_dict.items():

            self.cat_emb[key] = nn.Embedding(value+1, int(feature_dim))
            dynamic_feature_dim_sum += feature_dim

        num_extra_numerical_features = len_numerical_features - 1 

        self.num_proj = None
        if num_extra_numerical_features > 0:
            self.num_proj = nn.Linear(num_extra_numerical_features, feature_dim)
            dynamic_feature_dim_sum += feature_dim 

        
        self.total_d_model = dynamic_feature_dim_sum

        self.nhead = transformer_heads


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        decoder_norm = nn.LayerNorm(transformer_dim)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers,
                                             norm=decoder_norm,)
        self.reduction_dim_layer = nn.Linear(self.total_d_model, transformer_dim)

        self.output_layer_cat = nn.ModuleDict()
        for key,value  in self.num_classes_dict.items():

            self.output_layer_cat[key] = nn.Linear(transformer_dim, value)

        self.device = device
        self.transformer_dim = transformer_dim
        self.rezero = Rezero()



        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) /int(feature_dim)) for i in range((int(feature_dim)))],
            device=torch.device(self.device))

        self.order_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) /int(self.total_d_model)) for i in range((int(self.total_d_model)))],
            device=torch.device(self.device))

    # t, e_t, x_t, hist, non_padding_mask
    def forward(self, t, e, x, hist,cat_order):
        """

        :param x: dt and type seq representation, B x L_tgt
        :param t:
        :param h:
        :param memory_key_padding_mask:
        :return:
        """
        ##
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        order = torch.cat([torch.arange(x.size(1)).unsqueeze(0)]*x.size(0),dim=0)

        order = self.order_enc(order.to(x.device))

        time_embed = t.view(x.size(0), 1, int(self.feature_dim))
        t = torch.cat([time_embed]*x.size(1), dim=1)
        #e = self.event_emb(e) 
        combined_cat = []

        idx = 0 
        #for key,value in self.num_classes_dict.items():
        # breakpoint()
        for cat_name in cat_order:
            e_temp = e[:,:,idx]
            combined_cat.append(self.cat_emb[cat_name](e_temp))
            idx += 1
        
        all_features = [self.temporal_enc(x[:,:,0])]
        if self.num_proj is not None:
            all_features.append(self.num_proj(x[:,:,1:]))

        all_features.append(torch.cat(combined_cat,dim=-1))
        all_features.append(t)

        tgt = torch.cat(all_features, dim=-1) + order


        tgt = self.reduction_dim_layer(tgt)

        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        memory = hist.permute(1, 0, -1)
        
        tgt = tgt.permute(1, 0, -1)
        

        assert memory.size(1) == tgt.size(1), \
            f'Error: history embed batch size (got {memory.size(1)}) should equal to target\'s (got {tgt.size(1)})'
        assert memory.size(2) == tgt.size(2), \
            f'Error: history dim (got {hist.size(2)}) should equal to target\'s (got {tgt.size(2)})'
        
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask,
        )
        
        output = output.permute(1, 0, -1)
        ##TODO: we must use seperate output laye for different category
        #out = self.output_layer(output)
        out_list = []

        for cat_name in cat_order:
            out_list.append(self.output_layer_cat[cat_name](output))
        out = torch.cat(out_list,dim=-1)
        out = out.permute(0, 2, 1)
        out = self.rezero(out)
        return out

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def temporal_enc(self, dt_seq):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def order_enc(self, dt_seq):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = dt_seq.unsqueeze(-1) / self.order_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
