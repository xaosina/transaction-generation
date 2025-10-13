import torch
import torch.nn as nn
import math
from .temporal_pos_enc import ContinuousSinusoidalPosEmb


class TimeDenoisingModule(nn.Module):
    def __init__(self, transformer_dim=32, num_classes=10, n_steps=100, transformer_heads=2,
                 dim_feedforward=64, dropout=0.1, n_decoder_layers=1, device='cuda', batch_first=True,
                 x_dim=2, e_dim=2):  # 🚀 新增 x_dim 和 e_dim 参数
        """
        :param transformer_dim: 
        :param num_classes:
        :param n_steps:
        # ... (其他参数保持不变)
        :param x_dim: x 的特征维度 (如果 x 是原始时间间隔，x_dim=1)
        :param e_dim: e 的特征维度 (如果 e 是事件类型 ID，e_dim=1)
        """
        super(TimeDenoisingModule, self).__init__()

        self.device = device
        self.transformer_dim = transformer_dim
        self.x_dim = x_dim
        self.e_dim = e_dim

        self.num_classes_dict = num_classes
        self.cat_emb = nn.ModuleDict() 
        for key,value in self.num_classes_dict.items():
            self.cat_emb[key] = nn.Embedding(value+1, int(transformer_dim/8))

        # 🚀 移除原始的 event_emb 和 temporal_enc 相关的逻辑，因为 x 和 e 已是多特征输入
        # 除非你的 e 仍然是 event ID，那么 event_emb 需要修改输出维度
        
        # 如果 e 仍是事件 ID，则使用 Embedding Layer
        # if e_dim == 1:
        #     self.event_input_layer = nn.Embedding(num_classes + 1, int(transformer_dim/4), padding_idx=num_classes)
        # else:
        #     # 🚀 如果 e 是多特征序列，使用线性层映射到统一维度
        #     self.event_input_layer = nn.Linear(e_dim, int(transformer_dim/4))
            
        # 🚀 线性层：将 x 的多特征维度映射到统一维度
        self.x_input_layer = nn.Linear(x_dim, int(transformer_dim/4))

        # Diffusion time embedding: 保持不变
        self.time_pos_emb = ContinuousSinusoidalPosEmb(int(transformer_dim/4), n_steps)
        self.mlp = nn.Sequential(
            nn.Linear(int(transformer_dim/4), transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, int(transformer_dim/4))
        )

        self.nhead = transformer_heads

        # Decoder: 保持不变
        decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_dim, nhead=transformer_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)

        decoder_norm = nn.LayerNorm(transformer_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers,
                                             norm=decoder_norm)
        
        # 🚀 统一特征维度层：将所有编码后的特征 (x_emb, e_emb, t_emb, order_emb) 拼接后的维度映射到 transformer_dim
        # 这里的输入维度是 4 * (transformer_dim/4) = transformer_dim
        # 我们使用一个线性层来处理拼接后的特征。
        self.unified_feature_layer = nn.Linear(4 * int(transformer_dim/4), transformer_dim)

        self.output_layer = nn.Linear(transformer_dim, x_dim)

        self.num_classes = num_classes
        
        # Time encoding for inter-arrival time seq
        # 🚀 移除原始的 position_vec 和 order_vec 的定义，因为我们用更标准的 Order/Position Encoding

    def forward(self, x, e, t, hist,cat_list):
        """
        :param x: B x Seq_Len x X_Dim (如果 X_Dim > 1)
        :param e: B x Seq_Len x E_Dim (如果 E_Dim > 1)
        :param t: B
        :param hist: history representation, B x L_context x Dim
        """
        # 1. 时间步 t 编码 (保持不变，输出 B x d_model/4)
        t_emb = self.time_pos_emb(t)
        t_emb = self.mlp(t_emb) # B x d_model/4

        # 2. 顺序编码 order (输出 B x Seq_Len x d_model/4)
        order = torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        order_emb = self.order_enc(order.float())

        # 3. 将 t 扩展到序列长度 (输出 B x Seq_Len x d_model/4)
        t_emb_seq = t_emb.view(x.size(0), 1, -1).repeat(1, x.size(1), 1)

        # 4. x 特征编码 (输出 B x Seq_Len x d_model/4)
        x_emb = self.x_input_layer(x.float()) # 🚀 使用线性层映射

        # 5. e 特征编码 (输出 B x Seq_Len x d_model/4)
        # if self.e_dim == 1:
        #     e_emb = self.event_input_layer(e.long()) # 🚀 原始嵌入层 (假设 e 是 ID)
        # else:
        #     e_emb = self.event_input_layer(e.float()) # 🚀 线性层 (假设 e 是多特征)
    
        e_emb = []

        idx = 0 
        for key,value in self.num_classes_dict.items():
            e_temp = e[:,:,idx]
            e_emb.append(self.cat_emb[key](e_temp))
            idx += 1
        
        e_emb = torch.cat(e_emb,dim=-1)

        # 6. 拼接所有特征 (B x Seq_Len x (4 * d_model/4))
        combined_features = torch.cat([x_emb, e_emb, t_emb_seq, order_emb], dim=-1)
        
        # 7. 统一维度到 transformer_dim (B x Seq_Len x transformer_dim)
        tgt = self.unified_feature_layer(combined_features) # 🚀 新增统一层

        # Transformer Decoder 部分 (保持不变，除了 memory 维度检查)
        
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        memory = hist # 假设 hist 已经是 B x L_context x transformer_dim

        # memory 的维度检查需要修改，因为它现在可能不是 transformer_dim
        # 假设 hist 已经是 B x L_context x transformer_dim
        assert memory.size(2) == self.transformer_dim, \
            f'Error: history dim (got {memory.size(2)}) should equal to transformer_dim (got {self.transformer_dim})'

        # 维度转换：B x S x D -> S x B x D
        memory = hist.permute(1, 0, -1)
        tgt = tgt.permute(1, 0, -1)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        # 维度转换：S x B x D -> B x S x D
        output = output.permute(1, 0, -1)

        out = self.output_layer(output)

        return out

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def diff_step_enc(self, time, shape):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) * torch.ones(shape).to(self.device)
        result = result[:, 1, :].to(self.device)
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result.unsqueeze(1)

    def temporal_enc(self, dt_seq):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
    

    
    def order_enc(self, order_seq):
        """
        🚀 保持原始的 Order Encoding 逻辑，但使用 order_vec 保持不变，它是一个 d_model 维度的编码
        """
        d_model = self.transformer_dim
        # 使用 transformer_dim 作为 d_model
        position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=order_seq.device) # 确保在正确设备上
        
        result = order_seq.unsqueeze(-1) / position_vec[:int(d_model/4)] # 只取 d_model/4 维
        # 原始代码的 order_vec 维度是 transformer_dim，但用于 d_model/4 编码，这里保持 d_model/4
        
        # 我们为 order_enc 保持 d_model/4 的输出维度，然后和其他特征拼接。
        
        result = order_seq.unsqueeze(-1) / position_vec[:int(d_model/4)] 
        
        order_emb = torch.zeros(order_seq.size(0), order_seq.size(1), int(d_model/4), device=order_seq.device)
        order_emb[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        order_emb[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return order_emb
