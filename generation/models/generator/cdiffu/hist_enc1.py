import torch
import torch.nn as nn
import math

class HistoryEncoder(nn.Module):
    def __init__(self,
                 transformer_dim=128,
                 transformer_heads=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_encoder_layers=3,
                 num_classes=10, 
                 device='cuda',
                 len_numerical_features=1,
                 feature_dim = 8): 
        
        super(HistoryEncoder, self).__init__()


        #self.num_classes = num_classes
        self.device = device
        self.transformer_dim = transformer_dim
        self.feature_dim = feature_dim
        self.num_classes_dict = num_classes
        all_dynamic_vocab_sizes = self.num_classes_dict.values()
        num_all_numerical_features = len_numerical_features

        # --- 1. 动态类别特征处理（F 个特征，统一投影） ---
        
        if all_dynamic_vocab_sizes is None or not all_dynamic_vocab_sizes:
            raise ValueError("all_dynamic_vocab_sizes must be provided for all categorical features.")
        
        self.num_all_cat_features = len(all_dynamic_vocab_sizes)
        # N_cat = F

        # 为所有 F 个类别特征定义独立的嵌入层
        dynamic_feature_dim_sum = 0
        self.cat_emb = nn.ModuleDict() 
        for key,value in self.num_classes_dict.items():

            self.cat_emb[key] = nn.Embedding(value+1, int(feature_dim))
            dynamic_feature_dim_sum += feature_dim

        # --- 2. 动态数值特征处理（时间间隔 dt + M 个额外特征） ---
        
        self.num_all_numerical_features = num_all_numerical_features
        num_extra_numerical_features = num_all_numerical_features - 1 # 额外的 M 个数值特征

        # A. 时间间隔 dt 的编码位置向量 (用于时间编码)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / feature_dim) for i in range(feature_dim)],
            device=torch.device(self.device))
        
        dynamic_feature_dim_sum += feature_dim # 时间编码贡献 D 维度

        # B. 额外的 M 个数值特征的投影层
        if num_extra_numerical_features > 0:
            # 投影层：将 M 维投影到 D 维
            self.dynamic_num_proj = nn.Linear(num_extra_numerical_features, feature_dim)
            dynamic_feature_dim_sum += feature_dim # 额外数值特征贡献 D 维度

        # --- 3. 确定最终的 d_model ---
        
        # d_model = D (类别) + D (时间) + [D (额外数值)]
        self.total_d_model = dynamic_feature_dim_sum

        # --- 4. 编码器和输出层 ---

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.total_d_model, nhead=transformer_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,)
        encoder_norm = nn.LayerNorm(self.total_d_model)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_encoder_layers,
                                             norm=encoder_norm)

        self.output_layer = nn.Linear(self.total_d_model, transformer_dim)

    def forward(self, hist_x, hist_e):
        """
        :param hist_x: 所有数值型特征张量 [B, L', M+1]，第 0 列是 dt
        :param hist_e: 统一的类别型特征张量 [B, L', F]
        :return: memory: history_representation, B x L' x dim
        """
        # breakpoint()
        dynamic_features_list = []
        
        # --- 1. 统一动态类别特征处理 ---
        
        # A. 解包所有类别特征: [B, L', F] -> F 个 [B, L'] 的张量
        if hist_e.dim() == 3 and hist_e.size(-1) == self.num_all_cat_features:
            all_cat_inputs = torch.unbind(hist_e, dim=-1)
        else:
             raise ValueError(f"hist_e shape is incorrect. Expected 3D tensor with {self.num_all_cat_features} features in the last dim.")

        # B. 嵌入并拼接
        cat_embeddings = []
        cat_name = list(self.num_classes_dict.keys())
        for i in range(self.num_all_cat_features):
            key = cat_name[i]
            cat_input = all_cat_inputs[i] 
            cat_emb = self.cat_emb[key](cat_input) # [B, L', D]
            cat_embeddings.append(cat_emb)

        # C. 投影融合 [B, L', F*D] -> [B, L', D]
        cat_concat = torch.cat(cat_embeddings, dim=-1)
        dynamic_features_list.append(cat_concat)
        
        # --- 2. 动态数值特征处理 (hist_x) ---
        
        # A. 提取时间间隔 dt (第 0 列)
        hist_dt = hist_x[:, :, 0] # [B, L']
        
        # B. 时间编码 (dt)
        hist_time_encoding = self.temporal_enc(hist_dt) # [B, L', D]
        dynamic_features_list.append(hist_time_encoding)
        
        # C. 额外 M 个数值特征 (第 1 列及以后)
        num_extra_numerical_features = self.num_all_numerical_features - 1
        if num_extra_numerical_features > 0:
            # 提取额外的数值特征 [B, L', M]
            extra_num_features = hist_x[:, :, 1:] 
            
            # 投影 [B, L', M] -> [B, L', D]
            dynamic_num_emb = self.dynamic_num_proj(extra_num_features)
            dynamic_features_list.append(dynamic_num_emb)
            
        # --- 3. 最终拼接与编码 ---

        # 拼接所有动态特征 [B, L', D_total]
        ##
        src = torch.cat(dynamic_features_list, dim=-1)
            
        # 维度调整：[B, L', D_total] -> [L', B, D_total]
        src = src.permute(1, 0, -1)

        # Transformer 编码
        src_mask = self.generate_square_subsequent_mask(hist_dt.size(1)).to(src.device)
        src_mask = src_mask != 0
        
        memory = self.encoder(src, mask=src_mask)
        
        # 维度调回并降维
        memory = memory.permute(1, 0, -1)
        memory = self.output_layer(memory)

        return memory


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