import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import math

class Tokenizer(nn.Module):

    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + list(categories[:-1])).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.cat_weight = nn.Parameter(Tensor(sum(categories), d_token))
            nn.init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))
        self.d_token = d_token
        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):

        # x_num = x_num.permute(1,0,2)
        # x_cat = x_cat.permute(1,0,2)
        # if x_num is not None:
        #     B, L, D_num = x_num.shape
        # if x_cat is not None:
        #     B, L, D_cat = x_cat.shape

        
        # # --- 1. 形状重塑：L*B 扁平化为新的 Batch Size ---
        
        # # 将 (L, B, D) 转换为 (L * B, D)
        # if x_num is not None:
        #     x_num_flat = x_num.reshape(L * B, -1)
        # else:
        #     x_num_flat = None
            
        # if x_cat is not None:
        #     x_cat_flat = x_cat.reshape(L * B, -1)
        # else:
        #     x_cat_flat = None
        
        # x_cat,x_num = x_cat_flat,x_num_flat
        
        # --- 2. 使用原始 Tokenizer 逻辑进行标记化 ---
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
    
        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            for start, end in zip(self.category_offsets, torch.cat([self.category_offsets[1:], torch.tensor([x_cat.shape[1]], device=x_cat.device)])):
                if start < end:
                    x = torch.cat(
                        [x, x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None]],
                        dim=1,
                    )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )

            x = x + bias[None]
        # 
        # D_cat = 2
        # x = x[:,1:,:]
        # x = x.reshape(B,L,(D_num+D_cat)*self.d_token)
    

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression = None, value_compression = None):
  
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b , dim=-1)

        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
        
class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        activation = 'relu',
        prenormalization = True,
        initialization = 'kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
      
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat



def tensor_to_one_hot(X_index, categories) -> torch.Tensor:
    """
    将包含类别索引的张量 (L, B, D) 转换为 One-Hot 编码张量。
    D 维中的每个特征可以有不同的类别数量。

    Args:
        X_index: 形状为 (L, B, D) 的 LongTensor，其中 D 是类别特征的数量，
                 包含每个样本的类别索引。
        categories: 一个列表，长度等于 D (X_index.shape[-1])，
                    其中 categories[i] 是第 i 个特征的类别总数。

    Returns:
        One-Hot 编码后的张量，形状为 (L, B, sum(categories))。
    """
    
    # 检查输入张量是否为整数类型 (One-Hot 编码要求)
    if X_index.dtype not in [torch.long, torch.int]:
        raise TypeError(f"输入张量的数据类型必须是整数 (torch.long 或 torch.int)，但当前是 {X_index.dtype}")

    # 检查 D 维度是否与 categories 列表长度匹配
    D = X_index.shape[-1]
    if D != len(categories):
        raise ValueError(
            f"张量的最后一个维度 D ({D}) 必须与 categories 列表的长度 ({len(categories)}) 匹配。"
        )

    # 确保张量在内存中是连续的，以进行切片操作（可选，但推荐）
    X_index = X_index.contiguous()

    one_hot_list = []

    # 逐特征 (D 维度) 进行 One-Hot 编码
    for i in range(D):
        # 1. 提取第 i 个特征的索引。形状为 (L, B)
        feature_indices = X_index[..., i] 
        
        # 2. 获取当前特征的类别数
        num_classes = categories[i]
        
        # 3. 进行 One-Hot 编码
        # F.one_hot 的结果是 (L, B, num_classes)
        # 转换为 float，以便后续的嵌入/计算
        one_hot_feature = F.one_hot(feature_indices, num_classes=num_classes).float() 
        
        one_hot_list.append(one_hot_feature)

    # 4. 拼接 (Concatenation)
    # 沿最后一个维度（类别维度）拼接所有 One-Hot 结果
    X_one_hot_final = torch.cat(one_hot_list, dim=-1)

    return X_one_hot_final