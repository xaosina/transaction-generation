import torch
import torch.nn as nn
import math
from .temporal_pos_enc import ContinuousSinusoidalPosEmb


class TimeDenoisingModule(nn.Module):
    def __init__(
        self,
        transformer_dim=32,
        num_classes=10,
        n_steps=100,
        transformer_heads=2,
        dim_feedforward=64,
        dropout=0.1,
        n_decoder_layers=1,
        device="cuda",
        batch_first=True,
        feature_dim=8,
        len_numerical_features=2,
        outer_history_encoder_dim=None,
        prefix_dim=None,
    ):  # 🚀 新增 x_dim 和 e_dim 参数
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

        self.cat_features_number = num_classes

        # Diffusion time embedding: 保持不变
        self.time_pos_emb = ContinuousSinusoidalPosEmb(int(feature_dim), n_steps)
        
        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, int(feature_dim)),
        )
        dynamic_feature_dim_sum = 2* feature_dim

        self.cat_emb = nn.ModuleDict()
        for key, value in self.cat_features_number.items():
            self.cat_emb[key] = nn.Embedding(value + 1, int(feature_dim))
            dynamic_feature_dim_sum += feature_dim

        num_extra_numerical_features = len_numerical_features - 1

        self.num_proj = None
        if num_extra_numerical_features > 0:
            self.num_proj = nn.Linear(num_extra_numerical_features, int(feature_dim))
            dynamic_feature_dim_sum += feature_dim

        self.nhead = transformer_heads

        # Decoder: 保持不变
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        decoder_norm = nn.LayerNorm(transformer_dim)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=decoder_norm
        )

        self.feature_dim_sum = dynamic_feature_dim_sum
        self.reduction_dim_layer = nn.Linear(dynamic_feature_dim_sum, transformer_dim)

        self.output_layer = nn.Linear(transformer_dim, len_numerical_features)

        self.num_classes = num_classes

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / int(feature_dim)) for i in range(int(feature_dim))],
            device=self.device,
        )

        self.prefix_proj = None
        if outer_history_encoder_dim is not None:
            self.n_prefix = prefix_dim
            self.prefix_proj = nn.Sequential(
                nn.Linear(
                    outer_history_encoder_dim,
                    transformer_dim * self.n_prefix
                ),
                nn.Tanh(),
            )


    def forward(self, x, e, t, hist, cat_order, hist_emb=None):
        """
        :param x: B x Seq_Len x X_Dim (如果 X_Dim > 1)
        :param e: B x Seq_Len x E_Dim (如果 E_Dim > 1)
        :param t: B
        :param hist: history representation, B x L_context x Dim
        """

        t = self.time_pos_emb(t)
        t = self.mlp(t)  # B x d_model/4

        order = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        )
        order = self.order_enc(order.float())

        t = t.view(x.size(0), 1, -1).repeat(1, x.size(1), 1)

        combined_cat = []

        idx = 0
        # for key,value in self.num_classes_dict.items():
        # breakpoint()
        for cat_name in cat_order:
            # e_temp = e[:,:,idx] if e.dim() == 3 else e
            e_temp = e[:, :, idx]
            combined_cat.append(self.cat_emb[cat_name](e_temp))
            idx += 1

        all_features = [self.temporal_enc(x[:, :, 0])]

        if self.num_proj is not None:
            all_features.append(self.num_proj(x[:, :, 1:]))

        all_features.append(torch.cat(combined_cat, dim=-1))
        all_features.append(t)

        tgt = torch.cat(all_features, dim=-1) + order

        tgt = self.reduction_dim_layer(tgt)

        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        memory = hist

        assert (
            memory.size(2) == self.transformer_dim
        ), f"Error: history dim (got {memory.size(2)}) should equal to transformer_dim (got {self.transformer_dim})"

        memory = hist.permute(1, 0, -1) # Why it is so different from type_denoising_ml?

        if self.prefix_proj is not None:
            assert hist_emb is not None, "You set history embedding to be but do not provide any embeddings."

            prefix = self.prefix_proj(hist_emb)
            prefix = prefix.view(self.n_prefix, prefix.size(0), self.transformer_dim)

            memory = torch.cat([prefix, memory], dim=0)

        # 维度转换：B x S x D -> S x B x D
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
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
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
        保持原始的 Order Encoding 逻辑，但使用 order_vec 保持不变，它是一个 d_model 维度的编码
        """
        d_model = self.feature_dim_sum
        position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=order_seq.device,
        )

        result = order_seq.unsqueeze(-1) / position_vec[: int(d_model)]

        order_emb = torch.zeros(
            order_seq.size(0), order_seq.size(1), int(d_model), device=order_seq.device
        )
        order_emb[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        order_emb[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return order_emb
