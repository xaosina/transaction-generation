import torch
import torch.nn as nn
import math


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        transformer_dim=128,
        transformer_heads=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_encoder_layers=3,
        num_classes=10,
        device="cuda",
        len_numerical_features=1,
        num_feature_dim=8,
        cat_feature_dim=8,
        time_feature_dim=8,
        causal_mask=True,
        use_simple_time_proj=True,
    ):
        super(HistoryEncoder, self).__init__()

        self.device = device
        self.transformer_dim = transformer_dim
        self.use_causal_mask = causal_mask

        self.cat_features_number = num_classes
        all_dynamic_vocab_sizes = self.cat_features_number.values()

        if all_dynamic_vocab_sizes is None or not all_dynamic_vocab_sizes:
            raise ValueError(
                "all_dynamic_vocab_sizes must be provided for all categorical features."
            )

        self.num_all_cat_features = len(all_dynamic_vocab_sizes)


        dynamic_feature_dim_sum = time_feature_dim

        self.cat_emb = nn.ModuleDict()
        for key, value in self.cat_features_number.items():
            self.cat_emb[key] = nn.Embedding(value, int(cat_feature_dim))
            dynamic_feature_dim_sum += cat_feature_dim

        self.simple_time_proj = None
        if use_simple_time_proj:
            self.simple_time_proj = nn.Linear(1, time_feature_dim)
            dynamic_feature_dim_sum += time_feature_dim

        self.num_all_numerical_features = len_numerical_features
        num_extra_numerical_features = len_numerical_features - 1

        self.num_proj = None
        if num_extra_numerical_features > 0:
            self.num_proj = nn.Linear(
                num_extra_numerical_features, int(num_feature_dim)
            )
            dynamic_feature_dim_sum += num_feature_dim

        self.total_d_model = dynamic_feature_dim_sum

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dynamic_feature_dim_sum,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(dynamic_feature_dim_sum)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )
        self.output_layer = nn.Linear(dynamic_feature_dim_sum, transformer_dim)
        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / time_feature_dim)
                for i in range(int(time_feature_dim))
            ],
            device=self.device,
        )

    def forward(self, hist_x, hist_e):
        if hist_e.dim() == 3 and hist_e.size(-1) == self.num_all_cat_features:
            all_cat_inputs = torch.unbind(hist_e, dim=-1)
        else:
            raise ValueError(
                f"hist_e shape is incorrect. Expected 3D tensor with {self.num_all_cat_features} features in the last dim."
            )

        all_features = [self.temporal_enc(hist_x[:, :, 0])]
        if self.simple_time_proj is not None:
            all_features.append(self.simple_time_proj(hist_x[:, :, [0]]))

        combined_cat = []
        cat_name = list(self.cat_features_number.keys())
        for i in range(self.num_all_cat_features):
            key = cat_name[i]
            cat_input = all_cat_inputs[i]
            cat_emb = self.cat_emb[key](cat_input)  # [B, L', D]
            combined_cat.append(cat_emb)

        if self.num_proj is not None:
            all_features.append(self.num_proj(hist_x[:, :, 1:]))

        all_features.append(torch.cat(combined_cat, dim=-1))
        src = torch.cat(all_features, dim=-1)

        src = src.permute(1, 0, -1)

        src_mask = self.generate_square_subsequent_mask(hist_e.size(1)).to(src.device)

        memory = self.encoder(src, mask=src_mask if self.use_causal_mask else None)

        memory = memory.permute(1, 0, -1)
        out = self.output_layer(memory)

        return out

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def temporal_enc(self, dt_seq):
        if dt_seq.dim() == 1:
            dt_seq = dt_seq.unsqueeze(-1)
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
