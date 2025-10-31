import torch
import torch.nn as nn
import math
from .temporal_pos_enc import get_time_pos_emb


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
        len_numerical_features=None,
        num_feature_dim=8,
        cat_feature_dim=8,
        time_feature_dim=8,
        outer_history_encoder_dim=None,
        prefix_dim=None,
        use_post_norm=False,
        diffusion_t_type='discrete',
        use_simple_time_proj=True,
    ):
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
        super(TimeDenoisingModule, self).__init__()

        self.device = device
        self.transformer_dim = transformer_dim
        self.time_feature_dim = time_feature_dim
        self.cat_features_number = num_classes

        self.time_pos_emb = get_time_pos_emb(diffusion_t_type, time_feature_dim, n_steps)

        self.mlp = nn.Sequential(
            nn.Linear(int(time_feature_dim), transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, int(time_feature_dim)),
        )

        # time, time, t - time twice because it was in cdiff original
        dynamic_feature_dim_sum = 2 * time_feature_dim

        self.cat_emb = nn.ModuleDict()
        for key, value in self.cat_features_number.items():
            self.cat_emb[key] = nn.Embedding(value, int(cat_feature_dim))
            dynamic_feature_dim_sum += cat_feature_dim

        num_extra_numerical_features = len_numerical_features - 1
        self.simple_time_proj = None
        if use_simple_time_proj:
            self.simple_time_proj = nn.Linear(1, time_feature_dim)
            dynamic_feature_dim_sum += time_feature_dim

        self.num_proj = None
        if num_extra_numerical_features > 0:
            self.num_proj = nn.Linear(
                num_extra_numerical_features, int(num_feature_dim)
            )
            dynamic_feature_dim_sum += num_feature_dim

        self.nhead = transformer_heads

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        decoder_norm = nn.LayerNorm(transformer_dim)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=decoder_norm,
        )
        self.reduction_dim_layer = nn.Linear(dynamic_feature_dim_sum, transformer_dim)

        self.prefix_post_norm = nn.LayerNorm(self.transformer_dim, eps=1e-6) if use_post_norm else None

        self.feature_dim_sum = dynamic_feature_dim_sum

        self.output_layer = nn.Linear(transformer_dim, len_numerical_features)

        self.num_classes = num_classes

        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / int(time_feature_dim))
                for i in range(int(time_feature_dim))
            ],
            device=self.device,
        )

        self.order_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / int(dynamic_feature_dim_sum))
                for i in range((int(dynamic_feature_dim_sum)))
            ],
            device=torch.device(self.device),
        )

        self.prefix_proj = None
        if outer_history_encoder_dim is not None:
            self.n_prefix = prefix_dim
            self.prefix_proj = nn.Sequential(
                nn.Linear(outer_history_encoder_dim, transformer_dim * self.n_prefix),
                nn.Tanh(),
            )

    def forward(self, x, e, t, hist, cat_order, h_emb=None):
        """
        :param x: B x Seq_Len x X_Dim (如果 X_Dim > 1)
        :param e: B x Seq_Len x E_Dim (如果 E_Dim > 1)
        :param t: B
        :param hist: history representation, B x L_context x Dim
        """

        t = self.time_pos_emb(t)
        t = self.mlp(t)  # B x d_model/4
        order = torch.cat([torch.arange(x.size(1)).unsqueeze(0)] * x.size(0), dim=0)

        order = self.order_enc(order.to(x.device))

        time_embed = t.view(x.size(0), 1, int(self.time_feature_dim))
        t = torch.cat([time_embed] * x.size(1), dim=1)
        combined_cat = []

        idx = 0
        # for key,value in self.num_classes_dict.items():
        # breakpoint()
        for cat_name in cat_order:
            e_temp = e[:, :, idx]
            combined_cat.append(self.cat_emb[cat_name](e_temp))
            idx += 1

        all_features = [self.temporal_enc(x[:, :, 0])]
        if self.simple_time_proj is not None:
            all_features.append(self.simple_time_proj(x[:, :, [0]]))

        if self.num_proj is not None:
            all_features.append(self.num_proj(x[:, :, 1:]))

        all_features.append(torch.cat(combined_cat, dim=-1))
        all_features.append(t)

        tgt = torch.cat(all_features, dim=-1) + order

        tgt = self.reduction_dim_layer(tgt)

        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        memory = hist.permute(1, 0, -1)

        tgt = tgt.permute(1, 0, -1)

        assert memory.size(2) == tgt.size(
            2
        ), f"Error: history dim (got {memory.size(2)}) should equal to target's (got {tgt.size(2)})"

        if self.prefix_proj is not None:
            assert (
                h_emb is not None
            ), "You set history embedding to be but do not provide any embeddings."

            prefix = self.prefix_proj(h_emb)
            prefix = prefix.reshape(prefix.size(0), self.n_prefix, self.transformer_dim)
            prefix = prefix.permute(1, 0, -1)

            memory = torch.cat([prefix, memory], dim=0)
            if self.prefix_post_norm is not None:
                memory = self.prefix_post_norm(memory)

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
        )

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
