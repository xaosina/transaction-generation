import torch
import torch.nn as nn
import math
from .temporal_pos_enc import ContinuousSinusoidalPosEmb


class TimeDenoisingModule(nn.Module):
    def __init__(self, transformer_dim=32, num_classes=10, n_steps=100, transformer_heads=2,
                 dim_feedforward=64, dropout=0.1, n_decoder_layers=1, device='cuda', batch_first=True):
        """

        :param transformer_dim:
        :param num_classes:
        :param n_steps:
        :param transformer_heads:
        :param dim_feedforward:
        :param dropout:
        :param n_decoder_layers:
        :param device:
        """
        super(TimeDenoisingModule, self).__init__()

        self.device = device

        # event type embedding for event type seq
        self.event_emb = nn.Embedding(num_classes + 1, int(transformer_dim/4), padding_idx=num_classes)

        # Diffusion time embedding
        self.time_pos_emb = ContinuousSinusoidalPosEmb(int(transformer_dim/4), n_steps)
        self.mlp = nn.Sequential(
            nn.Linear(int(transformer_dim/4), transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, int(transformer_dim/4))
        )

        self.nhead = transformer_heads

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_dim, nhead=transformer_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)

        decoder_norm = nn.LayerNorm(transformer_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers,
                                             norm=decoder_norm)
        self.output_layer = nn.Linear(transformer_dim, 1)

        self.num_classes = num_classes
        self.device = device
        self.transformer_dim = transformer_dim

        # Time encoding for inter-arrival time seq
        # position vector, used for time encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / int(transformer_dim/4)) for i in range(int(transformer_dim/4))],
            device=torch.device(self.device))
        
        self.order_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) /int(transformer_dim)) for i in range((int(transformer_dim)))],
            device=torch.device(self.device))

        # event type embedding for event type seq
        self.event_emb = nn.Embedding(num_classes + 1, int(transformer_dim/4), padding_idx=num_classes)

    def forward(self, x, e, t, hist):
        """

        :param x: B x Seq_Len
        :param t:
        :param e:
        :param hist: history representation, B x L_context x Dim
        :param non_padding_mask:
        """
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        order = torch.cat([torch.arange(x.size(1)).unsqueeze(0)]*x.size(0),dim=0)

        order = self.order_enc(order.to(x.device))

        time_embed = t.view(x.size(0), 1, int(self.transformer_dim/4))
        t = torch.cat([time_embed]*x.size(1), dim=1)
        x = self.temporal_enc(x)

        e = self.event_emb(e)

        tgt = torch.cat([x,x,e,t], dim=-1) + order
        
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        memory = hist

        memory_mask = self.generate_square_subsequent_mask(hist.size(1)).to(hist.device)

        memory = hist.permute(1, 0, -1)
        tgt = tgt.permute(1, 0, -1)
        assert memory.size(2) == tgt.size(2), \
            f'Error: history dim (got {memory.size(2)}) should equal to target\'s (got {tgt.size(2)})'

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

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
    

    
    def order_enc(self, dt_seq):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = dt_seq.unsqueeze(-1) / self.order_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
