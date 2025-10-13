import torch
import torch.nn as nn
import math


# def get_attn_key_pad_mask(seq_k, seq_q, num_classes):
#     """ For masking out the padding part of key sequence. """

#     # expand to fit the shape of key query attention matrix
#     len_q = seq_q.size(1)
#     padding_mask = seq_k.eq(0)
#     padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
#     return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class HistoryEncoder(nn.Module):
    def __init__(self, transformer_dim=128, transformer_heads=6, dim_feedforward=1024, dropout=0.1, batch_first=True,
                 num_encoder_layers=3, num_classes=10, device='cuda'):
        """

        :param transformer_dim:
        :param transformer_heads:
        :param dim_feedforward:
        :param dropout:
        :param batch_first:
        :param num_encoder_layers:
        :param num_classes:
        :param device:
        """

        super(HistoryEncoder, self).__init__()

        self.num_classes = num_classes
        self.device = device

        # Time encoding for inter-arrival time seq
        # position vector, used for time encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / transformer_dim) for i in range(transformer_dim)],
            device=torch.device(self.device))

        # event type embedding for event type seq
        self.event_emb = nn.Embedding(num_classes + 1, transformer_dim, padding_idx=num_classes)

        # Encoder

        encoder_layer = nn.TransformerEncoderLayer(d_model=2 * transformer_dim, nhead=transformer_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)

        encoder_norm = nn.LayerNorm(2 * transformer_dim)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_encoder_layers,
                                             norm=encoder_norm)

        self.output_layer = nn.Linear(2 * transformer_dim, transformer_dim)

    def forward(self, hist_x, hist_e,  non_pad_mask):
        """

        :param non_pad_mask:
        :param hist_x:
        :param hist_e:
        :return: memory: history_representation, B x L' x dim
        """
        non_pad_mask_temporal_enc = (non_pad_mask.type(torch.float).unsqueeze(-1) - 1) * (-1)
        hist_dt_encoding = self.temporal_enc(hist_x, non_pad_mask_temporal_enc)
        # hist_time_stamps_encoding = self.temporal_enc(hist_time_stamps, non_pad_mask_temporal_enc)
        # hist_dt_encoding = hist_dt_encoding + hist_time_stamps_encoding
        if hist_e.dim() == 1:
            hist_e = hist_e.unsqueeze(-1)
        hist_type_emb = self.event_emb(hist_e)

        src = torch.cat((hist_type_emb, hist_dt_encoding), dim=-1)
        src = src.permute(1, 0, -1)
        src_mask = self.generate_square_subsequent_mask(hist_e.size(1)).to(src.device)
        src_mask = src_mask != 0
        src_key_padding_mask = non_pad_mask
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = memory.permute(1, 0, -1)

        memory = self.output_layer(memory)

        return memory

    # def get_non_pad_mask(self, seq):
    #     """
    #     Get the non-padding positions.
    #     Set num_classes as padding, then 0 is an actual event type.
    #     seq: should be history event sequence, the target sequence is fixed length, default 20, B x L'
    #     output: B x L'
    #     """

    #     assert seq.dim() == 2
    #     return seq.eq(self.num_classes)

    def temporal_enc(self, dt_seq, non_pad_mask):
        """
        dt_seq: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        if dt_seq.dim() == 1:
            dt_seq = dt_seq.unsqueeze(-1)
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
