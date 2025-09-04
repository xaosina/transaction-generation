from ebes.model.seq2seq import BaseSeq2Seq, PositionalEncoding
from ebes.types import Seq


class Transformer1(BaseSeq2Seq):
    def __init__(
        self,
        input_size: int,
        max_len: int,
        num_layers: int = 1,
        num_heads: int = 1,  # Dont change so we dont have to worry about input_size
        scale_hidden: int = 4,
        dropout: float = 0.0,
        pos_dropout: float = 0.0,
        pos_enc_type: str = "base",
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            input_size, pos_dropout, max_len, pos_enc_type
        )
        if pos_enc_type == "cat":
            input_size += 16
        self._out_dim = input_size
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=input_size * scale_hidden,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    @property
    def output_dim(self):
        return self._out_dim

    def generate_causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, seq: Seq) -> Seq:
        src = seq.tokens  # [L, B, D]
        src = self.pos_encoder(src)
        seq_len = src.size(0)
        tgt_mask = self.generate_causal_mask(seq_len).to(src.device)

        encoded_src = self.transformer_decoder(src, memory=None, tgt_mask=tgt_mask)
        return Seq(tokens=encoded_src, lengths=seq.lengths, time=seq.time)
