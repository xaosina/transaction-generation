import pandas as pd
from ebes.model.basemodel import BaseModel
from ebes.types import Seq, Batch
from ebes.model import Batch2Seq

from typing import Literal
import torch.nn as nn
import torch


class ConditionGRU(BaseModel):
    def __init__(
        self,
        # Preprocessor
        cat_cardinalities,
        num_features,
        cat_emb_dim=16,
        num_emb_dim=16,
        time_process: Literal["cat", "diff", "none"] = "cat",
        num_norm=True,
        # GRU
        hidden_size: int = 128,
        condition_path: str = None,
        num_layers: int = 1,
    ):
        """
        condition_path - full path to parquet with embeddings(numpy) for each user_id
        """
        super().__init__()
        self.conditions = pd.read_parquet(condition_path)
        self.cond_proj = nn.Linear(
            self.conditions.iloc[0].shape[0], num_layers * hidden_size
        )
        self.processor = Batch2Seq(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            time_process=time_process,
            num_norm=num_norm,
        )

        self._out_dim = hidden_size
        self.net = nn.GRU(
            input_size=self.processor.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, batch: Batch) -> Seq:
        conditions = torch.tensor(self.conditions[batch.index]).to(batch.time.device)
        conditions = self.cond_proj(conditions)  # shape - [B, num_l x hidden_size]
        conditions = conditions.view(conditions.shape[0], -1, self._out_dim).transpose(
            0, 1
        )  # shape [n_l, B, hidden]
        # TODO CHECK FOR SIZE

        seq = self.processor(batch)
        x, _ = self.net(seq.tokens, conditions)
        return Seq(tokens=x, lengths=seq.lengths, time=seq.time)
