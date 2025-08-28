from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig

from .base import BaselineLoss, NoOrderLoss, TailLoss, VAELoss
from .oneshot import DistLoss, GaussianDistLoss, MatchedLoss, TargetLoss
from .detpp.detpp import DeTPPLoss


@dataclass(frozen=True)
class LossConfig:
    name: Optional[str] = "BaselineLoss"
    params: dict = field(default_factory=dict)


class DummyLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        return {'loss': data}


def get_loss(data_conf: LatentDataConfig, config: LossConfig):
    name = config.name
    if name in globals():
        return globals()[name](data_conf, **config.params)
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")
