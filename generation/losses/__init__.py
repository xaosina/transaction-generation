from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from generation.data.data_types import GenBatch, LatentDataConfig

from .base import BaselineLoss, NoOrderLoss, TailLoss, VAELoss
from .oneshot import DistLoss, GaussianDistLoss, MatchedLoss, TargetLoss


@dataclass(frozen=True)
class LossConfig:
    name: Optional[str] = "baseline"
    params: dict = field(default_factory=dict)


class DummyLoss(Module):
    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        return {'loss': data}


def get_loss(data_conf: LatentDataConfig, config: LossConfig):
    name = config.name
    if name == "baseline":
        return BaselineLoss(data_conf, **config.params)
    elif name == "target":
        return TargetLoss(data_conf, **config.params)
    elif name == "distloss":
        return DistLoss(data_conf, **config.params)
    elif name == "gaussian_distloss":
        return GaussianDistLoss(data_conf, **config.params)
    elif name == "matched":
        return MatchedLoss(data_conf, **config.params)
    elif name == "tail":
        return TailLoss(data_conf, **config.params)
    elif name == "vae":
        return VAELoss(data_conf, init_beta=1.0, **config.params)
    elif name == "no_order_loss":
        return NoOrderLoss(data_conf, **config.params)
    elif name == "dummy":
        return DummyLoss()
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")
