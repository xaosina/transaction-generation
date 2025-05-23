"""GenBatch transforms for data loading pipelines."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
import pickle

import numpy as np
import pandas as pd
import torch

from ..utils import RateLimitFilter
from .data_types import DataConfig

logger = logging.getLogger(__name__)
logger.addFilter(RateLimitFilter(60))

MISSING_CAT_VAL = 0


class LoaderTransform(ABC):
    """Base class for all transforms.
    """

    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to the frame."""
        ...
    
    def new_dataconf(self, DataConfig) -> DataConfig:
        ...



@dataclass
class NewFeatureTransform(BatchTransform):
    """Does not modify old features. Only adds new ones. Always adds on the right (torch.cat(tensor, new_feature, dim=-1))"""

    @property
    def num_names(self) -> list[str] | None:
        return []

    @property
    def cat_cardinalities(self) -> list[str] | None:
        return {}

    @property
    def num_names_removed(self) -> list[str] | None:
        return []

    @property
    def cat_names_removed(self) -> list[str] | None:
        return []
    
    def new_focus_on(self, focus_on) -> list[str]:
        return focus_on
    
