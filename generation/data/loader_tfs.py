"""GenBatch transforms for data loading pipelines."""

from copy import deepcopy
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
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to the frame."""
        ...

    def new_dataconf(self, DataConfig) -> DataConfig: ...


@dataclass
class ExcludeCategories(LoaderTransform):

    feature_name: str
    exclude_categories: list[int]
    min_hist_len: int
    gen_len: int

    def __post_init__(self):
        self.min_len = self.gen_len + self.min_hist_len

    def remove_excluded(self, seq: list[int]) -> list[int]:
        return [e for e in seq if e not in self.exclude_categories]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.feature_name] = data[self.feature_name].apply(self.remove_excluded)
        data['_seq_len'] = data[self.feature_name].map(len)
        mask = data['_seq_len'] >= self.min_len
        return data.loc[mask].reset_index(drop=True)

    def new_dataconf(self, data_conf: DataConfig) -> DataConfig:
        new_data_conf = deepcopy(data_conf)
        new_data_conf.cat_cardinalities[self.feature_name] = (
            data_conf.cat_cardinalities[self.feature_name]
            - len(self.exclude_categories)
        )
        return new_data_conf
