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
    path: str
    min_hist_len: int
    gen_len: int
    disable: bool = True

    def __post_init__(self):
        self.min_len = self.gen_len + self.min_hist_len
        with open(self.path, "rb") as file:
            self.exclude_categories = np.load(file)

    def remove_excluded(self, seq: np.ndarray) -> np.ndarray:
        mask = ~np.isin(seq, self.exclude_categories)
        return seq[mask]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.disable:
            return data
        
        data[self.feature_name] = data[self.feature_name].apply(self.remove_excluded)
        data["_seq_len"] = data[self.feature_name].map(len)
        mask = data["_seq_len"] >= self.min_len
        return data.loc[mask].reset_index(drop=True)

    def new_dataconf(self, data_conf: DataConfig) -> DataConfig:
        if self.disable:
            return data_conf
        
        new_data_conf = deepcopy(data_conf)
        new_data_conf.cat_cardinalities[self.feature_name] = (
            data_conf.cat_cardinalities[self.feature_name]
            - len(self.exclude_categories)
        )
        return new_data_conf


@dataclass
class NumericalFilter(LoaderTransform):
    feature_name: str
    min_hist_len: int
    gen_len: int
    low_quantile: float = 0.01
    high_quantile: float = 0.99
    min_cut: bool = True
    max_cut: bool = True

    def cut(self, data: pd.DataFrame, func) -> pd.DataFrame:
        s = data["amount"].apply(func)
        lo, hi = s.quantile([self.low_quantile, self.high_quantile])
        return data[s.between(lo, hi)]

    def __post_init__(self):
        self.min_len = self.min_hist_len + self.gen_len

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min_cut:
            data = self.cut(data, min)
        if self.max_cut:
            data = self.cut(data, max)

        return data

    def new_dataconf(self, data_conf: DataConfig) -> DataConfig:
        return data_conf
