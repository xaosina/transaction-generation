import collections
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Dict, List, NewType, Union

import numpy as np
import pandas as pd
from Levenshtein import distance as lev_score
from sdmetrics.reports.single_table import QualityReport
from sklearn.metrics import accuracy_score

from ..data.data_types import DataConfig

UserStatistic = NewType("UserStatistic", Dict[int, Dict[str, Union[int, float]]])


class BaseMetric(ABC):
    data_conf: DataConfig

    @abstractmethod
    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame): ...

    @abstractmethod
    def __repr__(self): ...


class BinaryMetric(BaseMetric):
    target_key: str

    @abstractmethod
    def get_scores(self, row): ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        df = pd.concat(
            (orig[self.target_key], gen[self.target_key]),
            keys=["gt", "pred"],
            axis=1,
        ).map(lambda x: x[-self.data_conf.generation_len :])
        return df.apply(self.get_scores, axis=1).mean()


class Levenstein(BinaryMetric):
    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        lev_m = 1 - lev_score(gt, pred) / len(pred)
        return lev_m

    def __repr__(self):
        return f"Levenstein on {self.target_key}"


class Accuracy(BinaryMetric):

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        acc_m = accuracy_score(gt, pred)
        return acc_m

    def __repr__(self):
        return f"Accuracy on {self.target_key}"


class F1Metric(BaseMetric):

    def __init__(self, average="macro"):
        super().__init__()
        self.__average = average

    @staticmethod
    def f1_score_macro_unorder(cls_metric: UserStatistic) -> float:
        f1_score_sum = 0
        for _, value in cls_metric.items():
            f1_score_sum += (
                2
                * value["precision"]
                * value["recall"]
                / (value["precision"] + value["recall"])
                if (value["precision"] + value["recall"]) != 0
                else 0
            )
        return f1_score_sum / len(cls_metric)

    @staticmethod
    def f1_score_micro_unorder(cls_metric: UserStatistic) -> float:
        tp, fp, fn = 0, 0, 0
        for _, value in cls_metric.items():
            tp += value["tp"]
            fp += value["fp"]
            fn += value["fn"]
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def get_statistics(gt: List[int], pred: List[int]) -> UserStatistic:
        cls_metric = dict()
        gt_counter = collections.defaultdict(int, collections.Counter(gt))
        pred_counter = collections.defaultdict(int, collections.Counter(pred))
        for cls in set(gt + pred):
            gt_cls = gt_counter[cls]
            pred_cls = pred_counter[cls]
            tp = min(gt_cls, pred_cls)
            if gt_cls >= pred_cls:
                fn = gt_cls - pred_cls
                fp = 0
            else:
                fn = 0
                fp = pred_cls - gt_cls
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            cls_metric[cls] = {
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return cls_metric

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        f1_function = (
            self.f1_score_macro_unorder
            if self.__average == "macro"
            else self.f1_score_micro_unorder
        )
        f1 = f1_function(self.get_statistics(gt, pred))

        return f1

    def __repr__(self):
        return f"F1_{self.__average} on {self.target_key}"


class DistributionMetric(BaseMetric):

    target_key: str

    @abstractmethod
    def get_scores(self, p): ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        vals = np.array(list(gen[self.target_key].value_counts()), dtype=float)
        p = vals / vals.sum()
        return self.get_scores(p)


class Gini(DistributionMetric):

    def get_scores(self, p):

        p_sorted = np.sort(p)
        n = len(p_sorted)

        gini = 0
        for k in range(n):
            gini += (2 * (k + 1) - n - 1) * p_sorted[k]
        gini = gini / (n - 1)

        return gini

    def __repr__(self):
        return f"Gini on {self.target_key}"


class ShannonEntropy(DistributionMetric):

    def get_scores(self, p):

        shannon_entropy = -np.sum(p * np.log2(p))

        return shannon_entropy

    def __repr__(self):
        return f"Shannon entropy on {self.target_key}"


class GenVsHistoryMetric(BaseMetric):
    target_key: str
    overall: bool

    @abstractmethod
    def get_scores(row): ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        hist = gen[self.target_key].map(lambda x: x[: -self.data_conf.generation_len])
        preds = gen[self.target_key].map(lambda x: x[-self.data_conf.generation_len :])
        df = pd.concat((hist, preds), keys=["hists", "preds"], axis=1)
        if not self.overall:
            df = df.agg(lambda x: [np.concatenate(x.values)], axis=0)
        return df.apply(self.get_scores, axis=1).mean()


class CardinalityCoverage(GenVsHistoryMetric):
    def get_scores(row):
        hists, preds = row["hists"], row["preds"]
        return len(np.unique(preds)) / len(np.unique(hists))

    def __repr__(self):
        return self.overall * "Overall " + f"CardinalityCoverage on {self.target_key}"


class NoveltyScore(GenVsHistoryMetric):
    def get_scores(row):
        hists, preds = row["hists"], row["preds"]
        hists, preds = set(hists), set(preds)
        return len(preds - hists) / len(preds)

    def __repr__(self):
        return self.overall * "Overall " + f"NoveltyScore on {self.target_key}"


class Density(BaseMetric):
    log_cols: list[str] = field(default_factory=list)
    with_timediff: bool = False
    get_details: bool = False

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):

        def log10_scale(x):
            linear = (x <= np.e) & (x >= -np.e)  # to match the derivatives
            y = np.where(linear, 1, x)  # to avoid np.log10 warnings
            y = np.abs(y)
            return np.where(linear, x / (np.e * np.log(10)), np.sign(x) * np.log10(y))

        def preproc_parquet(df):
            assert df._seq_len.min() >= self.data_conf.generation_len
            if self.with_timediff:
                df["time_delta"] = df[self.data_conf.time_name].map(
                    lambda x: np.diff(x, prepend=0)
                )
            df[self.data_conf.seq_cols] = df[self.data_conf.seq_cols].map(
                lambda x: x[-self.data_conf.generation_len :]
            )
            df = df.explode(self.data_conf.seq_cols)
            return df

        # Preprocess
        gen = preproc_parquet(gen)
        orig = preproc_parquet(orig)
        # Prepare metadata
        metadata = {"columns": {}}
        for num_name in self.data_conf.num_names:
            metadata["columns"][num_name] = {"sdtype": "numerical"}
        for cat_name in self.data_conf.cat_cardinalities:
            metadata["columns"][cat_name] = {"sdtype": "categorical"}
        # metadata["sequence_key"] = self.data_conf.index_name  # TODO doesn't matter?
        # metadata["sequence_index"] = self.data_conf.time_name  # TODO doesn't matter?

        if self.with_timediff:
            metadata["columns"]["time_diff"] = {"sdtype": "numerical"}

        for col in self.log_cols:
            gen[col] = log10_scale(gen[col])
            orig[col] = log10_scale(orig[col])

        gen = gen[metadata["columns"].keys()]
        orig = orig[metadata["columns"].keys()]
        # Calculate
        qual_report = QualityReport()
        qual_report.generate(orig, gen, metadata)
        quality = qual_report.get_properties()
        Shape = quality["Score"][0]
        Trend = quality["Score"][1]
        res = dict(shape=Shape, trend=Trend)
        if self.get_details:
            shapes = qual_report.get_details(property_name="Column Shapes")
            trends = qual_report.get_details(property_name="Column Pair Trends")
            res |= {"Shape details": shapes, "Trend details:": trends}
        return res
