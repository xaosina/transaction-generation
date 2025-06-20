"""
Coverage and Gini Index were implemented following the approach described in https://www.mdpi.com/2078-2489/16/2/151
"""

import collections
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NewType, Union

import numpy as np
import pandas as pd
from Levenshtein import distance as lev_score
from sdmetrics.reports.single_table import QualityReport
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from ..data.data_types import DataConfig
from .pipelines.eval_detection import run_eval_detection

UserStatistic = NewType("UserStatistic", Dict[int, Dict[str, Union[int, float]]])


@dataclass
class BaseMetric(ABC):
    devices: list[str]
    data_conf: DataConfig
    log_dir: str

    @abstractmethod
    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame): ...

    @abstractmethod
    def __repr__(self): ...


class BatchCutMetric(BaseMetric):
    def __call__(self, orig, gen):
        return orig[self.data_conf.index_name].shape[0]
    
    def __repr__(self):
        return "BatchCutMetric"

@dataclass
class Reconstruction(BaseMetric):
    def __call__(self, orig, gen):
        assert (orig.columns == gen.columns).all()
        results = {}

        cat_cards = (self.data_conf.cat_cardinalities or {})
        for col in self.data_conf.focus_on:

            df = pd.concat(
                (orig[col], gen[col]),
                keys=["gt", "pred"],
                axis=1,
            ).map(lambda x: x[-self.data_conf.generation_len :])

            results[col] = df.apply(
                self._compute_accuracy if col in cat_cards else self._compute_mse,
                axis=1,
            ).mean()
        return {
            "overall": np.mean(list(results.values())),
            **results,
        }

    def _compute_mse(self, row):
        gt, pred = row["gt"], row["pred"]
        return r2_score(gt, pred)

    def _compute_accuracy(self, row):
        gt, pred = row["gt"], row["pred"]
        return accuracy_score(gt, pred)

    def __repr__(self):
        return "Reconstruction"


@dataclass
class EffectiveReconstruction(BaseMetric):
    def __call__(self, orig, gen):
        assert (orig.columns == gen.columns).all()
        results = {}

        cat_cards = (self.data_conf.cat_cardinalities or {})
        for col in self.data_conf.focus_on:

            df = pd.concat(
                (orig[col], gen[col]),
                keys=["gt", "pred"],
                axis=1,
            ).map(lambda x: x[-self.data_conf.generation_len :])

            results[col] = df.apply(
                self._compute_accuracy if col in cat_cards else self._compute_mse,
                axis=1,
            ).mean()
        return {
            "overall": np.mean(list(results.values())),
            **results,
        }

    def _compute_mse(self, row):
        gt, pred = row["gt"], row["pred"]
        return r2_score(gt, pred)

    def _compute_accuracy(self, row):
        gt, pred = row["gt"], row["pred"]
        return accuracy_score(gt, pred)

    def __repr__(self):
        return "Reconstruction"

@dataclass
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


@dataclass
class Levenshtein(BinaryMetric):
    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        lev_m = 1 - lev_score(gt, pred) / max(len(pred), len(gt))
        return lev_m

    def __repr__(self):
        return f"Levenstein on {self.target_key}"


@dataclass
class Accuracy(BinaryMetric):

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        acc_m = accuracy_score(gt, pred)
        return acc_m

    def __repr__(self):
        return f"Accuracy on {self.target_key}"


@dataclass
class PR(BinaryMetric):

    @staticmethod
    def get_statistics(gt: np.ndarray, pred: np.ndarray) -> UserStatistic:
        assert isinstance(gt, np.ndarray) and isinstance(pred, np.ndarray)
        if len(gt) == 0 and len(pred) == 0:
            return {}

        cls_metric = dict()
        gt_counter = collections.Counter(gt)
        pred_counter = collections.Counter(pred)

        all_classes = set(np.concatenate([gt, pred]))
        for cls in all_classes:
            gt_cls = gt_counter.get(cls, 0)
            pred_cls = pred_counter.get(cls, 0)

            tp = min(gt_cls, pred_cls)
            fn = max(0, gt_cls - pred_cls)
            fp = max(0, pred_cls - gt_cls)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0

            cls_metric[cls] = {
                "Precision": precision,
                "Recall": recall,
            }

        return cls_metric

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        stats = self.get_statistics(gt, pred)
        return stats

    def __repr__(self):
        return f"PR on {self.target_key}"


@dataclass
class F1Metric(BinaryMetric):
    average: str = "macro"

    @staticmethod
    def f1_score_macro_unorder(cls_metric: UserStatistic) -> float:
        if not cls_metric:
            return 1.0
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
        if not cls_metric:
            return 1.0

        tp, fp, fn = 0, 0, 0
        for _, value in cls_metric.items():
            tp += value["tp"]
            fp += value["fp"]
            fn += value["fn"]
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def get_statistics(gt: np.ndarray, pred: np.ndarray) -> UserStatistic:
        assert isinstance(gt, np.ndarray) and isinstance(pred, np.ndarray)
        if len(gt) == 0 and len(pred) == 0:
            return {}

        cls_metric = dict()
        gt_counter = collections.Counter(gt)
        pred_counter = collections.Counter(pred)

        all_classes = set(np.concatenate([gt, pred]))
        for cls in all_classes:
            gt_cls = gt_counter.get(cls, 0)
            pred_cls = pred_counter.get(cls, 0)

            tp = min(gt_cls, pred_cls)
            fn = max(0, gt_cls - pred_cls)
            fp = max(0, pred_cls - gt_cls)

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
            if self.average == "macro"
            else self.f1_score_micro_unorder
        )
        f1 = f1_function(self.get_statistics(gt, pred))

        return f1

    def __repr__(self):
        return f"F1_{self.average} on {self.target_key}"


@dataclass
class DistributionMetric(BaseMetric):
    target_key: str
    overall: bool = False

    @abstractmethod
    def get_scores(self, row) -> pd.Series | float: ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        df = pd.concat(
            (orig[self.target_key], gen[self.target_key]),
            keys=["gt", "pred"],
            axis=1,
        ).map(lambda x: x[-self.data_conf.generation_len :])
        if self.overall:
            df = df.agg(lambda x: [np.concatenate(x.values)], axis=0)
        max_c = df.map(max).max().max()
        assert isinstance(max_c, np.int64)

        def get_frequency(arr, max_c):
            frequency_array = np.zeros(max_c + 1, dtype=float)
            unique_values, counts = np.unique(arr, return_counts=True)
            frequency_array[unique_values] = counts
            return frequency_array / arr.size

        df = df.map(lambda x: get_frequency(x, max_c))
        df = df.apply(self.get_scores, axis=1).mean()
        if isinstance(df, pd.Series):
            return df.to_dict()
        elif isinstance(df, float):
            return df


@dataclass
class StatisticMetric(DistributionMetric):
    def get_scores(self, row) -> pd.Series:
        orig_score, gen_score = row.map(self.get_statistic)
        relative = (gen_score - orig_score) / (abs(orig_score) + 1e-8)
        score = 1 - (1 + abs(gen_score - orig_score))
        return pd.Series({"score": score, "relative": relative, "orig": orig_score})

    @abstractmethod
    def get_statistic(self, p) -> float: ...


@dataclass
class Gini(StatisticMetric):

    def get_statistic(self, p):
        p_sorted = np.sort(p)
        n = len(p_sorted)

        gini = 0
        for k in range(n):
            gini += (2 * (k + 1) - n - 1) * p_sorted[k]
        gini = gini / (n - 1) if n != 1 else 1.0

        return gini

    def __repr__(self):
        return self.overall * "Overall " + f"Gini on {self.target_key}"


@dataclass
class ShannonEntropy(StatisticMetric):

    def get_statistic(self, p):
        p = p[p > 0]
        shannon_entropy = -np.sum(p * np.log2(p))

        return shannon_entropy

    def __repr__(self):
        return self.overall * "Overall " + f"Shannon entropy on {self.target_key}"


@dataclass
class GenVsHistoryMetric(BaseMetric):
    target_key: str
    overall: bool = False

    @abstractmethod
    def get_scores(row): ...

    def score_for_df(self, df):
        gen_len = self.data_conf.generation_len
        hist = df[self.target_key].map(lambda x: x[:-gen_len])
        preds = df[self.target_key].map(lambda x: x[-gen_len:])
        df = pd.concat((hist, preds), keys=["hists", "preds"], axis=1)
        if self.overall:
            df = df.agg(lambda x: [np.concatenate(x.values)], axis=0)
        return df.apply(self.get_scores, axis=1).mean()

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        gen_score = self.score_for_df(gen)
        orig_score = self.score_for_df(orig)
        score = 1 - (1 + abs(gen_score - orig_score))
        return {"score": score, "gen": gen_score, "orig": orig_score}


@dataclass
class CardinalityCoverage(GenVsHistoryMetric):
    def get_scores(self, row):
        hists, preds = row["hists"], row["preds"]
        return len(np.unique(preds)) / len(np.unique(hists))

    def __repr__(self):
        return self.overall * "Overall " + f"CardinalityCoverage on {self.target_key}"


@dataclass
class NoveltyScore(GenVsHistoryMetric):
    def get_scores(self, row):
        hists, preds = row["hists"], row["preds"]
        hists, preds = set(hists), set(preds)
        return len(preds - hists) / len(preds)

    def __repr__(self):
        return self.overall * "Overall " + f"NoveltyScore on {self.target_key}"


@dataclass
class Density(BaseMetric):
    log_cols: list[str] = None
    with_timediff: bool = False
    save_details: bool = False
    verbose: bool = False

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        data_conf = deepcopy(self.data_conf)
        num_names = data_conf.num_names or []
        if self.with_timediff:
            num_names += ["time_delta"]
        cat_names = list(data_conf.cat_cardinalities or {})
        seq_cols = num_names + cat_names

        def log10_scale(x):
            linear = (x <= np.e) & (x >= -np.e)  # to match the derivatives
            y = np.where(linear, 1, x)  # to avoid np.log10 warnings
            y = np.abs(y)
            return np.where(linear, x / (np.e * np.log(10)), np.sign(x) * np.log10(y))

        def preproc_parquet(df):
            df = deepcopy(df)
            time_name, gen_len = data_conf.time_name, data_conf.generation_len
            assert df._seq_len.min() >= data_conf.generation_len

            if self.with_timediff:
                df["time_delta"] = df[time_name].map(lambda x: np.diff(x, prepend=0))

            df = df[seq_cols].map(lambda x: x[-gen_len:])
            df = df.explode(seq_cols)
            df[cat_names] = df[cat_names].astype("int64")
            df[num_names] = df[num_names].astype("float32")
            return df

        # Preprocess
        gen = preproc_parquet(gen)
        orig = preproc_parquet(orig)
        # Prepare metadata
        metadata = {"columns": {}}
        for num_name in num_names:
            metadata["columns"][num_name] = {"sdtype": "numerical"}
        for cat_name in cat_names:
            metadata["columns"][cat_name] = {"sdtype": "categorical"}

        if self.log_cols:
            for col in self.log_cols:
                gen[col] = log10_scale(gen[col].values)
                orig[col] = log10_scale(orig[col].values)

        gen = gen[metadata["columns"].keys()]
        orig = orig[metadata["columns"].keys()]
        # Calculate
        qual_report = QualityReport()
        qual_report.generate(orig, gen, metadata, verbose=self.verbose)
        quality = qual_report.get_properties()
        Shape = quality["Score"][0]
        Trend = quality["Score"][1]
        res = dict(shape=Shape, trend=Trend)
        if self.save_details:
            save_dir = f"{self.log_dir}/density"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{save_dir}/density.txt", "w") as f:
                f.write(f"Shape: {Shape}\n")
                f.write(f"Trend: {Trend}\n")
            shapes = qual_report.get_details(property_name="Column Shapes")
            trends = qual_report.get_details(property_name="Column Pair Trends")
            shapes.to_csv(f"{save_dir}/shape.csv")
            trends.to_csv(f"{save_dir}/trend.csv")
        return res

    def __repr__(self):
        return "Density"


@dataclass
class Detection(BaseMetric):
    dataset_config: str
    method_config: str = "gru"
    condition_len: int = 0
    verbose: bool = False

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        data_conf = deepcopy(self.data_conf)
        tail_len = data_conf.generation_len + self.condition_len
        discr_res = run_eval_detection(
            orig=orig,
            gen=gen,
            log_dir=self.log_dir,
            data_conf=data_conf,
            dataset=self.dataset_config,
            tail_len=tail_len,
            devices=self.devices,
            verbose=self.verbose,
        )
        acc = discr_res.loc["MulticlassAccuracy"].loc["mean"]
        err = (1 - acc) * 2
        return float(err)

    def __repr__(self):
        return f"Detection score ({self.condition_len} hist)"
