"""
Coverage and Gini Index were implemented following the approach described in https://www.mdpi.com/2078-2489/16/2/151
"""

import collections
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NewType, Union

import numpy as np
import pandas as pd
from Levenshtein import distance as lev_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy, gaussian_kde
from sdmetrics.reports.single_table import QualityReport
from sklearn.metrics import accuracy_score, r2_score as r2_sklearn, f1_score

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

        cat_cards = self.data_conf.cat_cardinalities or {}
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
        return r2_sklearn(gt, pred)

    def _compute_accuracy(self, row):
        gt, pred = row["gt"], row["pred"]
        return accuracy_score(gt, pred)

    def __repr__(self):
        return "Reconstruction"


def get_perfect_score(score, max_shift, gen_len):
    cost = np.transpose(score.mean(-1), (2, 0, 1))  # [B, L, L]
    if max_shift >= 0:
        i_indices = np.arange(gen_len)[:, None]  # L, 1
        j_indices = np.arange(gen_len)
        distance_from_diagonal = np.abs(i_indices - j_indices)  # L, L
        mask_outside_band = distance_from_diagonal > max_shift
        cost[:, mask_outside_band] = np.inf

    L, B, D = score.shape[1:]
    perfect_score = np.zeros_like(score, shape=(B, D))
    for b in range(B):
        workers, tasks = linear_sum_assignment(cost[b], maximize=True)
        perfect_score[b] = score[workers, tasks, b].sum(0)  # D
    return perfect_score.mean(0)


def r2_score(true_num, pred_num):
    """R2 score for numerical
    Input:
        true_num: [L, B, D]
        pred_num: [L, B, D]
    """
    gen_len = true_num.shape[0]
    denominator = ((true_num - true_num.mean(0)) ** 2).sum(
        axis=0, dtype=np.float64
    )  # B, D
    nominator = (pred_num[:, None] - true_num[None, :]) ** 2  # [L, L, B, D]
    denominator[nominator.sum(0).sum(0) == 0] = 1
    nominator[:, :, (denominator == 0)] = 1 / gen_len
    denominator[denominator == 0] = 1

    return 1 / gen_len - (nominator / denominator)  # [L, L, B, D]


def r1_score(true_num, pred_num):
    """R1 score for numerical(MAE analog for R2)
    Input:
        true_num: [L, B, D]
        pred_num: [L, B, D]
    """
    gen_len = true_num.shape[0]
    denominator = np.abs(true_num - np.median(true_num, 0)[0]).sum(
        axis=0, dtype=np.float64
    )  # B, D
    nominator = np.abs(pred_num[:, None] - true_num[None, :])  # [L, L, B, D]
    denominator[nominator.sum(0).sum(0) == 0] = 1
    nominator[:, :, (denominator == 0)] = 1 / gen_len
    denominator[denominator == 0] = 1

    return 1 / gen_len - (nominator / denominator)  # [L, L, B, D]


def smape_score(true_num, pred_num):
    """1 - sMAPE score
    Input:
        true_num: [L, B, D]
        pred_num: [L, B, D]
    """
    gen_len = true_num.shape[0]
    nominator = np.abs(pred_num[:, None] - true_num[None, :])  # [L, L, B, D]
    denominator = np.abs(pred_num[:, None]) + np.abs(true_num[None, :])  # [L, L, B, D]
    denominator[nominator == 0] = 1
    smape = nominator / denominator / gen_len  # L, L, B, D

    return 1 / gen_len - smape  # [L, L, B, D]


@dataclass
class OTD(BaseMetric):
    max_shift: int = -1
    num_metric: str = "r1"

    def __call__(self, orig, gen):
        assert (orig.columns == gen.columns).all()
        results = {}
        orig, gen = deepcopy(orig), deepcopy(gen)

        # Time to diff
        # time_name = self.data_conf.time_name
        # orig[time_name] = orig[time_name].map(lambda x: np.diff(x, 1))
        # gen[time_name] = gen[time_name].map(lambda x: np.diff(x, 1))

        # Cut gen_len
        gen_len = self.data_conf.generation_len
        seq_cols = self.data_conf.focus_on
        orig[seq_cols] = orig[seq_cols].map(lambda x: x[-gen_len:])
        gen[seq_cols] = gen[seq_cols].map(lambda x: x[-gen_len:])

        # Prepare num arrays
        num_metric = np.empty((gen_len, gen_len, orig.shape[0], 0))
        if self.data_conf.focus_num:
            true_num = orig[self.data_conf.focus_num].values  # B, D
            pred_num = gen[self.data_conf.focus_num].values  # B, D

            B, D = true_num.shape
            true_num = np.transpose(
                np.concatenate(true_num.ravel()).reshape((B, D, gen_len)),
                (2, 0, 1),
            )  # [gen_len, B, D]
            pred_num = np.transpose(
                np.concatenate(pred_num.ravel()).reshape((B, D, gen_len)),
                (2, 0, 1),
            )  # [gen_len, B, D]

            if self.num_metric == "r2":
                num_metric = r2_score(true_num, pred_num)
            elif self.num_metric == "r1":
                num_metric = r1_score(true_num, pred_num)
            elif self.num_metric == "smape":
                num_metric = smape_score(true_num, pred_num)

        # Prepare cat arrays
        accuracy = np.empty((gen_len, gen_len, orig.shape[0], 0))
        if self.data_conf.focus_cat:
            true_cat = orig[self.data_conf.focus_cat].values  # B, D
            pred_cat = gen[self.data_conf.focus_cat].values  # B, D

            B, D = true_cat.shape
            true_cat = np.transpose(
                np.concatenate(true_cat.ravel()).reshape((B, D, gen_len)),
                (2, 0, 1),
            )  # [gen_len, B, D]
            pred_cat = np.transpose(
                np.concatenate(pred_cat.ravel()).reshape((B, D, gen_len)),
                (2, 0, 1),
            )  # [gen_len, B, D]
            accuracy = (
                pred_cat[:, None] == true_cat[None, :]
            ) / gen_len  # [L, L, B, D]

        full_score = np.concatenate([num_metric, accuracy], axis=-1)  # [L, L, B, D]
        perfect_score = get_perfect_score(full_score, self.max_shift, gen_len)
        results = dict(
            zip(self.data_conf.focus_num + self.data_conf.focus_cat, perfect_score)
        )

        return {
            "overall": np.mean(list(results.values())),
            **results,
        }

    def __repr__(self):
        res = "OTD"
        if self.max_shift >= 0:
            res += f" {self.max_shift}"
        if self.num_metric != "r1":
            res += f" {self.num_metric}"
        return res


class KLDiv(BaseMetric):

    EPS = 1e-8  # сглаживание, чтобы не было нулевых вероятностей
    N_BINS = 50

    def __call__(self, orig, gen):
        assert (orig.columns == gen.columns).all()

        cat_cards = self.data_conf.cat_cardinalities or {}
        results = {}

        for col in self.data_conf.focus_on:
            df = pd.concat(
                (orig[col], gen[col]),
                keys=["gt", "pred"],
                axis=1,
            ).map(lambda x: x[-self.data_conf.generation_len :])

            results[col] = df.apply(
                lambda row: self._compute_kl(row, bins=cat_cards.get(col, None)),
                axis=1,
            ).mean()

        return {"overall": np.mean(list(results.values())), **results}

    def _compute_kl(self, row, bins=None):
        if bins is None:
            bins = self.N_BINS

        gt, pred = row["gt"], row["pred"]

        # общий диапазон
        range_ = (
            (min(gt.min(), pred.min()), max(gt.max(), pred.max()))
            if bins is not None
            else (0, bins)
        )

        p, _ = np.histogram(gt, bins=self.N_BINS, range=range_, density=False)
        q, _ = np.histogram(pred, bins=self.N_BINS, range=range_, density=False)

        p = p.astype(float) + self.EPS
        q = q.astype(float) + self.EPS
        p /= p.sum()
        q /= q.sum()
        return entropy(p, q)

    def __repr__(self):
        return "KLDiv"


class JSDiv(BaseMetric):

    EPS = 1e-8
    N_BINS = 50

    def __call__(self, orig, gen):
        assert (orig.columns == gen.columns).all()

        cat_cards = self.data_conf.cat_cardinalities or {}
        results = {}

        for col in self.data_conf.focus_on:
            df = pd.concat(
                (orig[col], gen[col]),
                keys=["gt", "pred"],
                axis=1,
            ).map(lambda x: x[-self.data_conf.generation_len :])

            results[col] = df.apply(
                lambda row: self._compute_kl(row, bins=cat_cards.get(col, None)),
                axis=1,
            ).mean()

        return {"overall": np.mean(list(results.values())), **results}

    def _compute_kl(self, row, bins=None):
        if bins is None:
            bins = self.N_BINS

        gt, pred = row["gt"], row["pred"]

        range_ = (
            (min(gt.min(), pred.min()), max(gt.max(), pred.max()))
            if bins is not None
            else (0, bins)
        )

        p, _ = np.histogram(gt, bins=self.N_BINS, range=range_, density=False)
        q, _ = np.histogram(pred, bins=self.N_BINS, range=range_, density=False)

        p = p.astype(float) + self.EPS
        q = q.astype(float) + self.EPS
        p /= p.sum()
        q /= q.sum()

        m = 0.5 * (p + q)

        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    def __repr__(self):
        return "JSDiv"


@dataclass
class BinaryMetric(BaseMetric):

    @abstractmethod
    def get_scores(self, row): ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        df = pd.concat(
            (orig[self.data_conf.target_token], gen[self.data_conf.target_token]),
            keys=["gt", "pred"],
            axis=1,
        ).map(lambda x: x[-self.data_conf.generation_len :])
        return df.apply(self.get_scores, axis=1).mean()


@dataclass
class NDCG(BinaryMetric):
    k: str = 1

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        set_gt = set(gt)
        pred_len = min(self.k, len(pred))
        ground_truth_len = min(self.k, len(gt))
        denom = [1 / np.log2(i + 2) for i in range(self.k)]
        dcg = sum(denom[i] for i in range(pred_len) if pred[i] in set_gt)
        idcg = sum(denom[:ground_truth_len])
        return dcg / idcg

    def __repr__(self):
        return f"NDCG@{self.k} on {self.data_conf.target_token}"


@dataclass
class Levenshtein(BinaryMetric):
    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        lev_m = 1 - lev_score(gt, pred) / max(len(pred), len(gt))
        return lev_m

    def __repr__(self):
        return f"Levenstein on {self.data_conf.target_token}"


@dataclass
class Accuracy(BinaryMetric):

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        acc_m = accuracy_score(gt, pred)
        return acc_m

    def __repr__(self):
        return f"Accuracy on {self.data_conf.target_token}"


@dataclass
class F1Score(BinaryMetric):
    average: str = "macro"

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        f1 = f1_score(gt, pred, average=self.average)
        return f1

    def __repr__(self):
        return f"F1 score {self.average} on {self.data_conf.target_token}"


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
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        return cls_metric

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        stats = self.get_statistics(gt, pred)
        return pd.Series(stats)

    def __repr__(self):
        return f"PR on {self.data_conf.target_token}"


@dataclass
class Precision(PR):
    average: str = "macro"

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        stats = self.get_statistics(gt, pred)
        perfs = stats.values()
        if self.average == "macro":
            ret = sum(m["Precision"] for m in perfs) / len(perfs)
        else:
            total_tp = sum(m["tp"] for m in perfs)
            total_fp = sum(m["fp"] for m in perfs)
            ret = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0

        return ret

    def __repr__(self):
        return f"Precision {self.average} on {self.data_conf.target_token}"


@dataclass
class Recall(PR):
    average: str = "macro"

    def get_scores(self, row):
        gt, pred = row["gt"], row["pred"]
        stats = self.get_statistics(gt, pred)
        perfs = stats.values()
        if self.average == "macro":
            ret = sum(m["Recall"] for m in perfs) / len(perfs)
        else:
            total_tp = sum(m["tp"] for m in perfs)
            total_fn = sum(m["fn"] for m in perfs)
            ret = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        return ret

    def __repr__(self):
        return f"Recall {self.average} on {self.data_conf.target_token}"


@dataclass
class MultisetF1Metric(BinaryMetric):
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
        return f"MultisetF1_{self.average} on {self.data_conf.target_token}"


@dataclass
class DistributionMetric(BaseMetric):
    overall: bool = False

    @abstractmethod
    def get_scores(self, row) -> pd.Series | float: ...

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        df = pd.concat(
            (orig[self.data_conf.target_token], gen[self.data_conf.target_token]),
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
        return self.overall * "Overall " + f"Gini on {self.data_conf.target_token}"


@dataclass
class ShannonEntropy(StatisticMetric):

    def get_statistic(self, p):
        p = p[p > 0]
        shannon_entropy = -np.sum(p * np.log2(p))

        return shannon_entropy

    def __repr__(self):
        return (
            self.overall * "Overall "
            + f"Shannon entropy on {self.data_conf.target_token}"
        )


@dataclass
class GenVsHistoryMetric(BaseMetric):
    overall: bool = False

    @abstractmethod
    def get_scores(row): ...

    def score_for_df(self, df):
        gen_len = self.data_conf.generation_len
        hist = df[self.data_conf.target_token].map(lambda x: x[:-gen_len])
        preds = df[self.data_conf.target_token].map(lambda x: x[-gen_len:])
        df = pd.concat((hist, preds), keys=["hists", "preds"], axis=1)
        if self.overall:
            df = df.agg(lambda x: [np.concatenate(x.values)], axis=0)
        return df.apply(self.get_scores, axis=1).mean()

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        gen_score = self.score_for_df(gen)
        orig_score = self.score_for_df(orig)
        # score = 1 - (1 + abs(gen_score - orig_score))
        return {"gen": gen_score, "orig": orig_score}  # "score": score,


@dataclass
class DiversityIndex(BaseMetric):
    def _simpson_cat(self, values):
        simpson = []
        for b in range(len(values)):
            _, counts = np.unique(values[b], return_counts=True)
            p = counts / len(values[b])
            simpson += [1 - np.sum(p**2)]
        return np.mean(simpson)

    def _simpson_num(self, values):
        simpson = []
        flattened = np.concatenate(values.values)
        x_min, x_max = flattened.min(), flattened.max()
        x = np.linspace(x_min, x_max, 512)
        for b in range(len(values)):
            kde = gaussian_kde(values[b])
            p = kde(x)
            integral = np.trapz(p**2, x)
            simpson += [integral]
        return np.mean(simpson)

    def __call__(self, orig: pd.DataFrame, gen: pd.DataFrame):
        data = {}
        data["orig"] = orig[self.data_conf.focus_on].map(
            lambda x: x[-self.data_conf.generation_len :]
        )
        data["gen"] = gen[self.data_conf.focus_on].map(
            lambda x: x[-self.data_conf.generation_len :]
        )
        collect = {"orig": [], "gen": []}
        res = {}
        for sample in collect:
            for feature in self.data_conf.focus_cat:
                res[f"{sample}_{feature}"] = self._simpson_cat(data[sample][feature])
                collect[sample] += [res[f"{sample}_{feature}"]]
            # for feature in self.data_conf.focus_num:
            #     if self.data_conf.time_name != feature:
            #         continue
            #     res[f"{sample}_{feature}"] = self._simpson_num(data[sample][feature])
            #     collect[sample] += [res[f"{sample}_{feature}"]]
            res[sample] = np.mean(collect[sample])
        return res

    def __repr__(self):
        return "DiversityIndex"

@dataclass
class CardinalityCoverage(GenVsHistoryMetric):
    def get_scores(self, row):
        hists, preds = row["hists"], row["preds"]
        return len(np.unique(preds)) / len(np.unique(hists))

    def __repr__(self):
        return (
            self.overall * "Overall "
            + f"CardinalityCoverage on {self.data_conf.target_token}"
        )


@dataclass
class Cardinality(GenVsHistoryMetric):
    def get_scores(self, row):
        hists, preds = row["hists"], row["preds"]
        return len(np.unique(preds))

    def __repr__(self):
        return (
            self.overall * "Overall " + f"Cardinality on {self.data_conf.target_token}"
        )


@dataclass
class NoveltyScore(GenVsHistoryMetric):
    def get_scores(self, row):
        hists, preds = row["hists"], row["preds"]
        hists, preds = set(hists), set(preds)
        return len(preds - hists) / len(preds)

    def __repr__(self):
        return (
            self.overall * "Overall " + f"NoveltyScore on {self.data_conf.target_token}"
        )


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
    '''
    Run GRU classifier to detect generated sequences.
    '''
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
            dataset=f"detection/{self.data_conf.dataset_name}",
            tail_len=tail_len,
            devices=self.devices,
            verbose=self.verbose,
        )
        acc = discr_res.loc["MulticlassAccuracy"].loc["mean"]
        err = (1 - acc) * 2
        return float(err)

    def __repr__(self):
        return f"Detection score ({self.condition_len} hist)"
