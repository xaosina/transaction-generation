from abc import ABC, abstractmethod


from typing import List, Set, Dict, Union, NewType

import collections

import numpy as np
import pandas as pd

from Levenshtein import distance as lev_score
from sklearn.metrics import accuracy_score

UserStatistic = NewType("UserStatistic", Dict[int, Dict[str, Union[int, float]]])


class BaseMetric(ABC):
    generation_len: int

    @abstractmethod
    def __call__(self, y_true: pd.DataFrame, y_gen: pd.DataFrame): ...

    @abstractmethod
    def __repr__(self): ...


class BinaryMetric(BaseMetric):
    target_key: str

    @abstractmethod
    def get_scores(self, row): ...

    def __call__(self, y_true: pd.DataFrame, y_gen: pd.DataFrame):
        df = pd.concat(
            (y_true[self.target_key], y_gen[self.target_key]),
            keys=["gt", "pred"],
            axis=1,
        ).map(lambda x: x[-self.generation_len :])
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

    def __call__(self, y_gen: pd.DataFrame):
        vals = np.array(list(y_gen[self.target_key].value_counts()), dtype=float)
        total = vals.sum()
        p = vals / total

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


class HistoryAwareMetric(BaseMetric):
    target_key: str
    userwise: bool

    @abstractmethod
    def get_scores(row): ...

    def __call__(self, y_hist: pd.DataFrame, y_gen: pd.DataFrame) -> pd.DataFrame:
        if self.userwise:
            df = pd.concat(
                (y_hist[self.target_key].agg(set), y_gen[self.target_key].agg(set)),
                keys=["hists", "preds"],
                axis=1,
            )
            return df.apply(self.get_scores, axis=1).mean()
        else:
            unique_hist = y_hist[self.target_key].nunique()
            unique_pred = y_gen[self.target_key].nunique()

            return unique_pred / unique_hist


class Coverage(HistoryAwareMetric):

    def get_scores(row):
        hists, preds = row["hists"], row["preds"]
        return len(preds) / len(hists)
    
    def __repr__(self):
        return self.userwise * "Userwise" + f"Coverage on {self.target_key}"

