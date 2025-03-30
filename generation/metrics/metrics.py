
from abc import ABC, abstractmethod


from typing import List, Set, Dict, Union, NewType

import collections

import numpy as np
import pandas as pd

from Levenshtein import distance as lev_score
from sklearn.metrics import accuracy_score
from .types import BinaryData, CoverageData, OnlyPredData

UserStatistic = NewType('UserStatistic', Dict[int, Dict[str, Union[int, float]]])


class BaseMetric(ABC):

    required_data_type = None

    def __init__(self, name: str):
        self.name = name
        self.result = None

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.result = self.compute(*args, **kwargs)
        return self.result

    def __repr__(self):
        return f"<Metric {self.name}: {self.result}>"


class Levenstein(BaseMetric):

    required_data_type = BinaryData

    def compute(self, data: BinaryData, seq_key, target_key):

        preds = data.y_pred.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        gts = data.y_true.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        df = pd.concat((gts, preds), keys=["gt", "pred"], axis=1)
        
        def get_scores(row):
            gt, pred = row.iloc[0]
            lev_m = 1 - lev_score(gt, pred) / len(pred)
            return pd.Series(data=[lev_m], index=["lev_score"])
        
        return df.groupby(level=0).apply(get_scores).mean()


class Accuracy(BaseMetric):

    required_data_type = BinaryData

    def compute(self, data: BinaryData, seq_key, target_key):

        preds = data.y_pred.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        gts = data.y_true.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        df = pd.concat((gts, preds), keys=["gt", "pred"], axis=1)
        
        def get_scores(row):
            gt, pred = row.iloc[0]
            acc_m = accuracy_score(gt, pred)
            return pd.Series(data=[acc_m ], 
                            index=["accuracy"])
        
        return df.groupby(level=0).apply(get_scores).mean()


class F1Metric(BaseMetric):

    required_data_type = BinaryData

    def __init__(self, name, average='macro'):
        super().__init__(name)
        self.__average = average

    def f1_score_macro_unorder(cls_metric: UserStatistic) -> float:
        f1_score_sum = 0
        for _, value in cls_metric.items():
            f1_score_sum += (
                2 * value['precision'] * value['recall'] / (value['precision'] + value['recall'])
                if (value['precision'] + value['recall']) != 0 
                else 0
            )
        return f1_score_sum / len(cls_metric)


    def f1_score_micro_unorder(cls_metric: UserStatistic) -> float:
        tp, fp, fn = 0, 0, 0
        for _, value in cls_metric.items():
            tp += value['tp']
            fp += value['fp']
            fn += value['fn']
        return 2 * tp / (2 * tp + fp + fn)

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
            cls_metric[cls] = {'precision': precision, 'recall': recall,'tp': tp, 'fp': fp, 'fn': fn}
        return cls_metric
    

    def compute(self, data: BinaryData, seq_key, target_key):
     
        preds = data.y_pred.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        gts = data.y_true.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
        df = pd.concat((gts, preds), keys=["gt", "pred"], axis=1)
        
        def get_scores(row):
            gt, pred = row.iloc[0]
            f1_function = self.f1_score_macro_unorder if self.__average == 'macro' else self.f1_score_micro_unorder
            f1 = f1_function(self.get_statistics(gt, pred))

            return pd.Series(data=[f1], index=[f"f1_{self.__average}"])
        
        return df.groupby(level=0).apply(get_scores).mean()
    

class Gini(BaseMetric):

    required_data_type = OnlyPredData

    def compute(self, data:OnlyPredData, seq_key, target_key):
        
        vals = np.array(list(data.y_pred[target_key].value_counts()), dtype=float)
        total = vals.sum()
        p = vals / total

        p_sorted = np.sort(p)
        n = len(p_sorted)
        
        gini = 0
        for k in range(n):
            gini += (2*(k+1) - n - 1) * p_sorted[k]
        gini = gini / (n - 1)
        
        return gini


class ShannonEntropy(BaseMetric):

    required_data_type = OnlyPredData

    def compute(self, data:OnlyPredData, seq_key, target_key):
        vals = np.array(list(data.y_pred[target_key].value_counts()), dtype=float)
        total = vals.sum()
        p = vals / total
        
        shannon_entropy = - np.sum(p * np.log2(p))
        
        return shannon_entropy


class Coverage(BaseMetric):

    required_data_type = CoverageData

    def __init__(self, name, userwise=False):
        super().__init__(name)
        self.__userwise = userwise

    def compute(self, *args, **kwargs):
        if self.__userwise:
            self.__compute_whole(*args, **kwargs)
        else:
            self.__compute_userwise(*args, **kwargs)

    def __compute_whole(self, data: CoverageData, _, target_key):
        unique_hist = data.y_hist[target_key].nunique() 
        unique_pred = data.y_pred[target_key].nunique()
        return unique_pred / unique_hist

    def __compute_userwise(self, data:CoverageData, seq_key, target_key):
        
        hists = data.y_hist.groupby(seq_key)[target_key].apply(lambda s: set(s))
        preds = data.y_pred.groupby(seq_key)[target_key].apply(lambda s: set(s))
        df = pd.concat((hists, preds), keys=["hists", "preds"], axis=1)
        
        def get_scores(row):
            hists, preds = row.iloc[0]
            coverage = len(preds) / len(hists)
            return coverage
        
        return df.groupby(level=0).apply(get_scores).mean()

