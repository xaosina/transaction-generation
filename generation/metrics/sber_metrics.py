from typing import List, Set, Dict, Union, NewType
import collections

import numpy as np
import pandas as pd

from Levenshtein import distance as lev_score
from sklearn.metrics import accuracy_score


UserStatistic = NewType('UserStatistic', Dict[int, Dict[str, Union[int, float]]])


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
 

def get_metric_per_mcc(users_stat: List[UserStatistic], name='recall') -> Dict[int, List[Union[int, float]]]:
    mcc_metric = {}
    for user in users_stat:
        for (mcc, metric) in user.items():
            mcc_metric.setdefault(mcc, []).append(metric[name])
    return mcc_metric


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


def recall_per_mcc(users_stat: List[UserStatistic]) -> Dict[int, float]:
    recall_per_mcc = {}
    mcc_recalls = get_metric_per_mcc(users_stat)
    for (mcc, data) in mcc_recalls.items():
        recall_per_mcc[mcc] = np.mean(data)
    return recall_per_mcc


def recall_macro(recall_per_mcc: Dict[int, float]) -> float:
    return np.mean(list(recall_per_mcc.values()))


def calc_f1_scores(df_gt, df_pred, seq_key, target_key, shuffle_clients=False, random_state=0):
    if set(df_pred[seq_key]) != set(df_gt[seq_key]):
        print("Warning, user ids in dataframes does not match")
    
    if shuffle_clients:
        rng = np.random.default_rng(seed=random_state)
        df_pred.client_id = rng.permutation(df_pred[seq_key].values)
    
    preds = df_pred.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
    gts = df_gt.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
    df = pd.concat((gts, preds), keys=["gt", "pred"], axis=1)
    
    def get_scores(row):
        gt, pred = row.iloc[0]
        f1_macro = f1_score_macro_unorder(get_statistics(gt, pred))
        f1_micro = f1_score_micro_unorder(get_statistics(gt, pred))
        return pd.Series(data=[f1_macro, f1_micro], index=["f1_macro", "f1_micro"])
    
    return df.groupby(level=0).apply(get_scores).mean()
    
    
def calc_lev_scores(df_gt, df_pred, seq_key, target_key, shuffle_clients=False, random_state=0):
    if set(df_pred.client_id) != set(df_gt.client_id):
        print("Warning, user ids in dataframes does not match")
        
    if shuffle_clients:
        rng = np.random.default_rng(seed=random_state)
        df_pred.client_id = rng.permutation(df_pred[seq_key].values)

    preds = df_pred.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
    gts = df_gt.groupby(seq_key)[target_key].apply(lambda s: s.tolist())
    df = pd.concat((gts, preds), keys=["gt", "pred"], axis=1)
    
    def get_scores(row):
        gt, pred = row.iloc[0]
        lev_m = 1 - lev_score(gt, pred) / len(pred)
        acc_m = accuracy_score(gt, pred)
        return pd.Series(data=[lev_m, acc_m ], 
                         index=["lev_score", "accuracy"])
    
    return df.groupby(level=0).apply(get_scores).mean()
                            