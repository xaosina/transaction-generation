import time
import wandb

import pandas as pd
import numpy as np

from tabsyn.utils.other_utils import (
    dictprettyprint,
)

# metrics
from tmetrics.eval_density import run_eval_density
from tmetrics.eval_detection import run_eval_detection

from .imp import METRICS

from .types import BinaryData, CoverageData, OnlyPredData


class MetricEstimator:

    def __init__(self, gt_save_path, gen_save_path, config, logger, train_state):
        self.gt_save_path = gt_save_path
        self.gen_save_path = gen_save_path
        self.name_postfix = ""
        self.name_prefix = ""
        self.config = config
        self.logger = logger
        self.train_state = train_state

        self.bmetrics = [metric(metric.__name__) for metric in METRICS]

    def set_name_fixes(self, name_postfix="", name_prefix=""):
        self.name_postfix = name_postfix
        self.name_prefix = name_prefix

    def __get_data(self, orig_path, generated_path, subject_key, target_key):
        total_len = self.config["EST_GEN_LEN"] + self.config["EST_HIST_LEN"]

        def process(path):
            df = pd.read_csv(path)[[subject_key, target_key]]
            df = df.groupby(subject_key).tail(total_len).reset_index(drop=True)
            groups = df.groupby(subject_key)
            assert (groups.size() >= total_len).all(), "Не хватает записей для некоторых групп"
            return {
                "hist": groups.head(self.config["EST_HIST_LEN"]).reset_index(drop=True),
                "preds": groups.tail(self.config["EST_GEN_LEN"]).reset_index(drop=True)
            }

        dfs = {
            "gt": process(orig_path),
            "gen": process(generated_path)
        }

        assert set(dfs['gt']['preds'][subject_key]) == set(dfs['gen']['preds'][subject_key]), (
            "Warning, subject ids in dataframes does not match"
        )
        return dfs

    
    def __prepare_data_to_metrics(self, dfs):
        seq_len = min(self.config["EST_GEN_LEN"], self.config["EST_HIST_LEN"])
        
        return {
            BinaryData: {
                "gt": BinaryData(
                    dfs["gt"]["preds"].groupby("client_id").head(seq_len).reset_index(drop=True),
                    dfs["gt"]["hist"].groupby("client_id").tail(seq_len).reset_index(drop=True),
                ),
                "gen": BinaryData(dfs['gt']['pred'], dfs['gen']['pred']),

            },
            CoverageData: {
                "gt": CoverageData(dfs['gt']['hist'], dfs['gt']['pred']),
                "gen": CoverageData(dfs['gen']['hist'], dfs['gen']['pred']),
            },
            OnlyPredData: {
                "gt": OnlyPredData(dfs['gt']['pred']),
                "gen": OnlyPredData(dfs['gen']['pred']),
            }
        }


    def __estimate_metrics(self, orig_path, generated_path, metrics=None, shuffle_clients=False, seed=0):
        if metrics is None:
            return None

        subject_key = 'client_id'
        target_key = 'event_type'

        dfs = self.__get_data(orig_path, generated_path, subject_key, target_key)
        
        prepared_data = self.__prepare_data_to_metrics(dfs, )
        
        # TODO: Сделать перемешивание клиентов если надо.
        # if shuffle_clients:
        #     rng = np.random.default_rng(seed=seed)
        #     dfs["gen"]["pred"].client_id = rng.permutation(dfs["gen"]["pred"][subject_key].values)

        results = {}

        for metric in self.metrics:
            data = prepared_data.get(metric.required_data_type)
            results[self.__ret_name('gtvsgt' + metric.name)] = metric(data['gt'])
            results[self.__ret_name('gtvsgen' + metric.name)] = metric(data['gen'])

    def estimate(self):
        start_time = time.perf_counter()
        metrics = self.__estimate_metrics() | self.__estimate_tmetrics()
        self.logger.info(f"Epoch: {self.train_state['epoch']}, some metrics eval:\n")
        self.logger.info(dictprettyprint(metrics))
        self.logger.info(f"---\ntime estimation: {time.perf_counter() - start_time} s")

    def __estimate_tmetrics(
        self,
        metric_type: str,
    ):

        orig_path, generated_path = self.gt_save_path, self.gen_save_path

        results = dict()
        for metric_type in ["discriminative", "density"]:
            if metric_type == "discriminative":
                discr_res = run_eval_detection(
                    "not_tabsyn",
                    generated_path,
                    orig_path,
                    0,
                    dataset="mbd_short",
                    match_users=False,
                    gpu_ids=self.config["device"],
                    verbose=False,
                )

                discr_score = discr_res.loc["MulticlassAUROC"].loc["mean"]
                results[self.__ret_name("discriminative")] = float(discr_score)
                # discr_score = 0
                # self.train_state["curr_discri"] = float(discr_score)
                # est_res_dict[self.__ret_name("discriminative")] = float(discr_score)

            elif metric_type == "density":
                sh_tr = run_eval_density(generated_path, orig_path)
                results[self.__ret_name("shape")] = float(sh_tr["shape"])
                results[self.__ret_name("trend")] = float(sh_tr["trend"])
            else:
                raise Exception(f"Unknown metric type '{metric_type}'!")

    def __ret_name(self, name):
        return self.name_prefix + name + self.name_postfix