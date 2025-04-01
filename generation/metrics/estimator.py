from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from .evaluation.eval_density import run_eval_density
from .evaluation.eval_detection import run_eval_detection

from generation.metrics import metrics
from generation.metrics.metrics import BaseMetric

from .types import BinaryData, CoverageData, OnlyPredData


class MetricEstimator:

    def __init__(
        self,
        gt_save_path: Optional[Union[str, Path]] = None,
        gen_save_path: Optional[Union[str, Path]] = None,
        name_prefix: str = "",
        metric_names: Optional[List[str]] = None,
        detection_config: str = None,
        log_dir: str = None,
        gen_len: int = 16,
        hist_len: int = 16,
        device: int = 0,
        subject_key: str = "client_id",
        target_key: str = "event_type",
    ):

        self.gt_save_path = Path(gt_save_path) if gt_save_path else None
        self.gen_save_path = Path(gen_save_path) if gen_save_path else None
        self.name_prefix = name_prefix
        self.gen_len = gen_len
        self.hist_len = hist_len
        self.device = device
        self.name_prefix = name_prefix
        self.subject_key = subject_key
        self.target_key = target_key
        self.detection_config = detection_config
        self.log_dir = log_dir

        self.metrics: list[BaseMetric] = [
            getattr(metrics, name) for name in metric_names
        ]

    def estimate(self) -> Dict[str, float]:
        return self.estimate_metrics() | self.estimate_tmetrics()

    def estimate_metrics(self) -> Dict[str, float]:
        if not self.gt_save_path or not self.gen_save_path:
            raise ValueError("Ground-truth or generated file path is not provided.")

        dfs = self.get_data()

        prepared_data = self.prepare_data_to_metrics(dfs)

        # TODO: Сделать перемешивание клиентов если надо.

        results = {}

        for metric in self.metrics:
            data = prepared_data.get(metric.required_data_type)
            results[self.__ret_name("gtvsgt_" + metric.name)] = metric(
                data["gt"], self.subject_key, self.target_key
            )
            results[self.__ret_name("gtvsgen_" + metric.name)] = metric(
                data["gen"], self.subject_key, self.target_key
            )
        return results

    def estimate_tmetrics(self) -> Dict[str, float]:

        results = dict()
        for metric_type in ["discriminative", "density"]:
            if metric_type == "discriminative":
                discr_res = run_eval_detection(
                    data=self.gen_save_path,
                    orig=self.gt_save_path,
                    log_dir=self.log_dir,
                    data_conf=self.data_conf,
                    dataset=self.detection_config,
                    tail_len=self.tail_len,
                    match_users=False,
                    gpu_ids=self.device,
                    verbose=False,
                )

                discr_score = discr_res.loc["MulticlassAUROC"].loc["mean"]
                # discr_score = 0.0
                results[self.__ret_name("discriminative")] = float(discr_score)

            elif metric_type == "density":
                sh_tr = run_eval_density(
                    self.gen_save_path,
                    self.gt_save_path,
                    self.data_conf,
                    self.log_cols,
                    self.tail_len,
                )
                results[self.__ret_name("shape")] = float(sh_tr["shape"])
                results[self.__ret_name("trend")] = float(sh_tr["trend"])
            else:
                raise Exception(f"Unknown metric type '{metric_type}'!")

        return results

    def get_data(self) -> Dict[str, pd.DataFrame]:

        dfs = {
            "gt": pd.read_csv(
                self.gt_save_path, usecols=[self.subject_key, self.target_key]
            ),
            "gen": pd.read_csv(
                self.gen_save_path, usecols=[self.subject_key, self.target_key]
            ),
        }

        return dfs

    def prepare_data_to_metrics(
        self, dfs: pd.DataFrame
    ) -> Dict[type, Dict[str, Union[BinaryData, OnlyPredData, CoverageData]]]:
        def slice_df(
            df: pd.DataFrame, start: int, end: int | None = None
        ) -> pd.DataFrame:

            return (
                df.groupby(self.subject_key)
                .apply(lambda x: x[start:end])
                .reset_index(drop=True)
            )

        return {
            BinaryData: {
                "gt": BinaryData(
                    slice_df(dfs["gt"], -self.gen_len, None),
                    slice_df(dfs["gt"], -2 * self.gen_len, -self.gen_len),
                ),
                "gen": BinaryData(
                    slice_df(dfs["gt"], -self.gen_len, None),
                    slice_df(dfs["gen"], -self.gen_len, None),
                ),
            },
            CoverageData: {
                "gt": CoverageData(
                    slice_df(dfs["gt"], None, -self.gen_len),
                    slice_df(dfs["gt"], -self.gen_len, None),
                ),
                "gen": CoverageData(
                    slice_df(dfs["gen"], None, -self.gen_len),
                    slice_df(dfs["gen"], -self.gen_len, None),
                ),
            },
            OnlyPredData: {
                "gt": OnlyPredData(slice_df(dfs["gt"], -self.gen_len, None)),
                "gen": OnlyPredData(slice_df(dfs["gen"], -self.gen_len, None)),
            },
        }

    def __ret_name(self, name: str) -> str:
        return self.name_prefix + name
