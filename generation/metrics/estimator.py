from pathlib import Path
from typing import Dict, List, Optional, Union

from generation.metrics import metrics
import pandas as pd

from tmetrics.eval_density import run_eval_density
from tmetrics.eval_detection import run_eval_detection

from .types import BinaryData, CoverageData, OnlyPredData


class MetricEstimator:

    def __init__(
        self,
        gt_save_path: Optional[Union[str, Path]] = None,
        gen_save_path: Optional[Union[str, Path]] = None,
        name_prefix: str = "",
        metric_names: Optional[List[str]] = None,
        gen_len: int = 16,
        hist_len: int = 16,
        device: int = 0,
    ):

        self.gt_save_path = Path(gt_save_path) if gt_save_path else None
        self.gen_save_path = Path(gen_save_path) if gen_save_path else None
        self.name_prefix = name_prefix
        self.gen_len = gen_len
        self.hist_len = hist_len
        self.device = device
        self.name_prefix = name_prefix

        self.metrics = [getattr(metrics, name) for name in metric_names]

    def estimate(self) -> Dict[str, float]:
        return self.estimate_metrics() | self.estimate_tmetrics()

    def estimate_metrics(self) -> Dict[str, float]:
        if not self.gt_save_path or not self.gen_save_path:
            raise ValueError("Ground-truth or generated file path is not provided.")

        subject_key = "client_id"
        target_key = "event_type"

        dfs = self.get_data(
            self.gt_save_path, self.gen_save_path, subject_key, target_key
        )

        prepared_data = self.prepare_data_to_metrics(dfs, subject_key)

        # TODO: Сделать перемешивание клиентов если надо.

        results = {}

        for metric in self.metrics:
            data = prepared_data.get(metric.required_data_type)
            results[self.__ret_name("gtvsgt" + metric.name)] = metric(data["gt"])
            results[self.__ret_name("gtvsgen" + metric.name)] = metric(data["gen"])

    def estimate_tmetrics(self) -> Dict[str, float]:

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
                    gpu_ids=self.config.device,
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

    def get_data(
        self, orig_path, generated_path, subject_key, target_key
    ) -> Dict[str, pd.DataFrame]:

        dfs = {
            "gt": pd.read_csv(orig_path)[[subject_key, target_key]],
            "gen": pd.read_csv(generated_path)[[subject_key, target_key]],
        }

        return dfs

    def prepare_data_to_metrics(
        self, dfs: pd.DataFrame, subject_key: str = "client_id"
    ) -> Dict[type, Dict[str, Union[BinaryData, OnlyPredData, CoverageData]]]:
        def slice_df(
            df: pd.DataFrame, start: int, end: int | None = None
        ) -> pd.DataFrame:

            return (
                df.groupby(subject_key)
                .apply(lambda x: x[start:end])
                .reset_index(drop=True)
            )

        return {
            BinaryData: {
                "gt": BinaryData(
                    slice_df(dfs["gt"], -2 * self.gen_len, -self.gen_len),
                    slice_df(dfs["gt"], -self.gen_len, None),
                ),
                "gen": BinaryData(
                    slice_df(dfs["gt"], -self.gen_len, None),
                    slice_df(dfs["gen"], -2 * self.gen_len, -self.gen_len),
                ),
            },
            CoverageData: {
                "gt": CoverageData(
                    slice_df(dfs["gt"], -self.hist_len - self.gen_len, -self.gen_len),
                    slice_df(dfs["gt"], -self.gen_len, None),
                ),
                "gen": CoverageData(
                    slice_df(dfs["gen"], -self.hist_len - self.gen_len, -self.gen_len),
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
