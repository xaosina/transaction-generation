import pandas as pd

from .types import BinaryData, CoverageData, OnlyPredData
from dataclasses import dataclass, field
from pathlib import Path

# metrics
from tmetrics.eval_density import run_eval_density
from tmetrics.eval_detection import run_eval_detection
import metrics


@dataclass
class MetricsConfig:
    gen_len: int = 16
    hist_len: int = 16
    device: int = 0
    metric_names: list[str]
    gt_save_path: str = ""
    gen_save_path: str = ""
    name_prefix: str = ""


def get_metrics(cfg: MetricsConfig = None):
    return MetricEstimator(
        cfg.gt_save_path,
        cfg.gen_save_path,
        cfg.metric_names,
        cfg.gen_len,
        cfg.hist_len,
        cfg.device,
        cfg.name_prefix,
    )


class MetricEstimator:

    def __init__(
        self,
        gt_save_path: str | Path = None,
        gen_save_path: str | Path = None,
        metric_names: list = None,
        gen_len: int = 16,
        hist_len: int = 16,
        device: int = 0,
        name_prefix: str = "",
    ):

        self.gt_save_path = gt_save_path
        self.gen_save_path = gen_save_path
        self.name_prefix = name_prefix
        self.gen_len = gen_len
        self.hist_len = hist_len
        self.device = device
        self.name_prefix = name_prefix

        if metric_names is None:
            raise ValueError("List of metrics is empty")
        else:
            self.metrics = [getattr(metrics, name) for name in metric_names]

    def estimate(self):
        return self.__estimate_metrics() | self.__estimate_tmetrics()

    def __estimate_metrics(self, orig_path, generated_path, metrics=None):

        subject_key = "client_id"
        target_key = "event_type"
        orig_path, generated_path = self.gt_save_path, self.gen_save_path

        dfs = self.__get_data(orig_path, generated_path, subject_key, target_key)

        prepared_data = self.__prepare_data_to_metrics(dfs)

        # TODO: Сделать перемешивание клиентов если надо.

        results = {}

        for metric in self.metrics:
            data = prepared_data.get(metric.required_data_type)
            results[self.__ret_name("gtvsgt" + metric.name)] = metric(data["gt"])
            results[self.__ret_name("gtvsgen" + metric.name)] = metric(data["gen"])

    def __estimate_tmetrics(self):

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

    def __get_data(self, orig_path, generated_path, subject_key, target_key):

        dfs = {
            "gt": pd.read_csv(orig_path)[[subject_key, target_key]],
            "gen": pd.read_csv(generated_path)[[subject_key, target_key]],
        }

        return dfs

    def __prepare_data_to_metrics(
        self, dfs: pd.DataFrame, subject_key: str = "client_id"
    ):
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

    def __ret_name(self, name):
        return self.name_prefix + name
