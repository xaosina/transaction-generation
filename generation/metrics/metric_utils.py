import time
import wandb

import pandas as pd

from tabsyn.utils.other_utils import (
    dictprettyprint,
)

# metrics
from tmetrics.eval_density import run_eval_density
from tmetrics.eval_detection import run_eval_detection
from tmetrics.eval_tct import run_eval_tct
from sber_metrics import calc_f1_scores as sber_f1
from sber_metrics import calc_lev_scores as we_dist


class MetricEstimator:
    
    METRICS = ["sber", "discriminative", "density"]

    def __init__(self, gt_save_path, gen_save_path, config, logger, train_state):
        self.gt_save_path = gt_save_path
        self.gen_save_path = gen_save_path
        self.name_postfix = ""
        self.name_prefix = ""
        self.config = config
        self.logger = logger
        self.train_state = train_state

    def set_name_fixes(self, name_postfix="", name_prefix=""):
        self.name_postfix = name_postfix
        self.name_prefix = name_prefix

    def estimate(self):

        for m_type in self.METRICS:
            self.__estimate_metric(m_type)

    def __estimate_metric(
        self,
        metric_type: str,
    ):
        start_time = time.perf_counter()

        orig_path, generated_path = self.gt_save_path, self.gen_save_path

        est_res_dict = dict()
        if metric_type == "discriminative":
            # discr_res = run_eval_detection(
            #     "not_tabsyn",
            #     generated_path,
            #     orig_path,
            #     0,
            #     dataset="mbd_short",
            #     match_users=False,
            #     gpu_ids=config["gpu_ids"],
            #     verbose=not config["no_tqdm_pbar"],
            # )

            # discr_score = discr_res.loc["MulticlassAUROC"].loc["mean"]
            # train_state["curr_discri"] = float(discr_score)
            # est_res_dict[self.__ret_name("discriminative")] = float(discr_score)
            discr_score = 0
            self.train_state["curr_discri"] = float(discr_score)
            est_res_dict[self.__ret_name("discriminative")] = float(discr_score)

        elif metric_type == "density":
            sh_tr = run_eval_density(generated_path, orig_path)
            est_res_dict[self.__ret_name("shape")] = float(sh_tr["shape"])
            est_res_dict[self.__ret_name("trend")] = float(sh_tr["trend"])

        elif metric_type == "sber":
            dfs = dict(
                gt={"hist": None, "preds": None},
                gen={"hist": None, "preds": None},
            )

            for parti, path in zip(["gt", "gen"], [orig_path, generated_path]):
                df = pd.read_csv(path)
                df = (
                    df[["client_id", "event_type"]]
                    .groupby("client_id")
                    .tail(self.config["EST_GEN_LEN"] + self.config["EST_HIST_LEN"])
                    .reset_index(drop=True)
                )
                assert (df.groupby("client_id").size() >= self.config["EST_GEN_LEN"] + self.config["EST_HIST_LEN"]).all()
                dfs[parti]["hist"] = df.groupby("client_id").head(self.config["EST_HIST_LEN"]).reset_index(drop=True)
                dfs[parti]["preds"] = df.groupby("client_id").tail(self.config["EST_GEN_LEN"]).reset_index(drop=True)

            # variant 1: whole gen vs whole gt

            res_sr_wh = sber_f1(
                dfs["gt"]["preds"], dfs["gen"]["preds"], "client_id", "event_type"
            )
            res_lev_wh = we_dist(
                dfs["gt"]["preds"], dfs["gen"]["preds"], "client_id", "event_type"
            )
            
            est_res_dict[self.__ret_name("et_f1_macro_whole")] = res_sr_wh[
                "f1_macro"
            ].item()
            est_res_dict[self.__ret_name("et_f1_micro_whole")] = res_sr_wh[
                "f1_micro"
            ].item()
            est_res_dict[self.__ret_name("et_accuracy_whole")] = res_lev_wh[
                "accuracy"
            ].item()
            est_res_dict[self.__ret_name("et_lev_whole")] = res_lev_wh[
                "lev_score"
            ].item()
            
            # variant 2: history gt vs preds gt
            seq_len = min(self.config["EST_GEN_LEN"], self.config["EST_HIST_LEN"])
            df_gt_history = (
                dfs["gt"]["hist"]
                .groupby("client_id")
                .tail(seq_len)
                .reset_index(drop=True)
            )

            df_gt_preds = (
                dfs["gt"]["preds"]
                .groupby("client_id")
                .head(seq_len)
                .reset_index(drop=True)
            )

            res_sr_gt = sber_f1(
                df_gt_history, df_gt_preds, "client_id", "event_type"
            )
            res_lev_gt = we_dist(
                df_gt_history, df_gt_preds, "client_id", "event_type"
            )
            est_res_dict[self.__ret_name("et_f1_macro_gtvsgt")] = res_sr_gt[
                "f1_macro"
            ].item()
            est_res_dict[self.__ret_name("et_f1_micro_gtvsgt")] = res_sr_gt[
                "f1_micro"
            ].item()
            est_res_dict[self.__ret_name("et_accuracy_gtvsgt")] = res_lev_gt[
                "accuracy"
            ].item()
            est_res_dict[self.__ret_name("et_lev_gtvsgt")] = res_lev_gt[
                "lev_score"
            ].item()

        # elif metric_type == 'tct':

        #     #########################
        #     # tct calculation

        #     tct_res_raw = run_eval_tct(generated_path, orig_path, length)
        #     tct_score = tct_res_raw[0].loc[0, 'mean']
        #     est_res_dict['tct' + name_postfix] = tct_score
        else:
            raise Exception(f"Unknown metric type '{metric_type}'!")

        self.logger.info(f"Epoch: {self.train_state['epoch']}, some metrics eval:\n")
        self.logger.info(dictprettyprint(est_res_dict))
        self.logger.info(f"---\ntime estimation: {time.perf_counter() - start_time} s")

        # wandb.log(est_res_dict, step=self.train_state["CURR_TRAIN_STEP"])

    def __ret_name(self, name):
        return self.name_prefix + name + self.name_postfix