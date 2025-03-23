from tqdm import tqdm
import torch
from .estimator import MetricEstimator
import pandas as pd
from ..data.types import Batch
from dataclasses import dataclass
from pathlib import Path
import time

from ..utils import dictprettyprint
from ..logger import Logger
from ..models.generator import Generator
from .estimator import get_metrics, MetricsConfig


@dataclass
class SamplerConfig:
    ckpt: Path
    model: Generator
    logger: Logger
    collator: ...


def get_sampler(cfg: SamplerConfig):
    return SampleEvaluator(
        cfg.model,
        cfg.collator,
        cfg.logger,
        cfg.ckpt,
    )


class SampleEvaluator:

    def __init__(self, model, collator, logger, ckpt):
        self.model = model
        self.collator = collator
        self.logger = logger
        self.ckpt = ckpt

    def evaluate(self, loader, blim=None, prefix=""):

        gt_df_save_path, gen_df_save_path = self.generate_samples(loader, blim)
        metrics = self.evaluate_and_save(prefix, gt_df_save_path, gen_df_save_path)
        start_time = time.perf_counter()
        self.logger.info(f"Metric eval:\n")
        self.logger.info(dictprettyprint(metrics))
        self.logger.info(f"---\ntime estimation: {time.perf_counter() - start_time} s")
        return metrics

    def generate_samples(self, data_loader, blim=None):
        self.model.eval()

        gen_df_save_path = self.ckpt / "validation_gen.csv"
        gt_df_save_path = self.ckpt / "validation_gt.csv"

        pbar = tqdm(data_loader, total=len(data_loader))

        for batch_idx, samp_inp in enumerate(pbar):
            if blim is not None and batch_idx > blim:
                break

            with torch.no_grad():
                samp_res = self.model.generate(samp_inp)
            gt, gen = concat_samples(samp_inp, samp_res)

            # syn_df["event_time"] = syn_df.groupby("client_id")[
            #     "time_diff_days"
            # ].cumsum()

            gt: pd.DataFrame = self.collator.reverse(gt)
            gen: pd.DataFrame = self.collator.reverse(gen)

            gt.to_csv(gt_df_save_path, mode="a", index=False)
            gen.to_csv(gen_df_save_path, mode="a", index=False)

        return gt_df_save_path, gen_df_save_path

    def evaluate_and_save(
        self, name_prefix, gt_df_save_path, gen_df_save_path, metrics_cfg
    ):

        metric_estimator = MetricEstimator(
            gt_df_save_path,
            gen_df_save_path,
            self.config,
            self.logger,
            name_prefix=name_prefix,
        )

        return metric_estimator.estimate()


def concat_samples(hist: Batch, pred: Batch) -> tuple[Batch, Batch]:
    assert (
        hist.lengths == pred.lengths
    ), "Mismatch in sequence lengths between hist and pred"

    hist.lengths = hist.lengths + hist.lengths

    hist.time = torch.cat([hist.time, hist.target_time])
    hist.num_features = torch.cat([hist.num_features, hist.target_num_features])
    hist.cat_features = torch.cat([hist.cat_features, hist.target_cat_features])

    pred.lengths = pred.lengths + pred.lengths
    pred.time = torch.cat([pred.time, pred.target_time])
    pred.cat_features = torch.cat([pred.cat_features, pred.target_cat_features])
    pred.num_features = torch.cat([pred.num_features, pred.target_num_features])

    return hist, pred
