from tqdm import tqdm
import torch
from .metric_utils import MetricEstimator
import delu
from ..data.types import Batch
from dataclasses import dataclass
from pathlib import Path
import time

from ..utils import dictprettyprint
from ..logger import Logger
from ..models.generator import Generator
from .metric_utils import get_metrics, MetricsConfig

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

        gen_samples, gt_samples, gt_client_ids = self.generate_samples(loader, blim)
        gt_df_save_path, gen_df_save_path = self.recover_and_save_samples(
            gen_samples, gt_samples, gt_client_ids
        )
        metrics = self.evaluate_and_save(prefix, gt_df_save_path, gen_df_save_path)
        start_time = time.perf_counter()
        self.logger.info(f"Metric eval:\n")
        self.logger.info(dictprettyprint(metrics))
        self.logger.info(f"---\ntime estimation: {time.perf_counter() - start_time} s")
        return metrics

    def generate_samples(self, data_loader, blim=None):
        self.model.eval()

        pbar = tqdm(data_loader, total=len(data_loader))

        gen_samples = []
        gt_samples = []
        gt_client_ids = (
            []
        )  # TODO. Нужно ли нам это теперь? Подумать как заменить ее в этом коде.

        for batch_idx, samp_inp in enumerate(pbar):
            if blim is not None and batch_idx > blim:
                break

            with torch.no_grad():
                samp_res = self.model.generate(samp_inp)
            gt, gen = concat_samples(samp_inp, samp_res)

            gt_samples.append(gt)
            gen_samples.append(gen)
        return gt_samples, gen_samples

    def recover_and_save_samples(self, gen_samples, gt_samples, gt_client_ids):

        gt_client_ids = torch.cat(gt_client_ids)

        _dfs = dict(
            gen=None,
            gt=None,
        )

        gen_df_save_path = self.ckpt / "validation_gen.csv"
        gt_df_save_path = self.ckpt / "validation_gt.csv"

        for parti, samples, df_save_path in zip(
            ["gen", "gt"],
            [gen_samples, gt_samples],
            [gen_df_save_path, gt_df_save_path],
        ):

            samples = torch.cat(samples).detach().cpu().numpy()

            syn_df["event_time"] = syn_df.groupby("client_id")[
                "time_diff_days"
            ].cumsum()

            syn_df.to_csv(df_save_path, index=False)
            _dfs[parti] = syn_df
            self.logger.info(f"Saving validation {parti} data to {df_save_path}")

        return gt_df_save_path, gen_df_save_path

    def evaluate_and_save(self, name_prefix, gt_df_save_path, gen_df_save_path, metrics_cfg):

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
