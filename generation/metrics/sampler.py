from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from ..data.data_types import GenBatch
from .estimator import MetricEstimator

from generation.models.generator import Generator
class SampleEvaluator:
    def __init__(
        self,
        ckpt: str,
        metrics: Optional[list[str]] = None,
        gen_len: int = 16,
        hist_len: int = 16,
        device: int = 0,
    ):
        self.ckpt = ckpt
        Path(ckpt).mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or []
        self.device = device
        self.gen_len = gen_len
        self.hist_len = hist_len

    def evaluate(self, model, loader, blim=None, prefix=""):
        gt_df_save_path, gen_df_save_path = self.generate_samples(
            model, loader, blim, prefix
        )
        return self.evaluate_and_save(prefix, gt_df_save_path, gen_df_save_path)

    def generate_samples(self, model: Generator, data_loader, blim=None, prefix=""):
        model.eval()

        gen_df_save_path = self.ckpt / f"validation_gen{prefix}.csv"
        gt_df_save_path = self.ckpt / f"validation_gt{prefix}.csv"

        pbar = tqdm(data_loader, total=len(data_loader))
        for batch_idx, batch_input in enumerate(pbar):
            batch_input = batch_input.to("cuda")
            if blim is not None and batch_idx > blim:
                break

            with torch.no_grad():
                batch_pred = model.generate(batch_input, self.gen_len)
            gt, gen = concat_samples(batch_input, batch_pred)

            gt = pd.DataFrame(data_loader.collate_fn.reverse(gt))
            gen = pd.DataFrame(data_loader.collate_fn.reverse(gen))

            gt.to_csv(gt_df_save_path, mode="a", index=False)
            gen.to_csv(gen_df_save_path, mode="a", index=False)

        return (
            gt_df_save_path,
            gen_df_save_path,
        )

    def evaluate_and_save(self, name_prefix, gt_save_path, gen_save_path):
        return MetricEstimator(
            gt_save_path,
            gen_save_path,
            name_prefix,
            self.metrics,
            self.gen_len,
            self.hist_len,
            device=self.device,
        ).estimate()


def concat_samples(hist: GenBatch, pred: GenBatch) -> tuple[GenBatch, GenBatch]:
    assert (
        hist.target_time.shape[0] == pred.time.shape[0]
    ), "Mismatch in sequence lengths between hist and pred"
    res = deepcopy(hist)

    res.target_time = pred.time
    res.target_num_features = pred.num_features
    res.target_cat_features = pred.cat_features
    return hist, res
