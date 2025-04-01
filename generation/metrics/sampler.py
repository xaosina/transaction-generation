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
        subject_key: str = 'client_id',
        target_key: str = 'event_type',
        device: int = 0,
    ):
        self.ckpt = ckpt
        Path(ckpt).mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or []
        self.device = device
        self.gen_len = gen_len
        self.hist_len = hist_len
        self.subject_key = subject_key
        self.target_key = target_key

    def evaluate(self, model, loader, blim=None, prefix="", buffer_size=None):
        gt_df_save_path, gen_df_save_path = self.generate_samples(
            model, loader, blim, prefix, buffer_size=buffer_size
        )
        breakpoint()
        return self.evaluate_and_save(prefix, gt_df_save_path, gen_df_save_path)

    def generate_samples(
        self, model: Generator, data_loader, blim=None, prefix="", buffer_size=None
    ):
        model.eval()
        gen_dir = self.ckpt / f"validation_gen{prefix}"
        gt_dir = self.ckpt / f"validation_gt{prefix}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        buffer_gt, buffer_gen = [], []
        part_counter = 0

        for batch_idx, batch_input in enumerate(tqdm(data_loader)):
            if blim and batch_idx >= blim:
                break

            batch_input = batch_input.to("cuda")
            with torch.no_grad():
                batch_pred = model.generate(batch_input, self.gen_len)
            gt, gen = concat_samples(batch_input, batch_pred)
            gt = data_loader.collate_fn.reverse(gt)
            gen = data_loader.collate_fn.reverse(gen)

            buffer_gt.append(gt)
            buffer_gen.append(gen)

            if buffer_size and len(buffer_gt) >= buffer_size:
                self._save_buffers(buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter)
                part_counter += 1
                buffer_gt, buffer_gen = [], []

        if buffer_gen:
            self._save_buffers(buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter)

        return gen_dir, gt_dir

    def _save_buffers(self, buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter):
        gt_file = gt_dir / f"part-{part_counter:04d}.parquet"
        gen_file = gen_dir / f"part-{part_counter:04d}.parquet"
        pd.concat(buffer_gt, ignore_index=True).to_parquet(gt_file, index=False)
        pd.concat(buffer_gen, ignore_index=True).to_parquet(gen_file, index=False)

    def evaluate_and_save(self, name_prefix, gt_save_path, gen_save_path):
        return MetricEstimator(
            gt_save_path,
            gen_save_path,
            name_prefix,
            self.metrics,
            self.gen_len,
            self.hist_len,
            device=self.device,
            subject_key=self.subject_key,
            target_key=self.target_key,
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
