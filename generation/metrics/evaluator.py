import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import metrics as m
import pandas as pd
import torch
from tqdm import tqdm

from generation.models.generator import Generator

from ..data.data_types import GenBatch
from ..utils import create_instances_from_module, get_unique_folder_suffix
from .evaluation.eval_density import run_eval_density
from .evaluation.eval_detection import run_eval_detection


class SampleEvaluator:
    def __init__(
        self,
        gen_path: str,
        metric_params: Optional[list[Mapping[str, Any] | str]] = None,
        generation_len: int = 16,
        history_len: int = 16,
    ):
        self.gen_path = Path(gen_path)
        self.gen_path.mkdir(parents=True, exist_ok=True)
        self.generation_len = generation_len
        self.history_len = history_len
        self.metrics = create_instances_from_module(m, metric_params) or []

    def evaluate(self, model, loader, blim=None, buffer_size=None, remove=False):
        gt_dir, gen_dir = self.generate_samples(model, loader, blim, buffer_size)

        gt = pd.read_parquet(gt_dir)
        gen = pd.read_parquet(gen_dir)

        results = {}
        for metric in self.metrics:
            values = metric(gt, gen)
            if isinstance(values, dict):
                for k, v in values.items():
                    results[f"{str(metric)}_{k}"] = v
            else:
                results[str(metric)] = values

        results |= self.estimate_tmetrics(gt_dir, gen_dir)

        if remove:
            shutil.rmtree(gt_dir), shutil.rmtree(gen_dir)
        return results

    def generate_samples(
        self, model: Generator, data_loader, blim=None, buffer_size=None
    ):
        gt_dir, gen_dir = self.gen_path / "gt", self.gen_path / "gen"
        gt_dir += get_unique_folder_suffix(gt_dir)
        gen_dir += get_unique_folder_suffix(gen_dir)

        gt_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        buffer_gt, buffer_gen = [], []
        part_counter = 0

        for batch_idx, batch_input in enumerate(tqdm(data_loader)):
            if blim and batch_idx >= blim:
                break

            batch_input = batch_input.to("cuda")
            with torch.no_grad():
                batch_pred = model.generate(batch_input, self.gen_len)
            gt, gen = _concat_samples(batch_input, batch_pred)
            gt = data_loader.collate_fn.reverse(gt)
            gen = data_loader.collate_fn.reverse(gen)

            buffer_gt.append(gt)
            buffer_gen.append(gen)

            if buffer_size and len(buffer_gt) >= buffer_size:
                _save_buffers(buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter)
                part_counter += 1
                buffer_gt, buffer_gen = [], []

        if buffer_gt:
            _save_buffers(buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter)

        return gt_dir, gen_dir


    def estimate_tmetrics(self) -> dict[str, float]:

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

        return results


def _save_buffers(buffer_gt, buffer_gen, gt_dir, gen_dir, part_counter):
    gt_file = gt_dir / f"part-{part_counter:04d}.parquet"
    gen_file = gen_dir / f"part-{part_counter:04d}.parquet"
    pd.concat(buffer_gt, ignore_index=True).to_parquet(gt_file, index=False)
    pd.concat(buffer_gen, ignore_index=True).to_parquet(gen_file, index=False)


def _concat_samples(gt: GenBatch, pred: GenBatch) -> tuple[GenBatch, GenBatch]:
    assert (
        get_unique_folder_suffix.target_time.shape[0] == pred.time.shape[0]
    ), "Mismatch in sequence lengths between hist and pred"
    gen = deepcopy(gt)

    gen.target_time = pred.time
    gen.target_num_features = pred.num_features
    gen.target_cat_features = pred.cat_features
    return gt, gen
