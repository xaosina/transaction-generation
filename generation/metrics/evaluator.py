import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import metrics as m
import pandas as pd
import torch
from tqdm import tqdm

from generation.models.generator import Generator

from ..data.data_types import DataConfig, GenBatch
from ..utils import create_instances_from_module, get_unique_folder_suffix
from .evaluation.eval_detection import run_eval_detection


@dataclass
class EvaluatorConfig:
    save_path: str
    detection_config: str = None
    devices: list[str] = ["cuda:0"]
    metrics: Optional[list[Mapping[str, Any] | str]] = None


class SampleEvaluator:
    def __init__(self, data_conf: DataConfig, eval_conf: EvaluatorConfig):
        self.save_path = Path(eval_conf.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.detection_config = eval_conf.detection_config
        self.devices = eval_conf.devices
        self.data_conf = data_conf
        self.metrics = (
            create_instances_from_module(m, eval_conf.metrics, {"data_conf": data_conf})
            or []
        )

    def evaluate(self, model, loader, blim=None, buffer_size=None, remove=False):
        gt_dir, gen_dir = self.generate_samples(model, loader, blim, buffer_size)

        results = self.estimate_metrics(gt_dir, gen_dir)
        results |= self.estimate_tmetrics(gt_dir, gen_dir)

        if remove:
            shutil.rmtree(gt_dir), shutil.rmtree(gen_dir)
        return results

    def generate_samples(
        self, model: Generator, data_loader, blim=None, buffer_size=None
    ):
        gt_dir, gen_dir = self.save_path / "gt", self.save_path / "gen"
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

    def estimate_metrics(self, gt_dir, gen_dir) -> dict:
        gt = pd.read_parquet(gt_dir)
        gen = pd.read_parquet(gen_dir)
        assert (gt[self.data_conf.index_name] == gen[self.data_conf.index_name]).all()
        results = {}
        for metric in self.metrics:
            values = metric(gt, gen)
            if isinstance(values, dict):
                for k, v in values.items():
                    results[f"{str(metric)}_{k}"] = v
            else:
                results[str(metric)] = values
        return results

    def estimate_tmetrics(self, gt_dir, gen_dir) -> dict[str, float]:
        results = dict()
        if self.detection_config:
            discr_res = run_eval_detection(
                data=gen_dir,
                orig=gt_dir,
                log_dir=self.save_path,
                data_conf=self.data_conf,
                dataset=self.detection_config,
                tail_len=self.data_conf.generation_len,
                gpu_ids=self.devices,
                verbose=False,
            )
            discr_score = discr_res.loc["MulticlassAUROC"].loc["mean"]
            results["Uncond Discriminative"] = float(discr_score)
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
