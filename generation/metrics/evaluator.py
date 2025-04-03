import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import metrics as m
import pandas as pd
import torch
from tqdm import tqdm

from generation.models.generator import Generator

from ..data.data_types import DataConfig, GenBatch
from ..utils import create_instances_from_module, get_unique_folder_suffix


@dataclass(frozen=True)
class EvaluatorConfig:
    data_conf: DataConfig
    save_path: str
    devices: list[str] = field(default_factory=lambda: ["cuda:0"])
    metrics: Optional[list[Mapping[str, Any] | str]] = None


class SampleEvaluator:
    def __init__(self, eval_config: EvaluatorConfig):
        Path(eval_config.save_path).mkdir(parents=True, exist_ok=True)
        self.data_config = eval_config.data_conf
        self.eval_config = eval_config
        self.metrics = (
            create_instances_from_module(
                m, eval_config.metrics, {"eval_config": eval_config}
            )
            or []
        )

    def evaluate(self, model, loader, blim=None, buffer_size=None, remove=False):
        gt_dir, gen_dir = self.generate_samples(model, loader, blim, buffer_size)

        results = self.estimate_metrics(gt_dir, gen_dir)

        if remove:
            shutil.rmtree(gt_dir), shutil.rmtree(gen_dir)
        return results

    def generate_samples(
        self, model: Generator, data_loader, blim=None, buffer_size=None
    ):
        gt_dir, gen_dir = (
            self.eval_config.save_path / "gt",
            self.eval_config.save_path / "gen",
        )
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
                    assert f"{str(metric)} {k}" not in results, "Dont overwrite metric"
                    results[f"{str(metric)} {k}"] = v
            else:
                assert str(metric) not in results, "Dont overwrite metric"
                results[str(metric)] = values
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
