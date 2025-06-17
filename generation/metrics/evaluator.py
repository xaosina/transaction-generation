import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import torch
from tqdm import tqdm

from generation.models.generator import BaseGenerator

from ..data.data_types import DataConfig, GenBatch
from ..utils import create_instances_from_module, get_unique_folder_suffix
from . import metrics as m

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluatorConfig:
    devices: list[str] = field(default_factory=lambda: ["cuda:0"])
    metrics: Optional[list[Mapping[str, Any] | str]] = None


class SampleEvaluator:
    def __init__(
        self,
        log_dir: str,
        data_conf: DataConfig,
        eval_config: EvaluatorConfig,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.data_config = data_conf
        self.device = device
        self.metrics = (
            create_instances_from_module(
                m,
                eval_config.metrics,
                {
                    "devices": eval_config.devices,
                    "data_conf": data_conf,
                    "log_dir": log_dir,
                },
            )
            or []
        )
        self.verbose = verbose

    def evaluate(self, model, loader, blim=None, buffer_size=None, remove=False):
        log_dir = Path(str(self.log_dir) + get_unique_folder_suffix(self.log_dir))
        for metric in self.metrics:
            metric.log_dir = log_dir
        assert loader.collate_fn.return_orig, "collator have to return orig seqs!"
        gt_dir, gen_dir = self.generate_samples(
            model, loader, log_dir, blim, buffer_size
        )
        logger.info("Sampling done.")
        results = self.estimate_metrics(gt_dir, gen_dir)
        logger.info("Metrics done.")
        if remove:
            shutil.rmtree(log_dir)
        return results

    def generate_samples(
        self, model: BaseGenerator, data_loader, log_dir, blim=None, buffer_size=None
    ):
        base_dir = log_dir / "samples"
        gt_dir, gen_dir = base_dir / "gt", base_dir / "gen"

        gt_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        buffer_gt, buffer_gen = [], []
        part_counter = 0

        for batch_idx, (batch_input, orig_seqs) in enumerate(
            tqdm(data_loader, disable=not self.verbose)
        ):
            if blim and batch_idx >= blim:
                break

            batch_input = batch_input.to(self.device)
            with torch.no_grad():
                batch_pred = model.generate(
                    deepcopy(batch_input), self.data_config.generation_len
                )
            gen = _concat_samples(batch_input, batch_pred)
            gen = data_loader.collate_fn.reverse(gen.to("cpu"))

            buffer_gt.append(orig_seqs)
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
        assert (
            gt[self.data_config.index_name] == gen[self.data_config.index_name]
        ).all()
        results = {}
        for metric in self.metrics:
            logger.info(f"{metric} is started")
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
        gt.target_time.shape[0] == pred.time.shape[0]
    ), "Mismatch in sequence lengths between hist and pred"
    gen = deepcopy(gt)

    gen.target_time = pred.time
    gen.target_num_features = pred.num_features
    gen.target_cat_features = pred.cat_features
    return gen
