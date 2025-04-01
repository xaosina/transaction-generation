from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from ..data.data_types import GenBatch
from .estimator import MetricEstimator

from generation.models.generator import Generator
from .evaluation.eval_density import run_eval_density
from .evaluation.eval_detection import run_eval_detection


class SampleEvaluator:
    def __init__(
        self,
        ckpt: str,
        metrics: Optional[list[str]] = None,
        gen_len: int = 16,
        hist_len: int = 16,
        subject_key: str = "client_id",
        target_key: str = "event_type",
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
        self.metrics: list[BaseMetric] = [
            getattr(metrics, name) for name in metric_names
        ]

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

    def estimate(self) -> Dict[str, float]:
        return self.estimate_metrics() | self.estimate_tmetrics()

    def estimate_metrics(self) -> Dict[str, float]:
        if not self.gt_save_path or not self.gen_save_path:
            raise ValueError("Ground-truth or generated file path is not provided.")

        gt = pd.read_parquet(self.gt_save_path)
        gen = pd.read_parquet(self.gen_save_path)
        dfs = self.get_data()

        prepared_data = self.prepare_data_to_metrics(dfs)

        # TODO: Сделать перемешивание клиентов если надо.

        results = {}

        for metric in self.metrics:
            data = prepared_data.get(metric.required_data_type)
            results[self.__ret_name("gtvsgt_" + metric.name)] = metric(
                data["gt"], self.subject_key, self.target_key
            )
            results[self.__ret_name("gtvsgen_" + metric.name)] = metric(
                data["gen"], self.subject_key, self.target_key
            )
        return results

    def estimate_tmetrics(self) -> Dict[str, float]:

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
                results[self.__ret_name("shape")] = float(sh_tr["shape"])
                results[self.__ret_name("trend")] = float(sh_tr["trend"])
            else:
                raise Exception(f"Unknown metric type '{metric_type}'!")

        return results

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

    def __ret_name(self, name: str) -> str:
        return self.name_prefix + name


def concat_samples(hist: GenBatch, pred: GenBatch) -> tuple[GenBatch, GenBatch]:
    assert (
        hist.target_time.shape[0] == pred.time.shape[0]
    ), "Mismatch in sequence lengths between hist and pred"
    res = deepcopy(hist)

    res.target_time = pred.time
    res.target_num_features = pred.num_features
    res.target_cat_features = pred.cat_features
    return hist, res
