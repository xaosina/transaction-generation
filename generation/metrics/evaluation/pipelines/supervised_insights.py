from collections.abc import Mapping
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ebes.data.utils import build_loaders
from ebes.model import build_model
from ebes.pipeline.base_runner import Runner
from ebes.pipeline.utils import (
    get_loss,
    get_metrics,
    get_optimizer,
    get_scheduler,
    suggest_conf,
)
from ebes.trainer import Trainer
from tqdm.autonotebook import tqdm
from torch import nn

logger = logging.getLogger()


class InsightsTrainer(Trainer):
    @torch.inference_mode()
    def validate_with_insights(self, loader=None) -> dict[str, Any]:
        assert self._model is not None
        if loader is None:
            if self._val_loader is None:
                raise ValueError("Either set val loader or provide loader explicitly")
            loader = self._val_loader

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)

        self._model.eval()
        for metric in self._metrics.values():
            metric.reset()

        manual_loss = nn.CrossEntropyLoss(reduction="none")
        loss_per_index = []
        for batch in tqdm(loader):
            batch.to(self._device)
            inp = batch
            gt = batch.pop_target()
            pred = self._model(inp)
            if self._loss is not None:
                loss = self._loss(pred, gt).cpu()
                self._metrics["loss"].update(loss.cpu())
            loss_per_index.extend(
                zip(
                    batch.index,
                    manual_loss(pred, gt).cpu().tolist(),
                    pred.argmax(1).eq(gt).cpu().tolist(),
                )
            )
            if gt is not None:
                gt = gt.to("cpu")

            for name, metric in self._metrics.items():
                if name != "loss":
                    pred = pred.to("cpu") if hasattr(pred, "to") else pred
                    metric.update(pred, gt)

        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)
        log_dir = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = os.path.dirname(handler.baseFilename)
                break
        if log_dir is None:
            raise ValueError("No log directory found in logger handlers")
        df = pd.DataFrame(loss_per_index, columns=["index", "loss", "correct"])
        df.to_csv(log_dir + "/losses_per_index.csv", index=False)
        return self.compute_metrics("val")


class InsightsRunner(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:

        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])

        net = build_model(config["model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        lr_scheduler = None
        if "lr_scheduler" in config:
            lr_scheduler = get_scheduler(opt, **config["lr_scheduler"])
        metrics = get_metrics(config["metrics"], "cpu")
        loss = get_loss(**config["main_loss"])

        trainer = InsightsTrainer(
            model=net,
            loss=loss,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            train_loader=loaders["train"],
            val_loader=loaders["train_val"],
            run_name=config["run_name"],
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "ckpt",
            device=config["device"],
            **config["trainer"],
        )

        trainer.run()
        trainer.load_best_model()

        train_metrics = trainer.validate(loaders["full_train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        hpo_metrics = trainer.validate_with_insights(loaders["hpo_val"])
        test_metrics = trainer.validate(test_loaders["test"])


        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**hpo_metrics, **train_metrics, **train_val_metrics, **test_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
