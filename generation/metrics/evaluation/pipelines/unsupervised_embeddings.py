from collections.abc import Mapping
import gc
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


class EmbeddingsTrainer(Trainer):
    @torch.inference_mode()
    def validate_with_embeddings(self, loader=None) -> dict[str, Any]:
        assert self._model is not None
        if loader is None:
            if self._val_loader is None:
                raise ValueError("Either set val loader or provide loader explicitly")
            loader = self._val_loader
        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()

        embeddings = []
        indices = []
        for batch in tqdm(loader):
            batch.to(self._device)
            inp = batch
            gt = batch.pop_target()
            pred = self._model(inp)
            indices.extend(batch.index.tolist())
            embeddings.extend(pred.cpu().tolist())
            if gt is not None:
                gt = gt.to("cpu")

        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)
        df = pd.DataFrame({"embedding": embeddings}, index=indices)
        return df


class UnsupervisedEmbedder(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        net = build_model(config["unsupervised_model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        lr_scheduler = None
        if "lr_scheduler" in config:
            lr_scheduler = get_scheduler(opt, **config["lr_scheduler"])
        loss = get_loss(**config["unsupervised_loss"])
        metrics = get_metrics(config.get("unsupervised_metrics"), "cpu")
        trainer = EmbeddingsTrainer(
            model=net,
            loss=loss,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
            train_loader=loaders["unsupervised_train"],
            val_loader=loaders["unsupervised_train_val"],
            run_name=config["run_name"] + "/pretrain",
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "pretrain" / "ckpt",
            device=config["device"],
            metrics=metrics,
            **config["unsupervised_trainer"],
        )
        trainer.run()

        del loaders["unsupervised_train"]  # type: ignore
        del loaders["unsupervised_train_val"]  # type: ignore
        gc.collect()

        df_train = trainer.validate_with_embeddings(loaders["full_train"])
        del loaders["full_train"]  # type: ignore
        df_val = trainer.validate_with_embeddings(loaders["train_val"])
        del loaders["train_val"]  # type: ignore

        df = pd.concat([df_train, df_val])
        df.to_parquet(trainer._ckpt_dir / "embeddings.parquet", index=True)

        return {"none": 0}

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
