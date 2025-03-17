from dataclasses import dataclass, field

import os
from collections.abc import Iterable, Sized
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torcheval.metrics import Mean, Metric
from tqdm.autonotebook import tqdm

from .utils import LoadTime, get_profiler, record_function
from .data.types import Batch
from .logger import Logger
from .metrics.sampler import SampleEvaluator


logger = Logger(__name__, log_to_file='log.log', )

@dataclass
class TrainConfig:
    """ Training config for Machine Learning """
    # The number of workers for training
    workers: int = field(default=8) # The number of workers for training
    # The number of aaaaworkers for training
    # The experiment name
    exp_name: str = field(default='default_exp')


def train(cfg: TrainConfig):
    pass

class Trainer:
    """A base class for all trainers."""

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        loss: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        train_loader: Iterable[Batch] | None = None,
        val_loader: Iterable[Batch] | None = None,
        metrics: Iterable[Metric] | None = None,
        run_name: str | None = None,
        total_iters: int | None = None,
        total_epochs: int | None = None,
        patience: int = -1,
        iters_per_epoch: int | None = 10_000,
        ckpt_dir: str | os.PathLike | None = None,
        ckpt_replace: bool = True,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: str | os.PathLike | None = None,
        device: str = "cpu",
        profiling: bool = False,
    ):
        """Initialize trainer.

        Args:
            model: model to train or validate.
            loss: loss function
            optimizer: torch optimizer for training.
            lr_scheduler: torch learning rate scheduler.
            train_loader: train dataloader.
            val_loader: val dataloader.
            metrics: metrics to compute every epoch
            run_name: for runs differentiation.
            total_iters: total number of iterations to train a model.
            total_epochs: total number of epoch to train a model. Exactly one of
                `total_iters` and `total_epochs` shoud be passed.
            patience: how many epochs trainer will go without improving
                validation ckpt_track_metric. -1 means never stop
                Assumes track_metric is MAXIMIZED
            iters_per_epoch: validation and checkpointing are performed every
                `iters_per_epoch` iterations.
            ckpt_dir: path to the directory, where checkpoints are saved.
            ckpt_replace: if `replace` is `True`, only the last and the best checkpoint
                are kept in `ckpt_dir`.
            ckpt_track_metric: if `ckpt_replace` is `True`, the best checkpoint is
                determined based on `track_metric`. All metrics except loss are assumed
                to be better if the value is higher.
            ckpt_resume: path to the checkpoint to resume training from.
            device: device to train and validate on.
            profiling: if profiling is `True`, the training trace will be saved to trace.json.  
                Use this option with caution, it may incur additional computational overhead.
        """
        assert (
            total_iters is None or total_epochs is None
        ), "Only one of `total_iters` and `total_epochs` shoud be passed."

        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime("%F_%T")
        )

        self._metrics = {}
        if metrics is not None:
            self._metrics.update({m.__class__.__name__: m for m in metrics})

        if loss is not None:
            self._metrics.update({"loss": Mean()})

        self._total_iters = total_iters
        self._total_epochs = total_epochs
        self._patience = patience
        self._iters_per_epoch = iters_per_epoch
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._ckpt_resume = ckpt_resume
        self._device = device

        self._model = None
        if model is not None:
            self._model = model.to(device)

        self._loss = None
        if loss is not None:
            self._loss = loss.to(device)

        self._profiler = get_profiler()

        self._opt = optimizer
        self._sched = lr_scheduler
        self._train_loader = train_loader
        self._val_loader = val_loader

        self._metric_values: dict[str, Any] | None = None
        self._last_iter = 0
        self._last_epoch = 0

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def train_loader(self) -> Iterable[Batch] | None:
        return self._train_loader

    @property
    def val_loader(self) -> Iterable[Batch] | None:
        return self._val_loader

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._opt

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        return self._sched

    @property
    def run_name(self):
        return self._run_name

    @property
    def device(self) -> str:
        return self._device
    
    def save_ckpt(self, ckpt_path: str | os.PathLike | None = None) -> None:
            """Save model, optimizer and scheduler states.

            Args:
                ckpt_path: path to checkpoints. If `ckpt_path` is a directory, the
                    checkpoint will be saved there with epoch, loss an metrics in the
                    filename. All scalar metrics returned from `compute_metrics` are used to
                    construct a filename. If full path is specified, the checkpoint will be
                    saved exectly there. If `None` `ckpt_dir` from construct is used with
                    subfolder named `run_name` from Trainer's constructor.
            """

            if ckpt_path is None and self._ckpt_dir is None:
                logger.warning(
                    "`ckpt_path` was not passned to `save_ckpt` and `ckpt_dir` "
                    "was not set in Trainer. No checkpoint will be saved."
                )
                return

            if ckpt_path is None:
                assert self._ckpt_dir is not None
                ckpt_path = self._ckpt_dir

            ckpt_path = Path(ckpt_path)
            ckpt_path.mkdir(parents=True, exist_ok=True)

            ckpt: dict[str, Any] = {
                "last_iter": self._last_iter,
                "last_epoch": self._last_epoch,
            }
            if self._model:
                ckpt["model"] = self._model.state_dict()
            if self._opt:
                ckpt["opt"] = self._opt.state_dict()
            if self._sched:
                ckpt["sched"] = self._sched.state_dict()

            if not ckpt_path.is_dir():
                torch.save(ckpt, ckpt_path)


    def load_ckpt(self, ckpt_fname: str | os.PathLike, strict: bool = True) -> None:
            """Load model, optimizer and scheduler states.

            Args:
                ckpt_fname: path to checkpoint.
            """

            assert self._model is not None
            ckpt = torch.load(ckpt_fname, map_location=self._device)

            if "model" in ckpt:
                msg = self._model.load_state_dict(ckpt["model"], strict=strict)
                logger.info(msg)
            if "opt" in ckpt:
                if self._opt is None:
                    logger.warning(
                        "optimizer was not passes, discarding optimizer state "
                        "in the checkpoint"
                    )
                else:
                    self._opt.load_state_dict(ckpt["opt"])
            if "sched" in ckpt:
                if self._sched is None:
                    logger.warning(
                        "scheduler was not passes, discarding scheduler state "
                        "in the checkpoint"
                    )
                else:
                    self._sched.load_state_dict(ckpt["sched"])
            self._last_iter = ckpt["last_iter"]
            self._last_epoch = ckpt["last_epoch"]

    def train(self, iters: int) -> dict[str, Any]:
        assert self._opt is not None, "Set an optimizer first"
        assert self._train_loader is not None, "Set a train loader first"
        assert self._model is not None
        assert self._loss is not None

        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        # loss_ema = 0.0
        losses: list[float] = []

        total_iters = iters
        if (
            hasattr(self._train_loader, "dataset")
            and isinstance(self._train_loader.dataset, Sized)  # type: ignore
            and (total_iters > len(self._train_loader))  # type: ignore
        ):
            total_iters = len(self._train_loader)  # type: ignore
        pbar = tqdm(zip(self._train_loader, range(total_iters)), total=total_iters)

        pbar.set_description_str(f"Epoch {self._last_epoch + 1: 3}")

        with self._profiler as prof:
            for batch, i in LoadTime(pbar, disable=pbar.disable):
                batch.to(self._device)
                inp = batch

                with self.record_function("forward"):
                    pred = self._model(inp)
                
                loss = self._loss(batch, pred)

                if torch.isnan(loss).any():
                    raise ValueError("None detected in loss. Terminating training.")
                
                with record_function("backward"):
                    loss.backward()

                loss_np = loss.item()

                losses.append(loss_np)
                # loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
                # pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

                self._opt.step()

                self._opt.zero_grad()
                self._last_iter += 1

                prof.step()

            logger.info(
                "Epoch %04d: avg train loss = %.4g", self._last_epoch + 1, np.mean(losses)
            )
            logger.info("Epoch %04d: train finished", self._last_epoch + 1)

    
    @torch.inference_mode()
    def validate(self, loader: Iterable[Batch] | None = None) -> dict[str, Any]:
        assert self._model is not None
        if loader is None:
            if self._val_loader is None:
                raise ValueError("Either set val loader or provide loader explicitly")
            loader = self._val_loader

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)

        self._model.eval()
        
        evaluator = SampleEvaluator(self._model, None, logger)
        self._metric_values = evaluator.evaluate(loader)
        logger.info(
            f"Epoch %04d: metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )

        logger.info("Validation finished")

        return None
    
    
    def run(self) -> None:
        """Train and validate model."""

        assert self._opt, "Set an optimizer to run full cycle"
        assert self._train_loader is not None, "Set a train loader to run full cycle"
        assert self._val_loader is not None, "Set a val loader to run full cycle"
        assert self._model is not None

        logger.info("run %s started", self._run_name)

        if self._ckpt_resume is not None:
            logger.info("Resuming from checkpoint '%s'", str(self._ckpt_resume))
            self.load_ckpt(self._ckpt_resume)

        self._model.to(self._device)

        if self._iters_per_epoch is None:
            logger.warning(
                "`iters_per_epoch` was not passed to the constructor. "
                "Defaulting to the length of the dataloader."
            )
            if not (
                hasattr(self._train_loader, "dataset")
                and isinstance(self._train_loader.dataset, Sized)  # type: ignore
            ):
                raise ValueError(
                    "You must explicitly set `iters_per_epoch` to use unsized loader"
                )

            self._iters_per_epoch = len(self._train_loader)  # type: ignore

        if self._total_iters is None:
            assert self._total_epochs is not None, "Set `total_iters` or `total_epochs`"
            self._total_iters = self._total_epochs * self._iters_per_epoch

        while self._last_iter < self._total_iters:
            train_iters = min(
                self._total_iters - self._last_iter,
                self._iters_per_epoch,
            )

            self.train(train_iters)
            if self._sched:
                self._sched.step()

            self._metric_values = None
            self.validate()

            self._last_epoch += 1
            self.save_ckpt()

        logger.info("run '%s' finished successfully", self._run_name)


    def load_best_model(self) -> None:
        """
        Loads the best model to self._model according to the track metric.
        """

        self.load_ckpt(self._ckpt_dir)