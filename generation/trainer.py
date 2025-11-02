import logging
import os
import subprocess
from collections.abc import Iterable, Sized
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from ema_pytorch import EMA
from torch import nn
from torcheval.metrics import Mean, Metric
from tqdm.autonotebook import tqdm

from generation.schedulers.schedulers import CompositeScheduler

from .data.data_types import GenBatch
from .metrics.evaluator import SampleEvaluator
from .utils import LoadTime, MeanDict, flatten_rnn_params, get_profiler, record_function

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    total_iters: Optional[int] = 100_000
    total_epochs: Optional[int] = None
    patience: int = -1
    iters_per_epoch: Optional[int] = 10_000
    ckpt_replace: bool = True
    ckpt_track_metric: str = "epoch"
    ckpt_resume: Optional[str] = None
    profiling: bool = False
    verbose: bool = True
    metrics_on_train: bool = False
    ema: Optional[dict] = None


class Trainer:
    """A base class for all trainers."""

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        loss: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: CompositeScheduler | None = None,
        train_loader: Iterable[GenBatch] | None = None,
        val_loader: Iterable[GenBatch] | None = None,
        evaluator: SampleEvaluator | None = None,
        run_name: str | None = None,
        total_iters: int | None = None,
        total_epochs: int | None = None,
        patience: int = -1,
        iters_per_epoch: int | None = 10_000,
        ckpt_dir: str | os.PathLike | None = None,
        ckpt_replace: bool = True,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: str | os.PathLike | None = None,
        ema: Optional[dict] = None,
        device: str = "cpu",
        profiling: bool = False,
        verbose: bool = True,
        grad_clip: float = 1,
        metrics_on_train: bool = False,
    ):
        """Initialize trainer.

        Args:
            model: model to train or validate.
            loss: loss function
            optimizer: torch optimizer for training.
            scheduler: scheduler.
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

        self._sample_evaluator = evaluator

        self._total_iters = total_iters
        self._total_epochs = total_epochs
        self._patience = patience
        self._iters_per_epoch = iters_per_epoch
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._ckpt_resume = ckpt_resume
        self._device = device
        self._verbose = verbose
        self._grad_clip = grad_clip
        self._metrics_on_train = metrics_on_train

        self._model = None
        if model is not None:
            self._model = model.to(device)

        if ema is None:
            self._ema_model = None
        else:
            self._ema_model = EMA(model, **ema)

        self._loss = None
        if loss is not None:
            self._loss = loss.to(device)

        self._profiler = get_profiler()

        self._opt = optimizer
        self._sched = scheduler
        self._train_loader = train_loader
        self._val_loader = val_loader

        self._train_collator = deepcopy(train_loader.collate_fn)
        self._train_random_end = deepcopy(train_loader.dataset.random_end)

        self._metric_values: dict[str, Any] | None = None
        self._last_iter = 0
        self._last_epoch = 0

    @property
    def ema_model(self) -> nn.Module | None:
        return self._ema_model

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def train_loader(self) -> Iterable[GenBatch] | None:
        return self._train_loader

    @property
    def val_loader(self) -> Iterable[GenBatch] | None:
        return self._val_loader

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._opt

    @property
    def scheduler(self):
        return self._sched

    @property
    def run_name(self):
        return self._run_name

    @property
    def device(self) -> str:
        return self._device

    def _make_key_extractor(self, key):
        def key_extractor(p: Path) -> float:
            metrics = {}
            for it in p.stem.split("_-_"):
                kv = it.split("__")
                assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                k = kv[0]
                # v = -float(kv[1]) if ("loss" in k) or ("mse" in k) else float(kv[1])
                v = float(kv[1])
                metrics[k] = v
            return metrics[key]

        return key_extractor

    def save_ckpt(self, ckpt_path: str | os.PathLike | None = None) -> None:
        """Save model, optimizer and scheduler states.

        Args:
            ckpt_path: path to checkpoints. If `ckpt_path` is a directory, the
                checkpoint will be saved there with epoch, loss and metrics in the
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
        if self._ema_model:
            ckpt["ema"] = self._ema_model.state_dict()
        if self._opt:
            ckpt["opt"] = self._opt.state_dict()
        if self._sched:
            ckpt["sched"] = self._sched.state_dict()

        if not ckpt_path.is_dir():
            torch.save(ckpt, ckpt_path)
            return
        assert self._metric_values

        metrics = {k: v for k, v in self._metric_values.items() if np.isscalar(v)}

        fname = f"epoch__{self._last_epoch:04d}"
        metrics_str = "_-_".join(
            f"{k}__{v:.4g}" for k, v in metrics.items() if k == self._ckpt_track_metric
        )

        if len(metrics_str) > 0:
            fname = "_-_".join((fname, metrics_str))
        fname += ".ckpt"

        torch.save(ckpt, ckpt_path / Path(fname))

        if not self._ckpt_replace:
            return

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))
        for p in all_ckpt:
            if p != best_ckpt:
                p.unlink()

    def load_ckpt(self, ckpt_fname: str | os.PathLike, strict: bool = True) -> None:
        """Load model, optimizer and scheduler states.

        Args:
            ckpt_fname: path to checkpoint.
        """

        assert self._model is not None
        ckpt = torch.load(ckpt_fname, map_location="cpu")

        if "model" in ckpt:
            msg = self._model.load_state_dict(ckpt["model"], strict=strict)
            logger.info(msg)
        if "ema" in ckpt:
            msg = self._ema_model.load_state_dict(ckpt["ema"], strict=strict)
            logger.info("EMA: " + str(msg))
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

        loss_ema = 0.0
        log_losses: MeanDict = MeanDict()

        total_iters = iters
        if (
            hasattr(self._train_loader, "dataset")
            and isinstance(self._train_loader.dataset, Sized)  # type: ignore
            and (total_iters > len(self._train_loader))  # type: ignore
        ):
            total_iters = len(self._train_loader)  # type: ignore
        pbar = tqdm(
            zip(self._train_loader, range(total_iters)),
            total=total_iters,
            disable=not self._verbose,
        )

        pbar.set_description_str(f"Epoch {self._last_epoch + 1: 3}")

        with self._profiler as prof:
            for batch, i in LoadTime(pbar, disable=pbar.disable):
                batch.to(self._device)

                with record_function("forward"):
                    pred = self._model(batch)

                loss_dict = self._loss(batch, pred)
                loss = loss_dict["loss"]
                log_losses.update(loss_dict)

                if torch.isnan(loss).any():
                    raise ValueError("None detected in loss. Terminating training.")

                with record_function("backward"):
                    loss.backward()
                loss_ema = loss.item() if i == 0 else 0.9 * loss_ema + 0.1 * loss.item()
                pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

                # torch.nn.utils.clip_grad_norm_(
                #     self._model.parameters(), max_norm=self._grad_clip
                # )
                self._opt.step()

                self._opt.zero_grad()
                self._last_iter += 1
                if self.ema_model:
                    self.ema_model.update()

                prof.step()

            logger.info(
                "Epoch %04d: avg train loss = %.4g",
                self._last_epoch + 1,
                log_losses.mean()["loss"],
            )
            logger.info("Epoch %04d: train finished", self._last_epoch + 1)
        return {"loss_ema": loss_ema} | log_losses.mean()

    @torch.inference_mode()
    def validate(
        self,
        loader: Iterable[GenBatch] | None = None,
        remove=True,
        get_loss: bool = True,
        get_metrics: bool = False,
        use_ema_model: bool = False,
    ) -> dict[str, Any]:
        _model = self.model
        if use_ema_model:
            _model = self._ema_model.ema_model
            flatten_rnn_params(_model)
        assert _model is not None
        assert get_loss or get_metrics, "Choose at least one: [loss, metrics]"
        if loader is None:
            if self._val_loader is None:
                raise ValueError("Either set val loader or provide loader explicitly")
            loader = self._val_loader
        logger.info(
            "Epoch %04d: %s validation started",
            self._last_epoch + 1,
            "EMA" if use_ema_model else "",
        )

        _model.eval()
        _metric_values = {}

        if get_loss:
            orig_collate, orig_random_end = loader.collate_fn, loader.dataset.random_end
            loader.collate_fn = self._train_collator
            loader.dataset.random_end = self._train_random_end
            log_losses: MeanDict = MeanDict()
            with torch.no_grad():
                for batch in tqdm(loader, disable=not self._verbose):
                    batch.to(self._device)
                    pred = _model(batch)
                    loss_dict = self._loss(batch, pred)
                    log_losses.update(loss_dict)
            loader.collate_fn, loader.dataset.random_end = orig_collate, orig_random_end
            _metric_values |= {k: -v for k, v in log_losses.mean().items()}

        if get_metrics:
            _metric_values |= self._sample_evaluator.evaluate(
                _model, loader, remove=remove
            )
        logger.info(
            "Epoch %04d: %s metrics: %s",
            self._last_epoch + 1,
            "EMA" if use_ema_model else "",
            str(_metric_values),
        )
        if not use_ema_model:
            self._metric_values = _metric_values

        return _metric_values

    def run(self) -> None:
        """Train and validate model."""

        assert self._opt, "Set an optimizer to run full cycle"
        assert self._train_loader is not None, "Set a train loader to run full cycle"
        assert self._val_loader is not None, "Set a val loader to run full cycle"
        assert self._model is not None
        logger.info("commit: %s", subprocess.getoutput("git rev-parse HEAD"))
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

        best_metric = float("-inf")
        patience = self._patience
        prev_lr = self._sched.schedulers[0].get_last_lr()[0]
        while self._last_iter < self._total_iters:
            train_iters = min(
                self._total_iters - self._last_iter,
                self._iters_per_epoch,
            )

            losses = self.train(train_iters)
            if self._sched:
                self._sched.step(loss=losses.pop("loss_ema"))
            current_lr = self._sched.schedulers[0].get_last_lr()[0]  # get current LR
            if current_lr != prev_lr:
                print(f"\n\n\n\nReducing learning rate to {current_lr:.2e}")
                prev_lr = current_lr
            self._metric_values = None
            if self._sample_evaluator is not None:
                self.validate(get_metrics=self._metrics_on_train)

                if self.ema_model is not None and (
                    self.ema_model.step.item() >= self.ema_model.update_after_step
                ):
                    self.validate(
                        get_metrics=self._metrics_on_train, use_ema_model=True
                    )

            self._last_epoch += 1
            self.save_ckpt()

            assert (
                self._metric_values is not None
                and self._ckpt_track_metric in self._metric_values
            )
            target_metric = self._metric_values[self._ckpt_track_metric]
            if target_metric > best_metric:
                best_metric = target_metric
                patience = self._patience
            else:
                patience -= 1
            if patience == 0:
                logger.info(
                    f"Patience has run out. Early stopping at {self._last_epoch} epoch"
                )
                break

        logger.info("run '%s' finished successfully", self._run_name)
        return

    def best_checkpoint(self) -> Path:
        """
        Return the path to the best checkpoint
        """
        assert self._ckpt_dir is not None
        ckpt_path = Path(self._ckpt_dir)

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))

        return best_ckpt

    def load_best_model(self) -> None:
        """
        Loads the best model to self._model according to the track metric.
        """

        best_ckpt = self.best_checkpoint()
        self.load_ckpt(best_ckpt)
