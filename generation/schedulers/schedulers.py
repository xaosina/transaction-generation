import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..losses import VAELoss
from .. import schedulers

logger = logging.getLogger(__name__)


class BaseScheduler(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @abstractmethod
    def step(self, epoch=None, loss=None, metrics=None): ...


class CompositeScheduler(BaseScheduler):
    def __init__(self, optimizer, loss, configs: Mapping[str, Mapping[str, Any] | str]):
        self.schedulers = []
        configs = configs.values()
        for config in configs:
            kwargs = {}
            if isinstance(config, str):
                klass = getattr(schedulers, config)
            else:
                for class_name, params in config.items():
                    klass = getattr(schedulers, class_name)
                    if isinstance(params, Mapping):
                        kwargs = params | kwargs
                    else:
                        raise TypeError("Class config has to be mapping")
                    break  # Only process first key-value pair in dict
            sig = inspect.signature(klass.__init__)
            if "optimizer" in sig.parameters:
                kwargs["optimizer"] = optimizer
            if "loss" in sig.parameters:
                kwargs["loss"] = loss
            self.schedulers.append(klass(**kwargs))

    def step(self, epoch=None, loss=None, metrics=None):
        for scheduler in self.schedulers:
            sig = inspect.signature(scheduler.step)
            kwargs = {}
            if "epoch" in sig.parameters and epoch is not None:
                kwargs["epoch"] = epoch
            if "loss" in sig.parameters and loss is not None:
                kwargs["loss"] = loss
            if "metrics" in sig.parameters:
                if metrics is not None:
                    kwargs["metrics"] = metrics
                else:
                    kwargs["metrics"] = loss

            scheduler.step(**kwargs)

    def state_dict(self) -> dict:
        return {
            str(idx): scheduler.state_dict()
            for idx, scheduler in enumerate(self.schedulers)
        }

    def load_state_dict(self, state_dict: dict):
        for idx, scheduler in enumerate(self.schedulers):
            key = str(idx)
            if key in state_dict:
                scheduler.load_state_dict(state_dict[key])
            else:
                scheduler.load_state_dict(state_dict[idx])


class BetaScheduler(BaseScheduler):
    def __init__(
        self,
        loss: VAELoss,
        init_beta: float = 1e-2,
        factor: float = 0.7,
        patience: int = 10,
        min_beta: float = 1e-5,
        verbose: bool = True,
    ):
        self._loss = loss
        self._beta = init_beta
        self._loss.update_beta(self._beta)
        self._factor = factor
        self._patience = patience
        self._min_beta = min_beta
        self._verbose = verbose

        self._best_loss = float("inf")
        self._num_bad_epochs = 0

    def get_beta(self) -> float:
        return self._beta

    def step(self, loss: float) -> None:
        if loss < self._best_loss:
            self._best_loss = loss
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self._patience:
                self._reduce_beta()
                self._num_bad_epochs = 0

    def _reduce_beta(self) -> None:
        old_beta = self._beta
        self._beta = max(old_beta * self._factor, self._min_beta)
        self._loss.update_beta(self._beta)

        if self._verbose:
            logger.info(f"BetaScheduler: Beta reduced from {old_beta} to {self._beta}")

    def state_dict(self) -> dict:
        return {
            "beta": self._beta,
            "best_loss": self._best_loss,
            "num_bad_epochs": self._num_bad_epochs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._beta = state_dict["beta"]
        self._best_loss = state_dict["best_loss"]
        self._num_bad_epochs = state_dict["num_bad_epochs"]
        self._loss.update_beta(self._beta)
