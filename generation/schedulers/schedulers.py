import inspect
from typing import Any, List, SupportsFloat


class CompositeScheduler:
    def __init__(self, schedulers: List[Any]):
        self.schedulers = schedulers

    def step(self, epoch=None, loss=None, metrics=None):
        for scheduler in self.schedulers:
            sig = inspect.signature(scheduler.step)
            kwargs = {}
            if "epoch" in sig.parameters and epoch is not None:
                kwargs["epoch"] = epoch
            if "loss" in sig.parameters and loss is not None:
                kwargs["loss"] = loss
            if "metrics" in sig.parameters and metrics is not None:
                kwargs["metrics"] = metrics

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
    
    def get_beta(self) -> float:
        for scheduler in self.schedulers:
            if isinstance(scheduler, BetaScheduler):
                return scheduler.get_beta()
        raise RuntimeError("No scheduler in CompositeScheduler supports get_beta()")


class BetaScheduler:
    def __init__(
        self,
        init_beta: float = 1e-2,
        factor=0.7,
        patience=10,
        min_beta=1e-5,
        verbose=True,
        optimizer=None,
    ):
        self.init_beta = init_beta
        self.beta = init_beta
        self.factor = factor
        self.patience = patience
        self.min_beta = min_beta
        self.verbose = verbose

        self.best_metric = float("inf")
        self.num_bad_epochs = 0

    def get_beta(self) -> float:
        return self.beta

    def step(self, metrics: SupportsFloat):
        if metrics < self.best_metric:
            self.best_metric = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self._reduce_beta()
                self.num_bad_epochs = 0

    def _reduce_beta(self):
        old_beta = self.beta
        new_beta = max(old_beta * self.factor, self.min_beta)
        self.beta = new_beta
        if self.verbose:
            print(f"BetaScheduler {old_beta} -> {new_beta}")

    def state_dict(self) -> dict:
        return {
            "beta": self.beta,
            "best_metric": self.best_metric,
            "num_bad_epochs": self.num_bad_epochs,
        }

    def load_state_dict(self, state_dict: dict):
        self.beta = state_dict["beta"]
        self.best_metric = state_dict["best_metric"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
