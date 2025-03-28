import logging
import os
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function, schedule


class DummyProfiler:
    def step(self):
        pass


@contextmanager
def dummy_profiler():
    yield DummyProfiler()


def get_profiler(activate: bool = False, save_path=None):
    assert save_path is None, "Set profiling save path"

    def on_trace_ready(prof):
        prof.export_chrome_trace("trace.json")

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=5,
            wait=1,
            warmup=2,
            active=10,
            repeat=1,
        ),
        record_shapes=True,
        on_trace_ready=on_trace_ready,
    )
    profiler_record = record_function

    return (profiler, record_function) if activate else (dummy_profiler(), None)


def dictprettyprint(data: Dict):
    return yaml.dump(data, default_flow_style=False)


class DataParallelAttrAccess(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def multi_gpu(model, config, dim=0):
    if config["multi_gpu_regime"]:
        return DataParallelAttrAccess(model, device_ids=config["gpu_ids"], dim=dim).to(
            config["device"]
        )
    return model


def get_state_dict(model, config):
    return (
        model.module.state_dict() if config["multi_gpu_regime"] else model.state_dict()
    )


class LoadTime:
    def __init__(self, loader):
        self.loader = loader
        self.full_time = 0
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.iterator is None:
            raise RuntimeError(
                "Iterator not initialized. Call __iter__() before __next__()."
            )
        try:
            start = time.time()
            value = next(self.iterator)
            self.full_time += time.time() - start
            return value
        except StopIteration:
            print(f"Data Loading took: {self.full_time} seconds")
            self.iterator = None
            self.full_time = 0
            raise


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    params: Optional[dict[str, Any]] = None


def get_optimizer(
    net_params: Iterable[torch.nn.Parameter], optim_conf: OptimizerConfig
):
    params = optim_conf.params or {}
    try:
        return getattr(torch.optim, optim_conf.name)(net_params, **params)
    except AttributeError:
        raise ValueError(f"Unknkown optimizer: {optim_conf.name}")


@dataclass
class SchedulerConfig:
    name: Optional[str] = "StepLR"
    params: Optional[dict[str, Any]] = None


def get_scheduler(optimizer: torch.optim.Optimizer, sch_conf: SchedulerConfig):
    params = sch_conf.params or {}
    try:
        return getattr(torch.optim.lr_scheduler, sch_conf.name)(optimizer, **params)
    except AttributeError:
        raise ValueError(f"Unknkown LR scheduler: {sch_conf.name}")


@dataclass
class LoginConfig:
    file_lvl: str = "info"
    cons_lvl: str = "warning"


@contextmanager
def log_to_file(filename: Path, log_cfg: LoginConfig):
    if isinstance(log_cfg.file_lvl, str):
        file_lvl = getattr(logging, log_cfg.file_lvl.upper())
    if isinstance(log_cfg.cons_lvl, str):
        cons_lvl = getattr(logging, log_cfg.cons_lvl.upper())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    fh = logging.FileHandler(filename)
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)
    logger = logging.getLogger()
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        yield
    finally:
        fh.close()
        logger.removeHandler(fh)
        logger.removeHandler(ch)


def get_unique_folder_suffix(folder_path):
    folder_path = str(folder_path)
    if not os.path.exists(folder_path):
        return ""
    n = 1
    while os.path.exists(f"{folder_path}({n})"):
        n += 1
    return f"({n})"
