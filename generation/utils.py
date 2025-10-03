import importlib
import inspect
import logging
import os
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np

import torch
from torch.nn import Module
import yaml
from omegaconf import DictConfig, ListConfig
from optuna import Trial
from torch.profiler import ProfilerActivity, profile, record_function, schedule


class MeanDict:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, d):
        for key in d:
            self.sums[key] = self.sums.get(key, 0) + d[key].item()
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self):
        assert self.sums.keys() == self.counts.keys()
        return {key: self.sums[key] / self.counts[key] for key in self.sums}


class RateLimitFilter(logging.Filter):
    def __init__(self, cooldown=60):
        super().__init__()
        self.cooldown = cooldown
        self.last_log = {}
        self.counter = {}

    def filter(self, r):
        t = time()
        if t - self.last_log.get((m := r.getMessage()), 0) < self.cooldown:
            return False
        self.last_log[m] = t
        return True


def freeze_module(m: Module):
    for param in m.parameters():
        param.requires_grad = False

    def train(mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        m.training = False
        for module in m.children():
            module.train(False)
        return m

    m.train = train
    return m


class DummyProfiler:
    def __enter__(self):
        return self  # ← позволяет делать `with self._profiler as prof`

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def step(self):
        pass


def get_profiler(activate: bool = False, save_path=None):
    assert save_path is None, "Set profiling save path"

    def on_trace_ready(prof):
        prof.export_chrome_trace("trace.json")

    return (
        profile(
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
        if activate
        else DummyProfiler()
    )


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
    def __init__(self, loader, disable=False):
        self.loader = loader
        self.full_time = 0
        self.iterator = None
        self._disable = disable

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.iterator is None:
            raise RuntimeError(
                "Iterator not initialized. Call __iter__() before __next__()."
            )
        try:
            start = time()
            value = next(self.iterator)
            self.full_time += time() - start
            return value
        except StopIteration:
            if not self._disable:
                print(f"Data Loading took: {self.full_time} seconds")
            self.iterator = None
            self.full_time = 0
            raise


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


def create_instances_from_module(
    module,
    configs: (
        list[Mapping[str, Any] | str] | Mapping[str, Mapping[str, Any] | str] | None
    ) = None,
    common_kwargs: dict = None,
) -> list[Any] | None:
    common_kwargs = common_kwargs or dict()
    instances = None
    if configs is not None:
        if isinstance(configs, Mapping):
            configs = configs.values()
        instances = []
        for config in configs:
            if isinstance(config, str):
                instances.append(getattr(module, config)(**common_kwargs))
                continue

            for class_name, params in config.items():
                klass = getattr(module, class_name)
                if isinstance(params, Mapping):
                    instances.append(klass(**(params | common_kwargs)))
                else:
                    raise TypeError("Class config has to be mapping")
                break  # Only process first key-value pair in dict
    return instances


def assign_by_name(config: dict | DictConfig, name: str, value: Any):
    field = config
    for k in name.split(".")[:-1]:
        if isinstance(field, Mapping):
            field = field[k]
        else:
            field = field[int(k)]
    field[name.split(".")[-1]] = value


def suggest_conf(suggestions: list, config: dict | DictConfig, trial: Trial):
    for names, suggestion in suggestions:
        if not isinstance(names, (list, ListConfig)):
            names = [names]
        first_name = names[0]
        value = getattr(trial, suggestion[0])(first_name, **suggestion[1])
        for name in names:
            assign_by_name(config, name, value)


def _auto_import_subclasses(current_dir, package_name, global_dict, parent_class):
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, parent_class)
                        and obj != parent_class
                    ):
                        global_dict[name] = obj

            except Exception as e:
                print(f"Error importing {full_module_name}: {e}")
                continue


# Printing dict values in a nice way

# --- Representers ---
def __numpy_array_representer(dumper, data):
    return dumper.represent_list(data.tolist())

__numpy_cast_types = {
    np.int16: 'int',
    np.int32: 'int',
    np.int64: 'int',
    np.float64: 'float',
    np.float32: 'float',
    np.float16: 'float',
    np.bool_: 'bool'
}

def __numpy_scalar_representer(dumper, data):
    return dumper.represent_scalar(
        u'tag:yaml.org,2002:' + __numpy_cast_types.get(type(data), 'str'),
        str(data.item())
    )

# Register handlers once
yaml.add_representer(np.ndarray, __numpy_array_representer)
for scalar_type in __numpy_cast_types.keys():
    yaml.add_multi_representer(scalar_type, __numpy_scalar_representer)

def dictprettyprint(data: Dict):
    return yaml.dump(data, default_flow_style=False)