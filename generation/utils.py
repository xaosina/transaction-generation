import yaml
from typing import Dict
from pathlib import Path
import yaml
from typing import Dict
import torch
from torch.utils.data import Subset, Dataset
import numpy as np
import time
from torch.profiler import profile, ProfilerActivity, schedule, record_function

from collections.abc import Iterable, Mapping
from typing import Any

def dictprettyprint(data: Dict):
    return yaml.dump(data, default_flow_style=False)


from contextlib import contextmanager, nullcontext

# "пустой" профайлер, который ничего не делает
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

    return (profiler, profiler_record) if activate else (dummy_profiler(), None)


def find_ar_paths(ar_id: str) -> Dict:
    basic_path = Path("tabsyn/hist_ckpt")

    def condition_ckp(path: Path):
        if not path.is_file():
            return False
        if path.name.find(ar_id) == -1:
            return False
        if path.name.endswith(".yaml"):
            return False
        return True

    def condition_cfg(path: Path):
        if not path.is_file():
            return False
        if path.name.find(ar_id) == -1:
            return False
        if not path.name.endswith(".yaml"):
            return False
        return True

    files = [x for x in basic_path.iterdir() if condition_ckp(x)]
    configs = [x for x in basic_path.iterdir() if condition_cfg(x)]

    if len(configs) == 0:
        raise Exception("Wrong parameters, ar config not found")
    if len(configs) > 1:
        raise Exception(f"Ambiguous parameters, found configs: f{configs}")
    config = configs[0]
    print(f'Used logdir: "{str(config)}"')

    if len(files) == 0:
        raise Exception("Wrong parameters, model checkpoint not found")
    if len(files) > 1:
        raise Exception(f"Ambiguous parameters, found checkpoints: f{files}")
    model_path = files[0]
    print(f'Used checkpoint: "{str(model_path)}"')
    return dict(ckpt=model_path, config=config)


def load_config(save_path: Path) -> dict:
    if not save_path.name.endswith(".yaml"):
        assert save_path.is_dir()
        save_path = save_path / "config.yaml"
    with open(save_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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


def get_optimizer(
    net_params: Iterable[torch.nn.Parameter],
    name: str = "Adam",
    params: Mapping[str, Any] | None = None,
):
    params = params or {}
    try:
        return getattr(torch.optim, name)(net_params, **params)
    except AttributeError:
        raise ValueError(f"Unknkown optimizer: {name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer, name: str, params: Mapping[str, Any] | None = None
):
    params = params or {}
    try:
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **params)
    except AttributeError:
        raise ValueError(f"Unknkown LR scheduler: {name}")
