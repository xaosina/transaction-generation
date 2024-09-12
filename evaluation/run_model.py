# ignoring all issues with config keys
# pyright: reportArgumentType=false

import argparse
from functools import partialmethod
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import signal
import pdb

from omegaconf import OmegaConf
from tqdm import tqdm

from ebes.pipeline.base_runner import Runner


def start_debugging(_, frame):
    pdb.Pdb().set_trace(frame)


def collect_config(
    dataset, method, experiment, specify=None, gpu=None
) -> dict[str, Any]:
    data_config = OmegaConf.load(Path(dataset))
    method_config = OmegaConf.load(Path(method))
    exp_config = OmegaConf.load(Path(experiment))
    configs = [data_config, method_config, exp_config]

    if specify is not None:
        specify_path = Path(specify)
        if specify_path.exists():
            configs.append(OmegaConf.load(specify_path))
        else:
            raise ValueError(f"No specification {specify}")

    config = OmegaConf.merge(*configs)
    if gpu is not None:
        assert config.runner.get("device_list") is None
        config["device"] = gpu
    return config  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-data", type=Path, default="orig_train")
    parser.add_argument("-d", "--dataset", type=str, default="datafusion")
    parser.add_argument("-m", "--method", type=str, default="configs/methods/gru.yaml")
    parser.add_argument("-e", "--experiment", type=str, default="test")
    parser.add_argument("-s", "--specify", type=str, default=None)
    parser.add_argument("--use-tqdm", action="store_true")
    return parser.parse_args()


def main(dataset: str, method: str, experiment: str, train_data: Path, specify: str=None, use_tqdm: bool = False):
    if isinstance(train_data, str):
        train_data = Path(train_data)
    signal.signal(signal.SIGUSR1, start_debugging)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not use_tqdm)  # type: ignore
    config = collect_config(dataset, method, experiment, specify)
    config.data.dataset.parquet_path = train_data
    config.run_name = train_data.parts[-2]

    runner = Runner.get_runner(config["runner"]["name"])
    res = runner.run(config)
    print(res)

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))