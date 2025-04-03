# ignoring all issues with config keys
# pyright: reportArgumentType=false

import argparse
from functools import partialmethod
from pathlib import Path
from typing import Any, List
from collections.abc import Mapping
import signal
import pdb
import pandas as pd

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


def main(
    dataset: str,
    method: str,
    experiment: str,
    train_data: Path,
    test_data: Path = None,
    log_dir: Path = None,
    specify: str = None,
    use_tqdm: bool = False,
    devices: List[int] | None = None,
    logging_lvl: str = "info",
) -> pd.DataFrame:
    """
    Output is a DataFrame which looks like:
                                       0          1          2       mean       std
    MulticlassAUROC             0.817996   0.805227   0.808816   0.810679  0.005377
    loss                        0.316283   0.311917   0.308628   0.312276  0.003136
    train_MulticlassAUROC       0.889246   0.856822   0.862033   0.869367  0.014217
    train_loss                  0.235925   0.266894   0.259100   0.253973  0.013152
    train_val_MulticlassAUROC   0.817843   0.805163   0.801275   0.808094  0.007074
    train_val_loss              0.310729   0.309233   0.314676   0.311546  0.002296
    test_MulticlassAUROC        0.000000   0.000000   0.000000   0.000000  0.000000
    test_loss                   0.000000   0.000000   0.000000   0.000000  0.000000
    memory_after               82.000000  82.000000  82.000000  82.000000  0.000000
    """
    assert logging_lvl in ["error", "info"]
    signal.signal(signal.SIGUSR1, start_debugging)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not use_tqdm)  # type: ignore
    config = collect_config(dataset, method, experiment, specify)
    config.logging.file_lvl = logging_lvl
    config.logging.cons_lvl = logging_lvl
    if devices is not None:
        config.device = devices[0]
        config.runner.device_list = devices
        config.runner.params.n_runs = len(devices)
        config.runner.params.n_workers = len(devices)

    if isinstance(train_data, (str, Path)):
        train_data = Path(train_data)
        config.data.dataset.parquet_path = train_data
        config.run_name = train_data.parts[-2]

    if isinstance(test_data, (str, Path)):
        test_data = Path(test_data)
        config.test_data.dataset.parquet_path = test_data

    if log_dir:
        config.log_dir = log_dir

    runner = Runner.get_runner(config["runner"]["name"])
    res = runner.run(config)
    print(res)
    return res


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
