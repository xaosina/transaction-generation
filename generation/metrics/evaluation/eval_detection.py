from argparse import ArgumentParser
from pathlib import Path
import tempfile
import os
from typing import List
import pandas as pd

from .preprocess.detection import prepare_data
from .run_model import main as run_model
from ...data.data_types import DataConfig

DIRPATH = os.path.dirname(__file__) + "/.."


def run_eval_detection(
    data: Path,
    orig: Path,
    log_dir: str,
    data_conf: DataConfig,  # dataset config with column names
    dataset: str,  # dataset config for detection
    method: str = DIRPATH + "/configs/methods/gru.yaml",
    experiment: str = DIRPATH + "/configs/experiments/detection.yaml",
    tail_len: int = None,
    gpu_ids: List[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / data.stem
        print("Temp parquet located in:", temp_dir)
        data = Path(data)
        prepare_data(
            data=data,
            orig=orig,
            save_path=temp_dir,
            data_conf=data_conf,
            tail_len=tail_len,
        )
        return run_model(
            dataset=dataset,
            method=method,
            experiment=experiment,
            train_data=temp_dir,
            log_dir=log_dir,
            use_tqdm=verbose,
            gpu_ids=gpu_ids,
            logging_lvl="info" if verbose else "error",
        )
