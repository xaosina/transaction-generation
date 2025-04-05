from argparse import ArgumentParser
from pathlib import Path
import tempfile
import os
from typing import List
import pandas as pd

from .preprocess.detection import prepare_data
from .run_model import main as run_model
from ...data.data_types import DataConfig


def run_eval_detection(
    orig: pd.DataFrame,
    gen: pd.DataFrame,
    log_dir: str,
    data_conf: DataConfig,  # dataset config with column names
    dataset: str,  # dataset config path for detection
    method: str = "gru",
    experiment: str = "detection",
    tail_len: int = None,
    devices: List[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "detection"
        print("Temp parquet located in:", temp_dir)
        prepare_data(
            orig=orig,
            gen=gen,
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
            devices=devices,
            logging_lvl="info" if verbose else "error",
        )
