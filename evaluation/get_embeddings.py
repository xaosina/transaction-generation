from argparse import ArgumentParser
import json
from pathlib import Path
import random
import tempfile
import os
from typing import List
import numpy as np
import pandas as pd

from .preprocess.common import csv_to_parquet

from .run_model import main as run_model

DIRPATH = os.path.dirname(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", help="data csv path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mbd_short")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def run_embeddings(
    data: Path,
    dataset: str = "mbd_short",
    method: str = DIRPATH + "/configs/methods/coles.yaml",
    experiment=DIRPATH + "/configs/experiments/basic_run.yaml",
    gpu_ids: List[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    data = Path(data)
    if dataset == "mbd_short":
        dataset = DIRPATH + "/configs/datasets/mbd_embeddings.yaml"
        with open(data.with_name("metadata_for_detection.json"), "r") as f:
            metadata = json.load(f)["METADATA"]
            metadata["target_columns"] = []
    else:
        raise NotImplementedError(f"There is no preprocess for {dataset} Dataset")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / data.stem
        print("Temp parquet located in:", temp_dir)
        random.seed(42)
        np.random.seed(42)
        csv_to_parquet(
            data=data,
            save_path=temp_dir,
            metadata=metadata,
            cat_codes_path=None,
            overwrite=False,
        )
        return run_model(
            dataset=dataset,
            method=method,
            experiment=experiment,
            train_data=temp_dir / "data",
            use_tqdm=verbose,
            gpu_ids=gpu_ids,
            logging_lvl="info" if verbose else "error",
        )


if __name__ == "__main__":
    args = parse_args()
    vars_args = vars(args)
    print(vars_args)
    for name in ["dataset", "method", "experiment"]:
        if not vars_args[name]:
            del vars_args[name]
    run_embeddings(**vars(args))
