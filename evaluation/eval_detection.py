from argparse import ArgumentParser
from pathlib import Path
import tempfile
import os
from typing import List
import pandas as pd

from .preprocess.mbd_detection import main as prepare_mbd
from .preprocess.datafusion_detection import main as prepare_datafusion
from .run_model import main as run_model

DIRPATH = os.path.dirname(__file__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--data-type", 
        type=str, choices=["general", "tabsyn"], default="tabsyn"
    )
    parser.add_argument("-n", "--n-rows", type=int, required=True)
    parser.add_argument("-m", "--match-users", action="store_true")
    parser.add_argument(
        "-d", "--data", 
        help="generated data csv path", 
        type=Path, required=True
    )
    parser.add_argument(
        "-o", "--orig",
        help="Path to orig dataset containing CSV files",
        type=Path, required=True
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument('--gpu_ids', type=int, nargs='*', default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def run_eval_detection(
    data_type: str,
    data: Path,
    orig: Path,
    n_rows: int,
    match_users: bool,
    dataset: str = "datafusion",
    method: str = DIRPATH + "/configs/methods/gru.yaml",
    experiment= DIRPATH + "/configs/experiments/detection.yaml",
    gpu_ids: List[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if dataset == "mbd":
        dataset = DIRPATH + "/configs/datasets/mbd_detection.yaml"
        prepare_data = prepare_mbd
    elif dataset == "datafusion":
        dataset = DIRPATH + "/configs/datasets/datafusion_detection.yaml"
        prepare_data = prepare_datafusion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / data.stem
        print("Temp parquet located in:", temp_dir)
        data = Path(data)
        prepare_data(
            data_type=data_type,
            data=data,
            orig=orig,
            n_rows=n_rows,
            match_users=match_users,
            save_path=temp_dir,
        )
        return run_model(
            dataset=dataset,
            method=method,
            experiment=experiment,
            train_data=temp_dir / "data",
            use_tqdm=verbose,
            gpu_ids=gpu_ids,
            logging_lvl= 'info' if verbose else 'error'
        )


if __name__ == "__main__":
    args = parse_args()
    vars_args = vars(args)
    print(vars_args)
    for name in ['dataset', 'method', 'experiment']:
        if not vars_args[name]:
            del vars_args[name]
    run_eval_detection(**vars(args))
