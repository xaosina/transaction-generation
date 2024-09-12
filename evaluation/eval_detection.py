from argparse import ArgumentParser
from pathlib import Path
import tempfile

from preprocess.datafusion_detection import main as prepare_data
from run_model import main as run_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--data-type", type=str, choices=["general", "tabsyn"], required=True
    )
    parser.add_argument("-n", "--n-rows", type=int)
    parser.add_argument("-m", "--match-users", action="store_true")
    parser.add_argument(
        "-d", "--data", help="generated data csv path", type=Path, required=True
    )
    parser.add_argument(
        "--orig",
        help="Path to orig dataset containing CSV files",
        default="data/datafusion/preprocessed_with_id_test.csv",
        type=Path,
    )
    parser.add_argument("--dataset", type=str, default="configs/datasets/datafusion_detection.yaml")
    parser.add_argument("--tqdm", action="store_true")
    return parser.parse_args()


def prepare_and_detect(
    data_type: str,
    data: Path,
    orig: Path,
    n_rows: int,
    match_users: bool,
    dataset: str = "configs/datasets/datafusion_detection.yaml",
    method: str = "configs/methods/gru.yaml",
    experiment="configs/experiments/detection.yaml",
    tqdm: bool = False,
):
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
        run_model(
            dataset=dataset,
            method=method,
            experiment=experiment,
            train_data=temp_dir / "data",
            use_tqdm=tqdm,
        )


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    prepare_and_detect(**vars(args))
