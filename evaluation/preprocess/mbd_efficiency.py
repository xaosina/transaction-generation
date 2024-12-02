from argparse import ArgumentParser
import json
from pathlib import Path
import random
import tempfile
import pandas as pd
import numpy as np
from .common import csv_to_parquet


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--tabsyn", type=bool, action="store_true"
    )
    parser.add_argument("-n", "--n-rows", type=int)
    parser.add_argument("-s", "--sample-size", type=int)
    parser.add_argument(
        "-d", "--data", help="generated data csv path", type=Path, required=True
    )
    parser.add_argument(
        "-s",
        "--save-path",
        help="Where to save preprocessed parquets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--orig",
        help="Path to orig dataset containing CSV files",
        default="data/mbd/mbd_test.csv",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    return parser.parse_args()


def prepare_tabsyn(data_path, n_rows, metadata):
    print("Prepare tabsyn started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    assert (df.shape[0] % n_rows) == 0
    df["client_id"] = np.repeat(np.arange(df.shape[0] // n_rows), n_rows)
    print(df.columns)
    df[metadata["ordering_columns"][0]] = df.groupby("client_id")[
        "time_diff_days"
    ].cumsum()

    df["client_id"] = df["client_id"].astype(str)
    n_users = df["client_id"].nunique()
    print(f"Prepare tabsyn finished. {n_users} sequences total.")
    return df


def prepare_orig(data_path, n_rows, sample_size, metadata):
    print("Prepare orig started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    print("Downloaded orig")
    client_ids = df["client_id"].unique()
    if (sample_size > 0) and (sample_size < len(client_ids)):
        print(
            f"Reducing original data from {len(client_ids)} to {sample_size}."
        )
        client_ids = df["client_id"].unique()
        gen = np.random.default_rng(0)
        train_ids = pd.Series(
            gen.choice(client_ids, size=sample_size, replace=False),
            name="client_id",
        )
        df = df.merge(train_ids, on="client_id")

    def sample_subsequences(df):
        # Slice
        lens = df.groupby("client_id").size()
        firsts = df.reset_index().groupby("client_id")["index"].first()
        start_indexes = np.random.randint(firsts, firsts + lens - n_rows + 1)
        slices = np.concatenate(
            [range(start, start + n_rows) for start in start_indexes]
        )
        df = df.loc[slices]
        return df

    if n_rows == 1:
        print("Using all rows in orig, assuming full reconstruction.")
        range_number = df.groupby("client_id").cumcount()
        df["client_id"] = range_number.astype(str) + "_" + df["client_id"].astype(str)
        df[metadata["ordering_columns"][0]] = df["time_diff_days"]
    elif n_rows > 1:
        # Filter
        df = df.groupby("client_id").filter(lambda x: len(x) >= n_rows)
        df = sample_subsequences(df)

        time_offset = df.groupby("client_id")[
            metadata["ordering_columns"][0]
        ].transform("first") - df.groupby("client_id")["time_diff_days"].transform(
            "first"
        )
        df[metadata["ordering_columns"][0]] -= time_offset

    df["client_id"] = "orig_" + df["client_id"].astype(str)
    print(f"Prepare orig finished. {df['client_id'].nunique()} sequences total.")
    return df


def main(
    is_tabsyn: bool,
    data: Path,  # train
    orig: Path,  # test
    n_rows: int,
    save_path: Path,  # Going to save in temp_dir/train and temp_dir/test
    sample_size: int = -1,  # n_users from train
    overwrite: bool = False,
):
    with open(orig.with_name("metadata_for_efficiency.json"), "r") as f:
        metadata = json.load(f)["METADATA"]
    random.seed(42)
    np.random.seed(42)
    if is_tabsyn:
        gen_df = prepare_tabsyn(data, n_rows, metadata)
    else:
        gen_df = prepare_orig(data, n_rows, sample_size, metadata)

    orig_df = prepare_orig(orig, n_rows, -1, metadata)

    assert set(orig_df.columns) == set(gen_df.columns), (
        orig_df.columns,
        gen_df.columns,
    )

    # Create a temporary file to save the CSV
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_csv_file:
        print("Temp csv", temp_csv_file.name)
        temp_csv_path = temp_csv_file.name
        gen_df.to_csv(temp_csv_path)
        csv_to_parquet(
            data=temp_csv_path,
            save_path=save_path / "train",
            metadata=metadata,
            cat_codes_path=None,
            overwrite=overwrite,
        )
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_csv_file:
        print("Temp csv", temp_csv_file.name)
        temp_csv_path = temp_csv_file.name
        orig_df.to_csv(temp_csv_path)
        csv_to_parquet(
            data=temp_csv_path,
            save_path=save_path / "test",
            metadata=metadata,
            cat_codes_path=save_path / "train/cat_codes",
            overwrite=overwrite,
        )
        


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    main(**vars(args))
