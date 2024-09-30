from argparse import ArgumentParser
from pathlib import Path
import random
import tempfile
import pandas as pd
import numpy as np
from .common import csv_to_parquet

METADATA = {
    "cat_features": ["event_type", "event_subtype", "currency", "src_type11", "src_type12", "dst_type11", "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32"],
    "num_features": ["amount"],
    "index_columns": ["client_id", "generated"],
    "ordering_columns": ["days_since_first_tx"],
}


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


def prepare_tabsyn(data_path, n_rows):
    print("Prepare tabsyn started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    assert (df.shape[0] % n_rows) == 0
    df["client_id"] = np.repeat(np.arange(df.shape[0] // n_rows), n_rows)
    df["days_since_first_tx"] = df.groupby("client_id")["time_diff_days"].cumsum()

    df["generated"] = 1

    df["client_id"] = pd.factorize(df["client_id"])[0]
    n_users = df["client_id"].nunique()
    print("Prepare tabsyn finished")
    return df, n_users


def prepare_generated(data_path):
    print("Prepare generated started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df["generated"] = 1
    df["client_id"] = pd.factorize(df["client_id"])[0]
    n_users = df["client_id"].nunique()
    print("Prepare tabsyn finished")
    return df, n_users


def prepare_orig(data_path, n_rows, n_users):
    print("Prepare orig started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

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

    if n_rows > 0:
        # Filter
        df = df.groupby("client_id").filter(lambda x: len(x) >= n_rows)
        df = sample_subsequences(df)

        time_offset = df.groupby("client_id")["days_since_first_tx"].transform(
            "first"
        ) - df.groupby("client_id")["time_diff_days"].transform("first")
        df["days_since_first_tx"] -= time_offset

    client_ids = df["client_id"].unique()
    if (n_users > 0) and (n_users < len(client_ids)):
        print(
            f"Reducing original data from {len(client_ids)} to {n_users} users to match the numbers"
        )
        client_ids = df["client_id"].unique()
        gen = np.random.default_rng(0)
        train_ids = pd.Series(
            gen.choice(client_ids, size=n_users, replace=False),
            name="client_id",
        )
        df = df.merge(train_ids, on="client_id")
    df["client_id"] = pd.factorize(df["client_id"])[0]
    df["client_id"] += n_users
    df["generated"] = 0
    print("Prepare orig finished")
    return df


def main(
    data_type: str,
    data: Path,
    orig: Path,
    n_rows: int,
    match_users: bool,
    save_path: Path,
    overwrite: bool = False,
):
    random.seed(42)
    np.random.seed(42)
    if data_type == "tabsyn":
        gen_df, n_users = prepare_tabsyn(data, n_rows)
    else:
        gen_df, n_users = prepare_generated(data)

    if not match_users:
        n_users = -1

    orig_df = prepare_orig(orig, n_rows, n_users)

    assert set(orig_df.columns) == set(gen_df.columns), (
        orig_df.columns,
        gen_df.columns,
    )

    final_df = pd.concat([gen_df, orig_df], ignore_index=True)

    # Create a temporary file to save the CSV
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_csv_file:
        print("Temp csv", temp_csv_file.name)
        temp_csv_path = temp_csv_file.name
        final_df.to_csv(temp_csv_path)

        # Convert CSV to Parquet
        csv_to_parquet(
            data=temp_csv_path,
            save_path=save_path,
            metadata=METADATA,
            cat_codes_path=None,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    main(**vars(args))
