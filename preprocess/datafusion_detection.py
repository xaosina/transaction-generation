from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from common import csv_to_parquet


METADATA = {
    "cat_features": ["mcc_code", "currency_rk", "customer_age"],
    "num_features": ["transaction_amt"],
    "index_columns": ["user_id", "generated"],
    "ordering_columns": ["days_since_first_tx"],
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--type", type=str, choices=["general", "tabsyn"], required=True
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
        default="data/datafusion/preprocessed_with_id_test.csv",
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
    df["user_id"] = np.repeat(np.arange(df.shape[0] // n_rows), n_rows)
    df["days_since_first_tx"] = df.groupby("user_id")["time_diff_days"].cumsum()

    df["customer_age"] = df.groupby("user_id")["customer_age"].transform(lambda x: x.mode()[0])    
    age_uniqueness = df.groupby("user_id")["customer_age"].nunique()
    assert age_uniqueness.eq(1).all(), age_uniqueness[age_uniqueness > 1]
    df["generated"] = 1

    df["user_id"] = pd.factorize(df["user_id"])[0]
    n_users = df["user_id"].nunique()
    print("Prepare tabsyn finished")
    return df, n_users


def prepare_generated(data_path):
    print("Prepare generated started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df["generated"] = 1
    df["user_id"] = pd.factorize(df["user_id"])[0]
    n_users = df["user_id"].nunique()
    print("Prepare tabsyn finished")
    return df, n_users


def prepare_orig(data_path, n_rows, n_users):
    print("Prepare orig started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    def sample_subsequences(df):
        # Slice
        lens = df.groupby("user_id").size()
        firsts = df.reset_index().groupby("user_id")["index"].first()
        start_indexes = np.random.randint(firsts, firsts + lens - n_rows + 1)
        slices = np.concatenate(
            [range(start, start + n_rows) for start in start_indexes]
        )
        df = df.loc[slices]
        return df
    if n_rows > 0:
        # Filter
        df = df.groupby("user_id").filter(lambda x: len(x) >= n_rows)
        df = sample_subsequences(df)

        time_offset = (
            df.groupby("user_id")["days_since_first_tx"].transform("first") -
            df.groupby("user_id")["time_diff_days"].transform("first")
        )
        df["days_since_first_tx"] -= time_offset

    client_ids = df["user_id"].unique()
    if (n_users > 0) and (n_users < len(client_ids)):
        print(f"Reducing original data from {len(client_ids)} to {n_users} users to match the numbers")
        client_ids = df["user_id"].unique()
        gen = np.random.default_rng(0)
        train_ids = pd.Series(
            gen.choice(client_ids, size=n_users, replace=False),
            name="user_id",
        )
        df = df.merge(train_ids, on="user_id")
    df["user_id"] = pd.factorize(df["user_id"])[0]
    df["user_id"] += n_users
    df["generated"] = 0
    print("Prepare orig finished")
    return df


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    if args.type == "tabsyn":
        gen_df, n_users = prepare_tabsyn(args.data, args.n_rows)
    else:
        gen_df, n_users = prepare_generated(args.data)
    if not args.match_users:
        n_users = -1
    orig_df = prepare_orig(args.orig, args.n_rows, n_users)
    assert set(orig_df.columns) == set(gen_df.columns), (orig_df.columns, gen_df.columns)
    final_df = pd.concat([gen_df, orig_df], ignore_index=True)
    final_df.to_csv(f"log/generation/temp/{str(args.data.stem)}.csv")
    csv_to_parquet(
        data=f"log/generation/temp/{str(args.data.stem)}.csv",
        save_path=args.save_path,
        metadata=METADATA,
        cat_codes_path=None,
        overwrite=args.overwrite,
    )
