from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from datafusion_age_prediction import csv_to_parquet

CAT_FEATURES = ["mcc_code", "currency_rk"]
NUM_FEATURES = ["days_since_first_tx", "transaction_amt"]
INDEX_COLUMNS = ["user_id", "customer_age"]
ORDERING_COLUMNS = ["days_since_first_tx"]
TARGET_VALS = [0, 1, 2, 3]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-path",
        help="Path to directory containing CSV files",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--save-path",
        help="Where to save preprocessed parquets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-c",
        "--cat-codes-path",
        help="Path where to save codes for categorical features",
        type=Path,
        default="data/datafusion/orig_train/cat_codes",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument("-n", "--n-rows", type=int)
    return parser.parse_args()


def slice_tabsyn(data_path, n_rows):
    print("Prettyfication started")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    assert (df.shape[0] % n_rows) == 0
    df["user_id"] = np.repeat(np.arange(df.shape[0] // n_rows), n_rows)
    df['days_since_first_tx'] = df.groupby('user_id')['time_diff_days'].cumsum()

    most_common_age = df.groupby('user_id')['customer_age'].agg(lambda x: x.mode()[0])
    age_mapping = most_common_age.to_dict()
    df['customer_age'] = df['user_id'].map(age_mapping)
    age_uniqueness  = df.groupby('user_id')['customer_age'].nunique()
    assert age_uniqueness.eq(1).all(), age_uniqueness[age_uniqueness > 1]

    new_data_path = data_path.with_stem(data_path.stem + "_pretty")
    df.to_csv(new_data_path, index=False)
    print("Pretty finished")
    return new_data_path

if __name__ == "__main__":
    args = vars(parse_args())
    new_data_path = slice_tabsyn(args.pop("data_path"), args.pop("n_rows"))
    csv_to_parquet(new_data_path, **args)
