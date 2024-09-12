from argparse import ArgumentParser
from pathlib import Path
from common import csv_to_parquet

METADATA = {
    "cat_features": ["mcc_code", "currency_rk"],
    "num_features": ["transaction_amt"],
    "index_columns": ["user_id", "customer_age"],
    "ordering_columns": ["days_since_first_tx"],
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
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
        default="data/datafusion/downstream/orig_train/cat_codes",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    csv_to_parquet(**vars(args), metadata=METADATA)
