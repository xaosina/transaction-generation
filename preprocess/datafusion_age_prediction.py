from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists, CatMap


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
        default="data/datafusion/orig_train/cat_codes"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    return parser.parse_args()


def main(data_path, save_path, cat_codes_path=None, overwrite=False):
    mode = "overwrite" if overwrite else "error"
    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    df = spark.read.csv(data_path.as_posix(), header=True)

    assert not (
        set(INDEX_COLUMNS) & set(CAT_FEATURES)
        or set(INDEX_COLUMNS) & set(NUM_FEATURES)
        or set(NUM_FEATURES) & set(CAT_FEATURES)
    ), "Sets intersect"
    for_selection = []
    for_selection += [F.col(name).cast(LongType()) for name in INDEX_COLUMNS]
    for_selection += [F.col(name).cast(LongType()) for name in CAT_FEATURES]
    for_selection += [F.col(name).cast(FloatType()) for name in NUM_FEATURES]
    df = df.select(*for_selection)

    if cat_codes_path is None:
        print("Creating new cat codes.")
        vcs = cat_freq(df, CAT_FEATURES)
        for vc in vcs:
            df = vc.encode(df)
            vc.write(save_path / "cat_codes" / vc.feature_name, mode=mode)
    else:
        print("Reading cat codes.")
        for cat_col in CAT_FEATURES:
            vc = CatMap.read(cat_codes_path / cat_col)
            df = vc.encode(df)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    df.coalesce(1).write.parquet((save_path / "data").as_posix(), mode=mode)


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    main(**vars(args))
