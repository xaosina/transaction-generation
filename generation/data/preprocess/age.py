from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["small_group", "age"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "bins"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing CSV files",
        default="data/age",
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        default="data/age/preprocessed",
        type=Path,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed used to split the data on train and test",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    df, df_kag_train = None, None

    df_kag_train = spark.read.csv(
        (args.data_path / "transactions_train.csv").as_posix(), header=True
    )
    df_kag_train = df_kag_train.select(
        F.col("client_id").cast(LongType()),
        F.col("trans_date").cast(LongType()),
        F.col("small_group").cast(LongType()),
        F.col("amount_rur").cast(FloatType()),
    )

    df_label = spark.read.csv(
        (args.data_path / "train_target.csv").as_posix(), header=True
    ).select(F.col("client_id").cast(LongType()), F.col("bins").cast(LongType()))

    df_kag_train = df_kag_train.join(df_label, on="client_id")
    df_kag_train = df_kag_train.withColumn("age", F.col("bins"))


    df = df_kag_train

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        vc.write(args.save_path / "cat_codes" / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    stratify_col = "bins"
    stratify_col_vals = TARGET_VALS

    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed,
    )

    train_df.repartition(20).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.repartition(3).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
