from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import LongType, StringType, TimestampType

from common import cat_freq, collect_lists


CAT_FEATURES = ["item_id", "item_category", "behavior_type"]
INDEX_COLUMNS = ["user_id"]
ORDERING_COLUMNS = ["time_since_tuesday"]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path CSV train user",
        default="data/taobao",
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        default="data/taobao/preprocessed",
        type=Path,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df = spark.read.csv(
        (args.data_path / "tianchi_mobile_recommend_train_user.csv").as_posix(), header=True
    )
    df = df.select(
        F.col("user_id").cast(StringType()),
        F.col("item_id").cast(LongType()),
        F.col("item_category").cast(LongType()),
        F.col("behavior_type").cast(LongType()),
        F.col("time").cast(TimestampType()),
    )

    def extract_data(
        df: DataFrame,
        start_date: str,
        mid_date: str,
    ) -> DataFrame:
        # Choose time interval
        w_hist = df.filter(f"time >= '{start_date}' and time < '{mid_date}'")
        # fix time
        w_hist = w_hist.withColumn(
            "time_since_tuesday",
            (
                F.unix_timestamp(F.col("time"))
                - F.unix_timestamp(F.to_timestamp(F.lit(start_date), "yyyy-MM-dd"))
            )
            / (60 * 60 * 24),
        )
        w_hist = w_hist.drop("first_event_time")
        return w_hist

    df_train = extract_data(df, "2014-11-18", "2014-12-02")
    df_test = extract_data(df, "2014-12-02", "2014-12-16")

    vcs = cat_freq(df_train.union(df_test), CAT_FEATURES)
    for vc in vcs:
        df_train = vc.encode(df_train)
        df_test = vc.encode(df_test)
        vc.write(args.save_path / "cat_codes" / vc.feature_name, mode=mode)

    train_df = collect_lists(
        df_train, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS
    )
    test_df = collect_lists(df_test, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    train_df.repartition(20).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(3).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )


if __name__ == "__main__":
    main()
