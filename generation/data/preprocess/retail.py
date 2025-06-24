from argparse import ArgumentParser
from pathlib import Path

from pyspark.sql import functions as F, Window
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType, TimestampType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = [
    # "store_id",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "segment_id",
    # "brand_id",
    # "vendor_id",
    "gender",
    "is_own_trademark",
    "is_alcohol",
]
INDEX_COLUMNS = [
    "client_id",
    # "first_issue_date",
    # "first_redeem_date",
]
ORDERING_COLUMNS = [
    "transaction_datetime",
]
AGE_BOUNDS = [10.0, 35.0, 45.0, 60.0, 90.0]
TEST_FRACTION = 0.2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing CSV files",
        default="data/retail/data",
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        default="data/retail/preprocessed",
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

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df_prods = (
        spark.read.csv((args.data_path / "products.csv").as_posix(), header=True)
        .withColumn("netto", F.col("netto").cast(FloatType()))
        .withColumn("is_own_trademark", F.col("is_own_trademark").cast(LongType()))
        .withColumn("is_alcohol", F.col("is_alcohol").cast(LongType()))
    )

    df_clients = (
        spark.read.csv((args.data_path / "clients.csv").as_posix(), header=True)
        .withColumn("first_issue_date", F.col("first_issue_date").cast(TimestampType()))
        .withColumn(
            "first_redeem_date", F.col("first_redeem_date").cast(TimestampType())
        )
        .withColumn("age", F.col("age").cast(FloatType()))
        .filter("age >= 10.0 and age <= 90.0")  # as in CoLES
    )

    df_tx = (
        spark.read.csv((args.data_path / "purchases.csv").as_posix(), header=True)
        .withColumn(
            "transaction_datetime", F.col("transaction_datetime").cast(TimestampType())
        )
        .withColumn(
            "regular_points_received",
            F.col("regular_points_received").cast(FloatType()),
        )
        .withColumn(
            "express_points_received",
            F.col("express_points_received").cast(FloatType()),
        )
        .withColumn(
            "regular_points_spent", F.col("regular_points_spent").cast(FloatType())
        )
        .withColumn(
            "express_points_spent", F.col("express_points_spent").cast(FloatType())
        )
        .withColumn("purchase_sum", F.col("purchase_sum").cast(FloatType()))
        .withColumn("product_quantity", F.col("product_quantity").cast(FloatType()))
        .withColumn("trn_sum_from_iss", F.col("trn_sum_from_iss").cast(FloatType()))
        .withColumn("trn_sum_from_red", F.col("trn_sum_from_red").cast(FloatType()))
    )

    mode = "overwrite" if args.overwrite else "error"

    df = df_tx.join(df_prods, on="product_id").join(df_clients, on="client_id")

    df = df.drop(
        "transaction_id",
        "product_id",
        "first_issue_date",
        "first_redeem_date",
        "brand_id",
        "trn_sum_from_red",
        "vendor_id",
        "store_id",
    )

    # Filter out clients that have any NaN or NULL in 'netto'
    original_clients = df.select("client_id").distinct().count()
    w = Window.partitionBy("client_id")
    df_with_flag = df.withColumn(
        "has_nan_or_null",
        F.max(F.when(F.isnan("netto") | F.col("netto").isNull(), 1).otherwise(0)).over(
            w
        ),
    )
    df = df_with_flag.filter(F.col("has_nan_or_null") == 0).drop("has_nan_or_null")
    filtered_clients = df.select("client_id").distinct().count()
    print(f"Number of clients dropped: {original_clients - filtered_clients}")

    def filter_time(df, start_date, end_date):
        w_hist = df.filter(
            f"transaction_datetime >= '{start_date}' and transaction_datetime < '{end_date}'"
        )
        # fix time
        w_hist = w_hist.withColumn(
            "transaction_datetime",
            (
                F.unix_timestamp(F.col("transaction_datetime"))
                - F.unix_timestamp(F.to_timestamp(F.lit(start_date), "yyyy-MM-dd"))
            )
            / (60 * 60 * 24),
        )
        return w_hist
    
    df = filter_time(df, "2018-11-22", "2019-03-19")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        vc.write(args.save_path / "cat_codes" / vc.feature_name, mode=mode)

    # def extract_data(
    #     df,
    #     start_date: str,
    #     mid_date: str,
    # ):
    #     # Choose time interval
    #     w_hist = df.filter(
    #         f"transaction_datetime >= '{start_date}' and transaction_datetime < '{mid_date}'"
    #     )
    #     # fix time
    #     w_hist = w_hist.withColumn(
    #         "transaction_datetime",
    #         (
    #             F.unix_timestamp(F.col("transaction_datetime"))
    #             - F.unix_timestamp(F.to_timestamp(F.lit(start_date), "yyyy-MM-dd"))
    #         )
    #         / (60 * 60 * 24),
    #     )
    #     return w_hist

    # # Min-Max ('2018-11-21 21:02:33'), Timestamp('2019-03-18 23:40:03')

    # df_train = extract_data(df, "2018-11-22", "2019-01-21")
    # df_test = extract_data(df, "2019-01-20", "2019-03-19")

    # train_df = collect_lists(
    #     df_train, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS
    # )
    # test_df = collect_lists(df_test, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        random_seed=args.split_seed,
    )

    train_df.repartition(20).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(3).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )


if __name__ == "__main__":
    main()
