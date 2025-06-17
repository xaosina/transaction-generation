import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from generation.data.preprocess.quantile_transformer import QuantileTransformerTorch
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from .common import csv_to_parquet


def spark_connection():
    return SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore


def select_clients(df, feature_name, cn):
    clients_with_counts = df.groupBy(feature_name).count()
    sampled_clients = clients_with_counts.orderBy(F.rand()).limit(cn)
    selected_client_ids = sampled_clients.select(feature_name)
    filtered_df = df.join(selected_client_ids, on=feature_name, how="inner")
    return filtered_df


def set_time_features(df, user_id, time_feature):

    window_spec = Window.partitionBy(user_id).orderBy(time_feature)
    df = df.withColumn("previous_event_time", F.lag(time_feature).over(window_spec))
    df = df.withColumn(
        "time_diff_days",
        (
            F.unix_timestamp(F.col(time_feature))
            - F.unix_timestamp(F.col("previous_event_time"))
        )
        / (60 * 60 * 24),
    )

    window_spec_min = Window.partitionBy(user_id)
    df = df.withColumn("first_event_time", F.min(time_feature).over(window_spec_min))
    df = df.withColumn(
        "days_since_first_tx",
        (
            F.unix_timestamp(F.col(time_feature))
            - F.unix_timestamp(F.col("first_event_time"))
        )
        / (60 * 60 * 24),
    )
    df = df.drop("previous_event_time", "first_event_time")
    df = df.withColumn(
        time_feature, F.unix_timestamp(time_feature).cast("double") / (24 * 60 * 60)
    )
    return df.dropna()


def split_train_test_by_date(
    df: DataFrame, time_col: str = "datetime", cutoff_date: str = "2023-03-15"
) -> tuple[DataFrame, DataFrame]:
    df_dates = df.withColumn("_event_date", F.to_date(F.col(time_col)))

    cutoff = F.to_date(F.lit(cutoff_date), "yyyy-MM-dd")

    train_df = df_dates.filter(F.col("_event_date") <= cutoff).drop("_event_date")
    test_df = df_dates.filter(F.col("_event_date") > cutoff).drop("_event_date")

    return train_df, test_df


def seq_len_filter(df, user_id, min_seq_len=512):
    seq_len = df.groupby(user_id).agg(F.count("*").alias("seq_len"))
    return (
        df.join(seq_len, on=user_id, how="inner")
        .filter(F.col("seq_len") > min_seq_len)
        .drop("seq_len")
    )


METADATA = {
    "cat_features": [
        "session_id",
        "track_id",
    ],
    "num_features": [
        "play_duration",
        "datetime",
        "time_diff_days",
    ],
    "index_columns": ["user_id"],
    "target_columns": [],
    "ordering_columns": ["days_since_first_tx"],
}


def prepocess_full_mbd(data_path, temp_path: Path, clients_number: int):

    spark = spark_connection()
    spark_df = spark.read.parquet(data_path)
    spark_df = spark_df.dropna()

    spark_df = seq_len_filter(spark_df, "user_id")

    spark_df = select_clients(spark_df, "user_id", 50_000)

    user_counts = spark_df.select("user_id").distinct().count()

    print(f"Total user counts {user_counts}")

    train_dataset, test_dataset = split_train_test_by_date(
        df=spark_df, time_col="datetime", cutoff_date="2023-03-15"
    )
    time_feature = "datetime"
    user_id = "user_id"

    train_dataset = set_time_features(train_dataset, user_id, time_feature)
    test_dataset = set_time_features(test_dataset, user_id, time_feature)

    train_dataset.write.parquet(
        (temp_path / ".train.parquet").as_posix(), header=True, mode="overwrite"
    )
    test_dataset.write.parquet(
        (temp_path / ".test.parquet").as_posix(), header=True, mode="overwrite"
    )

    spark.stop()


def main(data_path, dataset_name="mbd-50k", clients_number=1_000):
    temp_path = Path("data/temp")
    dataset_path = Path(f"data/{dataset_name}")

    prepocess_full_mbd(data_path, temp_path, clients_number)

    csv_to_parquet(
        temp_path / ".train.csv",
        save_path=dataset_path,
        cat_codes_path=None,
        metadata=METADATA,
        overwrite=True,
    )

    csv_to_parquet(
        temp_path / ".test.csv",
        save_path=dataset_path,
        cat_codes_path=dataset_path / "cat_codes",
        idx_codes_path=dataset_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )

    for file in temp_path.glob("quantile_transform_*.pt"):
        target = dataset_path / file.name
        shutil.move(file, target)
        print(f"Moved: {file} -> {target}")

    if temp_path.exists() and temp_path.is_dir():
        shutil.rmtree(temp_path)
        print(f"Temp directory was removed: {temp_path}")
    else:
        print(f"!WARNING! Temp directory is not found: {temp_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and convert MBD dataset to Parquet"
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--dataname", type=str, default="mbd-50k", help="Dataset name")
    parser.add_argument(
        "--remove-temp", type=bool, default=False, help="Cleanup temp files"
    )
    parser.add_argument("--nc", type=int, default=1_000, help="Number of clients")
    args = parser.parse_args()

    main(data_path=args.data_path, dataset_name=args.dataname, clients_number=args.nc)
