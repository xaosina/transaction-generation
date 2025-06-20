import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from generation.data.preprocess.quantile_transformer import QuantileTransformerTorch
from generation.data.preprocess.common import code_categories, code_indexes
from .common import csv_to_parquet


def spark_connection():
    return SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore


def select_clients(df, cn):
    clients_with_counts = df.groupBy("client_id").count()
    sampled_clients = clients_with_counts.orderBy(F.rand()).limit(cn)
    selected_client_ids = sampled_clients.select("client_id")
    filtered_df = df.join(selected_client_ids, on="client_id", how="inner")
    return filtered_df


def set_time_features(df):

    window_spec = Window.partitionBy("client_id").orderBy("event_time")
    df = df.withColumn("previous_event_time", F.lag("event_time").over(window_spec))
    df = df.withColumn(
        "time_diff_days",
        (F.col("event_time").cast("long") - F.col("previous_event_time").cast("long"))
        / (60 * 60 * 24),
    )

    window_spec_min = Window.partitionBy("client_id")
    df = df.withColumn("first_event_time", F.min("event_time").over(window_spec_min))
    df = df.withColumn(
        "days_since_first_tx",
        (F.unix_timestamp("event_time") - F.unix_timestamp("first_event_time"))
        / (60 * 60 * 24),
    )
    df = df.drop("previous_event_time", "first_event_time")
    df = df.withColumn(
        "event_time", F.unix_timestamp("event_time").cast("double") / (24 * 60 * 60)
    )
    return df.dropna()


def choose_years(df, years):
    df = df.withColumn("year", F.year("event_time")).withColumn(
        "month", F.month("event_time")
    )
    df = df.withColumn("year_month", F.expr("year * 100 + month"))
    return df.filter(df.year.isin(years))


def filter_by_trx_in_month(df, min_trx, max_trx):

    window_spec = Window.partitionBy("client_id").orderBy("year_month")
    df = df.withColumn(
        "month_diff", F.col("year_month") - F.lag("year_month", 1).over(window_spec)
    )
    monthly_df = df.groupBy("client_id").agg(
        F.count_distinct("year_month").alias("cum_year_month"),
    )

    monthly_df = monthly_df.filter(monthly_df.cum_year_month == 24)
    df = df.join(monthly_df.select("client_id"), on="client_id", how="inner")
    trx_count = df.groupBy("client_id", "year_month").agg(
        F.count("*").alias("sum"),
    )

    trx_counter = trx_count.groupby("client_id").agg(
        F.min("sum").alias("min_trx_count"), F.max("sum").alias("max_trx_count")
    )

    trx_counter = trx_counter.filter(
        ((F.col("min_trx_count") > min_trx) & (F.col("max_trx_count") < max_trx))
    )

    return df.join(trx_counter.select("client_id"), on="client_id", how="inner")


def split_train_test(df):
    return df.filter(df.year == "2021"), df.filter(df.year == "2022")


METADATA = {
    "cat_features": [
        "event_type",
        "event_subtype",
        "currency",
        "src_type11",
        "src_type12",
        "dst_type11",
        "dst_type12",
        "src_type21",
        "src_type22",
        "src_type31",
        "src_type32",
    ],
    "num_features": [
        "amount",
        "event_time",
        "time_diff_days",
    ],
    "index_columns": ["client_id"],
    "target_columns": [],
    "ordering_columns": ["days_since_first_tx"],
}


def prepocess_full_mbd(temp_path: Path, clients_number: int, dataset_path: Path):

    spark = spark_connection()
    spark_df = spark.read.parquet("/home/dev/sb-proj/data/mbd-dataset/detail/trx").limit(10000).cache()
    spark_df = spark_df.dropna()

    spark_df = choose_years(spark_df, [2021, 2022])
    spark_df = filter_by_trx_in_month(spark_df, 2, 100)
    spark_df = select_clients(spark_df, clients_number)

    spark_df.write.parquet((temp_path / ".temp.parquet").as_posix())

    spark_df = spark.read.parquet((temp_path / ".temp.parquet").as_posix())
    save_quantile_statistic(spark_df, temp_path, features_to_transform=["amount"])
    
    spark_df = code_indexes(spark_df, dataset_path, METADATA['index_columns'])
    spark_df = code_categories(spark_df, dataset_path, METADATA['cat_features'])

    train_dataset, test_dataset = split_train_test(spark_df)

    train_dataset = set_time_features(train_dataset)
    test_dataset = set_time_features(test_dataset)

    train_dataset.write.csv(
        (temp_path / ".train.csv").as_posix(), header=True, mode="overwrite"
    )
    test_dataset.write.csv(
        (temp_path / ".test.csv").as_posix(), header=True, mode="overwrite"
    )

    spark.stop()

    return #path


def save_quantile_statistic(
    spark_df, temp_path: Path | str, features_to_transform=None
):
    assert (
        features_to_transform is not None
    ), "feature for quantile transform should be defined."

    for feature_name in features_to_transform:
        amount_list = (
            spark_df.select(feature_name)
            .rdd.map(lambda row, name=feature_name: row[name])
            .collect()
        )

        amount_np = np.array(amount_list, dtype=np.float64)
        amount_tensor = torch.tensor(amount_np, dtype=torch.float64)

        qt = QuantileTransformerTorch(n_quantiles=1000, output_distribution="normal")
        qt.fit(amount_tensor)

        qt.save((temp_path / f"quantile_transform_{feature_name}.pt").as_posix())


def main(dataset_name="mbd-50k", clients_number=1_000):
    temp_path = Path("data/temp-mbd")
    dataset_path = Path(f"data/{dataset_name}")

    prepocess_full_mbd(temp_path, clients_number, dataset_path)

    csv_to_parquet(
        temp_path / ".train.csv",
        save_path=dataset_path,
        cat_codes_path=dataset_path / "cat_codes",
        idx_codes_path=dataset_path / "idx",
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
    parser.add_argument("--dataname", type=str, default="mbd-50k", help="Dataset name")
    parser.add_argument(
        "--remove-temp", type=bool, default=False, help="Cleanup temp files"
    )
    parser.add_argument("--nc", type=int, default=1_000, help="Number of clients")
    args = parser.parse_args()

    main(dataset_name=args.dataname, clients_number=args.nc)
