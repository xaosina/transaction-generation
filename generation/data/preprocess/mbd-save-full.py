import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import Window
from pyspark.sql import functions as F
from generation.data.preprocess.common import code_indexes, code_categories
from generation.data.preprocess.quantile_transformer import QuantileTransformerTorch
from generation.data.preprocess.utils import (
    read_data,
    select_clients,
    set_time_features,
    spark_connection,
)

from .common import save_to_parquet


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
    "raw_time_name": "event_time",
    "ordering_columns": ["days_since_first_tx"],
}


def prepocess_full_mbd(
    spark,
    raw_data_path,
    output_path,
    temp_path: Path,
    clients_number: int,
    debug=False,
):

    assert os.path.isdir(raw_data_path)
    df = read_data(spark, raw_data_path, debug=debug)

    df = choose_years(df, [2021, 2022])
    df = filter_by_trx_in_month(df, 2, 100)
    df = select_clients(df, METADATA["index_columns"][0], clients_number)

    df = df.persist()
    _ = df.count()  

    # save_quantile_statistic(df, temp_path, features_to_transform=["amount"])

    df = code_indexes(df, output_path, METADATA["index_columns"])
    df = code_categories(df, output_path, METADATA["cat_features"])
    full_dataset = set_time_features(
        df, METADATA["index_columns"][0], METADATA["raw_time_name"], denom=(60 * 60 * 24), time_name=METADATA["ordering_columns"][0]
    )
    full_dataset.write.csv(
        (temp_path / ".full.csv").as_posix(), header=True, mode="overwrite"
    )

    print('saved')

    return full_dataset


def main(
    raw_data_path,
    dataset_name,
    clients_number=1_000,
    debug=False,
    quantile_transform=False,
):
    spark = spark_connection()
    spark.sparkContext.setLogLevel("ERROR")

    temp_path = Path(f"data/temp-{dataset_name}")
    output_path = Path(f"data/{dataset_name}")

    df = prepocess_full_mbd(spark, raw_data_path, output_path, temp_path, clients_number, debug=debug)

    save_to_parquet(
        df,
        save_path=output_path / "full",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )
    spark.stop()

    print('parqueted')

    if temp_path.exists() and temp_path.is_dir():
        shutil.rmtree(temp_path)
        print(f"Temp directory was removed: {temp_path}")
    else:
        print(f"!WARNING! Temp directory is not found: {temp_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and convert MBD dataset to Parquet"
    )

    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name")
    parser.add_argument("--remove-temp", action="store_true", help="Cleanup temp files")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--nc", type=int, default=1_000, help="Number of clients")
    args = parser.parse_args()

    main(
        raw_data_path=args.data_path,
        dataset_name=args.dataname,
        clients_number=args.nc,
        debug=args.debug,
    )
