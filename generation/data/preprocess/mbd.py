import pickle
from typing import Dict, List

import numpy as np
import torch
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder

from .common import csv_to_parquet


# TODO: Эту функцию в препроцесс паркетов
def encode_client_ids(train_clients: Dict, test_clients: Dict, config) -> List[Dict]:
    clients = np.hstack((train_clients, test_clients))

    encoder = LabelEncoder()
    encoder.fit(clients)
    train_clients = encoder.transform(train_clients)
    test_clients = encoder.transform(test_clients)

    config["client_ids_encoder_save_path"] = (
        f"{config['vae_ckpt_dir']}/client_ids_encoder.pickle"
    )

    with open(config["client_ids_encoder_save_path"], "wb") as file:
        pickle.dump(encoder, file)

    return torch.tensor(train_clients), torch.tensor(test_clients)


def spark_connection():
    return (
        SparkSession.builder.appName("ClientDataProcessing")
        .config("spark.executor.cores", "22")
        .config("spark.executor.memory", "32g")
        .config("spark.driver.cores", "22")
        .config("spark.driver.memory", "32g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.shuffle.partitions", "1500")
        .config(
            "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors",
            "G1 Young Generation",
        )
        .config(
            "spark.eventLog.gcMetrics.oldGenerationGarbageCollectors",
            "G1 Old Generation",
        )
        .getOrCreate()
    )


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


def prepocess_full_mbd():
    spark = spark_connection()
    spark_df = spark.read.parquet("/home/dev/sb-proj/data/mbd-dataset/detail/trx")

    spark_df = spark_df.dropna()

    spark_df = choose_years(spark_df, [2021, 2022])

    spark_df = filter_by_trx_in_month(spark_df, 2, 100)

    spark_df = select_clients(spark_df, 50_000)

    spark_df.write.csv("data/temp/.temp.csv", header=True)

    spark_df = spark.read.csv("data/temp/.temp.csv", header=True, inferSchema=True)

    train_dataset, test_dataset = split_train_test(spark_df)

    train_dataset = set_time_features(train_dataset)
    test_dataset = set_time_features(test_dataset)

    train_dataset.write.csv("data/temp/.train.csv", header=True, mode="overwrite")
    test_dataset.write.csv("data/temp/.test.csv", header=True, mode="overwrite")

    spark.stop()


def main():
    prepocess_full_mbd()

    csv_to_parquet(
        "data/temp/.train.csv",
        save_path="data/mbd-50k/",
        cat_codes_path=None,
        metadata=METADATA,
        overwrite=True,
    )

    csv_to_parquet(
        "data/temp/.test.csv",
        save_path="data/mbd-50k/",
        cat_codes_path="data/mbd-50k/cat_codes/",
        idx_codes_path="data/mbd-50k/idx",
        metadata=METADATA,
        overwrite=True,
    )


if __name__ == "__main__":
    # TODO: Make args for path to dataset (in/out).
    # TODO: Delete temp files after script.
    # TODO: Refactor paths and strs and consts.
    main()
