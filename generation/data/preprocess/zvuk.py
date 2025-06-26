import os
import argparse
import shutil
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from .common import save_to_parquet
from generation.data.preprocess.common import code_indexes, code_categories
from generation.data.preprocess.utils import (
    seq_len_filter,
    read_data,
    select_clients,
    set_time_features,
    spark_connection
)


METADATA = {
    "cat_features": [
        "session_id",
        "track_id",
        "artist_id",
        "cluster_id",
    ],
    "num_features": [
        "play_duration",
        "datetime",
        "time_diff_days",
    ],
    "index_columns": ["user_id"],
    "target_columns": [],
    "raw_time_name": "datetime",
    "ordering_columns": ["hours_since_first_tx"],
}


def split_train_test_by_date(
    df: DataFrame, time_col: str, cutoff_date: str = "2023-03-15"
) -> tuple[DataFrame, DataFrame]:
    df_dates = df.withColumn("_event_date", F.to_date(F.col(time_col)))

    cutoff = F.to_date(F.lit(cutoff_date), "yyyy-MM-dd")

    train_df = df_dates.filter(F.col("_event_date") <= cutoff).drop("_event_date")
    test_df = df_dates.filter(F.col("_event_date") > cutoff).drop("_event_date")

    return train_df, test_df


def prepocess_full_mbd(
    raw_data_path,
    output_path,
    temp_path: Path,
    clients_number: int,
    debug=False,
):

    spark = spark_connection()
    assert os.path.isdir(raw_data_path)

    df = read_data(spark, raw_data_path + "/zvuk-interactions.parquet", debug=debug)
    
    df_ext = read_data(spark, raw_data_path + "/zvuk-track_artist_embedding.parquet", debug=debug)
    df_ext = df_ext.select("track_id", "artist_id", "cluster_id")

    df = df.join(
        df_ext,
        on="track_id",
        how="left"
    )

    df = seq_len_filter(df, METADATA["index_columns"][0], 128)

    df = select_clients(df, METADATA["index_columns"][0], clients_number)

    df = code_indexes(df, output_path, METADATA["index_columns"])
    df = code_categories(df, output_path, METADATA["cat_features"])

    train_dataset, test_dataset = split_train_test_by_date(
        df=df, time_col=METADATA["raw_time_name"], cutoff_date="2023-03-15"
    )

    train_dataset = set_time_features(
        train_dataset, METADATA["index_columns"][0], METADATA["raw_time_name"], denom=(60 * 60), time_name=METADATA["ordering_columns"][0]
    )
    test_dataset = set_time_features(
        test_dataset, METADATA["index_columns"][0], METADATA["raw_time_name"], denom=(60 * 60), time_name=METADATA["ordering_columns"][0]
    )

    train_dataset.write.parquet(
        (temp_path / ".train.parquet").as_posix(), header=True, mode="overwrite"
    )
    test_dataset.write.parquet(
        (temp_path / ".test.parquet").as_posix(), header=True, mode="overwrite"
    )

    spark.stop()


def main(
    raw_data_path,
    dataset_name,
    clients_number=1_000,
    debug=False,
    quantile_transform=False,
):
    temp_path = Path(f"data/temp-{dataset_name}")
    output_path = Path(f"data/{dataset_name}")

    prepocess_full_mbd(
        raw_data_path, output_path, temp_path, clients_number, debug=debug
    )

    save_to_parquet(
        temp_path / ".train.csv",
        save_path=output_path / "train",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )

    save_to_parquet(
        temp_path / ".test.csv",
        save_path=output_path / "test",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )

    if quantile_transform:
        for file in temp_path.glob("quantile_transform_*.pt"):
            target = output_path / file.name
            shutil.move(file, target)
            print(f"Moved: {file} -> {target}")

    if temp_path.exists() and temp_path.is_dir():
        shutil.rmtree(temp_path)
        print(f"Temp directory was removed: {temp_path}")
    else:
        print(f"!WARNING! Temp directory is not found: {temp_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and convert ZVUK dataset to Parquet"
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
