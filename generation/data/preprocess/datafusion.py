import argparse
from pathlib import Path

from typing import Optional, Sequence
import numpy as np
import pandas as pd
from generation.data.preprocess.common import code_indexes, code_categories
from pyspark.sql import SparkSession
from .common import save_to_parquet
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

METADATA = {
    "cat_features": [
        "mcc_code",
    ],
    "num_features": [
        "transaction_amt",
    ],
    "index_columns": ["user_id"],
    "target_columns": [],
    "ordering_columns": ["days_since_zero"],
}


def encode_data(train_data, test_data, path):

    df_concated = train_data.unionByName(test_data).repartition(256)

    _ = code_indexes(df_concated, METADATA["index_columns"], save_path=path)
    _ = code_categories(df_concated, METADATA["cat_features"], save_path=path)


def feature_drop(
    df: pd.DataFrame,
    feature_name: str,
    drop_cats: Optional[Sequence] = None,
    remain_cats: Optional[Sequence] = None,
    user_col: str = "app_id",
) -> pd.DataFrame:
    assert (drop_cats is None) ^ (
        remain_cats is None
    ), "Задайте ровно один из параметров: drop_cats или remain_cats."

    if remain_cats is not None:
        allowed = set(remain_cats)
        mask_allowed = df[feature_name].isin(allowed)
        bad_ids = df.loc[
            ~mask_allowed, user_col
        ].unique()  # есть хоть одно запрещённое значение
        keep_mask = ~df[user_col].isin(bad_ids)
        return df.loc[keep_mask]

    # drop_cats is not None
    forbidden = set(drop_cats)
    bad_ids = df.loc[df[feature_name].isin(forbidden), user_col].unique()
    keep_mask = ~df[user_col].isin(bad_ids)
    return df.loc[keep_mask]


def split_by_user(df: pd.DataFrame, test_part: float = 0.1, random_state: int = 0):
    users = df["user_id"].unique()
    train_users, test_users = train_test_split(
        users, test_size=test_part, random_state=random_state
    )
    return df[df["user_id"].isin(train_users)].reset_index(drop=True), df[
        df["user_id"].isin(test_users)
    ].reset_index(drop=True)


def extract_time_feature(df: pd.DataFrame):
    df["transaction_dttm"] = pd.to_datetime(df["transaction_dttm"]).dt.floor("us")

    min_date = df["transaction_dttm"].min()
    df["days_since_zero"] = (df["transaction_dttm"] - min_date).dt.days.astype("int32")
    return df.drop(columns=["transaction_dttm"])


def preprocess_full_datafusion(
    raw_data_path,
    clients_number: int,
    output_path: Path,
) -> list:
    assert isinstance(raw_data_path, str)
    df = pd.read_csv(raw_data_path)
    print("Original size:", df.shape)
    df = feature_drop(df, "currency_rk", remain_cats=[48], user_col="user_id")
    df = feature_drop(df, 'mcc_code', drop_cats=df['mcc_code'].value_counts().index[-20:].tolist(), user_col='user_id')

    df = extract_time_feature(df)
    train, test = split_by_user(df, 0.3)

    train.to_parquet(output_path / "temp_train.parquet")
    test.to_parquet(output_path / "temp_test.parquet")


def main(
    raw_data_path,
    dataset_name,
    clients_number=1_000,
):
    output_path = Path(f"data/{dataset_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info("Start data preprocessing")

    preprocess_full_datafusion(raw_data_path, clients_number, output_path)
    logging.info("Dataset was preprocessed")

    spark = SparkSession.builder.master("local[32]").getOrCreate()

    train_data = spark.read.parquet(str(output_path / "temp_train.parquet"))
    test_data = spark.read.parquet(str(output_path / "temp_test.parquet"))

    encode_data(train_data, test_data, path=output_path)
    logging.info("Encoding was saved.")

    train_data = code_indexes(
        train_data, METADATA["index_columns"], load_path=output_path / "idx"
    )
    logging.info("Indexes was created")
    train_data = code_categories(
        train_data, METADATA["cat_features"], load_path=output_path / "cat_codes"
    )
    logging.info("Codes was created")

    test_data = code_indexes(
        test_data, METADATA["index_columns"], load_path=output_path / "idx"
    )
    logging.info("Indexes was created")
    test_data = code_categories(
        test_data, METADATA["cat_features"], load_path=output_path / "cat_codes"
    )
    logging.info("Codes was created")

    logging.info("Saving to parquet...")
    save_to_parquet(
        train_data,
        save_path=output_path / "train",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )
    logging.info("Train was saved.")

    save_to_parquet(
        test_data,
        save_path=output_path / "test",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )
    logging.info("Test was saved.")
    logging.info("Dataset was successfully preprocessed.")
    spark.stop()
    logging.info("Spark was successfully stopped.")


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
    )
