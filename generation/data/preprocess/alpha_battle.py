import argparse
from pathlib import Path

from typing import Optional, Sequence
import numpy as np
import pandas as pd
from generation.data.preprocess.common import code_indexes, code_categories
from pyspark.sql import SparkSession
from .common import save_to_parquet
import logging
logging.basicConfig(level=logging.INFO)

MONTH_DAYS = 30
MONTHS = 12

def filter_by_trx_in_month(df: pd.DataFrame, min_trx, max_trx) -> pd.DataFrame:

    df["month_before"] = (df["days_before"] // MONTH_DAYS) + 1
    assert df["month_before"].max() == 12, df["month_before"].max()
    assert df["month_before"].min() == 1, df["month_before"].min()

    trx_count = (
        df.groupby(["app_id", "month_before"]).size().reset_index(name="trx_sum")
    )

    trx_counter = (
        trx_count.groupby("app_id")["trx_sum"]
        .agg(
            min_trx_count="min",
            max_trx_count="max",
            month_number="count",
        )
        .reset_index()
    )

    trx_counter = trx_counter[
        (trx_counter["min_trx_count"] >= min_trx)
        & (trx_counter["max_trx_count"] <= max_trx)
        & (trx_counter["month_number"] == MONTHS)
    ]

    return df[df["app_id"].isin(trx_counter["app_id"])]


METADATA = {
    "cat_features": [
        "currency",
        "operation_kind",
        "card_type",
        "operation_type",
        "operation_type_group",
        "ecommerce_flag",
        "payment_system",
        "income_flag",
        "mcc",
        "country",
        "city",
        "mcc_category",
    ],
    "num_features": [
        "day_of_week",
        "hour",
        "amnt",
        "days_before",
        "weekofyear",
        "hour_diff",
        "transaction_number",
    ],
    "index_columns": ["app_id"],
    "target_columns": [],
    # "raw_time_name": "event_time",
    "ordering_columns": ["hours_since_first_tx"],
}


def select_clients(df: pd.DataFrame, feature_name, cn) -> pd.DataFrame:
    sampled_clients = np.random.choice(df[feature_name].unique(), cn, replace=False)
    return df[df[feature_name].isin(sampled_clients)]


def encode_data(train_data, test_data, path):

    df_concated = train_data.unionByName(test_data).repartition(256)

    _ = code_indexes(
        df_concated, METADATA["index_columns"], save_path=path
    )
    _ = code_categories(
        df_concated, METADATA["cat_features"], save_path=path
    )

def currency_drop(df: pd.DataFrame, remain_currency: int):
    currency_per_user = df.groupby("app_id")["currency"].nunique()
    keep_ids = currency_per_user[currency_per_user==remain_currency]
    return df[df["app_id"].isin(keep_ids)]

def feature_drop(
    df: pd.DataFrame,
    feature_name: str,
    drop_cats: Optional[Sequence] = None,
    remain_cats: Optional[Sequence] = None,
    user_col: str = "app_id",
) -> pd.DataFrame:
    assert (drop_cats is None) ^ (remain_cats is None), "Задайте ровно один из параметров: drop_cats или remain_cats."

    if remain_cats is not None:
        allowed = set(remain_cats)
        mask_allowed = df[feature_name].isin(allowed)
        bad_ids = df.loc[~mask_allowed, user_col].unique()  # есть хоть одно запрещённое значение
        keep_mask = ~df[user_col].isin(bad_ids)
        return df.loc[keep_mask]

    # drop_cats is not None
    forbidden = set(drop_cats)
    bad_ids = df.loc[df[feature_name].isin(forbidden), user_col].unique()
    keep_mask = ~df[user_col].isin(bad_ids)
    return df.loc[keep_mask]

def extract_time_feature(df: pd.DataFrame):
    assert df["hour_diff"].min() == -1
    df["hour_diff"] = df["hour_diff"].clip(lower=0)
    df["hours_since_first_tx"] = df.groupby("app_id")["hour_diff"].cumsum()
    return df

def preprocess_full_alphabattle(
    raw_data_path,
    clients_number: int,
    output_path: Path,
) -> list:
    assert isinstance(raw_data_path, tuple)
    clients_number = [clients_number, int(clients_number * (0.35 if clients_number != -1 else 1))]
    print(clients_number)
    for p, cn, pth in zip(['train', 'test'], clients_number, raw_data_path):
        df = pd.read_parquet(pth)
        print(pth, cn)
        print('Original size:', df.shape)
        df = feature_drop(df, 'currency', remain_cats=[1], user_col='app_id')
        df = feature_drop(df, 'payment_system', drop_cats=df['payment_system'].value_counts().index[-1:].tolist(), user_col='app_id')
        df = feature_drop(df, 'mcc_category', drop_cats=df['mcc_category'].value_counts().index[-1:].tolist(), user_col='app_id')
        df = feature_drop(df, 'card_type', drop_cats=df['card_type'].value_counts().index[-20:].tolist(), user_col='app_id')
        df = extract_time_feature(df)
        # df = filter_by_trx_in_month(df, 2, 300)
        if cn != -1:
            df = select_clients(df, METADATA["index_columns"][0], cn)
        print('Filtered size:', df.shape)
        df.to_parquet(output_path / f'temp_{p}.parquet')


def main(
    raw_data_path,
    dataset_name,
    clients_number=1_000,
    debug=False,
    quantile_transform=False,
):
    output_path = Path(f"data/{dataset_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info("Start data preprocessing")

    preprocess_full_alphabattle(raw_data_path, clients_number, output_path)
    logging.info("Dataset was preprocessed")

    spark = SparkSession.builder.master("local[32]").getOrCreate()

    train_data = spark.read.parquet(str(output_path / 'temp_train.parquet'))
    test_data  = spark.read.parquet(str(output_path / 'temp_test.parquet'))

    encode_data(train_data, test_data, path=output_path)
    logging.info("Encoding was saved.")

    train_data = code_indexes(train_data, METADATA["index_columns"], load_path=output_path / 'idx')
    logging.info("Indexes was created")
    train_data = code_categories(train_data, METADATA["cat_features"], load_path=output_path / 'cat_codes')
    logging.info("Codes was created")

    test_data = code_indexes(test_data, METADATA["index_columns"], load_path=output_path / 'idx')
    logging.info("Indexes was created")
    test_data = code_categories(test_data, METADATA["cat_features"], load_path=output_path / 'cat_codes')
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
    logging.info("Spart was successfully stopped.")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and convert MBD dataset to Parquet"
    )

    parser.add_argument(
        "--train-path", type=str, required=True, help="Path to raw data"
    )
    parser.add_argument("--test-path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name")
    parser.add_argument("--remove-temp", action="store_true", help="Cleanup temp files")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--nc", type=int, default=1_000, help="Number of clients")
    args = parser.parse_args()

    main(
        raw_data_path=(args.train_path, args.test_path),
        dataset_name=args.dataname,
        clients_number=args.nc,
        debug=args.debug,
    )
