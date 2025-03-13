import torch
import numpy as np
import pickle
import pandas as pd

from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from collections.abc import Iterable
from pathlib import Path
from typing import Any


from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import FloatType, LongType, StringType


# TODO: Эту функцию в препроцесс паркетов
def encode_client_ids(train_clients: Dict, test_clients: Dict, config) -> List[Dict]:
    clients = np.hstack((
        train_clients, test_clients
    ))
    
    encoder = LabelEncoder()
    encoder.fit(clients)
    train_clients = encoder.transform(train_clients)
    test_clients = encoder.transform(test_clients)
    
    config['client_ids_encoder_save_path'] = f"{config['vae_ckpt_dir']}/client_ids_encoder.pickle"
    
    with open(config['client_ids_encoder_save_path'], 'wb') as file:
        pickle.dump(encoder, file)
    
    return torch.tensor(train_clients), torch.tensor(test_clients)


def spark_connection():
    return SparkSession.builder \
        .appName("ClientDataProcessing") \
        .config("spark.executor.cores", "22") \
        .config("spark.executor.memory", "32g") \
        .config("spark.driver.cores", "22") \
        .config("spark.driver.memory", "32g") \
        .config("spark.driver.maxResultSize", "8g") \
        .config("spark.sql.shuffle.partitions", "1500") \
        .config("spark.eventLog.gcMetrics.youngGenerationGarbageCollectors", "G1 Young Generation") \
        .config("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Generation") \
        .getOrCreate()

def select_clients(df, cn):
    clients_with_counts = df.groupBy('client_id').count()
    sampled_clients = clients_with_counts.orderBy(F.rand()).limit(cn)
    selected_client_ids = sampled_clients.select('client_id')
    filtered_df = df.join(selected_client_ids, on='client_id', how='inner')
    return filtered_df

def set_time_features(df):
        
    window_spec = Window.partitionBy('client_id').orderBy('event_time')
    df = df.withColumn('previous_event_time', F.lag('event_time').over(window_spec))
    df = df.withColumn('time_diff_days', (F.col('event_time').cast('long') - F.col('previous_event_time').cast('long')) / (60 * 60 * 24))
    
    window_spec_min = Window.partitionBy('client_id')
    df = df.withColumn('first_event_time', F.min('event_time').over(window_spec_min))
    df = df.withColumn(
        'days_since_first_tx',
        (F.unix_timestamp('event_time') - F.unix_timestamp('first_event_time')) / (60 * 60 * 24)
    )
    df = df.drop('previous_event_time', 'first_event_time')
    df = df.withColumn('event_time', F.unix_timestamp('event_time').cast('double') / (24 * 60 * 60))
    return df.dropna()

def choose_years(df, years):
    df = df.withColumn("year", F.year("event_time")).withColumn("month", F.month("event_time"))
    df = df.withColumn("year_month", F.expr("year * 100 + month"))
    return df.filter(df.year.isin(years))

def filter_by_trx_in_month(df, min_trx, max_trx):
     
    window_spec = Window.partitionBy("client_id").orderBy("year_month")
    df = df.withColumn("month_diff", F.col("year_month") - F.lag("year_month", 1).over(window_spec))
    monthly_df = df.groupBy('client_id').agg(
        F.count_distinct('year_month').alias('cum_year_month'),
        )

    monthly_df = monthly_df.filter(monthly_df.cum_year_month == 24)
    df = df.join(monthly_df.select('client_id'), on='client_id', how='inner')
    trx_count = df.groupBy('client_id', 'year_month').agg(
        F.count('*').alias('sum'),
    )

    trx_counter = trx_count.groupby('client_id').agg(
        F.min('sum').alias('min_trx_count'),
        F.max('sum').alias('max_trx_count')
    )

    trx_counter = trx_counter.filter(
        ((F.col('min_trx_count') > min_trx) & (F.col('max_trx_count') < max_trx))
    )

    return df.join(trx_counter.select('client_id'), on='client_id', how='inner')

def split_train_test(df):
    return df.filter(df.year == '2021'), df.filter(df.year == '2022')


NA_VALUE = 0


# TODO: docs
class CatMap:
    def __init__(self, map_df: DataFrame):
        if (
            len(map_df.columns) != 2
            or map_df.columns[1] != "_code"
            or map_df.schema["_code"].dataType != LongType()
        ):
            raise ValueError(
                "DataFrame must contain exactly 2 columns. "
                "The 2nd one must be named `_code` and have type LongType"
            )

        self._df = map_df

    @property
    def feature_name(self):
        return self._df.columns[0]

    @property
    def df(self):
        return self._df

    def encode(self, df: DataFrame) -> DataFrame:
        return (
            df.join(self._df, on=self.feature_name, how="left")
            .drop(self.feature_name)
            .na.fill(NA_VALUE, subset=["_code"])
            .withColumnRenamed("_code", self.feature_name)
        )

    def decode(self, df: DataFrame) -> DataFrame:
        return (
            df.join(
                self._df.select(
                    F.col(self.feature_name).alias("_code"),
                    F.col("_code").alias(self.feature_name),
                ),
                on=self.feature_name,
                how="left",
            )
            .drop(self.feature_name)
            .withColumnRenamed("_code", self.feature_name)
        )

    def write(self, path: Path, mode: str | None = None):
        self._df.write.parquet(path.as_posix(), mode=mode)

    @classmethod
    def read(cls, path: Path):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore
        df = spark.read.parquet(path.as_posix())
        return cls(df)


def collect_lists(
    df: DataFrame,
    group_by: str | Iterable[str],
    order_by: str | Iterable[str],
) -> DataFrame:
    """Collect lists and add auxiliary columns.

    The function collect all sequence elements in the dataframe in lists grouping by
    the `group_by` columns and ordering by the `order_by` columns. It also computes the
    auziliary information: sequence lengths and the last value(s) in the `order_by`
    column(s). The latter columns are named as `order_by` columns with prefix "_last_",
    the column containing sequence lengths has name "_seq_len".

    Args:
        df: DataFrame containing all sequences.
        group_by: column(s) identifying a sequence.
        order_by: column(s) used for ordering sequences.

    Return:
        a dataframe with collected lists and auxiliary columns.
    """

    if isinstance(order_by, str):
        order_by = (order_by,)
    order_by = list(order_by)

    if isinstance(group_by, str):
        group_by = (group_by,)
    group_by = list(group_by)

    seq_cols = list(set(df.columns) - set(group_by) - set(order_by))

    return (
        df.select(*group_by, F.struct(*order_by, *seq_cols).alias("s"))
        .groupBy(*group_by)
        .agg(F.sort_array(F.collect_list("s")).alias("s"))
        .select(
            *group_by,
            *map(lambda c: "s." + c, order_by + seq_cols),
            F.size("s").alias("_seq_len"),
            *map(lambda c: F.element_at("s." + c, -1).alias("_last_" + c), order_by),
        )
    )


def cat_freq(df: DataFrame, cols: Iterable[str]) -> list[CatMap]:
    """Computes the value frequency ranks for columns.

    Counts occurencies of each value in col (excluding NULL and NaN values) and returns
    dataframes containing mappings from values in col to their frequency ranks (starting
    from 1 for the most frequent value). Each dataframe has 2 columns: 'col' with column
    values and '_code' with the frequency ranks.

    Args:
        df: dataframe.
        cols: columns for which to count the occurencies.

    Returns:
        list of dataframes with values frequency ranks for each column.
    """

    val_counts = []
    for col in cols:
        map_df = (
            df.select(col)
            .dropna(subset=[col])
            .groupBy(col)
            .count()
            .select(
                col,
                (
                    F.row_number().over(
                        # dummy partition F.lit(0) to suppress WindowExec warning
                        # "No Partition Defined for Window operation! ..."
                        Window.partitionBy(F.lit(0)).orderBy(F.col("count").desc())
                    )
                )
                .cast(LongType())
                .alias("_code"),
            )
        )
        val_counts.append(CatMap(map_df))

    return val_counts


def train_test_split(
    df: DataFrame,
    test_frac: float,
    index_col: str,
    stratify_col: str | None = None,
    stratify_col_vals: list[Any] | None = None,
    random_seed: int = 0,
) -> tuple[DataFrame, DataFrame]:
    """Split dataset on train and test parts.

    Stratified random splitting dataframe rows on train and test. It uses
    `pyspark.sql.DataFrame.sampleBy` internally with `col` parameter set to
    `stratify_col`.

    Args:
        df: dataframe to split.
        test_frac: test fraction (0 <= test_frac <= 1).
        index_col: a column with dataset items index to sample.
        stratify_col: a column to stratify by.
        stratify_col_vals: unique values of the stratification colums. Uf not set, the
            values are calculated using `pyspark.sql.DataFrame.distinct()`.
        random_seed: random seed for reproducibility.

    Returns:
        a tuple of train and test dataframes.
    """

    if stratify_col is None:
        index = df.select(index_col).coalesce(1).sort(index_col).cache()
        test_index = index.sample(fraction=test_frac, seed=random_seed).cache()
        test_df = df.join(test_index, on=index_col)
        train_index = index.select(index_col).subtract(test_index)
        train_df = df.join(train_index, on=index_col)
        return train_df, test_df

    fractions = None
    if stratify_col_vals is not None:
        fractions = {val: test_frac for val in stratify_col_vals}
    else:
        df.cache()
        fractions = (
            df.select(stratify_col)
            .distinct()
            .withColumn("_fraction", F.lit(test_frac))
            .rdd.collectAsMap()
        )

    index = df.select(index_col, stratify_col).coalesce(1).sort(index_col).cache()

    test_index = (
        index.sampleBy(stratify_col, fractions, random_seed).select(index_col).cache()
    )
    test_df = df.join(test_index, on=index_col)

    train_index = index.select(index_col).subtract(test_index)
    train_df = df.join(train_index, on=index_col)

    return train_df, test_df

def csv_to_parquet(data, save_path, metadata, cat_codes_path=None, overwrite=False):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(cat_codes_path, str):
        cat_codes_path = Path(cat_codes_path)
    mode = "overwrite" if overwrite else "error"
    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    if isinstance(data, Path) or isinstance(data, str):
        data = Path(data)
        df = spark.read.csv(data.as_posix(), header=True)
    elif isinstance(data, pd.DataFrame):
        df = spark.createDataFrame(data)
    else:
        raise TypeError
    index_columns = metadata.get("index_columns", [])
    target_columns = metadata.get("target_columns", [])
    cat_features = metadata.get("cat_features", [])
    num_features = metadata.get("num_features", [])
    ordering_columns = metadata.get("ordering_columns", [])

    assert len(index_columns + cat_features + num_features + ordering_columns) == len(
        set(index_columns + cat_features + num_features + ordering_columns)
    ), "Sets intersect"
    for_selection = []
    for_selection += [F.col(name).cast(StringType()) for name in index_columns]
    for_selection += [F.col(name).cast(LongType()) for name in target_columns]
    for_selection += [F.col(name).cast(LongType()) for name in cat_features]
    for_selection += [F.col(name).cast(FloatType()) for name in num_features]
    for_selection += [F.col(name).cast(FloatType()) for name in ordering_columns]
    df = df.select(*for_selection)

    if cat_codes_path is None:
        print("Creating new cat codes.")
        vcs = cat_freq(df, cat_features)
        for vc in vcs:
            df = vc.encode(df)
            vc.write(save_path / "cat_codes" / vc.feature_name, mode=mode)
    else:
        print("Reading cat codes.")
        for cat_col in cat_features:
            vc = CatMap.read(cat_codes_path / cat_col)
            df = vc.encode(df)

    df = collect_lists(
        df,
        group_by=index_columns + target_columns,
        order_by=ordering_columns,
    )
    df_repartitioned = df.repartition(20)

    df_repartitioned.write.parquet((save_path / ("train" if cat_codes_path is None else 'test')).as_posix(), 
                                   mode=mode)

METADATA = {
    "cat_features": ['event_type', 'event_subtype', 'currency', 'src_type11', 'src_type12', 'dst_type11', 'dst_type12', 'src_type21', 'src_type22', 'src_type31', 'src_type32',],
    "num_features": ['amount', 'event_time', 'time_diff_days',],
    "index_columns": ['client_id'],
    "target_columns": [],
    "ordering_columns": ['days_since_first_tx'],
}

def prepocess_full_mbd():
    spark = spark_connection()
    spark_df = spark.read.parquet(f'/home/dev/sb-proj/data/mbd-dataset/detail/trx')

    spark_df = spark_df.dropna()

    spark_df = choose_years(spark_df, [2021, 2022])

    spark_df = filter_by_trx_in_month(spark_df, 2, 100)

    spark_df = select_clients(spark_df, 50_000)
    
    spark_df.write.csv('data/temp/.temp.csv', header=True)

    spark_df = spark.read.csv(f'data/temp/.temp.csv', header=True, inferSchema=True)

    train_dataset, test_dataset = split_train_test(spark_df)

    train_dataset = set_time_features(train_dataset)
    test_dataset = set_time_features(test_dataset)

    train_dataset.write.csv(f'data/temp/.train.csv', header=True, mode='overwrite')
    test_dataset.write.csv(f'data/temp/.test.csv', header=True, mode='overwrite')
    
    spark.stop()

def main():
    prepocess_full_mbd()

    csv_to_parquet('data/temp/.train.csv', save_path='data/mbd-50k/', 
                   cat_codes_path=None,
                   metadata=METADATA, overwrite=True)
    
    csv_to_parquet('data/temp/.test.csv', save_path='data/mbd-50k/', 
                   cat_codes_path='data/mbd-50k/cat_codes/',
                   metadata=METADATA, overwrite=True)



if __name__ == '__main__':
    # TODO: Make args for path to dataset (in/out).
    # TODO: Delete temp files after script.
    # TODO: Refactor paths and strs and consts.
    main()