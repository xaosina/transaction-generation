from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

_DAYS_DENOMINATOR = 60 * 60 * 24

def seq_len_filter(df, user_id, min_seq_len):
    seq_len = df.groupby(user_id).agg(F.count('*').alias("seq_len"))
    df = df.join(seq_len, on=user_id, how="inner").filter(F.col("seq_len") > min_seq_len).drop('seq_len')
    return df


def read_data(spark, dpath, debug):
    df = spark.read.parquet(dpath)
    if debug:
        df = df.limit(10000).cache()
    df = df.dropna()
    return df

def spark_connection():
    return SparkSession.builder.master("local[32]").getOrCreate()


def select_clients(df, feature_name, cn):
    clients_with_counts = df.groupBy(feature_name).count()
    sampled_clients = clients_with_counts.orderBy(F.rand()).limit(cn)
    selected_client_ids = sampled_clients.select(feature_name)
    filtered_df = df.join(selected_client_ids, on=feature_name, how="inner")
    return filtered_df


def set_time_features(df, user_id, time_feature, denom=_DAYS_DENOMINATOR, time_name="days_since_first_tx"):

    window_spec = Window.partitionBy(user_id).orderBy(time_feature)
    df = df.withColumn("previous_event_time", F.lag(time_feature).over(window_spec))
    df = df.withColumn(
        "time_diff_days",
        (
            F.unix_timestamp(F.col(time_feature))
            - F.unix_timestamp(F.col("previous_event_time"))
        )
        / denom,
    )

    window_spec_min = Window.partitionBy(user_id)
    df = df.withColumn("first_event_time", F.min(time_feature).over(window_spec_min))
    df = df.withColumn(
        time_name,
        (
            F.unix_timestamp(F.col(time_feature))
            - F.unix_timestamp(F.col("first_event_time"))
        )
        / denom,
    )
    df = df.drop("previous_event_time", "first_event_time")
    df = df.withColumn(
        time_feature, F.unix_timestamp(time_feature).cast("double") / denom
    )
    return df.dropna()
