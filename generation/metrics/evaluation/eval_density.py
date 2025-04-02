import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from ...data.data_types import DataConfig

DIRPATH = os.path.dirname(__file__)


def log10_scale(x):
    linear = (x <= np.e) & (x >= -np.e)  # to match the derivatives
    y = np.where(linear, 1, x)  # to avoid np.log10 warnings
    y = np.abs(y)
    return np.where(linear, x / (np.e * np.log(10)), np.sign(x) * np.log10(y))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--orig",
        help="Path to orig dataset containing CSV files",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-d", "--data", help="generated data csv path", type=Path, required=True
    )
    parser.add_argument(
        "-m",
        "--max-rows",
        help="Maximum rows to download form csv",
        type=int,
        default=None,
    )
    parser.add_argument("--save-res", action="store_const", const=True, default=False)
    parser.add_argument(
        "--dataset", type=str, choices=["datafusion", "mdb"], default="datafusion"
    )
    return parser.parse_args()


def preproc_parquet(path, data_conf: DataConfig, tail_len):
    df = pd.read_parquet(path)
    if tail_len:
        assert df._seq_len.min() >= tail_len
        df[data_conf.seq_cols] = df[data_conf.seq_cols].map(lambda x: x[-tail_len:])
    df = df.explode(data_conf.seq_cols)
    return df


def run_eval_density(
    syn_path: Path,
    real_path: Path,
    data_conf: DataConfig,
    log_cols: list[str],
    tail_len: int = None,
    save_results: bool = False,
):
    # Load
    syn_data = preproc_parquet(syn_path, data_conf, tail_len)
    print("Synth ready")
    real_data = preproc_parquet(real_path, data_conf, tail_len)
    print("Real ready")
    # Prepare metadata
    metadata = {"columns": {}}
    for num_name in data_conf.num_names:
        metadata["columns"][num_name] = {"sdtype": "numerical"}
    for cat_name in data_conf.cat_cardinalities:
        metadata["columns"][cat_name] = {"sdtype": "categorical"}
    metadata["sequence_key"] = data_conf.index_name
    metadata["sequence_index"] = data_conf.time_name
    # Preprocess
    for col in log_cols:
        print(col)
        syn_data[col] = log10_scale(syn_data[col])
        real_data[col] = log10_scale(real_data[col])
    syn_data = syn_data[
        [col for col in syn_data.columns if col in metadata["columns"].keys()]
    ]
    real_data = real_data[
        [col for col in real_data.columns if col in metadata["columns"].keys()]
    ]
    # Calculate
    qual_report = QualityReport()
    qual_report.generate(real_data, syn_data, metadata)
    quality = qual_report.get_properties()
    Shape = quality["Score"][0]
    Trend = quality["Score"][1]
    if save_results:
        save_dir = f"log/density/{syn_path.stem}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/quality.txt", "w") as f:
            f.write(f"{Shape}\n")
            f.write(f"{Trend}\n")
        shapes = qual_report.get_details(property_name="Column Shapes")
        trends = qual_report.get_details(property_name="Column Pair Trends")
        shapes.to_csv(f"{save_dir}/shape.csv")
        trends.to_csv(f"{save_dir}/trend.csv")
    return dict(shape=Shape, trend=Trend)


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    run_eval_density(
        args.data,
        args.orig,
        dataset=args.dataset,
        save_results=args.save_res,
        max_rows=args.max_rows,
    )
