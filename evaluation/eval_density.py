import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport


def log10_scale(x):
    linear = (x <= np.e) & (x >= -np.e)  # to match the derivatives
    y = np.where(linear, 1, x)  # to avoid np.log10 warnings
    y = np.abs(y)
    return np.where(linear, x / (np.e * np.log(10)), np.sign(x) * np.log10(y))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig",
        help="Path to orig dataset containing CSV files",
        default="data/datafusion/preprocessed_with_id_test.csv",
        type=Path,
    )
    parser.add_argument(
        "-d", "--data", help="generated data csv path", type=Path, required=True
    )
    parser.add_argument(
        "--log-cols", type=list, default=["transaction_amt"]
    )
    return parser.parse_args()

def run_eval_density(
    data: Path, 
    orig: Path=Path("data/datafusion/preprocessed_with_id_test.csv"), 
    log_cols = ["transaction_amt"]):
    
    class Args:
        pass
    args = Args()
    args.data = data
    args.orig = orig
    args.log_cols = log_cols

    syn_path = args.data
    real_path = args.orig
    # Load
    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    with open(real_path.with_name("metadata_for_density.json"), "r") as f:
        metadata = json.load(f)["metadata"]
    # Preprocess
    for col in args.log_cols:
        print(col)
        syn_data[col] = log10_scale(syn_data[col])
        real_data[col] = log10_scale(real_data[col])
    syn_data = syn_data[[col for col in syn_data.columns if col in metadata["columns"].keys()]]
    real_data = real_data[[col for col in real_data.columns if col in metadata["columns"].keys()]]
    # Calculate 
    qual_report = QualityReport()
    qual_report.generate(real_data, syn_data, metadata)
    quality = qual_report.get_properties()
    Shape = quality["Score"][0]
    Trend = quality["Score"][1]
    return dict(shape=Shape, trend=Trend)

if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    syn_path = args.data
    real_path = args.orig
    # Load
    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    with open(real_path.with_name("metadata_for_density.json"), "r") as f:
        metadata = json.load(f)["metadata"]
    # Preprocess
    for col in args.log_cols:
        print(col)
        syn_data[col] = log10_scale(syn_data[col])
        real_data[col] = log10_scale(real_data[col])
    syn_data = syn_data[[col for col in syn_data.columns if col in metadata["columns"].keys()]]
    real_data = real_data[[col for col in real_data.columns if col in metadata["columns"].keys()]]
    # Calculate 
    qual_report = QualityReport()
    qual_report.generate(real_data, syn_data, metadata)
    quality = qual_report.get_properties()
    Shape = quality["Score"][0]
    Trend = quality["Score"][1]
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