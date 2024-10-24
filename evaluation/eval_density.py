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
        "-o", "--orig",
        help="Path to orig dataset containing CSV files",
        type=Path, required=True
    )
    parser.add_argument(
        "-d", "--data", 
        help="generated data csv path", 
        type=Path, required=True
    )
    parser.add_argument(
        '--save-res', action='store_const', const=True, default=False)
    return parser.parse_args()

def run_eval_density(
    data: Path,
    orig: Path,
    save_results: bool = False):
    syn_path = data
    real_path = orig
    # Load
    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    with open(real_path.with_name("metadata_for_density.json"), "r") as f:
        metadata = json.load(f)["metadata"]
    # Preprocess
    for col in metadata["log_cols_for_density"]:
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
    run_eval_density(args.data, args.orig, save_results=args.save_res)
