import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport

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


def run_eval_density(
    data: Path,
    orig: Path,
    dataset: str = "datafusion",
    save_results: bool = False,
    max_rows=None,
):
    syn_path = data
    real_path = orig
    # Load
    syn_data = pd.read_csv(syn_path, nrows=max_rows)
    print("Synth ready")
    real_data = pd.read_csv(real_path, nrows=max_rows)
    print("Real ready")

    try:

        metadata_path = DIRPATH + f"/data/{dataset}/metadata_for_density.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)["metadata"]

    except FileNotFoundError:
        metadata = {
            "columns": {
                "amount": {"sdtype": "numerical"},
                "event_type": {"sdtype": "categorical"},
                "src_type11": {"sdtype": "categorical"},
                "dst_type11": {"sdtype": "categorical"},
                "src_type32": {"sdtype": "categorical"},
                "time_diff_days": {"sdtype": "numerical"},
            },
            "sequence_key": "client_id",
            "sequence_index": "event_time",
            "log_cols_for_density": ["amount"],
        }
    # Preprocess
    for col in metadata["log_cols_for_density"]:
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
