import argparse
import logging

import os
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_unique_folder_suffix(folder_path):
    folder_path = str(folder_path)
    if not os.path.exists(folder_path):
        return ""
    n = 1
    while True:
        suffix = f"({n})"
        if not os.path.exists(folder_path + suffix):
            return suffix
        n += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, default="data/datafusion/preprocessed_with_id_train.csv"
    )
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=128)

    parser.add_argument("-s", "--sample-size", type=float, default=1.)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:3")
    parser.add_argument("-n", "--name", type=str, default="test")
    args = parser.parse_args()
    return args


def process_datafusion(df, sample_size=1.0):
    df["mcc_code"] = df["mcc_code"].astype("category")
    df["currency_rk"] = df["currency_rk"].astype("category")
    client_ids = df["user_id"].unique()
    n_clients = int(len(client_ids) * sample_size)

    gen = np.random.default_rng(0)
    train_ids = pd.Series(
        gen.choice(client_ids, size=n_clients, replace=False),
        name="user_id",
    )
    dataset = df.merge(train_ids, on="user_id")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)
    metadata.update_columns_metadata(
        {
            "user_id": {"sdtype": "id"},
            "mcc_code": {"sdtype": "categorical"},
        }
    )
    metadata.set_sequence_key("user_id")
    metadata.set_sequence_index("days_since_first_tx")
    return dataset, metadata


if __name__ == "__main__":
    args = parse_args()
    dataset, metadata = process_datafusion(pd.read_csv(args.data_path), args.sample_size)

    synthesizer = PARSynthesizer(
        metadata,
        context_columns=["customer_age", "dummy_binclass"],
        epochs=args.epochs,
        verbose=True,
        cuda=args.gpu,
    )
    synthesizer.fit(dataset)

    log_dir = f"log/generation/{args.name}"
    log_dir = log_dir + get_unique_folder_suffix(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    synthesizer.save(f"{log_dir}/my_synthesizer.pkl")
    # synthesizer = PARSynthesizer.load("log/my_synthesizer.pkl") # CAUSES GRU CONTUGUOUS HUINYU
    synthetic_data = synthesizer.sample(num_sequences=20_000)
    synthetic_data.to_csv(f"{log_dir}/synthetic.csv")
