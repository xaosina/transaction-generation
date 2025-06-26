from argparse import ArgumentParser
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TEST_FRACTION = 0.2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--save-path",
        help="Path to directory containing CSV files",
        default="data/shakespeare",
        type=Path,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed used to split the data on train and test",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()

    # 1. Save raw text
    input_file_path = os.path.join(args.save_path, "input.txt")
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    # 2. Load raw text
    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # 3. Preprocess, put in dataframe
    speeches = data.split("\n\n")
    speeches = [speach.split(":\n")[1] for speach in speeches if ":\n" in speach]
    df = pd.DataFrame({"char": speeches})
    df["_seq_len"] = df["char"].map(len)
    print((df["_seq_len"] <= 32).sum(), "dropped too short")
    df = df[df["_seq_len"] > 32]
    print((df["_seq_len"] >= 400).sum(), "dropped too long")
    df = df[df["_seq_len"] < 400]
    df["speach_id"] = range(df.shape[0])

    # 4. Encode characters
    filtered_text = "\n".join(df["char"])
    char_counts = Counter(filtered_text)
    cat_codes = pd.DataFrame(
        char_counts.most_common(), columns=["character", "frequency"]
    )
    cat_codes["_code"] = range(1, len(cat_codes) + 1)
    cat_codes.index = cat_codes["character"]
    (args.save_path / "preprocessed/cat_codes").mkdir(parents=True)
    print(cat_codes)
    cat_codes.to_parquet(args.save_path / "preprocessed/cat_codes/char", index=False)
    df["char"] = df["char"].map(
        lambda x: np.array([cat_codes.loc[c, "_code"] for c in x], np.int64)
    )
    df["char_number"] = df["char"].map(lambda x: np.arange(len(x), dtype=np.float32))

    # 5. Splitting on train and test
    df = df.sample(frac=1, random_state=args.split_seed, replace=False).reset_index(
        drop=True
    )
    split_id = int(len(df) * TEST_FRACTION)
    test_df = df[:split_id]
    train_df = df[split_id:]

    def save_partitioned_parquet(df, save_path, num_shards=20):
        # Add a dummy shard column
        print(f"df size: {len(df)}")
        df = df.copy()
        df.loc[:, "shard"] = np.arange(len(df)) % num_shards
        # Save the DataFrame as a partitioned Parquet file
        df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow")

    save_partitioned_parquet(train_df, args.save_path / "preprocessed/train", 20)
    save_partitioned_parquet(test_df, args.save_path / "preprocessed/test", 3)

if __name__ == "__main__":
    main()
