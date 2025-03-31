from pathlib import Path
import pandas as pd
from ....data.data_types import DataConfig


def prepare_data(
    data: Path, orig: Path, save_path: Path, data_conf: DataConfig, tail_len=None
):
    # Load data
    gen = pd.read_parquet(data)
    orig = pd.read_parquet(orig)
    # Classification labels
    gen["generated"] = 1
    orig["generated"] = 0
    # Index names disctintions
    orig[data_conf.index_name] = "orig_" + orig[data_conf.index_name].astype(str)
    gen[data_conf.index_name] = gen[data_conf.index_name].astype(str)
    # Concat data
    final_df = pd.concat([gen, orig], ignore_index=True)
    # If tail_len - cut only tails in each sequence
    if tail_len:
        assert final_df._seq_len.min() >= tail_len
        final_df[data_conf.seq_cols] = final_df[data_conf.seq_cols].map(lambda x: x[-tail_len:])
    # Save
    final_df.to_parquet(save_path, index=False)