import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import batch_tfs
import numpy as np
import pandas as pd
from collator import SequenceCollator
from dataset import ParquetShardDataset
from torch.utils.data import DataLoader

logger = logging.getLogger()


def save_partitioned_parquet(df, save_path, num_shards=20):
    # Check if the shard column already exists
    if "shard" in df.columns:
        logger.warning("'shard' column already exists. Overwriting...")
    # Add a dummy shard column
    df["shard"] = np.arange(len(df)) % num_shards
    # Save the DataFrame as a partitioned Parquet file
    df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow")


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    batch_size: int
    shuffle: bool = False
    num_workers: int = 4
    max_seq_len: int = 0

    time_name: str
    cat_cardinalities: Mapping[str, int] | None = None
    num_names: list[str] | None = None
    index_name: str | None = None
    batch_transforms: list[Mapping[str, Any] | str] | None = None
    padding_side: str = "start"
    padding_value: float = 0


def get_collator(data_conf: DataConfig) -> SequenceCollator:

    tfs = None
    if data_conf.batch_transforms is not None:
        tfs = []
        for bt in data_conf.batch_transforms:
            if isinstance(bt, str):
                tfs.append(getattr(batch_tfs, bt)())
                continue

            for name, params in bt.items():  # has params
                klass = getattr(batch_tfs, name)
                if isinstance(params, Mapping):
                    tfs.append(klass(**params))
                elif isinstance(params, Sequence):
                    tfs.append(klass(*params))
                else:
                    tfs.append(klass(params))
                break

    return SequenceCollator(
        time_name=data_conf.time_name,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        index_name=data_conf.index_name,
        max_seq_len=data_conf.max_seq_len,
        batch_transforms=tfs,
        padding_side=data_conf.padding_side,
        padding_value=data_conf.padding_value,
    )


def get_dataloader(data_conf: DataConfig):
    dataset = ParquetShardDataset(
        data_conf.train_path, data_conf.batch_size, data_conf.shuffle
    )
    collator = get_collator(data_conf)
    dataloader = DataLoader(
        dataset, batch_size=None, collate_fn=collator, num_workers=data_conf.num_workers
    )
    return dataloader
