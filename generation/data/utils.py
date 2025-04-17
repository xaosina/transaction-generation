import logging
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import create_instances_from_module
from . import batch_tfs
from .collator import SequenceCollator
from .data_types import DataConfig
from .dataset import ShardDataset

logger = logging.getLogger()


def save_partitioned_parquet(df, save_path, num_shards=20):
    # Check if the shard column already exists
    if "shard" in df.columns:
        logger.warning("'shard' column already exists. Overwriting...")
    # Add a dummy shard column
    df["shard"] = np.arange(len(df)) % num_shards
    # Save the DataFrame as a partitioned Parquet file
    df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow")


def get_collator(
    data_conf: DataConfig,
    batch_transforms: (
        Mapping[str, Mapping[str, Any] | str] | list[Mapping[str, Any] | str] | None
    ) = None,
) -> SequenceCollator:
    tfs = create_instances_from_module(batch_tfs, batch_transforms)
    return SequenceCollator(
        time_name=data_conf.time_name,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        index_name=data_conf.index_name,
        max_seq_len=data_conf.max_seq_len,
        batch_transforms=tfs,
        padding_value=data_conf.padding_value,
    )


def get_dataloaders(data_conf: DataConfig, seed: int):
    # Create datasets
    train_dataset, val_dataset = ShardDataset.train_val_split(
        data_conf.train_path, data_conf, split_seed=seed
    )
    test_dataset = ShardDataset(
        data_conf.test_path,
        data_conf,
        seed=0,
        random_end=data_conf.val_random_end,
        shuffle=False,
    )
    # Create collators (val and test has same collators)
    train_collator = get_collator(data_conf, data_conf.train_transforms)
    val_collator = get_collator(data_conf, data_conf.val_transforms)
    # Create loaders
    gen = torch.Generator().manual_seed(seed)  # for shard splits between workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        collate_fn=train_collator,
        generator=gen,
        num_workers=data_conf.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        collate_fn=val_collator,
        num_workers=data_conf.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        collate_fn=val_collator,
        num_workers=data_conf.num_workers,
    )
    return train_loader, val_loader, test_loader
