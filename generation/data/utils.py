from copy import copy
import logging
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import create_instances_from_module
from . import batch_tfs
from .batch_tfs import NewFeatureTransform
from .collator import SequenceCollator
from .data_types import DataConfig, LatentDataConfig
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
    return_orig: bool = False,
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
        return_orig=return_orig,
    )


def get_latent_dataconf(collator: SequenceCollator, data_conf) -> LatentDataConfig:
    cat_cardinalities = copy(collator.cat_cardinalities)
    num_names = copy(collator.num_names)
    focus_on = copy(data_conf.focus_on)

    if collator.batch_transforms:
        for tfs in collator.batch_transforms:
            if isinstance(tfs, NewFeatureTransform):
                for num_name in tfs.num_names:
                    num_names += [num_name]
                for cat_name, card in tfs.cat_cardinalities.items():
                    cat_cardinalities[cat_name] = card
                num_names = [n for n in num_names if n not in tfs.num_names_removed] if num_names is not None else None
                cat_cardinalities = {
                    k: v
                    for k, v in cat_cardinalities.items()
                    if k not in tfs.cat_names_removed
                }
                focus_on = tfs.new_focus_on(focus_on)

    return LatentDataConfig(
        cat_cardinalities=cat_cardinalities,
        num_names=num_names,
        focus_on=focus_on,
        time_name=data_conf.time_name
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
    val_collator = get_collator(data_conf, data_conf.val_transforms, return_orig=True)
    internal_dataconf = get_latent_dataconf(train_collator, data_conf)
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
    return (train_loader, val_loader, test_loader), internal_dataconf
