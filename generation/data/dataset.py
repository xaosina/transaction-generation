import logging
import random
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset
from data_types import DataConfig

logger = logging.getLogger(__name__)  # noqa: F821

class ShardDataset(IterableDataset):
    # TODO different collators
    # TODO function to sample batch from a general dataframe
    # TODO option to create in memory, or memory efficient dataset
    # TODO getitem

    def __init__(
        self,
        data_path: str | list[str],
        data_conf: DataConfig,
        seed: int | None = None,
        n_resamples: int = 1,
        random_end: Literal["index", "time", "none"] = "none",
        shuffle: bool = False,
    ):
        super().__init__()
        if isinstance(data_path, list):
            self.partitions = data_path
        elif isinstance(data_path, str):
            self.partitions = pq.ParquetDataset(data_path).files
        else:
            raise ValueError(f"{type(data_path)} is not supported for data path.")

        if len(self.partitions) < data_conf.num_workers:
            logger.warning(
                f"Only {len(self.partitions)} workers will be utilized, as there are no more partitions."
            )

        self.data_conf = data_conf
        self.seed = seed  # Seed for preserving same test set
        self.n_resamples = n_resamples
        self.random_end = random_end
        self.shuffle = shuffle

    @classmethod
    def train_val_split(
        cls, data_path, data_conf: DataConfig
    ) -> tuple["ShardDataset", "ShardDataset"]:
        assert isinstance(data_path, str)
        fragments = {
            f.path: f.count_rows() for f in pq.ParquetDataset(data_path).fragments
        }
        total_rows = sum(fragments.values())
        paths = list(fragments.keys())
        random.shuffle(paths)
        val_size = int(len(paths) * data_conf.val_ratio)
        val_paths = paths[:val_size]
        train_paths = paths[val_size:]
        actual_val_size = sum(fragments[p] for p in val_paths) / total_rows
        logger.info(f"Actual val size is {actual_val_size}")

        train_dataset = cls(
            train_paths,
            data_conf,
            n_resamples=data_conf.train_resamples,
            random_end=data_conf.train_random_end,
            shuffle=True,
        )
        val_dataset = cls(
            val_paths,
            data_conf,
            seed=0,
            random_end=data_conf.val_random_end,
            shuffle=False,
        )

        return train_dataset, val_dataset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        base_seed = (
            torch.initial_seed()
            if worker_info is None
            else worker_info.seed - worker_id
        ) % (2**32 - 1)
        shuffle_seed = self.seed if self.seed is not None else base_seed
        data_conf = self.data_conf

        partitions_copy = list(self.partitions)
        partitions_copy = partitions_copy * self.n_resamples
        if self.shuffle:
            random.Random(shuffle_seed).shuffle(partitions_copy)

        # Split shards across workers
        worker_shards = partitions_copy[worker_id::num_workers]

        remaining_data = None
        for shard_path in worker_shards:
            data = pd.read_parquet(
                shard_path,
                columns=[data_conf.index_name, data_conf.time_name]
                + data_conf.num_names
                + list(data_conf.cat_cardinalities.keys()),
            )
            data = self._preprocess(data)

            if self.shuffle:
                data = data.sample(frac=1, random_state=self.seed)

            if remaining_data is not None:
                data = pd.concat([remaining_data, data], ignore_index=True)
                remaining_data = None

            for start in range(0, len(data), data_conf.batch_size):
                batch = data.iloc[start : start + data_conf.batch_size]
                if len(batch) == data_conf.batch_size:
                    yield batch
                else:
                    remaining_data = batch

        if remaining_data is not None:
            yield remaining_data

    def _preprocess(self, data):
        data_conf = self.data_conf
        min_seq_len = data_conf.min_history_len + data_conf.generation_len

        # Calculate global time boundaries
        times = data[data_conf.time_name]
        max_time = times.map(max).max()
        min_time = times.map(min).min()

        # Filter sequences that meet minimum length requirement
        data = data[data._seq_len >= min_seq_len].reset_index(drop=True)

        if self.random_end != "none":
            # Initialize seeded random generator
            rng = np.random.default_rng(self.seed if self.seed is not None else None)

            # Determine slicing strategy
            if self.random_end == "index":
                end_values = rng.integers(min_seq_len, data._seq_len, endpoint=True)
                slice_func = slice_row_by_index
            elif self.random_end == "time":
                # Generate random end times in 48 discrete steps
                end_values = (
                    min_time
                    + (max_time - min_time)
                    * rng.integers(1, 48, data.shape[0], endpoint=True)
                    / 48
                )
                slice_func = slice_row_by_time

            # Apply slicing to all rows
            data = data.apply(
                lambda row: slice_func(row, end_values, data_conf), axis=1
            )

            # Post-process for time-based slicing
            data = data[data._seq_len >= min_seq_len].reset_index(drop=True)

        return data


def slice_row_by_index(row, end_indices, data_conf):
    end_id = end_indices[row.name]
    return _update_row_slice(row, end_id, data_conf)


def slice_row_by_time(row, end_times, data_conf):
    end_time = end_times[row.name]
    end_id = np.searchsorted(row[data_conf.time_name], end_time, side="right")
    return _update_row_slice(row, end_id, data_conf)


def _update_row_slice(row, end_id, data_conf):
    """Common helper to truncate sequence columns and update length"""
    seq_cols = data_conf.seq_cols
    # Stack sequence columns vertically and slice
    row[seq_cols] = list(
        np.vstack(row.values[row.index.get_indexer(seq_cols)])[:, :end_id]
    )
    row._seq_len = end_id
    return row


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())    
    data_conf = DataConfig(
        train_path="/home/dev/24/test_20",
        test_path="/home/dev/24/test_20",
        batch_size=16,
        num_workers=4,
        val_ratio=0.15,
        #
        train_resamples=1,
        max_seq_len=1000,
        min_history_len=16,
        generation_len=16,
        train_random_end="none",
        val_random_end="none",
        #
        time_name="Time",
        cat_cardinalities={"MechVent": 3, "Gender": 5, "ICUType": 6},
        num_names=[
            "Na",
            "Age",
            "TroponinT",
            "Platelets",
            "PaO2",
            "Albumin",
            "PaCO2",
            "RespRate",
            "WBC",
            "MAP",
            "ALP",
            "Creatinine",
            "Cholesterol",
            "HCT",
            "Temp",
            "Glucose",
            "HR",
            "Bilirubin",
            "GCS",
            "Height",
            "SysABP",
            "SaO2",
            "Mg",
            "NIMAP",
            "pH",
            "TroponinI",
            "AST",
            "NIDiasABP",
            "BUN",
            "DiasABP",
            "K",
            "NISysABP",
            "FiO2",
            "Weight",
            "Urine",
            "HCO3",
            "Lactate",
            "ALT",
        ],
        index_name="RecordID",
        train_transforms=[
            {"RescaleTime": {"loc": 0.0, "scale": 48.0}},
            {"TimeToFeatures": {"process_type": "cat"}},
            {"FillNans": {"fill_value": -1.0}},
        ],
        val_transforms=[
            {"RescaleTime": {"loc": 0.0, "scale": 48.0}},
            {"TimeToFeatures": {"process_type": "cat"}},
            {"FillNans": {"fill_value": -1.0}},
            {"CutTargetSequence": {"n_gen": 32}},
        ],
    )
    train_dataset, val_dataset = ShardDataset.train_val_split(
        data_conf.train_path, data_conf
    )
    test_dataset = ShardDataset(
        data_conf.test_path, data_conf, seed=0, random_end=data_conf.val_random_end
    )
    print("\nPartitions:")
    print([path.split("/")[-2] for path in train_dataset.partitions])
    print([path.split("/")[-2] for path in val_dataset.partitions])
    print([path.split("/")[-2] for path in test_dataset.partitions])
    assert (set(train_dataset.partitions) | set(val_dataset.partitions)) == set(
        test_dataset.partitions
    )
    print("\nSeeds:")
    print(f"{train_dataset.seed=}")
    print(f"{val_dataset.seed=}")
    print(f"{test_dataset.seed=}")
    print("\nN_resamples:")
    print(f"{train_dataset.n_resamples=}")
    print(f"{val_dataset.n_resamples=}")
    print(f"{test_dataset.n_resamples=}")
    print("\nRandom_end:")
    print(f"{train_dataset.random_end=}")
    print(f"{val_dataset.random_end=}")
    print(f"{test_dataset.random_end=}")
    print("\nShuffle:")
    print(f"{train_dataset.shuffle=}")
    print(f"{val_dataset.shuffle=}")
    print(f"{test_dataset.shuffle=}")

    for batch in train_dataset:
        print(batch)
    # TODO check speed, everything right, test returns all and same, train random, resamples work
    # TODO for time slice, index slice, none slice

    # TODO n_resamples train - shuffle/cut each time differently, but test/val same. returns full dataset multiple times
    # TODO n_workers - 1 then works good, more return all data reliably
    # TODO gen_len, hist_len - data filter check how much wasted
    # TODO seed - for train each call independed. For val and test - allways same no matter what
    # TODO random_end - time check correct np sortarray insert. index check mean len. check test val stable. check train different (and with resamples)
    # TODO big batch_size - check cutting. that resamples help.
    # TODO DataLoader check that workers work independetly. Return full dataset. For train - correctly distribute dataset. 
    # TODO train assert every shard is shuffled differently each time
    # TODO random_end - check speed drop. and correct lens after cut
    # TODO CutTargetSequence - check correct
    # TODO collator - checck padding and batch transforms
    breakpoint()
