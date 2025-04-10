import logging
import random
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from .data_types import DataConfig

logger = logging.getLogger(__name__)  # noqa: F821


def searchsorted_vectorized(seqs, values):
    positions = np.zeros_like(values, dtype=np.int64)
    for idx, value in enumerate(values):
        positions[idx] = np.searchsorted(seqs[idx], value, side="left")
    return positions


class ShardDataset(IterableDataset):
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

    def __len__(self):
        """
        Due to the sharded nature of the dataset, it is impossible to determine the exact number of batches that will be yielded.
        However, we can provide an upper bound, which is what this method does.
        """
        all_rows = sum(
            [f.count_rows() for f in pq.ParquetDataset(self.partitions).fragments]
        )
        upper_bound = (
            self.n_resamples * all_rows
        ) // self.data_conf.batch_size + self.data_conf.num_workers
        return upper_bound

    @classmethod
    def train_val_split(
        cls, data_path, data_conf: DataConfig, split_seed: int = None
    ) -> tuple["ShardDataset", "ShardDataset"]:
        assert isinstance(data_path, str)
        fragments = {
            f.path: f.count_rows() for f in pq.ParquetDataset(data_path).fragments
        }
        total_rows = sum(fragments.values())
        paths = list(fragments.keys())
        random.Random(split_seed).shuffle(paths)
        val_size = int(len(paths) * data_conf.val_ratio)
        val_paths = paths[:val_size]
        train_paths = paths[val_size:]
        actual_val_size = sum(fragments[p] for p in val_paths) / total_rows
        logger.warning(f"Actual val size is {actual_val_size}")

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
        # When we set generator in DataLoader, this seeds will repeat each run.
        worker_seed = torch.initial_seed() if worker_info is None else worker_info.seed
        base_seed = (worker_seed - worker_id) % (2**32 - 1)
        common_rng = np.random.default_rng(self.seed or base_seed)
        worker_rng = np.random.default_rng(self.seed or worker_seed)
        data_conf = self.data_conf

        partitions_copy = list(self.partitions)
        partitions_copy = partitions_copy * self.n_resamples
        if self.shuffle:
            common_rng.shuffle(partitions_copy)

        # Split shards across workers
        worker_shards = partitions_copy[worker_id::num_workers]

        remaining_data = None
        for shard_path in worker_shards:
            data = pd.read_parquet(
                shard_path,
                columns=[data_conf.index_name, "_seq_len", data_conf.time_name]
                + (data_conf.num_names or [])
                + (list(data_conf.cat_cardinalities) if data_conf.cat_cardinalities else []),
            )
            data = self._preprocess(data, worker_rng)

            if self.shuffle:
                data = data.sample(frac=1, random_state=worker_rng)

            if remaining_data is not None:
                data = pd.concat([remaining_data, data], ignore_index=True)
                remaining_data = None

            for start in range(0, len(data), data_conf.batch_size):
                batch = data.iloc[start : start + data_conf.batch_size]
                if len(batch) == data_conf.batch_size:
                    yield batch.reset_index(drop=True)
                else:
                    remaining_data = batch

        if remaining_data is not None:
            yield remaining_data.reset_index(drop=True)

    def _preprocess(self, data, rng):
        data_conf = self.data_conf
        min_seq_len = data_conf.min_history_len + data_conf.generation_len

        # Calculate global time boundaries
        times = data[data_conf.time_name]
        max_time = times.map(max).max()
        min_time = times.map(min).min()
        # Filter sequences that meet minimum length requirement
        data = data[data._seq_len >= min_seq_len].reset_index(drop=True)

        if self.random_end != "none":

            # Determine slicing strategy
            if self.random_end == "index":
                end_indices = rng.integers(min_seq_len, data._seq_len, endpoint=True)
            elif self.random_end == "time":
                # Generate random end times in 48 discrete steps
                end_times = (
                    min_time
                    + (max_time - min_time)
                    * rng.integers(1, 48, data.shape[0], endpoint=True)
                    / 48
                )
                end_indices = searchsorted_vectorized(
                    data[data_conf.time_name], end_times
                )

            data = self._slice_rows(data, end_indices)

            # Post-process for time-based slicing
            data = data[data._seq_len >= min_seq_len].reset_index(drop=True)

        return data

    def _slice_rows(self, data, end_indices):
        seq_lens = data["_seq_len"].values
        n_rows = len(data)
        cumulative_lengths = np.concatenate([[0], np.cumsum(seq_lens)])
        row_indices = np.repeat(np.arange(n_rows), seq_lens)
        pos_in_row = np.arange(len(row_indices)) - cumulative_lengths[row_indices]
        keep_mask = pos_in_row < end_indices[row_indices]
        split_indices = np.cumsum(end_indices[:-1])

        for col in self.data_conf.seq_cols:
            concatenated = np.concatenate(data[col].values)
            truncated = concatenated[keep_mask]
            data[col] = np.split(truncated, split_indices)
        data["_seq_len"] = end_indices
        return data


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    data_conf = DataConfig(
        train_path="/home/dev/24/test_20",
        test_path="/home/dev/24/test_20",
        batch_size=512,
        num_workers=4,
        val_ratio=0.15,
        #
        train_resamples=10,
        max_seq_len=1000,
        min_history_len=16,
        generation_len=32,
        train_random_end="time",
        val_random_end="time",
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
            {"TimeToFeatures": {"process_type": "diff"}},
        ],
        val_transforms=[
            {"RescaleTime": {"loc": 0.0, "scale": 48.0}},
            {"TimeToFeatures": {"process_type": "diff"}},
            {"CutTargetSequence": {"target_len": 32}},
        ],
    )
    train_dataset, val_dataset = ShardDataset.train_val_split(
        data_conf.train_path, data_conf
    )
    test_dataset = ShardDataset(
        data_conf.test_path, data_conf, seed=0, random_end=data_conf.val_random_end
    )

    from torch.utils.data import DataLoader
    from utils import get_collator

    collator = get_collator(data_conf, data_conf.train_transforms)
    dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        collate_fn=collator,
        num_workers=data_conf.num_workers,
    )
    from time import time

    # from copy import deepcopy
    start = time()
    import numpy as np
    from tqdm import tqdm

    # for batch, seqs in tqdm(train_loader):
    #     seqs_r = train_loader.collate_fn.reverse(batch)

    #     for s in range(len(seqs)):
    #         so = seqs.iloc[s]
    #         sr = seqs_r.iloc[s]
    #         assert set(so.index) == set(sr.index)
    #         res = []
    #         for id in so.index:
    #             if isinstance(so[id], np.ndarray):
    #                 if not np.allclose(so[id], sr[id], 1e-6, equal_nan=True):
    #                     res = res + [id]
    #             elif so[id] != sr[id]:
    #                 res = res + [id]
    #         if (res != []):
    #             breakpoint()

    for batch in tqdm(dataloader):
        pass
    print(time() - start)

    # random_end - with last modifications - almost none. Very good optimization.
    # train val split is ok. Train val test shuffles are ok. Resamples work fine.
    # n_resamples work for train and test, for shuffles and for cuts.
    # n_workers = 1 then works good, more - return all data reliably
    # gen_len, hist_len - data filter check how much wasted. ok, 30%. depends on data
    # seed - for train each call independed. For val and test - allways same no matter what
    # DataLoader check that workers work independetly. Return full dataset. For train - correctly distribute dataset.
    # train assert every shard is shuffled differently each time
    # big batch_size - check cutting. that resamples help.
    # random_end - time check correct np sortarray insert. index check mean len. check test val stable. check train different (and with resamples)
    # CutTargetSequence - check correct
    # collator - check padding and batch transforms
