import random
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class ParquetShardDataset(IterableDataset):
    # TODO train val split
    # TODO different collators
    # TODO warning if less partitions than workers
    # TODO function to sample batch from a general dataframe
    # TODO option to create in memory, or memory efficient dataset
    # TODO getitem
    # TODO select columns on loading
    # TODO test loader with random finish window. on collator level, BUT random splits are hardcoded 
    def __init__(self, data_path, batch_size, shuffle=True):
        super().__init__()
        self.data_path = data_path
        self.partitions = pq.ParquetDataset(data_path).files
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        base_seed = (
            torch.initial_seed()
            if worker_info is None
            else worker_info.seed - worker_info.id
        ) % (2**32 - 1)
        if self.shuffle is not None:
            partitions_copy = list(self.partitions)
            random.Random(base_seed).shuffle(partitions_copy)

        # Split shards across workers
        worker_shards = partitions_copy[worker_id::num_workers]

        for shard_path in worker_shards:
            # Load shard data
            data = pd.read_parquet(shard_path)
            # Shuffle samples within shard
            if self.shuffle:
                data = data.sample(frac=1, random_state=base_seed)

            for start in range(0, len(data), self.batch_size):
                yield data.iloc[start : start + self.batch_size]
