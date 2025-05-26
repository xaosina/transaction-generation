from copy import deepcopy
import os
import sys

import pyrallis
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from main import PipelineConfig

from generation.data.utils import get_dataloaders
from generation.metrics.evaluator import SampleEvaluator
from generation.models.generator import (
    BaselineRepeater,
    Generator,
    GroundTruthGenerator,
)
from generation.data.batch_tfs import ShuffleBatch
from generation.data.data_types import GenBatch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def create_batch():
    lengths = torch.tensor([5, 4, 3])

    time = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 1.0],
        [1.0, 2.0, 2.0],
        [2.0, 3.0, 0.0],
        [3.0, 0.0, 0.0]
    ])  # shape: [5, 3]

    # num_features: [seq_len, batch_size, num_feature_dim]
    num_features = torch.tensor([
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
        [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]],
        [[4.0, 5.0], [4.0, 5.0], [0.0, 0.0]],
        [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]
    ])  # shape: [5, 3, 2]

    # cat_features: [seq_len, batch_size, 1]
    cat_features = torch.tensor([
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 7],
        [10, 8, 0],
        [7, 0, 0]
    ]).unsqueeze(-1)

    # Create instance
    batch = GenBatch(
        lengths=lengths,
        time=time,
        num_features=num_features,
        cat_features=cat_features,
        cat_features_names=["category"],
        num_features_names=["feature_1", "feature_2"],
        monotonic_time=True
    )
    return batch

def test_evaluator():

    # while True:
    #     batch = create_batch()
    #     o_b = deepcopy(batch)
    #     ShuffleBatch([None, -3])(batch)
    #     i = int(input())
    #     print(o_b.num_features[:, i])
    #     print(batch.num_features[:, i])

    cfg = pyrallis.parse(config_class=PipelineConfig, config_path="/home/transaction-generation/log/generation/gru_shuffle/shuffle_pred/seed_0/config.yaml")
    logger.info("Config ready")
    (train_loader, val_loader, test_loader), latent_config = get_dataloaders(cfg.data_conf, 0)
    actual_len = 0
    for batch, orig in tqdm(test_loader):
        print(batch.num_features[:, 0, 0])
        print(batch.target_num_features[:, 0, 0])
        breakpoint()
        
        actual_len += 1
    print(actual_len)
    assert actual_len == 284


if __name__ == "__main__":
    test_evaluator()
