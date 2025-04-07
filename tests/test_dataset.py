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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_evaluator():

    cfg = pyrallis.parse(config_class=PipelineConfig, config_path="spec_config.yaml")
    logger.info("Config ready")
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data_conf)
    actual_len = 0
    for batch in tqdm(test_loader):
        actual_len += 1
    print(actual_len)
    assert actual_len == 284


if __name__ == "__main__":
    test_evaluator()
