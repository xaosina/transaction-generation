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
    evaluator = SampleEvaluator("tests/log/evaluation", cfg.data_conf, cfg.evaluator)
    logger.info(evaluator.metrics)
    # model = Generator(cfg.data_conf, cfg.model).to("cuda")
    model = BaselineRepeater(cfg.data_conf)
    # model = GroundTruthGenerator()
    # Setup ready
    metrics = evaluator.evaluate(model, test_loader, blim=10, buffer_size=50)
    print(metrics)
    assert True


if __name__ == "__main__":
    test_evaluator()
