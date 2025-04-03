import os
import sys

import pyrallis
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PipelineConfig

from generation.metrics.evaluator import SampleEvaluator
from generation.models.generator import (
    BaselineRepeater,
    Generator,
    GroundTruthGenerator,
)
from generation.data.utils import get_dataloaders


def prepare_evaluator():
    cfg = pyrallis.parse(config_class=PipelineConfig, config_path="spec_config.yaml")
    print("Config ready")
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data_conf)
    sample_evaluator = SampleEvaluator("tests/log", cfg.data_conf, cfg.evaluator)
    # model = Generator(cfg.data_conf, cfg.model).to("cuda")
    # model = BaselineRepeater(cfg.data_conf)
    model = GroundTruthGenerator()
    # Setup ready
    metrics = sample_evaluator.evaluate(model, test_loader, blim=100, buffer_size=50)
    print(metrics)
    breakpoint()


if __name__ == "__main__":
    prepare_evaluator()
