from dataclasses import asdict
from pathlib import Path

import pyrallis
import pytest
import torch
from ebes.types import Seq
from generation.data.data_types import GenBatch, PredBatch
from generation.data.utils import get_dataloaders
from generation.losses import LossConfig, get_loss
from generation.models.autoencoders.vae import (
    VAE,
    Decoder,
    Encoder,
)
from generation.models.generator import VAE
from generation.trainer import Trainer
from generation.utils import (
    get_optimizer,
    get_scheduler,
)
from main import GenerationRunner, PipelineConfig, RunnerConfig


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )
def test_vae_train(config: PipelineConfig):
    cfg = config
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.data_conf, cfg.common_seed
    )
    model = VAE(cfg.data_conf, cfg.model).to(cfg.device)
    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    lr_scheduler = get_scheduler(optimizer, cfg.scheduler)
    loss = get_loss(cfg.loss)
    log_dir = Path(cfg.log_dir) / cfg.run_name

    # batch = next(iter(test_loader))
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        evaluator=None,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=cfg.run_name,
        ckpt_dir= log_dir / "ckpt",
        device=cfg.device,
        **asdict(cfg.trainer),
    )

    trainer.run()