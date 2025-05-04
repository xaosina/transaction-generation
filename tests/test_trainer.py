from dataclasses import asdict
from pathlib import Path

import pyrallis
import pytest
from generation.data.utils import get_dataloaders
from generation.losses import get_loss
from generation.metrics.evaluator import SampleEvaluator
from generation.models.generator import VAE, Generator
from generation.trainer import Trainer
from generation.schedulers import CompositeScheduler
from generation.utils import (
    get_optimizer,
)
from main import PipelineConfig


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )


def test_generator_train(config: PipelineConfig):
    cfg = config
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.data_conf, cfg.common_seed
    )
    model = Generator(cfg.data_conf, cfg.model).to(cfg.device)
    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    scheduler = CompositeScheduler(optimizer, cfg.schedulers)
    loss = get_loss(cfg.data_conf, cfg.loss)
    log_dir = Path(cfg.log_dir) / cfg.run_name
    sample_evaluator = SampleEvaluator(
        log_dir / "evaluation", cfg.data_conf, cfg.evaluator, device=cfg.device
    )
    # batch = next(iter(test_loader))
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=sample_evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=cfg.run_name,
        ckpt_dir=log_dir / "ckpt",
        device=cfg.device,
        **asdict(cfg.trainer),
    )

    trainer.run()


def test_vae_train(config: PipelineConfig):
    cfg = config
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.data_conf, cfg.common_seed
    )
    model = VAE(cfg.data_conf, cfg.model).to(cfg.device)
    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    loss = get_loss(cfg.data_conf, cfg.loss)
    scheduler = CompositeScheduler(optimizer, loss, cfg.schedulers)
    log_dir = Path(cfg.log_dir) / cfg.run_name
    sample_evaluator = SampleEvaluator(
        log_dir / "evaluation", cfg.data_conf, cfg.evaluator, device=cfg.device
    )
    # batch = next(iter(test_loader))
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=sample_evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=cfg.run_name,
        ckpt_dir=log_dir / "ckpt",
        device=cfg.device,
        **asdict(cfg.trainer),
    )

    trainer.run()
