import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import pytest
import torch
from ebes.types import Seq
from generation.data.data_types import DataConfig, GenBatch, PredBatch
from generation.data.preprocess.vae.models.vae.model import (
    VAE,
    Decoder,
    Encoder,
    VaeConfig,
)
from generation.data.utils import get_dataloaders
from generation.losses import LossConfig
from generation.models.generator import ModelConfig
from generation.utils import LoginConfig, OptimizerConfig, SchedulerConfig


@dataclass
class TrainConfig:
    total_iters: Optional[int] = 100_000
    total_epochs: Optional[int] = None
    patience: int = -1
    iters_per_epoch: Optional[int] = 10_000
    ckpt_replace: bool = True
    ckpt_track_metric: str = "epoch"
    ckpt_resume: Optional[str | os.PathLike] = None
    profiling: bool = False


@dataclass
class MetricConfig:
    names: list[str] = field(default_factory=list)
    subject_key: str = "client_id"
    target_key: str = "event_type"


@dataclass
class PipelineConfig:
    run_name: str = "debug"
    log_dir: Path = "log/generation"
    device: str = "cuda:0"
    metrics: MetricConfig = field(default_factory=MetricConfig)
    data_conf: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    # model_conf: Mapping[str, Any] = field(default_factory=lambda: {})
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoginConfig = field(default_factory=LoginConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )


def test_vae_only_categorical(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to("cuda")

    batch.num_features = None
    batch.num_features_names = None

    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=None,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=config.device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.num_features is None


def test_vae_only_numerical(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to("cuda")

    batch.cat_features = None
    batch.cat_features_names = None

    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=None,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=config.device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.cat_features is None


def test_vae(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to("cuda")
    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=config.device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch)


def test_decoder(config: PipelineConfig):
    decoder = Decoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        num_names=config.data_conf.num_names,
        cat_cardinalities=config.data_conf.cat_cardinalities,
    )

    x = Seq(
        tokens=torch.rand(
            size=(
                64,
                32,
                6
                * (
                    len(config.data_conf.num_names)
                    + 1  # Time
                    + len(config.data_conf.cat_cardinalities.values())
                ),
            )
        ),
        lengths=torch.randint(
            0,
            120,
            size=[
                32,
            ],
        ),
        time=None,
    )

    predbatch = decoder(x)
    assert predbatch.num_features.shape[-1] == config.data_conf.num_names.__len__()


def test_encoder(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch = next(iter(train_loader)).to("cuda")
    encoder = Encoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=config.device)

    seq = encoder(batch)
    assert isinstance(seq, Seq)


def test_encoder_pretrain(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to("cuda")
    encoder = Encoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=config.device)

    seq = encoder(batch)
    assert (
        isinstance(seq[0], Seq)
        and seq[1]["mu_z"].shape[1]
        == batch.num_features.shape[-1] + batch.cat_features.shape[-1]
    )
