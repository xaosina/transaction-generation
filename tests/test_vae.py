import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import pytest
import torch
from ebes.types import Seq
from generation.data.data_types import GenBatch, PredBatch
from generation.data.preprocess.vae.models.vae.model import (
    VAE,
    Decoder,
    Encoder,
)
from generation.data.utils import get_dataloaders
from main import PipelineConfig


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )


def test_vae_only_categorical(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    device = config.devices[0]
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to(device)

    batch.num_features = None
    batch.num_features_names = None

    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=None,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.num_features is None


def test_vae_only_numerical(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    device = config.devices[0]
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to(device)

    batch.cat_features = None
    batch.cat_features_names = None

    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=None,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.cat_features is None


def test_vae(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    vae_conf = config.vae
    device = config.devices[0]
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch: GenBatch = next(iter(train_loader)).to(device)
    vae = VAE(
        vae_conf=vae_conf,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch)


def test_decoder(config: PipelineConfig):
    device = config.devices[0]
    decoder = Decoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        num_names=config.data_conf.num_names,
        cat_cardinalities=config.data_conf.cat_cardinalities,
    ).to(device)

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
        ).to(device),
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
    device = config.devices[0]
    train_loader, _, _ = get_dataloaders(config.data_conf)
    batch = next(iter(train_loader)).to(device)
    encoder = Encoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=device)

    seq = encoder(batch)
    assert isinstance(seq, Seq)


def test_encoder_pretrain(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    train_loader, _, _ = get_dataloaders(config.data_conf)
    device = config.devices[0]
    batch: GenBatch = next(iter(train_loader)).to(device)
    encoder = Encoder(
        config.vae.num_layers,
        config.vae.d_token,
        config.vae.n_head,
        config.vae.factor,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
        pretrain=True,
    ).to(device=device)

    seq = encoder(batch)
    assert (
        isinstance(seq[0], Seq)
        and seq[1]["mu_z"].shape[1]
        == batch.num_features.shape[-1] + batch.cat_features.shape[-1]
    )
