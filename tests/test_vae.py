import pyrallis
import pytest
import torch
from ebes.types import Seq
from generation.data.data_types import GenBatch, PredBatch
from generation.models.autoencoders.vae import (
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
    device = config.device
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch: GenBatch = next(iter(train_loader)).to(device)

    batch.num_features = None
    batch.num_features_names = None

    vae = VAE(
        vae_conf=config.model.vae,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=None,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.num_features is None


def test_vae_only_numerical(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    device = config.device
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch: GenBatch = next(iter(train_loader)).to(device)

    batch.cat_features = None
    batch.cat_features_names = None

    vae = VAE(
        vae_conf=config.model.vae,
        cat_cardinalities=None,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch) and predbatch.cat_features is None


def test_vae(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    device = config.device
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch: GenBatch = next(iter(train_loader)).to(device)
    vae = VAE(
        vae_conf=config.model.vae,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=device)

    predbatch, loss_params = vae(batch)
    assert isinstance(predbatch, PredBatch)


def test_decoder(config: PipelineConfig):
    device = config.device
    decoder = Decoder(
        config.model.vae,
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
        ).to(device),
        time=None,
    )

    predbatch: PredBatch = decoder(x)
    assert (
        predbatch.num_features.shape[-1] == config.data_conf.num_names.__len__()
        and predbatch.num_features.shape[0] == predbatch.time.shape[0]
    )


def test_encoder(config: PipelineConfig):
    data_conf = config.data_conf
    pre_conf = config.model.preprocessor
    device = config.device
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch = next(iter(train_loader)).to(device)
    encoder = Encoder(
        config.model.vae,
        cat_cardinalities=data_conf.cat_cardinalities,
        num_names=data_conf.num_names,
        batch_transforms=pre_conf.batch_transforms,
    ).to(device=device)

    seq = encoder(batch)
    if config.model.vae.pretrained:
        assert isinstance(seq, Seq)
    else:
        assert (
            isinstance(seq[0], Seq)
            and seq[1]["mu_z"].shape[1]
            == batch.num_features.shape[-1] + batch.cat_features.shape[-1]
        )
