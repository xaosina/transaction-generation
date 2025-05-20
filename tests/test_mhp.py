import pyrallis
import pytest
from generation.data.data_types import GenBatch, PredBatch
from generation.data.utils import get_dataloaders
from main import PipelineConfig
from generation.models.generator import BaselineHP, BaselineHMM

@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )

def test_mhp(config: PipelineConfig):
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch: GenBatch = next(iter(train_loader)).to('cpu')

    batch.num_features = None
    batch.num_features_names = None

    baseline_hp = BaselineHP(
        model_config=config.model
    )

    baseline_hmm = BaselineHMM(
        model_config=config.model
    )

    gen_batch_hp = baseline_hp.generate(batch, 32)
    gen_batch_hmm = baseline_hmm.generate(batch, 32)

    assert isinstance(batch, GenBatch)


def test_hmm(config: PipelineConfig):
    train_loader, _, _ = get_dataloaders(config.data_conf, config.common_seed)
    batch: GenBatch = next(iter(train_loader)).to('cpu')

    batch.num_features = None
    batch.num_features_names = None

    baseline = BaselineHMM(
        model_config=config.model
    )

    baseline.generate(batch, 32)

    assert isinstance(batch, GenBatch)
