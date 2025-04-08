import pyrallis
import pytest
import torch.nn as nn
import torch.optim as optim
from generation.schedulers.schedulers import BetaScheduler, CompositeScheduler
from generation.utils import get_schedulers
from main import PipelineConfig


def get_dummy_optimizer():
    model = nn.Linear(10, 1)
    return optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )


def test_composite_scheduler_state_dict_isolated_restore(config: PipelineConfig):
    optimizer = get_dummy_optimizer()
    comp_sched = get_schedulers(optimizer, config.schedulers)

    beta_sched = next(
        (s for s in comp_sched.schedulers if isinstance(s, BetaScheduler)), None
    )
    assert beta_sched is not None

    first_metrics = [1.0, 1.1]
    for metric in first_metrics:
        optimizer.step()
        comp_sched.step(metrics=metric)

    saved_state = comp_sched.state_dict()

    optimizer.step()
    comp_sched.step(metrics=1.2)
    beta_after_extra_step = beta_sched.get_beta()

    optimizer2 = get_dummy_optimizer()
    comp_sched_restored = get_schedulers(optimizer2, config.schedulers)
    beta_sched_restored = next(
        (s for s in comp_sched_restored.schedulers if isinstance(s, BetaScheduler)),
        None,
    )

    comp_sched_restored.load_state_dict(saved_state)

    optimizer2.step()
    comp_sched_restored.step(metrics=1.2)
    restored_beta = beta_sched_restored.get_beta()

    assert abs(restored_beta - beta_after_extra_step) < 1e-12


def test_composite_scheduler_state_dict(config: PipelineConfig):
    optimizer = get_dummy_optimizer()
    comp_sched = get_schedulers(optimizer, config.schedulers)
    assert comp_sched is not None

    beta_sched = next(
        (s for s in comp_sched.schedulers if isinstance(s, BetaScheduler)), None
    )
    assert beta_sched is not None

    metrics = [1.0, 1.1, 1.2]
    for metric in metrics:
        optimizer.step()
        comp_sched.step(metrics=metric)

    saved_state = comp_sched.state_dict()
    saved_beta = beta_sched.get_beta()
    saved_num_bad_epochs = beta_sched.num_bad_epochs

    optimizer2 = get_dummy_optimizer()
    comp_sched2 = get_schedulers(optimizer2, config.schedulers)
    beta_sched2 = next(
        (s for s in comp_sched2.schedulers if isinstance(s, BetaScheduler)), None
    )

    assert beta_sched2.get_beta() == 1e-2

    comp_sched2.load_state_dict(saved_state)

    assert abs(beta_sched2.get_beta() - saved_beta) < 1e-12
    assert beta_sched2.num_bad_epochs == saved_num_bad_epochs


def test_composite_scheduler_step_with_beta(config: PipelineConfig):
    optimizer = get_dummy_optimizer()
    comp_sched = get_schedulers(optimizer, config.schedulers)
    assert comp_sched is not None
    assert isinstance(comp_sched, CompositeScheduler)

    beta_sched = None
    for s in comp_sched.schedulers:
        if isinstance(s, BetaScheduler):
            beta_sched = s
            break

    assert (
        beta_sched is not None
    ), "В конфиге есть BetaScheduler, но почему-то не найден в CompositeScheduler?"

    assert abs(beta_sched.get_beta() - 1e-2) < 1e-12, beta_sched.get_beta()

    metrics = [1.0, 1.1, 1.2, 1.3, 1.4]
    for epoch, metric in enumerate(metrics):
        optimizer.step()
        comp_sched.step(metrics=metric)

    beta = beta_sched.get_beta()
    expected_beta = 0.01 * 0.7 * 0.7
    assert beta == expected_beta


def test_beta_scheduler():
    scheduler = BetaScheduler(
        init_beta=0.01, factor=0.7, patience=3, min_beta=1e-5, verbose=False
    )

    scheduler.step(0.5)
    assert scheduler.best_metric == 0.5
    assert scheduler.num_bad_epochs == 0
    beta_before = scheduler.get_beta()

    scheduler.step(0.4)

    assert scheduler.best_metric == 0.4
    assert scheduler.num_bad_epochs == 0
    assert scheduler.get_beta() == beta_before


def test_beta_scheduler_no_improvement():
    scheduler = BetaScheduler(
        init_beta=0.01, factor=0.7, patience=3, min_beta=1e-5, verbose=False
    )
    scheduler.step(0.5)
    scheduler.step(0.6)
    assert scheduler.num_bad_epochs == 1

    scheduler.step(0.7)
    assert scheduler.num_bad_epochs == 2

    scheduler.step(0.8)
    assert scheduler.num_bad_epochs == 0
    assert scheduler.beta == 0.01 * 0.7


def test_beta_scheduler_min_beta():
    scheduler = BetaScheduler(
        init_beta=0.001, factor=0.7, patience=1, min_beta=0.0005, verbose=False
    )

    scheduler.step(0.5)
    assert scheduler.best_metric == 0.5

    scheduler.step(0.6)
    new_beta = scheduler.get_beta()
    assert new_beta == 0.0007

    scheduler.step(0.7)
    new_beta = scheduler.get_beta()
    assert new_beta == 0.0005


def test_beta_scheduler_state_dict_and_load():
    scheduler = BetaScheduler(
        init_beta=0.01, factor=0.7, patience=2, min_beta=0.001, verbose=False
    )

    scheduler.step(0.5)
    scheduler.step(0.6)
    state = scheduler.state_dict()

    new_scheduler = BetaScheduler(
        init_beta=10, factor=10, patience=10, min_beta=0.002, verbose=False
    )
    new_scheduler.load_state_dict(state)

    assert new_scheduler.get_beta() == state["beta"]
    assert new_scheduler.best_metric == state["best_metric"]
    assert new_scheduler.num_bad_epochs == state["num_bad_epochs"]
