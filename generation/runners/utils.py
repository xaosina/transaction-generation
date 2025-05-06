from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from generation.data.data_types import DataConfig
from generation.losses import LossConfig
from generation.metrics.evaluator import EvaluatorConfig
from generation.models.generator import ModelConfig
from generation.trainer import TrainConfig
from generation.utils import (
    LoginConfig,
    OptimizerConfig,
)


@dataclass
class NRunsConfig:
    n_runs: int
    n_workers: int


@dataclass
class RunnerConfig:
    name: str = "GenerationRunner"
    run_type: str = "simple"
    seed_keys: list[str] = field(default_factory=lambda: ["common_seed"])
    params: NRunsConfig = field(default_factory=NRunsConfig)
    device_list: Optional[list[str]] = None


@dataclass(frozen=True)
class OptunaParams:
    target_metric: str
    n_trials: int = 50
    n_startup_trials: int = 3
    request_list: list[dict] = field(default_factory=list)
    multivariate: bool = True
    group: bool = True


@dataclass(frozen=True)
class OptunaConfig:
    suggestions: list
    params: OptunaParams = field(default_factory=OptunaParams)


@dataclass
class PipelineConfig:
    run_name: str = "debug"
    log_dir: str = "log/generation"
    device: str = "cuda:0"
    common_seed: int = 0
    # Config from this yaml path will override any field.
    spec_config: Optional[str] = None
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    data_conf: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedulers: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoginConfig = field(default_factory=LoginConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    optuna: Optional[OptunaConfig] = field(default_factory=OptunaConfig)
