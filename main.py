from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Any

import pyrallis
import pyrallis.parsers
from dacite import from_dict
from ebes.pipeline import Runner
from ebes.pipeline.utils import suggest_conf

from generation.data.data_types import DataConfig
from generation.data.utils import get_dataloaders
from generation.losses import LossConfig, get_loss
from generation.metrics.evaluator import EvaluatorConfig, SampleEvaluator
from generation.models.generator import Generator, ModelConfig, VAE
from generation.trainer import TrainConfig, Trainer
from generation.schedulers import CompositeScheduler
from generation.utils import (
    LoginConfig,
    OptimizerConfig,
    get_optimizer,
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


@dataclass
class PipelineConfig:
    run_name: str = "debug"
    log_dir: Path = Path("log/generation")
    device: str = "cuda:0"
    common_seed: int = 0
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    data_conf: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedulers: Optional[list[Mapping[str, Any] | str]] = None
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoginConfig = field(default_factory=LoginConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


class GenerationRunner(Runner):
    def pipeline(self, cfg: Mapping) -> dict[str, float]:
        cfg = from_dict(PipelineConfig, cfg)

        train_loader, val_loader, test_loader = get_dataloaders(
            cfg.data_conf, cfg.common_seed
        )
        model = VAE(cfg.data_conf, cfg.model).to(cfg.device)
        optimizer = get_optimizer(model.parameters(), cfg.optimizer)
        loss = get_loss(cfg.loss)
        scheduler = CompositeScheduler(optimizer, loss, cfg.schedulers)
        # batch = next(iter(test_loader))
        log_dir = Path(cfg.log_dir) / cfg.run_name
        sample_evaluator = SampleEvaluator(
            log_dir / "evaluation", cfg.data_conf, cfg.evaluator, device=cfg.device
        )
        trainer = Trainer(
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluator=sample_evaluator,
            train_loader=train_loader,
            val_loader=val_loader,
            run_name=cfg.run_name,
            ckpt_dir= log_dir / "ckpt",
            device=cfg.device,
            **asdict(cfg.trainer),
        )

        trainer.run()
        trainer.load_best_model()

        _ = trainer.validate(train_loader)
        _ = trainer.validate(val_loader)
        _ = trainer.validate(test_loader)


        # train_val_metrics = trainer.validate(loaders["train_val"])
        # hpo_metrics = trainer.validate(loaders["hpo_val"])
        # test_metrics = trainer.validate(test_loaders["test"])
        return {}

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config


@pyrallis.wrap("spec_config.yaml")
def main(cfg: PipelineConfig):
    config = asdict(cfg)
    runner = Runner.get_runner(config["runner"]["name"])
    res = runner.run(config)
    print(res)


if __name__ == "__main__":
    main()
