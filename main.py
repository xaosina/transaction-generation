from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Optional, List

import pyrallis
from generation.data.utils import get_dataloaders
from generation.data.data_types import DataConfig
from generation.models.generator import Generator

from generation.metrics.sampler import SampleEvaluator
from generation.losses import get_loss, LossConfig
from generation.utils import get_optimizer, get_scheduler
from generation.trainer import Trainer
from generation.utils import OptimizerConfig, SchedulerConfig
from generation.models.generator import ModelConfig

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
class PipelineConfig:
    run_name: str = "debug"
    log_dir: str = "log/generation"
    device: str = "cuda:0"
    metrics: List[str] = field(default_factory=list)
    data_conf: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    # model_conf: Mapping[str, Any] = field(default_factory=lambda: {})
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@pyrallis.wrap()
def main(cfg: PipelineConfig):
    pyrallis.dump(cfg, open("run_config.yaml", "w"))
    print(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data_conf)
    model = Generator(cfg.data_conf, cfg.model)
    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    lr_scheduler = get_scheduler(optimizer, cfg.scheduler)
    # loss = get_loss(cfg.loss)
    batch = next(iter(train_loader))
    loss = get_loss(config=cfg.loss)
    out = model(batch)
    # batch = next(iter(test_loader))
    loss_out = loss(batch, out)
    metrics, loss = None, None

    sample_evaluator = SampleEvaluator(
        ckpt=Path(cfg.log_dir) / cfg.run_name / "ckpt",
        metrics=cfg.metrics,
        gen_len=cfg.data_conf.generation_len,
        hist_len=cfg.data_conf.min_history_len,  # Здесь надо как-то по-другому делать
        device=cfg.device,
    )

    sample_evaluator.evaluate(model, test_loader)

    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        sample_evaluator=sample_evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=cfg.run_name,
        ckpt_dir=Path(cfg.log_dir) / cfg.run_name / "ckpt",
        device=cfg.device,
        **asdict(cfg.trainer),
    )

    trainer.run()
    trainer.load_best_model()

    train_metrics = trainer.validate(loaders["full_train"])
    train_val_metrics = trainer.validate(loaders["train_val"])
    hpo_metrics = trainer.validate(loaders["hpo_val"])
    test_metrics = trainer.validate(test_loaders["test"])


if __name__ == "__main__":
    main()
