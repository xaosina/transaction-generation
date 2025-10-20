from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from dacite import from_dict, Config
from ebes.pipeline import Runner

from ..models import generator as gen_models
from ..data.utils import get_dataloaders
from ..losses import get_loss
from ..metrics.evaluator import SampleEvaluator
from ..schedulers import CompositeScheduler
from ..trainer import Trainer
from ..utils import (
    get_optimizer,
    suggest_conf,
)

from .utils import PipelineConfig


class GenerationTrainer(Runner):
    def pipeline(self, cfg: Mapping) -> dict[str, float]:
        cfg = from_dict(PipelineConfig, cfg, Config(strict=True))
        assert isinstance(cfg, PipelineConfig)

        (train_loader, val_loader, test_loader), (internal_dataconf, cfg.data_conf) = (
            get_dataloaders(cfg.data_conf, cfg.common_seed)
        )
        model = getattr(gen_models, cfg.model.name)(internal_dataconf, cfg.model).to(
            cfg.device
        )
        optimizer = get_optimizer(model.parameters(), cfg.optimizer)
        loss = get_loss(internal_dataconf, cfg.loss)
        scheduler = CompositeScheduler(optimizer, loss, cfg.schedulers)
        log_dir = Path(cfg.log_dir) / cfg.run_name
        sample_evaluator = SampleEvaluator(
            log_dir / "evaluation",
            cfg.data_conf,
            cfg.evaluator,
            device=cfg.device,
            verbose=cfg.trainer.verbose,
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
            ckpt_dir=log_dir / "ckpt",
            device=cfg.device,
            **asdict(cfg.trainer),
        )
        trainer.run()
        trainer.load_best_model()

        train_loader.collate_fn = val_loader.collate_fn
        train_loader.dataset.random_end = val_loader.dataset.random_end

        val_metrics = trainer.validate(val_loader, get_loss=True, get_metrics=True)
        # train_metrics = trainer.validate(train_loader, get_loss=False, get_metrics=True)
        test_metrics = trainer.validate(test_loader, get_loss=True, get_metrics=True)

        val_metrics = {k: v for k, v in val_metrics.items()}
        # train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(
            # **train_metrics,
            **val_metrics,
            **test_metrics,
        )

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)

        return trial, config
