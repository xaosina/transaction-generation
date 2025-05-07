from dataclasses import asdict
import logging
from pathlib import Path
from typing import Mapping

import torch
from dacite import from_dict
from ebes.pipeline import Runner

from ..models import generator as gen_models
from ..data.utils import get_dataloaders
from ..metrics.evaluator import SampleEvaluator
from ..trainer import Trainer


from .utils import PipelineConfig

logger = logging.getLogger(__name__)


class GenerationEvaluator(Runner):
    def pipeline(self, cfg: Mapping) -> dict[str, float]:
        cfg = from_dict(PipelineConfig, cfg)
        assert isinstance(cfg, PipelineConfig)

        train_loader, val_loader, test_loader = get_dataloaders(
            cfg.data_conf, cfg.common_seed
        )
        model = getattr(gen_models, cfg.model.name)(cfg.data_conf, cfg.model).to(
            cfg.device
        )
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
            evaluator=sample_evaluator,
            train_loader=train_loader,
            val_loader=val_loader,
            run_name=cfg.run_name,
            ckpt_dir=log_dir / "ckpt",
            device=cfg.device,
            **asdict(cfg.trainer),
        )
        if cfg.trainer.ckpt_resume is not None:
            logger.info("Resuming from checkpoint '%s'", str(cfg.trainer.ckpt_resume))
            trainer.load_ckpt(cfg.trainer.ckpt_resume)

        train_loader.collate_fn = val_loader.collate_fn
        train_loader.dataset.random_end = val_loader.dataset.random_end

        val_metrics = trainer.validate(val_loader, remove=False)
        train_metrics = trainer.validate(train_loader, remove=False)
        test_metrics = trainer.validate(test_loader, remove=False)

        val_metrics = {k: v for k, v in val_metrics.items()}
        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**train_metrics, **val_metrics, **test_metrics)
