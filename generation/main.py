from dataclasses import dataclass, field
from typing import Any, Mapping

import pyrallis
from data.utils import DataConfig, get_dataloader
from ebes import build_model
from trainer import TrainConfig


@dataclass
class PipelineConfig:
    data_conf: DataConfig = field(default_factory=DataConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    model_conf: Mapping[str, Any]
    optimizer: 

@pyrallis.wrap()
def main(cfg: PipelineConfig):
    train_loader, val_loader = get_dataloader(cfg.data_conf)
    test_loader = get_dataloader(cfg.data_conf)
    model = build_model(cfg.model_conf)
    optimizer = get_optimizer(model.parameters, cfg.optimizer)
    metrics = get_metrics(cfg.metrics)
    loss = get_loss(cfg.loss)

    trainer = Trainer(
    model=net,
    loss=loss,
    optimizer=opt,
    lr_scheduler=lr_scheduler,
    metrics=metrics,
    train_loader=loaders["train"],
    val_loader=loaders["train_val"],
    run_name=config["run_name"],
    ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "ckpt",
    device=config["device"],
    **config["trainer"],
    )

    trainer.run()
    trainer.load_best_model()

    train_metrics = trainer.validate(loaders["full_train"])
    train_val_metrics = trainer.validate(loaders["train_val"])
    hpo_metrics = trainer.validate(loaders["hpo_val"])
    test_metrics = trainer.validate(test_loaders["test"])
    pyrallis.dump(cfg, open('run_config.yaml','w'))
    print(cfg)

if __name__ == "__main__":
    main()