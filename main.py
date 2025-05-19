import logging
import sys
from dataclasses import asdict
from typing import Mapping

import pyrallis
from ebes.pipeline import Runner
from omegaconf import OmegaConf
import omegaconf

from generation.runners import PipelineConfig

logger = logging.getLogger(__name__)


def get_dotlist_args(cfg: Mapping, parent_key: str = "") -> list[str]:
    keys = []
    for key in cfg:
        full_key = f"{parent_key}.{key}" if parent_key else key
        try:
            value = cfg[key]
            if isinstance(value, Mapping):
                keys.extend(get_dotlist_args(value, full_key))
            else:
                keys.append("--" + full_key)
        except omegaconf.errors.InterpolationKeyError:
            keys.append("--" + full_key)
    return keys


def main():
    args = sys.argv[1:]
    cfg = pyrallis.parse(PipelineConfig, "spec_config.yaml", args)
    config = OmegaConf.create(asdict(cfg))
    if config.spec_config is not None:
        specs = OmegaConf.load(config.spec_config)
        config = OmegaConf.merge(config, specs)
        overwritten = set(args) & set(get_dotlist_args(specs))
        if overwritten:
            raise ValueError(
                f"Conflicting args: {overwritten}. "
                "Specify either in command line or spec_config, not both."
            )

    runner = Runner.get_runner(config["runner"]["name"])
    res = runner.run(config)
    print(res)


if __name__ == "__main__":
    main()
