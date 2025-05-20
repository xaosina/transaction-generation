import logging
import sys
import tempfile
from dataclasses import asdict

import pyrallis
from ebes.pipeline import Runner
from omegaconf import OmegaConf

from generation.runners import PipelineConfig

logger = logging.getLogger(__name__)


def pop_arg(args, key):
    i = 0
    new_args = []
    value = None
    while i < len(args):
        if args[i] == key:
            value = args[i + 1]
            i += 2
        else:
            new_args += [args[i]]
            i += 1
    return new_args, value


def run_config_factory(config_path, config_factory):
    config_paths = [config_path or "config.yaml"]
    if config_factory is not None:
        config_paths = [
            f"configs/{name}.yaml" for name in config_factory
        ] + config_paths
    configs = [OmegaConf.load(path) for path in config_paths]
    merged_config = OmegaConf.merge(*configs)
    merged_config.pop("config_factory")


def main():
    args = sys.argv[1:]
    cfg = pyrallis.parse(PipelineConfig, "config.yaml", args)
    args, _ = pop_arg(args, "--config_factory")
    args, config_path = pop_arg(args, "--config_path")
    merged_config = run_config_factory(config_path, cfg.config_factory)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmpfile:
        OmegaConf.save(config=merged_config, f=tmpfile.name)
        temp_config_path = tmpfile.name
        print(f"Saved temporary config: {temp_config_path}")
        cfg = pyrallis.parse(PipelineConfig, temp_config_path, args)
        config = OmegaConf.create(asdict(cfg))
        runner = Runner.get_runner(config["runner"]["name"])
        res = runner.run(config)
        print(res)


if __name__ == "__main__":
    main()
