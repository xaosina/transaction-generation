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
            if key in ["--config_factory", "--append_factory"]:
                assert value[0] == "[" and value[-1] == "]", "Wrong factory format"
                value = value[1:-1].split(",")
            i += 2
        else:
            new_args += [args[i]]
            i += 1
    return new_args, value


def run_config_factory(config_path, config_factory):
    if config_factory is not None:
        config_paths = [f"configs/{name}.yaml" for name in config_factory]
    else:
        config_paths = []
    config_paths += [config_path or "config.yaml"]
    configs = [OmegaConf.load(path) for path in config_paths]
    merged_config = OmegaConf.merge(*configs)
    merged_config["config_factory"] = None
    return merged_config


def main():
    args = sys.argv[1:]
    args, config_factory = pop_arg(args, "--config_factory")
    args, append_factory = pop_arg(args, "--append_factory")
    args, config_path = pop_arg(args, "--config_path")
    path = config_path or "config.yaml"
    config_factory = config_factory or OmegaConf.load(path).get("config_factory")
    if append_factory is not None:
        config_factory += append_factory
    merged_config = run_config_factory(config_path, config_factory)
    #breakpoint()
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
