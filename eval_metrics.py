import sys

from omegaconf import OmegaConf
from pathlib import Path
from dataclasses import asdict

sys.path.append('..') # to import from the root of the transaction-generation repo
from generation.metrics.evaluator import SampleEvaluator
from generation.runners.utils import PipelineConfig
from dacite import from_dict, Config

repo_path = Path("/trinity/home/j.chen/2.SeqDiff/test/transaction-generation-multi-features") #YOUR PATH TO REPO

# config_relpath = Path("./log/generation/age/age/only_cat") # YOUR PATH TO CONFIG (CHANGE METRICS TO NEW ONES!)
# log_relpath = Path("./log/generation/age/evaluation_run") # seems that this path not actually used
# samples_relpath = Path("./log/generation/age/evaluation_run/temp1.0_topk1(1)") # YOUR PATH TO GENERATED SAMPLES

# config_path = (repo_path / config_relpath) / 'seed_0' / 'config.yaml'
# log_path = repo_path / log_relpath
# gen_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gen'
# gt_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gt'

config_relpath = Path("./log/generation/age/cdiff") # YOUR PATH TO CONFIG (CHANGE METRICS TO NEW ONES!)
log_relpath = Path("./log/generation/age/evaluation_run") # seems that this path not actually used
samples_relpath = Path("./log/generation/age/evaluation_run/temp1.0_topk1/") # YOUR PATH TO GENERATED SAMPLES

config_path = (repo_path / config_relpath) / 'seed_0' / 'config.yaml'
log_path = repo_path / log_relpath
gen_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gen'
gt_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gt'

#config_relpath = Path("./log/generation/alphabattle_small/only_cat/seed_0") # YOUR PATH TO CONFIG (CHANGE METRICS TO NEW ONES!)
#log_relpath = Path("./log/generation/age/evaluation_run") # seems that this path not actually used
#samples_relpath = Path("./log/generation/alphabattle_small/only_cat/seed_0") # YOUR PATH TO GENERATED SAMPLES

#config_path = (repo_path / config_relpath) / 'seed_0' / 'config.yaml'
#log_path = repo_path / log_relpath
#gen_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gen'
#gt_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gt'

cfg = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(cfg, resolve=True) # to dict
cfg = from_dict(PipelineConfig, cfg, Config(strict=True))


sample_evaluator = SampleEvaluator(
    log_path,
    cfg.data_conf,
    cfg.evaluator,
    device=cfg.device,
    verbose=cfg.trainer.verbose,
)

res = sample_evaluator.estimate_metrics(gt_dir, gen_dir)
print(res)
