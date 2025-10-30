import sys
import os
os.chdir("/home/dev/2025/transaction-generation")

from omegaconf import OmegaConf
from pathlib import Path
from dataclasses import asdict
from typing import Literal
import pandas as pd

from generation.metrics.evaluator import SampleEvaluator
from generation.runners.utils import PipelineConfig
from dacite import from_dict, Config

dataset: Literal['age', 'alphabattle_small'] = 'age'

repo_path = Path("/home/dev/2025/transaction-generation") #YOUR PATH TO REPO
samples_relpath = Path(f"./log/generation/{dataset}/evaluation/full-old-params/temp1.0_topk1(4)") # YOUR PATH TO GENERATED SAMPLES
log_relpath = Path(f"./log/generation/{dataset}/evaluation") # seems that this path not actually used
config_path = (repo_path / samples_relpath) / 'seed_0' / 'config.yaml'
gen_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gen'
gt_dir = (repo_path / samples_relpath) / "seed_0" / 'evaluation' / 'samples' / 'gt'

cfg = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(cfg, resolve=True) # to dict
cfg = from_dict(PipelineConfig, cfg, Config(strict=True))

sample_evaluator = SampleEvaluator(
    repo_path / log_relpath,
    cfg.data_conf,
    cfg.evaluator,
    device=cfg.device,
    verbose=True,
)
results = sample_evaluator.estimate_metrics(gt_dir, gen_dir)
res = dict()
for k, v in results.items():
    res[k] = [v,]
df = pd.DataFrame(res)
print(results)
print(df.T)
