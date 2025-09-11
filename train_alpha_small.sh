#!/bin/bash
python main.py --config_factory "[start,datasets/alphabattle_small/alphabattle_small,metrics/default,methods/adiff4tpp/base,methods/adiff4tpp/alphabattle_small]" --runner.name GenerationTrainer --model.name AsynDiffGenerator  --run_name AsynDiffGenerator
