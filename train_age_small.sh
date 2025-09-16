#!/bin/bash
python main.py --config_factory "[start,datasets/age_small/age_small,metrics/default,methods/adiff4tpp/base,methods/adiff4tpp/age_test]" --runner.name GenerationTrainer --model.name AsynDiffGenerator  --run_name AsynDiffGenerator
