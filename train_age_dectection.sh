#!/bin/bash
python main.py --config_factory "[start,datasets/age/age,metrics/detection,methods/adiff4tpp/base,methods/adiff4tpp/age]" --runner.name GenerationTrainer --model.name AsynDiffGenerator  --run_name AsynDiffGenerator
