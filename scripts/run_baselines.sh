#!/bin/bash

# Define the list of model names to cycle through
MODELS=("GroundTruthGenerator" "ModeGenerator" "BaselineRepeater" "BaselineHistSampler")
# MODELS=("GroundTruthGenerator")

# Loop over each model
for MODEL in "${MODELS[@]}"; do
    echo "Running with model: $MODEL"
    python main.py --config_factory "[start,datasets/shakespeare/shakespeare,methods/gru,metrics/default,1to1]" --runner.name GenerationEvaluator --model.name "$MODEL" --run_name "$MODEL"
done

# python main.py --run_name GRU