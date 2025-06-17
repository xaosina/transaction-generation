#!/bin/bash

# Define the list of model names to cycle through
MODELS=("GroundTruthGenerator" "ModeGenerator" "BaselineRepeater" "BaselineHistSampler")

# Loop over each model
for MODEL in "${MODELS[@]}"; do
    echo "Running with model: $MODEL"
    python main.py --runner.name GenerationEvaluator --model.name "$MODEL" --run_name "$MODEL"
done

python main.py --run_name GRU