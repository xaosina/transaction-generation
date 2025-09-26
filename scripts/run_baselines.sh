#!/bin/bash

# Define the list of model names to cycle through
MODELS=("gt" "mode" "repeat" "hist_sampler")

# Loop over each model
for MODEL in "${MODELS[@]}"; do
    echo "Running with model: $MODEL"
    python main.py --config_factory "[start,datasets/age/age,methods/baselines/$MODEL]"  --run_name "debug/$MODEL"
done

# python main.py --run_name GRU
# data = pd.read_parquet("/home/dev/2025/transaction-generation/log/generation/age/BaselineRepeater(9)/seed_0/evaluation/samples/gen/part-0000.parquet")
# data.explode(["trans_date", "amount_rur", "small_group"]).to_parquet("age-repeat.parquet")