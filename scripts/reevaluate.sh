#!/bin/bash

EXPERIMENT_ROOT="$1"
RUN_NAME_BASE="${2:-evaluation_run/-}"
DEVICE="${3:-cuda:2}"

DATASET="${EXPERIMENT_ROOT#log/generation/}"
DATASET="${DATASET%%/*}"
CONFIG_FILE="${EXPERIMENT_ROOT}/seed_0/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_0/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt | head -n1)

# Define parameter ranges — temperature starts at 1
TEMPERATURES=(1.0)
#  1.5 2.0 5.0 10.0 20.0 50.0 100.0)
TOPK_VALUES=(1)
#  5 10 20 40 80)

# Grid search loop
for temp in "${TEMPERATURES[@]}"; do
  for topk in "${TOPK_VALUES[@]}"; do
    RUN_NAME="${RUN_NAME_BASE}/temp${temp}_topk${topk}"
    echo "🚀 Running: $RUN_NAME"

    python main.py \
      --config_path "${CONFIG_FILE}" \
      --trainer.ckpt_resume "${CHECKPOINT_FILE}" \
      --run_name "${RUN_NAME}" \
      --device "${DEVICE}" \
      --runner.name GenerationEvaluator \
      --runner.run_type simple \
      --runner.params.n_runs 1 \
      --trainer.verbose True \
      --evaluator.topk ${topk} \
      --evaluator.temperature ${temp} \
      --overwrite_factory "[metrics/with_detection/${DATASET}]" \
      --device "cuda:0"
    # Optional: prevent resource spikes
    sleep 2
  done
done

echo "✅ Grid search completed."