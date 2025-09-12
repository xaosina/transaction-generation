#!/bin/bash

EXPERIMENT_ROOT="$1"
SEED="${2:-0}"
RUN_NAME_BASE="${3:-evaluation_run}"
DEVICE="${4:-cuda:2}"
TOPK="${5:-1}"
TEMP="${6:-1}"

CONFIG_FILE="${EXPERIMENT_ROOT}/seed_${SEED}/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_${SEED}/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt | head -n1)

RUN_NAME="${RUN_NAME_BASE}"
echo "🚀 Running: $RUN_NAME"


python main.py \
  --config_path "${CONFIG_FILE}" \
  --tail_factory "[metrics/paper]" \
  --trainer.ckpt_resume "${CHECKPOINT_FILE}" \
  --run_name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --runner.name GenerationEvaluator \
  --runner.run_type simple \
  --runner.params.n_runs 1 \
  --trainer.verbose True \
  --evaluator.topk "${TOPK}" \
  --evaluator.temperature "${TEMP}"