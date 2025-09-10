#!/bin/bash
# -*- coding: utf-8 -*-

EXPERIMENT_ROOT="$1"
RUN_NAME_BASE="${2:-evaluation_run}"
DEVICE="${3:-cuda:0}"

CONFIG_FILE="${EXPERIMENT_ROOT}/seed_0/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_0/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt | head -n1)

topk=1
temp=1.0

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
  --evaluator.temperature ${temp}

echo "✅ DONE!"
