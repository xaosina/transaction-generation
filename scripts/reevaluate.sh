#!/bin/bash

EXPERIMENT_ROOT="$1"
RUN_NAME="${2:-evaluation_run}"
DEVICE="${3:-cuda:2}"

CONFIG_FILE="${EXPERIMENT_ROOT}/seed_0/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_0/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt | head -n1)

# echo "${CONFIG_FILE}"
# echo "${CHECKPOINT_FILE}"
# echo "${RUN_NAME}"
# echo "${DEVICE}"

python main.py \
  --config_path "${CONFIG_FILE}" \
  --trainer.ckpt_resume "${CHECKPOINT_FILE}" \
  --config_factory [dataset/mbd/metric/evaluate] \
  --run_name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --runner.params.n_runs 1