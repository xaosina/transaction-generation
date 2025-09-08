#!/usr/bin/env bash
#
# simple-cycle.sh _ __________ _ _______ ___ _______ __ _____,
#                  ____________ ________ __ _________ exp<number>_
#
# Usage:
#   ./simple-cycle.sh <config-dir> [time] [login] [start_exp] [end_exp]
#     config-dir _ _____ _ .yaml
#     time       _ _____ Slurm (__ _________ 06-00)
#     login      _ _____ ___ e-mail (__ _________ whoami)
#     start_exp  _ ______ _______ _______ exp (__ _________ 1)
#     end_exp    _ _______ _______ _______ exp (__ _________ 10)

CONFIG_DIR=$1
LOCAL_SHUFFLE=${2:--1}
TIME=${3:-06-00}
LOGIN=${4:-$(whoami)}
START_EXP=${5:-1}
END_EXP=${6:-10}

if [ -z "$CONFIG_DIR" ]; then
  echo "Usage: $0 <config-dir> [time] [login] [start_exp] [end_exp]"
  exit 1
fi

BASE="/home/${LOGIN}/dev/transaction-generation"
# BASE="/home/dev/2025/transaction-generation"
CONFIG_DIR="${BASE%/}/${CONFIG_DIR}"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: directory '$CONFIG_DIR' not found."
  exit 1
fi

# if [ "$LOCAL_SHUFFLE" -eq -1 ]; then
  # local_shuffle="local_shuffle_n1"
# else
local_shuffle="local_shuffle_$LOCAL_SHUFFLE"
# fi

shopt -s nullglob
for cfg in "$CONFIG_DIR"*.yaml; do
  rel_cfg=${cfg#*transaction-generation/}
  job_name=${rel_cfg#zhores/configs/best_params/}
  job_name=${job_name%.yaml}/$local_shuffle

  echo "Submitting job: $job_name"
  # exit 1
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=ais-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${LOGIN}@skoltech.ru
#SBATCH --output=outputs/${job_name}/%j.txt
#SBATCH --time=${TIME}
#SBATCH --mem=$((1 * 100))G
#SBATCH --nodes=1
#SBATCH -c $((8 * 1))
#SBATCH --gpus=1

srun singularity exec --bind /home/${LOGIN}/dev:/home -f --nv image_trans.sif bash -lc '
  cd /home/transaction-generation
  nvidia-smi
  python main.py \
    --config_path ${rel_cfg} \
    --run_name ${job_name} \
    --device cuda:0 \
    --append_factory [specs/${local_shuffle}] \
    --trainer.verbose False \
    --runner.params.n_runs 3 \
'
EOT

done
shopt -u nullglob

