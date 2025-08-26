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
TIME=${2:-06-00}
LOGIN=${3:-$(whoami)}
START_EXP=${4:-1}
END_EXP=${5:-10}

if [ -z "$CONFIG_DIR" ]; then
  echo "Usage: $0 <config-dir> [time] [login] [start_exp] [end_exp]"
  exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: directory '$CONFIG_DIR' not found."
  exit 1
fi

shopt -s nullglob
for cfg in "$CONFIG_DIR"*.yaml; do
  rel_cfg=${cfg#*transaction-generation/}
  job_name=$(basename "${cfg#*/}" .yaml)
  echo $rel_cfg
  echo $cfg
  # ____ ____ _______ exp<number>_, _________ __ _________
  if [[ $job_name =~ ^exp([0-9]+)- ]]; then
    idx=${BASH_REMATCH[1]}
    if (( idx < START_EXP || idx > END_EXP )); then
      echo "Skipping $job_name (exp index $idx outside [$START_EXP,$END_EXP])"
      continue
    fi
  fi

  echo "Submitting job: $job_name"
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

srun singularity exec --bind /gpfs/gpfs0/${LOGIN}:/home -f --nv image_trans.sif bash -lc '
  cd /home/transaction-generation
  nvidia-smi
  python main.py \
    --config_path ${rel_cfg} \
    --run_name ${job_name} \
    --device cuda:0 \
    --append_factory [specs/local_shuffle_16] \
    --trainer.verbose False
'
EOT

done
shopt -u nullglob