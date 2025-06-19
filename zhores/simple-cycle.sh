#!/usr/bin/env bash
#
# simple-cycle.sh — отправляет в очередь все конфиги из папки
#
# Usage:
#   ./simple-cycle.sh <config-dir> [time] [login] [max_exp]
#     config-dir — папка с .yaml
#     time       — лимит Slurm (по умолчанию 06-00)
#     login      — логин для путей и e-mail (по умолчанию whoami)
#     max_exp    — если префикс в run_name начинается с exp<number>_, 
#                  отправить только номера ≤ max_exp (по умолчанию 10)

CONFIG_DIR=$1
TIME=${2:-06-00}
LOGIN=${3:-$(whoami)}
MAX_EXP=${4:-10}

if [ -z "$CONFIG_DIR" ]; then
  echo "Usage: $0 <config-dir> [time] [login] [max_exp]"
  exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: directory '$CONFIG_DIR' not found."
  exit 1
fi

# Чтобы *.yaml не остался как литерал, если нет файлов
shopt -s nullglob

for cfg in "$CONFIG_DIR"/*.yaml; do
  job_name=$(basename "$cfg" .yaml)

  # Если имя в формате exp<number>_*, берём только первые MAX_EXP
  if [[ $job_name =~ ^exp([0-9]+)- ]]; then
    idx=${BASH_REMATCH[1]}
    if (( idx > MAX_EXP )); then
      echo "Skipping $job_name (exp index $idx > $MAX_EXP)"
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

srun singularity exec \\
    --bind /gpfs/gpfs0/${LOGIN}:/home \\
    -f --nv image_trans.sif bash -lc '
      cd /home/transaction-generation
      nvidia-smi
      python main.py \\
        --config_path ${cfg} \\
        --run_name ${job_name} \\
        --device cuda:0 \\
        --trainer.verbose False
    '
EOT

done

# Отключаем nullglob
shopt -u nullglob
