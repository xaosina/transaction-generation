#!/usr/bin/env bash
#
# simple-cycle.sh — отправляет в очередь все конфиги из папки,
#                  поддерживает диапазон по префиксам exp<number>_
#
# Usage:
#   ./simple-cycle.sh <config-dir> [time] [login] [start_exp] [end_exp]
#     config-dir — папка с .yaml
#     time       — лимит Slurm (по умолчанию 06-00)
#     login      — логин для e-mail (по умолчанию whoami)
#     start_exp  — нижняя граница индекса exp (по умолчанию 1)
#     end_exp    — верхняя граница индекса exp (по умолчанию 10)

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
for cfg in "$CONFIG_DIR"/*.yaml; do
  job_name=$(basename "$cfg" .yaml)

  # если есть префикс exp<number>_, фильтруем по диапазону
  if [[ $job_name =~ ^exp([0-9]+)_ ]]; then
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
    --config_path ${cfg} \
    --run_name ${job_name} \
    --device cuda:0 \
    --trainer.verbose False
'
EOT

done
shopt -u nullglob
