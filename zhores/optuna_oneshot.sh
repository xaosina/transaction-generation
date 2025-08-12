#!/bin/bash

n_gpus=1
n_days=6
login="d.osin"

# Define arrays
models=(gru transformer)
datasets=(age mbd megamarket retail shakespeare taobao zvuk)
losses=(target dist matched)

# Loop through all combinations
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for loss in "${losses[@]}"; do
      echo "Submitting job for dataset=$dataset, model=$model, loss=$loss"

      # Create unique output path
      output_dir="outputs/optuna/${dataset}/${model}_${loss}"
      sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=optuna/${dataset}/${model}_${loss}
#SBATCH --partition=ais-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${login}@skoltech.ru
#SBATCH --output=${output_dir}/%j.txt
#SBATCH --time=${n_days}-00:00
#SBATCH --mem=$((n_gpus * 100))G
#SBATCH --nodes=1
#SBATCH -c $((8 * n_gpus))
#SBATCH --gpus=${n_gpus}

srun singularity exec --bind /gpfs/gpfs0/${login}:/home -f --nv image_trans.sif bash -c '
    cd /home/transaction-generation;
    nvidia-smi;
    python main.py \
        --run_name optuna/${model}_${loss} \
        --config_factory [start,datasets/${dataset}/${dataset},methods/oneshot/${model},methods/oneshot/${loss},metrics/default,optuna,allto1]
'
EOT
    done
  done
done

echo "Submitted $((${#models[@]} * ${#datasets[@]} * ${#losses[@]})) jobs."