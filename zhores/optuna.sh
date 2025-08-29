#!/bin/bash

n_gpus=1

dataset=${1:-alphabattle}
method=${2:-vae}

array_range=${3:-"0-0"}

n_days=${4:-6}
login=${5:-d.osin}

# Generate the sbatch script dynamically
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=optuna/${dataset}/${method}

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=${login}@skoltech.ru

#SBATCH --array=${array_range}

#SBATCH --output=outputs/optuna/${dataset}/${method}/%j_%a.txt

#SBATCH --time=${n_days}-00

#SBATCH --mem=$((n_gpus * 100))G

#SBATCH --nodes=1

#SBATCH -c $((8 * n_gpus))

#SBATCH --gpus=${n_gpus}

srun singularity exec --bind /gpfs/gpfs0/${login}:/home -f --nv image_trans.sif bash -c '
    cd /home/transaction-generation;
    nvidia-smi;
    python main.py \
        --run_name optuna/${method} \
        --config_factory [start,metrics/default,datasets/${dataset}/${dataset},methods/${method},optuna]
'
EOT

# sh transaction-generation/zhores/simple.sh optuna_loss/gpt
# python main.py --run_name optuna/gru --config_factory [start,datasets/age/age,methods/gru,metrics/default,optuna]