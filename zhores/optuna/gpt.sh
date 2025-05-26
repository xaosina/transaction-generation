#!/bin/bash

# Input arguments
# dataset=${1:-m}
# method=$2
# script_name=$3
n_gpus=1

job_name=${1:-optuna/gpt}

array_range=${2:-"0-0"}

n_days=${3:-6}
login=${4:-d.osin}

# Generate the sbatch script dynamically
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${job_name}

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=${login}@skoltech.ru

#SBATCH --array=${array_range}

#SBATCH --output=outputs/${job_name}/%j_%a.txt

#SBATCH --time=${n_days}-00

#SBATCH --mem=$((n_gpus * 100))G

#SBATCH --nodes=1

#SBATCH -c $((8 * n_gpus))

#SBATCH --gpus=${n_gpus}

srun singularity exec --bind /gpfs/gpfs0/${login}:/home -f --nv image_trans.sif bash -c '
    cd /home/transaction-generation;
    nvidia-smi;
    python main.py \
        --device 'cuda:0' \
        --trainer.verbose False \
        --run_name ${job_name} \
        --runner.run_type optuna \
        --config_factory [datasets/mbd/mbd,methods/gpt,datasets/mbd/metrics/default]
'
EOT
