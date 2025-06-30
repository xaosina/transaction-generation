#!/bin/bash

# Input arguments
# dataset=${1:-m}
# method=$2
# script_name=$3
n_gpus=1

job_name=${1:-optuna}

array_range=${2:-"0-0"}

n_days=${3:-6}
login=${4:-d.osin}

PATH_TO_CHECK=/gpfs/gpfs0/${login}/transaction-generation/zhores/configs/${job_name}.yaml

if [ ! -e "$PATH_TO_CHECK" ]; then
    echo "Error: The path '$PATH_TO_CHECK' does not exist."
    exit 1
fi

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
        --config_path zhores/configs/${job_name}.yaml \
        --run_name ${job_name} \
        --device 'cuda:0' \
        --trainer.verbose False \
        --runner.run_type optuna

'
EOT

# sh transaction-generation/zhores/simple.sh optuna_loss/gpt