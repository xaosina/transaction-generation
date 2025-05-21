#!/bin/bash

n_gpus=1

job_name=${1:-simple}

time=${2:-"00-03"}
login=${3:-d.osin}

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

#SBATCH --output=outputs/${job_name}/%j.txt

#SBATCH --time=${time}

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
'
EOT
