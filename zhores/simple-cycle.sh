#!/bin/bash

n_gpus=1

config_dir=${1:-zhores/configs}

time=${2:-"00-03"}
login=${3:-d.osin}

if [ ! -d "$config_dir"]; then
    echo "Error: Directory '$config_dir' is not found"
    exit 1
fi

for cfg in "$config_dir"/*.yaml; do
    job_name=$(baseline "$cfg" .yaml)
    echo "Submitting job: $job_name"

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

done
# sh transaction-generation/zhores/simple.sh mbd/all_to_one 6-00 e.surkov