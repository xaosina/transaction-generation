#!/bin/bash

# Input arguments
# dataset=${1:-m}
# method=$2
# script_name=$3
login=${4:-d.osin}
n_days=${5:-6}
n_gpus=${6:-3}

# Job name
# job_name="${dataset}/${method}/${script_name}"
job_name=${1:-optuna_gru}

# Generate the sbatch script dynamically
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${job_name}

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=${login}@skoltech.ru

#SBATCH --output=outputs/${job_name}.txt

#SBATCH --time=${n_days}-00

#SBATCH --mem=$((n_gpus * 100))G

#SBATCH --nodes=1

#SBATCH -c $((8 * n_gpus))

#SBATCH --gpus=${n_gpus}

srun singularity exec --bind /gpfs/gpfs0/${login}:/home -f --nv image_trans.sif bash -c '
    cd /home/transaction-generation;
    nvidia-smi;
    (sleep 0; python main.py --device 'cuda:0' --trainer.verbose False --run_name optuna_gru --runner.run_type optuna) &
    (sleep 5; python main.py --device 'cuda:1' --trainer.verbose False --run_name optuna_gru --runner.run_type optuna) &
    (sleep 10; python main.py --device 'cuda:2' --trainer.verbose False --run_name optuna_gru --runner.run_type optuna) &
    wait
'
EOT
