#!/bin/bash

job_name=${1:-simple}
device=${2:-"cuda:0"}

PATH_TO_CHECK=/home/transaction-generation/zhores/configs/${job_name}.yaml

if [ ! -e "$PATH_TO_CHECK" ]; then
    echo "Error: The path '$PATH_TO_CHECK' does not exist."
    exit 1
fi

# Generate the sbatch script dynamically

cd /home/transaction-generation;
nvidia-smi;
python main.py \
    --config_path zhores/configs/${job_name}.yaml \
    --run_name ${job_name} \
    --device ${device} \
    # --trainer.verbose False \
