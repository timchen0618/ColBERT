#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=eval
#SBATCH --output=/scratch/dq2024/ColBERT/get_data.out
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/colbert.ext3:ro \
            /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate colbert;

python /scratch/dq2024/ColBERT/get_training_data.py
"
