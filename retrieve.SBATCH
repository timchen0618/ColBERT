#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=train_colbert
#SBATCH --mail-type=END
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --output=round1_retr.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --gres=gpu:l40s:1

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/dq2024/colbert.ext3


singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/dq2024/ColBERT/; conda activate colbert; python indexing.py --do_indexing --dataset qampari_full" 
