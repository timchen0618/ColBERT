#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --mem=500GB
#SBATCH --job-name=retrieve_round1_top6_concatenated_colbert
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=retrieve_round1_top6_concatenated_colbert.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/colbert.ext3


singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/ColBERT/; conda activate colbert; python indexing.py --do_retrieval --dataset /scratch/hc3337/projects/ColBERT/data/round1_top6_concatenated.json" 
