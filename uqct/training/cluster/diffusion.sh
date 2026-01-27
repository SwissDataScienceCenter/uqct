#!/bin/bash
#SBATCH --job-name=diffusion-training       # name of the job
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=96:00:00
#SBATCH --array=0-1

set -euo pipefail
set -x

# locate python and project root
GITROOT=${HOME}/uq-xray-ct
export PYTHONPATH=${GITROOT}

cd $GITROOT
source ".venv/bin/activate"

# Experiment settings
DATASETS=(lamino lung)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# run the actual job
python -m uqct.training.diffusion \
	--dataset "$DATASET" \
	--epochs 500 \
	--batch-size 64 \
	--learning-rate 0.0001 \
	--cond True
