#!/bin/bash
#SBATCH --job-name=unet-training
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=96:00:00
#SBATCH --array=0-19

module eth_proxy load

set -euo pipefail
set -x

# Locate python and project root
GITROOT=${HOME}/uq-xray-ct
export PYTHONPATH="${GITROOT}"

cd "${GITROOT}"
source ".venv/bin/activate"

# Experiment settings
DATASETS=(composite lamino)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10))
DATASET=${DATASETS[$dataset_idx]}
SEED=$seed

# Run
python -m uqct.training.unet \
	--dataset "${DATASET}" \
	--epochs 500 \
	--batch-size 64 \
	--learning-rate 3e-5 \
	--seed "${SEED}" \
	--sparse
