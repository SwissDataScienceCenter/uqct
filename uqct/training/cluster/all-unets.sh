#!/bin/bash
#SBATCH --job-name=unet-training
#SBATCH --output=/cluster/scratch/mgaetzner/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/mgaetzner/logs/%x_%A_%a.err
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
PYTHONBIN=/cluster/home/mgaetzner/micromamba/bin/python3
GITROOT=/cluster/home/mgaetzner/uq-xray-ct
export PYTHONPATH="${GITROOT}"

cd "${GITROOT}"

# Experiment settings
DATASETS=(composite lamino)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10))
DATASET=${DATASETS[$dataset_idx]}
SEED=$seed

# Base directory for checkpoints
CKPT_BASE=/cluster/scratch/mgaetzner/uqct/runs/unet_sparse

# (Optional) Construct checkpoint folder path
# CKPT="${CKPT_BASE}/2025-09-17_17-37_${DATASET}_64_500_0.0001_0.37_0.0043_${SEED}/"
# CKPT=""

# Run
"${PYTHONBIN}" "${GITROOT}/uqct/training/unet.py" \
	--dataset "${DATASET}" \
	--epochs 500 \
	--batch-size 64 \
	--learning-rate 3e-5 \
	--seed "${SEED}" \
	--sparse \
	--load-model-ckpt "${CKPT}"
