#!/bin/bash
#SBATCH --job-name=unet-training       # name of the job
#SBATCH --output=/cluster/scratch/mgaetzner/logs/%x_%A_%a.out  # stdout
#SBATCH --error=/cluster/scratch/mgaetzner/logs/%x_%A_%a.err   # stderr
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=96:00:00
#SBATCH --array=0-29

module eth_proxy load

set -euo pipefail
set -x

# Locate python and project root
PYTHONBIN=/cluster/home/mgaetzner/micromamba/bin/python3
GITROOT=/cluster/home/mgaetzner/uq-xray-ct
export PYTHONPATH=${GITROOT}

cd $GITROOT

# Experiment settings
DATASETS=(composite lung lamino)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10))
DATASET=${DATASETS[$dataset_idx]}
SEED=$seed

# run the actual job
"$PYTHONBIN" "${GITROOT}/uqct/training/unet.py" \
	--dataset "$DATASET" \
	--epochs 500 \
	--batch-size 64 \
	--learning-rate 0.0001 \
	--seed "$SEED"
