#!/bin/bash
#SBATCH --job-name=unet-training
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=96:00:00

set -euo pipefail
set -x

# Locate python and project root
GITROOT=${HOME}/uq-xray-ct
export PYTHONPATH="${GITROOT}"

cd "${GITROOT}"
source ".venv/bin/activate"

# Experiment settings
DATASET="lung"
SEED=7

# Run
python -m uqct.training.unet \
	--dataset "${DATASET}" \
	--epochs 500 \
	--batch-size 64 \
	--learning-rate 0.0001 \
	--seed "${SEED}"
