#!/bin/bash
#SBATCH --job-name=uqct-sparse-eval
#SBATCH --array=0-899
#SBATCH --output=/cluster/scratch/mgaetzner/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/mgaetzner/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00

# Usage: sbatch cluster_scripts/sparse_eval.sh

set -euo pipefail

# Root of the repo
PROJECT_ROOT="${HOME}/uq-xray-ct"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

# Check for python executable in venv
if [ ! -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "Error: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Please run 'uv sync' to create it."
    exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

# Run evaluation
python -m uqct.eval.cli run --job-id "${SLURM_ARRAY_TASK_ID}" --sparse
