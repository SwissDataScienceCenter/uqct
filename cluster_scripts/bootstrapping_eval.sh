#!/bin/bash
#SBATCH --job-name=uqct-bootstrapping
#SBATCH --array=0-359
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00

# Usage: sbatch cluster_scripts/bootstrapping_eval.sh

set -euo pipefail

# Root of the repo (Adjusted to match user's environment variable usage if possible, else standard)
# User's sparse_eval.sh used PROJECT_ROOT="${HOME}/uq-xray-ct"
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

# Load environment variables from .env file
set -o allexport
[[ -f ${PROJECT_ROOT}/.env ]] && source .env
set +o allexport

# Determine Job ID (Normal vs Retry)
if [ "${1:-}" != "" ] && [ -f "$1" ]; then
    # Retry mode: read from file
    # SLURM_ARRAY_TASK_ID is 0-indexed, sed uses 1-indexed lines
    LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
    JOB_ID=$(sed -n "${LINE_NUM}p" "$1")
    
    if [ -z "$JOB_ID" ]; then
        echo "No job ID found at line ${LINE_NUM} of $1. Assuming array task out of range for fewer failed jobs."
        exit 0
    fi
    echo "Retry Mode: Mapped Array ID ${SLURM_ARRAY_TASK_ID} to Job ID ${JOB_ID} from $1"
else
    # Normal mode
    JOB_ID="${SLURM_ARRAY_TASK_ID}"
    echo "Normal Mode: Job ID ${JOB_ID}"
fi

# Run bootstrapping evaluation
echo "Running Bootstrapping Job ${JOB_ID}"
python -m uqct.eval.cli bootstrapping --job-id "${JOB_ID}" --sparse
echo "JOB ${JOB_ID} FINISHED"
