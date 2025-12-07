#!/bin/bash
#SBATCH --job-name=uqct-eval
#SBATCH --array=0-899
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00

# Usage: sbatch cluster_scripts/sparse_eval.sh

set -euo pipefail

# Root of the repo
# Try to find project root if not set
if [ -z "${PROJECT_ROOT:-}" ]; then
    # Assuming script is in cluster_scripts/
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

# Check for python executable in venv
if [ ! -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "Error: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Please run 'uv sync' to create it."
    exit 1
fi

PYTHON_EXEC="${PROJECT_ROOT}/.venv/bin/python"

echo "Using Python executable: ${PYTHON_EXEC}"

# Run evaluation
${PYTHON_EXEC} -m uqct.eval.cli run --job-id "${SLURM_ARRAY_TASK_ID}" --sparse
