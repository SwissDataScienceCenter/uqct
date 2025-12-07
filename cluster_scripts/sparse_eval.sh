#!/bin/bash
#SBATCH --job-name=uqct-eval
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00

# Usage: sbatch cluster_scripts/sparse_eval.sh <MODEL_NAME>
# Models: fbp, mle, map, unet, diffusion

set -euo pipefail

# Check if model argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <MODEL_NAME>"
    exit 1
fi

MODEL=$1

# Root of the repo
# Try to find project root if not set
if [ -z "${PROJECT_ROOT:-}" ]; then
    # Assuming script is in cluster_scripts/
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

# Run evaluation
# We use 'uv run' to ensure the environment is correct, or direct python if source-ed
# Assuming ~/.local/bin/uv is available or environment is set up.
# "uqctEnv" is not defined, we rely on the caller environment or implicit activation.
# But for SLURM, we usually need to activate.
# The user's run_unet_training.sh sets PYTHONPATH and uses PYTHON_BIN.

# Let's assume standard python usage:
python3 -m uqct.eval.cli run --model "${MODEL}" --job-id "${SLURM_ARRAY_TASK_ID}" --sparse
