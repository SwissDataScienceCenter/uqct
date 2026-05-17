#!/bin/bash
# Generic SLURM array job that bundles multiple `uqct.eval.cli` grid cells per
# task, so per-task wall time is ~1 h or more (cluster-friendly when individual
# cells are short -- EB ~9 min/cell, SK-ROCK ~2 min/cell).
#
# Required env vars (pass via `sbatch --export=ALL,METHOD=...,BUNDLE=...,GRID_SIZE=...`):
#   METHOD     subcommand name on `uqct.eval.cli` ("equivariant-bootstrapping" | "skrock").
#   BUNDLE     number of grid cells run sequentially per task.
#   GRID_SIZE  total grid size (1800 for both methods with current settings.toml).
#
# Array index range must satisfy `0 <= SLURM_ARRAY_TASK_ID < ceil(GRID_SIZE/BUNDLE)`.
# The last task may run fewer than BUNDLE cells if GRID_SIZE is not a multiple of BUNDLE.

#SBATCH --job-name=bundled
#SBATCH --output=/cluster/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g

set -euo pipefail

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/uq-xray-ct}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
    RUN=(python -m uqct.eval.cli)
else
    RUN=(uv run -m uqct.eval.cli)
fi

BUNDLE="${BUNDLE:?must set BUNDLE via --export}"
GRID_SIZE="${GRID_SIZE:?must set GRID_SIZE via --export}"
METHOD="${METHOD:?must set METHOD via --export}"

START=$(( SLURM_ARRAY_TASK_ID * BUNDLE ))
END=$(( START + BUNDLE - 1 ))
if [ "$END" -ge "$GRID_SIZE" ]; then
    END=$(( GRID_SIZE - 1 ))
fi
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): cells ${START}..${END}  method=${METHOD}"

for i in $(seq "$START" "$END"); do
    echo "--- cell ${i} ---"
    # --no-duplicate is the default; pre-existing cells (e.g. from the seed-0 array)
    # are skipped, so resubmissions and overlapping arrays stay idempotent.
    "${RUN[@]}" "${METHOD}" --sparse --job-id "$i"
done
