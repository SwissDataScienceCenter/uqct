#!/bin/bash
# SLURM array job: generate predictions on the alpha-calibration window
# (image_range=(0,10), seed 0) for one method, bundled to ~1h per task.
#
# Required env vars (pass via `sbatch --export=ALL,METHOD=...,BUNDLE=...`):
#   METHOD       skrock | equivariant_bootstrapping | bootstrapping_fbp |
#                bootstrapping_unet | boundary
#   BUNDLE       number of (dataset, intensity) cells per task (5 for skrock,
#                6 for eb, 9 for unet-bootstrap, 18 for fbp-bootstrap, 2 for boundary).
#   TOTAL_CELLS  defaults to 18 = 3 datasets * 6 intensities.
#
# Array index range = ceil(TOTAL_CELLS / BUNDLE) - 1.

#SBATCH --job-name=calib_window
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
    PY=(python)
else
    PY=(uv run python)
fi

METHOD="${METHOD:?must set METHOD via --export}"
BUNDLE="${BUNDLE:?must set BUNDLE via --export}"
TOTAL_CELLS="${TOTAL_CELLS:-18}"

START=$(( SLURM_ARRAY_TASK_ID * BUNDLE ))
END=$(( START + BUNDLE ))
if [ "$END" -gt "$TOTAL_CELLS" ]; then
    END=$TOTAL_CELLS
fi
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): method=${METHOD} cells ${START}..$((END-1))"

for ((CELL=START; CELL<END; CELL++)); do
    echo "--- cell ${CELL} ---"
    "${PY[@]}" scripts/run_calibration_window.py --method "${METHOD}" --cell-idx "${CELL}"
done
