#!/bin/bash
# SLURM array job for the sparse equivariant-bootstrap evaluation grid
# (Tachella & Pereyra, AISTATS 2024; estimator = FBPUNet).
#
# Each array task runs one cell of `uqct.eval.cli`'s equivariant-bootstrapping grid
# (build_equivariant_bootstrapping_grid: datasets x total_intensity_values x seed_range x
# chunks-of-10 images; rotation_std_deg / flip are taken from the per-cell
# [[eval-sparse.equivariant_bootstrapping.calibrated]] overrides in uqct/settings.toml,
# falling back to the section defaults). With the current settings that is
#   3 datasets x 6 intensities x 10 seeds x 10 chunks = 1800 tasks
# -> --array=0-1799 (the --job-id is 0-indexed). Re-check the count after editing
# settings.toml with `uv run -m uqct.eval.cli equivariant-bootstrapping --job-id 999999`.
#
# Prerequisites: the FBPUNet checkpoint bundle must be on the cluster (UQCT_CKPT_DIR /
# checkpoints/), plus the dataset bundle (UQCT_DATA_DIR / data/).
#
# --no-duplicate is the subcommand default, so a cell whose results parquet already
# exists under results/runs/ is skipped -- safe to resubmit after preemption. NOTE: the
# old equivariant_bootstrapping_* results in results/runs/ predate the calibration
# overrides (and an earlier model name / chunking) -- delete them first for a clean run:
#   rm results/runs/equivariant_bootstrapping*
#
# Submit with:  sbatch scripts/submit_equivariant_bootstrapping.sh
# (adapt the #SBATCH lines / paths below to your cluster.)

#SBATCH --job-name=eqbootstrap_eval
#SBATCH --output=/cluster/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --array=0-1799%50

set -euo pipefail

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/uq-xray-ct}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
# Prefer the project venv; fall back to `uv run`.
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
    RUN=(python -m uqct.eval.cli)
else
    RUN=(uv run -m uqct.eval.cli)
fi

echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
"${RUN[@]}" equivariant-bootstrapping --sparse --job-id "${SLURM_ARRAY_TASK_ID}"
