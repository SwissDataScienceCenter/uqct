#!/bin/bash
# SLURM array job for the sparse SK-ROCK evaluation grid.
#
# Each array task runs one cell of `uqct.eval.cli`'s skrock grid
# (build_skrock_grid: datasets x total_intensity_values x seed_range x
# chunks-of-10 images). With the defaults in uqct/settings.toml that is
#   3 datasets x 6 intensities x 10 seeds x 10 chunks = 1800 tasks
# -> --array=0-1799 (the --job-id is 0-indexed). To re-check the count after
# editing settings.toml, run e.g. `uv run -m uqct.eval.cli skrock --job-id 999999`
# (it prints "Job ID 999999 out of range (0-<N-1>)").
#
# --no-duplicate is the default in the skrock subcommand, so a cell whose
# results parquet already exists under results/runs/ is skipped -- safe to
# resubmit after preemption, and it does NOT touch the existing fbp / mle /
# unet / diffusion / bootstrap results (those come from `uqct.eval.cli run` and
# its own --no-duplicate, or the bootstrapping/equivariant subcommands; just
# don't re-run those if you want them reused as-is).
#
# Submit with:  sbatch scripts/submit_skrock.sh
# (adapt the #SBATCH lines / paths below to your cluster.)

#SBATCH --job-name=skrock_eval
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
"${RUN[@]}" skrock --sparse --job-id "${SLURM_ARRAY_TASK_ID}"
