# Principled Uncertainty Quantification for Deep Computed Tomography

## Installation

Install `uv` and run

```bash
  uv sync
```

## Usage

We describe how to reproduce the results from the paper.

### Structure of the repository

The repository is structured as follows:

- `uqct`: The main module containing the uncertainty quantification methods.
- `uqct.data`: The data module containing the data loading and preprocessing functions (e.g. for loading the Lung dataset).
- `uqct.ct`: The CT module containing the CT-specific functions (e.g. for computing a FBP).
- `uqct.metrics`: The metrics module containing the metrics functions (PSNR, SSIM etc.).
- `uqct.uq`: The uncertainty quantification module containing the uncertainty quantification functions (e.g. for computing confidence intervals).
- `uqct.utils`: The utils module containing the utils functions (e.g. finding paths to checkpoints, results, caches etc.).
- `uqct.training`: The training module containing the training functions (U-Nets and Diffusion models).

### Reproducing the results

We focus on how to reproduce the results in the main body of the paper in this text. The results from the dense setting can be reproduced using `uqct.evaluation.eval_dense` and `notebooks/Eval_dense.ipynb`.

To reproduce the results from the main body of the paper, first download the datasets and place them into `data/{lung,lamino_tiff,composite}`.
To train the models, run

```bash
uv run -m uqct.training.diffusion --epochs 500 --batch-size 32 --learning-rate 0.0001 --cond True --dataset lung
uv run -m uqct.training.unet
```

### Confidence Sequences

Run the scripts in `uqct/eval/` to compute confidence sequences.
E.g.

```bash
uv run -m uqct.eval.diffusion --sparse True --dataset lung --total-intensity 1e6 --image-range 10 20 --schedule-length 32 --schedule-start 10 --seed 0 --cond True --replicates 16
```

generates confidence sequences for the diffusion model on the Lung dataset for the images 10 to 20 with a total intensity of 1e6 and a schedule length of 32 and a schedule start of 10 (pre-scan using 10 angles) and a seed of 0 and 16 replicates.

The results are stored in `results/`.
To plot the results, run

```bash
uv run -m uqct.vis.plot_scaling --filter-intensities
uv run -m uqct.vis.plot_uq
uv run -m uqct.vis.plot_rotation
uv run -m uqct.vis.plot_examples
uv run -m uqct.vis.plot_example_reconstructions
uv run -m uqct.vis.plot_boundary_samples
```
