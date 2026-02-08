# Principled Confidence Estimation for Deep Computed Tomography

This repository contains the code accompanying the paper  
*“Principled Confidence Estimation for Deep Computed Tomography”*  
by Matteo Gätzner and Johannes Kirschner (2026).

ArXiv link: [https://arxiv.org/abs/2602.05812](https://arxiv.org/abs/2602.05812)
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
- `uqct.eval`: The evaluation module containing the evaluation code (e.g. for computing confidence sequences).


### Datasets and Checkpoints

Data is sourced from public datasets. Preprocessed data and checkpoints are currently not included and are available on request. We are working on providing a zenodo link for the data and checkpoints.

---

### Reproducing the results: Sparse Setting

We focus on how to reproduce the results in the main body of the paper in this text. The results from the dense setting can be reproduced using `uqct.evaluation.eval_dense` and `notebooks/Eval_dense.ipynb`.


To train the models, run

```bash
uv run -m uqct.training.diffusion --epochs 500 --batch-size 32 --learning-rate 0.0001 --cond True --dataset lung
uv run -m uqct.training.unet
```

#### Confidence Sequences

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


---

### Reproducing the results: Dense Setting

```bash
python uqct/eval/dense.py --help

OUTPUT_DIR="./results/"

# Dataset in ("lung", "lamino", "composite")
DATASET="lung"

# Ground truth: generate nnl scores for ground truth image
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --model gt

# FBP
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --model fbp

# MLE
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --iterative_num_gradient_steps=100 --iterative_lr 1e-1 --model mle

# U-Net
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --model unet

# U-Net Ensemble
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --model unet_ensemble

# FBP Bootstrap
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --num_bootstrap_samples 100 --model fbp_bootstrap

# U-Net Bootstrap
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --num_bootstrap_samples 100 --model unet_bootstrap

# Conditional Diffusion
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --guidance_num_gradient_steps 10 --guidance_end 0 --guidance_lr 1e-3 --diffusion_num_inference_steps 50 --model cond_diffusion

# Conditional Diffusion: Boundary Sampling  (needs results from Conditional Diffusion in the same output directory)
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --guidance_num_gradient_steps 10 --guidance_end 0 --guidance_lr 1e-3 --diffusion_num_inference_steps 50 --model diverse_cond_diffusion

# Unconditional Diffusion
python uqct/eval/dense.py --output_dir $OUTPUT_DIR --initial_intensity 1e4  --total_intensity 1e9 --num_steps 30 --num_images 100 --seeds 0-10 --dataset $DATASET --guidance_num_gradient_steps 10 --guidance_end 0 --guidance_lr 1e-1 --guidance_lr_decay --diffusion_num_inference_steps 50 --model diffusion
```

#### Plots

See notebooks/Eval_dense_ICML.ipynb for code to generate plots from the dense evaluation.

---

## Citation

To cite our work, please use the following BibTeX entry:
```
@article{gaetzner2024principled,
	title        = {Principled Confidence Estimation for Deep Computed Tomography},
	author       = {G{\"a}tzner, Matteo and Kirschner, Johannes},
	year         = 2026,
	journal      = {arXiv preprint arXiv:2602.05812},
	url          = {https://arxiv.org/abs/2602.05812}
}
```
