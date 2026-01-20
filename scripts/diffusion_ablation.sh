#!/bin/bash

# diffusion_ablation.sh
# Script for running diffusion ablation experiments

cd /mydata/chip/johannes/uq-xray-ct

uv sync --active

dataset="lamino"
total_intensity="1e7"

# Default values for diffusion parameters
diffusion_num_inference_steps=50
diffusion_num_gradient_steps=5
diffusion_lr=1e-3
diffusion_end=5
batch_size=18

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --diffusion_num_inference_steps) diffusion_num_inference_steps="$2"; shift 2;;
        --diffusion_num_gradient_steps)  diffusion_num_gradient_steps="$2"; shift 2;;
        --diffusion_lr)                 diffusion_lr="$2"; shift 2;;
        --diffusion_end)                diffusion_end="$2"; shift 2;;
        --total_intensity)              total_intensity="$2"; shift 2;;
        --dataset)                     dataset="$2"; shift 2;;
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
done

python uqct/evaluation/eval_dense.py \
    --output_dir results/diffusion_ablation \
    --model cond_diffusion \
    --total_intensity $total_intensity \
    --schedule exponential \
    --base=2 \
    --dataset $dataset \
    --diffusion_num_inference_steps $diffusion_num_inference_steps \
    --diffusion_num_gradient_steps $diffusion_num_gradient_steps \
    --diffusion_lr $diffusion_lr \
    --diffusion_end $diffusion_end \
    --batch_size $batch_size