#!/bin/bash

# diffusion_ablation.sh
# Script for running diffusion ablation experiments

cd /mydata/chip/johannes/uq-xray-ct

uv sync --active

dataset="lamino"
total_intensity="1e9"
initial_intensity="1e6"

# Default values for diffusion parameters
diffusion_num_inference_steps=50
guidance_num_gradient_steps=5
guidance_lr=1e-3
guidance_lr_decay=false
guidance_end=5
batch_size=-1
# Set output_dir to results/{current_date} by default
current_date=$(date +%Y-%m-%d)
output_dir="results/$current_date"
existing="skip"
rotation=false
seeds="0"
num_images=50
init_fraction=false
num_steps=10
num_samples=10
bypass_inverse_crime=false
verbose=false
iterative_num_gradient_steps=80
iterative_lr=1e-1

# Default model(s) and dataset(s)
models=("fbp")
datasets=("lamino")


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --diffusion_num_inference_steps) diffusion_num_inference_steps="$2"; shift 2;;
        --guidance_num_gradient_steps)  guidance_num_gradient_steps="$2"; shift 2;;
        --guidance_lr)                 guidance_lr="$2"; shift 2;;
        --guidance_lr_decay)           guidance_lr_decay=true; shift ;;
        --guidance_end)                guidance_end="$2"; shift 2;;
        --iterative_num_gradient_steps) iterative_num_gradient_steps="$2"; shift 2;;
        --iterative_lr)                iterative_lr="$2"; shift 2;;
        --total_intensity)              total_intensity="$2"; shift 2;;
        --initial_intensity)            initial_intensity="$2"; shift 2;;
        --existing)                     existing="$2"; shift 2;;
        --rotation)                rotation="$2"; shift 2;;
        --dataset)
            shift
            datasets=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --output_dir)                   output_dir="$2"; shift 2;;
        --seeds)                        seeds="$2"; shift 2;;
        --num_images)                   num_images="$2"; shift 2;;
        --num_samples)                  num_samples="$2"; shift 2;;
        --batch_size)                    batch_size="$2"; shift 2;;
        --init_fraction)                init_fraction="$2"; shift 2;;
        --num_steps)                    num_steps="$2"; shift 2;;
        --bypass_inverse_crime)          bypass_inverse_crime=true; shift ;;
        --model)
            shift
            models=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                models+=("$1")
                shift
            done
            ;;
        --verbose) verbose=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
done



for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Set batch_size depending on model if batch_size is -1
        if [[ "$batch_size" -eq -1 ]]; then
            if [[ "$model" == "cond_diffusion" ||  "$model" == "beta_cond_diffusion" ||  "$model" == "diverse_cond_diffusion" ||  "$model" == "diffusion" ]]; then
            model_batch_size=12  # 0.2 A100
            # model_batch_size=  # 0.5 A100
            elif [[ "$model" == "gt" ]]; then
            model_batch_size=20  # 0.2 A100
            else
            model_batch_size=100 # 0.2 A100
            # model_batch_size=300 # 0.5 A100
            fi
        else
            model_batch_size="$batch_size"
        fi

        cmd=(
            python uqct/evaluation/eval_dense.py
            --output_dir "$output_dir"
            --model "$model"
            --total_intensity "$total_intensity"
            --initial_intensity "$initial_intensity"
            --num_samples "$num_samples"
            --schedule exponential
            --dataset "$dataset"
            --iterative_num_gradient_steps "$iterative_num_gradient_steps"
            --iterative_lr "$iterative_lr"
            --diffusion_num_inference_steps "$diffusion_num_inference_steps"
            --guidance_num_gradient_steps "$guidance_num_gradient_steps"
            --guidance_lr "$guidance_lr"
            $( [[ "$guidance_lr_decay" == "true" ]] && echo --guidance_lr_decay )
            --guidance_end "$guidance_end"
            --batch_size "$model_batch_size"
            --num_steps "$num_steps"
            --existing "$existing"
            $( [[ "$rotation" != "false" ]] && echo --rotation "$rotation" )
            $( [[ "$init_fraction" != "false" ]] && echo --init_fraction "$init_fraction" )
            --seeds "$seeds"
            --num_images "$num_images"
            $( [[ "$bypass_inverse_crime" != "false" ]] && echo --bypass_inverse_crime )
            $( [[ "$verbose" == "true" ]] && echo --verbose )
        )
        echo "${cmd[@]}"
        "${cmd[@]}"

        
    done
done
