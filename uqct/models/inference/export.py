from pathlib import Path

import torch

# --- Import your model ---
from uqct.models.diffusion import Diffusion
from uqct.utils import get_checkpoint_dir

# --- Configuration ---
DEVICE = torch.device("cuda")

# TensorRT Optimization Profile Config
# MIN: Smallest batch you'll ever send (1)
# OPT: The batch size you use most often (32) - TRT optimizes for this!
# MAX: The absolute largest batch you will ever allow (64)
MIN_BATCH = 1
OPT_BATCH = 32
MAX_BATCH = 64

N_CHANNELS = 1
RES = 128
DTYPE = torch.float  # FP32 weights

datasets = ("lung", "composite", "lamino")

if __name__ == "__main__":
    for dataset in datasets:
        onnx_file = (
            get_checkpoint_dir()
            / "diffusion"
            / Path(f"ddpm_conditional_128_{dataset}.onnx")
        )

        if not onnx_file.exists():
            print(f"Exporting to {onnx_file}...")

            # Initialize model
            unet = Diffusion(dataset, cond=True).unet.float().to(DEVICE)  # type: ignore
            unet.eval()

            # Create Dummy Inputs (using OPT_BATCH size)
            x_t_b = torch.rand(
                OPT_BATCH, N_CHANNELS, RES, RES, device=DEVICE, dtype=DTYPE
            )
            fbps_b = torch.rand(
                OPT_BATCH, N_CHANNELS, RES, RES, device=DEVICE, dtype=DTYPE
            )

            # KEY CHANGE: Using Int32 for timesteps
            timesteps_b = torch.randint(
                0, 999, (OPT_BATCH,), device=DEVICE, dtype=torch.int32
            )

            intensities_norm_b = (
                torch.rand(OPT_BATCH, device=DEVICE, dtype=DTYPE) * 2 - 1
            )
            n_angles_norm_b = torch.rand_like(
                intensities_norm_b, device=DEVICE, dtype=DTYPE
            )

            input_names = [
                "x_t",
                "fbps",
                "timesteps",
                "intensities_norm",
                "n_angles_norm",
            ]

            # Define dynamic axes
            dynamic_axes = {k: {0: "batch_size"} for k in input_names}

            torch.onnx.export(
                unet,
                (x_t_b, fbps_b, timesteps_b, intensities_norm_b, n_angles_norm_b),
                onnx_file,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=input_names,
                output_names=["pred"],
                dynamic_axes=dynamic_axes,
            )

            # Cleanup PyTorch memory before starting TensorRT
            del (
                unet,
                x_t_b,
                fbps_b,
                timesteps_b,
                intensities_norm_b,
                n_angles_norm_b,
            )
            torch.cuda.empty_cache()
            print("ONNX Export complete.")
        else:
            print(f"ONNX file {onnx_file} already exists. Skipping export.")
