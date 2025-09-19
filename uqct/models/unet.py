from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel

# Project utilities
from uqct.ct import AstraParallelOp3D, get_astra_geometry_3d
from uqct.datasets.utils import (KWARGS_COMPOSITE, KWARGS_LAMINO, KWARGS_LUNG,
                                 get_dataset)
# Metrics
from uqct.metrics import get_metrics
# Reuse from training:
from uqct.training.unet import \
    sample_fbp  # forward -> Poisson -> bin -> FBP (LR)
from uqct.training.unet import (  # shared geometry/exposure constants
    MAX_EXPOSURE, MIN_EXPOSURE, N_ANGLES, build_unet, sample_fbp_sparse)


def load_unet(ckpt_path: Path, sparse: bool) -> UNet2DModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = build_unet(sparse).to(device)  # type: ignore
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["unet"]
    # Handle potential _orig_mod prefix
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    unet.load_state_dict(sd, strict=True)
    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}"
    )
    return unet  # type: ignore


@torch.inference_mode()
def predict(
    unet: UNet2DModel,
    fbp_lr: torch.Tensor,
    I0_lr: torch.Tensor,
    class_labels: torch.Tensor | None,
) -> torch.Tensor:
    """
    fbp_lr: (B,128,128) in [0,1]
    I0_lr: (B,1,1) exposure per sample (LR)
    returns: (B,128,128) in [0,1]
    """
    device = unet.device
    x_in = (fbp_lr * 2.0 - 1.0).unsqueeze(1)  # (B,1,128,128), [-1,1]
    exposure_norm = (
        (I0_lr * N_ANGLES - MIN_EXPOSURE) / (MAX_EXPOSURE - MIN_EXPOSURE) * 999
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
    ):
        y = unet(
            x_in,
            timestep=exposure_norm.flatten(),
            class_labels=class_labels,
            return_dict=False,
        )[
            0
        ]  # (B,1,128,128)
    y = ((y + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(1)  # (B,128,128)
    return y


@click.command()
@click.option(
    "--ckpt-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to training checkpoint (.pt)",
)
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lamino", "lung", "composite"]),
    help="Dataset name",
)
@click.option(
    "--num-examples", default=5, type=int, help="How many examples to visualize"
)
@click.option(
    "--sparse-model",
    default=False,
    type=bool,
    is_flag=True,
    help="Whether its as 'sparsely trained' U-Net",
)
@click.option(
    "--sparse-data",
    default=False,
    type=bool,
    is_flag=True,
    help="Whether to use the 'sparse' data distribution",
)
def main(
    ckpt_path: Path,
    dataset: str,
    num_examples: int,
    sparse_model: bool,
    sparse_data: bool,
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Dataset (reuse training settings: rescale=256)
    settings = {
        "composite": {"kwargs": KWARGS_COMPOSITE, "filetype": "nii"},
        "lamino": {"kwargs": KWARGS_LAMINO, "filetype": "tiff"},
        "lung": {"kwargs": KWARGS_LUNG, "filetype": "h5"},
    }
    for v in settings.values():
        v["kwargs"]["rescale"] = 256

    _, test_set = get_dataset(
        settings[dataset]["kwargs"], settings[dataset]["filetype"]
    )
    num_examples = min(num_examples, len(test_set))
    xs = torch.stack([test_set[i] for i in range(num_examples)], dim=0).to(
        device
    )  # (N,1,256,256), [0,1]

    # Geometry & projector (reuse training)
    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))
    proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(angles, 256, num_examples)
    proj_geom_lr, vol_geom_lr = get_astra_geometry_3d(angles, 128, num_examples)
    op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)

    # Load model
    unet = load_unet(ckpt_path, sparse_model).to(device)  # type: ignore
    unet.eval()

    # Build LR FBP and predict
    n_angles = torch.ones(num_examples, device=device) * 200
    if sparse_data:
        fbp_lr, I0_lr, n_angles = sample_fbp_sparse(xs)  # (N,128,128), (N,1,1)
    else:
        fbp_lr, I0_lr = sample_fbp(
            xs, op, proj_geom_lr, vol_geom_lr, device
        )  # (N,128,128), (N,1,1)
    preds_lr = predict(
        unet,
        fbp_lr,
        I0_lr,
        class_labels=n_angles,
    )  # (N,128,128)

    # Prepare for plotting (uniform 256×256 display)
    gt = xs.squeeze(1).clamp(0, 1).detach().cpu()  # (N,256,256)
    fbp_up = (
        F.interpolate(
            fbp_lr.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False
        )
        .squeeze(1)
        .clamp(0, 1)
        .cpu()
    )
    pred_up = (
        F.interpolate(
            preds_lr.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False
        )
        .squeeze(1)
        .clamp(0, 1)
        .cpu()
    )

    # ---- Metrics at native 128×128 resolution ----
    gt_lr = (
        F.interpolate(gt.unsqueeze(1), size=(128, 128), mode="area")
        .squeeze(1)
        .to(fbp_lr.device)
    )  # match FBP/Pred resolution

    fbp_metrics = []
    pred_metrics = []
    for i in range(num_examples):
        m_fbp = get_metrics(gt_lr[i], fbp_lr[i], normalize_range=True, constrained=True)
        m_pred = get_metrics(
            gt_lr[i], preds_lr[i], normalize_range=True, constrained=True
        )
        fbp_metrics.append(m_fbp)
        pred_metrics.append(m_pred)

    # Plot: 3 rows (GT / FBP / Pred) × N columns, show PSNR & SSIM on each column
    _, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.0))
    if num_examples == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for i in range(num_examples):
        # GT
        axes[0, i].imshow(gt[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("GT", fontsize=10)

        # FBP + metrics
        axes[1, i].imshow(fbp_up[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
        axes[1, i].set_title(
            f"FBP  PSNR {fbp_metrics[i]['PSNR']:.2f}  SSIM {fbp_metrics[i]['SS']:.3f}",
            fontsize=8,
        )

        # Pred + metrics
        axes[2, i].imshow(pred_up[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, i].axis("off")
        axes[2, i].set_title(
            f"Pred PSNR {pred_metrics[i]['PSNR']:.2f}  SSIM {pred_metrics[i]['SS']:.3f}",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
