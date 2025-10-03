from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from uqct.ct import AstraParallelOp3D, get_astra_geometry_3d
from uqct.datasets.utils import get_dataset
from uqct.metrics import get_metrics
from uqct.training.unet import (MAX_EXPOSURE, MIN_EXPOSURE, N_ANGLES,
                                build_unet, sample_fbp_dense,
                                sample_fbp_sparse)


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


@torch.inference_mode()
def eval_mse_over_dataset(
    dataset,
    unet: UNet2DModel,
    device: torch.device,
    sparse: bool,
    batch_size: int = 64,
    num_workers: int = 0,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))

    total_sqerr = 0.0
    total_pixels = 0

    unet.eval()
    for x in tqdm(loader):
        x = x.to(device, non_blocking=True)  # (B,1,256,256) in [0,1]
        B = x.shape[0]

        # Build ASTRA geoms for this batch size (like training)
        side_length = x.shape[-1]
        proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(angles, side_length, B)
        proj_geom_lr, vol_geom_lr = get_astra_geometry_3d(angles, 128, B)
        op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            if sparse:
                # Use your provided sparse sampler to stay identical:
                fbp, I_0, n_angles = sample_fbp_sparse(x)
                _n_angles = n_angles
                # ---- mirror loss_fn exactly: scale inputs to [-1,1]
                x_in = x * 2.0 - 1.0
                fbp_in = (
                    (fbp * 2.0 - 1.0).unsqueeze(1)
                    if fbp.ndim == 3
                    else (fbp * 2.0 - 1.0)
                )
                exposure_norm = (
                    (I_0 * _n_angles - MIN_EXPOSURE)
                    / (MAX_EXPOSURE - MIN_EXPOSURE)
                    * 999
                )
                y = unet(
                    fbp_in,
                    timestep=exposure_norm.flatten(),
                    class_labels=_n_angles - 1,
                    return_dict=False,
                )[0]
            else:
                fbp, I_0 = sample_fbp_dense(x, op, proj_geom_lr, vol_geom_lr, device)
                x_in = x * 2.0 - 1.0
                fbp_in = (
                    (fbp * 2.0 - 1.0).unsqueeze(1)
                    if fbp.ndim == 3
                    else (fbp * 2.0 - 1.0)
                )
                exposure_norm = (
                    (I_0 * N_ANGLES - MIN_EXPOSURE)
                    / (MAX_EXPOSURE - MIN_EXPOSURE)
                    * 999
                )
                y = unet(fbp_in, timestep=exposure_norm.flatten(), return_dict=False)[0]

            # Downsample GT to pred resolution (area), both are in [-1,1]
            x_lr = F.interpolate(x_in.unsqueeze(1), size=y.shape[-2:], mode="area")
            # True mean per-pixel MSE
            total_sqerr += F.mse_loss(y, x_lr, reduction="sum").item()
            total_pixels += y.numel()

        # free
        del (
            x,
            fbp,
            I_0,
            y,
            x_lr,
            op,
            proj_geom_hr,
            proj_geom_lr,
            vol_geom_hr,
            vol_geom_lr,
        )

    return total_sqerr / max(1, total_pixels)


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
    dataset: Literal["lamino", "lung", "composite"],
    num_examples: int,
    sparse_model: bool,
    sparse_data: bool,
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Deterministic split & sampling
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    train_set, test_set = get_dataset(dataset, True)

    # ---- Create val split from train_set (5%) ----
    dataset_size = len(train_set)
    val_size = int(0.05 * dataset_size)
    train_size = dataset_size - val_size
    train_subset, val_subset = random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),  # reproducible split
    )

    # ---- Load model ----
    unet = load_unet(ckpt_path, sparse_model).to(device)  # type: ignore
    unet.eval()

    # ---- Quick viz on a few test examples (kept from your script) ----
    num_examples = min(num_examples, len(test_set))
    xs = torch.stack([test_set[i] for i in range(num_examples)], dim=0).to(
        device
    )  # (N,1,256,256)

    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))
    proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(angles, 256, num_examples)
    proj_geom_lr, vol_geom_lr = get_astra_geometry_3d(angles, 128, num_examples)
    op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)

    # Build LR FBP and predict for the small viz batch
    if sparse_data:
        fbp_lr, I0_lr, n_angles_tensor = sample_fbp_sparse(xs)
        n_angles_disp = n_angles_tensor
    else:
        fbp_lr, I0_lr = sample_fbp_dense(xs, op, proj_geom_lr, vol_geom_lr, device)
        n_angles_disp = torch.full((num_examples,), N_ANGLES, device=device)

    class_labels = n_angles_disp if sparse_model else None
    preds_lr = predict(unet, fbp_lr, I0_lr, class_labels)

    # Prepare for plotting (uniform 256×256 display)
    gt = xs.squeeze(1).clamp(0, 1).detach().cpu()
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

    # ---- Metrics at native 128×128 for the viz batch ----
    gt_lr = (
        F.interpolate(gt.unsqueeze(1), size=(128, 128), mode="area")
        .squeeze(1)
        .to(fbp_lr.device)
    )
    mse_batch = F.mse_loss(gt_lr, preds_lr)
    print(f"[Viz batch] MSE: {mse_batch:.6f}")

    fbp_metrics, pred_metrics = [], []
    for i in range(num_examples):
        m_fbp = get_metrics(gt_lr[i], fbp_lr[i], normalize_range=True, constrained=True)
        m_pred = get_metrics(
            gt_lr[i], preds_lr[i], normalize_range=True, constrained=True
        )
        fbp_metrics.append(m_fbp)
        pred_metrics.append(m_pred)

    n_angles_np = n_angles_disp.detach().float().cpu().numpy()
    I0_vals = I0_lr.detach().float().view(num_examples, -1).mean(dim=1).cpu().numpy()

    # Plot
    _, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.0))
    if num_examples == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for i in range(num_examples):
        # GT
        axes[0, i].imshow(gt[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("GT", fontsize=10)
        overlay_txt = f"{int(round(n_angles_np[i]))} angles | I0={I0_vals[i]:.3g}"
        axes[0, i].text(
            6,
            14,
            overlay_txt,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
        )

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

    # ---- Full-dataset MSEs (batched, no full-set GPU loads) ----
    print("Evaluating full datasets (batched, bs=64)...")
    # train_mse = eval_mse_over_dataset(train_subset, unet, device, sparse_model)
    val_mse = eval_mse_over_dataset(val_subset, unet, device, sparse_model)
    test_mse = eval_mse_over_dataset(test_set, unet, device, sparse_model)
    # print(f"[Train] MSE: {train_mse:.8f}")
    print(f"[Val]   MSE: {val_mse:.8f}")
    print(f"[Test]  MSE: {test_mse:.8f}")


if __name__ == "__main__":
    main()
