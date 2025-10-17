from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uqct.datasets.utils import get_dataset
from uqct.metrics import get_metrics
from uqct.training.unet import (
    MAX_TOTAL_INTENSITY,
    MIN_TOTAL_INTENSITY,
    N_ANGLES,
    build_unet,
    sample_fbp_dense,
    sample_fbp_sparse,
)


class FBPUNet(nn.Module):
    def __init__(self, ckpt_path: Path, sparse: bool):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = build_unet(sparse).to(device)  # type: ignore
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["unet"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        unet.load_state_dict(sd, strict=True)
        print(
            f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}"
        )
        self.unet = unet.eval()  # inference only

    # TODO: Take experiment
    def forward(
        self,
        fbp_lr: torch.Tensor,  # (N,H,W) or (N,1,H,W) in [0,1]
        intensity_lr: torch.Tensor,  # (N,1,1)
        class_labels: torch.Tensor | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        out_device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Batched inference for large inputs.
        Always uses a DataLoader internally.
        Returns (N,H,W) in [0,1].
        """
        device = next(self.unet.parameters()).device
        if fbp_lr.ndim == 3:
            fbp_lr = fbp_lr.unsqueeze(1)  # (N,1,H,W)

        if class_labels is None:
            dataset = torch.utils.data.TensorDataset(
                fbp_lr.to("cpu"), intensity_lr.to("cpu")
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                fbp_lr.to("cpu"), intensity_lr.to("cpu"), class_labels.to("cpu")
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        preds = []
        torch.set_grad_enabled(False)
        for batch in loader:
            if class_labels is None:
                fbp_b, intensity_b = batch
                cls_b = None
            else:
                fbp_b, intensity_b, cls_b = batch

            x = fbp_b.to(device, non_blocking=True) * 2.0 - 1.0
            exposure_norm = (
                (
                    intensity_b.to(device, non_blocking=True) * N_ANGLES
                    - MIN_TOTAL_INTENSITY
                )
                / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY)
                * 999
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                y = self.unet(
                    x,
                    timestep=exposure_norm.flatten(),
                    class_labels=(cls_b.to(device) if cls_b is not None else None),
                    return_dict=False,
                )[0]

            if out_device:
                y = y.to(out_device)
            preds.append(((y + 1.0) / 2.0).clamp(0.0, 1.0).to(out_device))

        return torch.cat(preds, dim=0)


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
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    _, test_set = get_dataset(dataset, True)

    # ---- Quick viz on a few test examples (kept from your script) ----
    num_examples = min(num_examples, len(test_set))
    gt = torch.stack([test_set[i] for i in range(num_examples)], dim=0).to(
        device
    )  # (N,1,256,256)

    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))

    # Load model
    model = FBPUNet(ckpt_path, sparse_model)
    model.eval()

    # Build LR FBP and predict
    # Keep a separate tensor for DISPLAY so we don't lose values when class_labels=None
    if sparse_data:
        # (N,128,128), (N,1,1), (N,)
        fbp, intensity_lr, n_angles_tensor = sample_fbp_sparse(gt)
        n_angles_disp = n_angles_tensor  # per-sample angle counts
    else:
        fbp, intensity_lr = sample_fbp_dense(gt, angles, device)  # (N,128,128), (N,1,1)
        n_angles_disp = torch.full((num_examples,), N_ANGLES, device=device)

    # Class labels only if using sparse_model (as in training)
    class_labels = n_angles_disp if sparse_model else None

    # (N,128,128)
    preds = model(fbp, intensity_lr, class_labels=class_labels)

    # Prepare for plotting (uniform 256×256 display)
    gt_lr = F.interpolate(gt, size=(128, 128), mode="area").squeeze(1).to(fbp.device)

    fbp_metrics, pred_metrics = [], []
    for i in range(num_examples):
        m_fbp = get_metrics(gt_lr[i], fbp[i], normalize_range=True, constrained=True)
        m_pred = get_metrics(gt_lr[i], preds[i], normalize_range=True, constrained=True)
        fbp_metrics.append(m_fbp)
        pred_metrics.append(m_pred)

    # Values for overlay: angles (int) and intensity per sample (float)
    n_angles_np = n_angles_disp.detach().float().cpu().numpy()
    intensity_vals = (
        intensity_lr.detach().float().view(num_examples, -1).mean(dim=1).cpu().numpy()
    )

    # Plot: 3 rows (GT / FBP / Pred) × N columns, show PSNR & SSIM on each column
    _, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.0))
    if num_examples == 1:
        axes = np.asarray(axes).reshape(3, 1)

    # Prep for plotting
    gt_lr = gt_lr.to("cpu")
    fbp = fbp.squeeze(1).to("cpu")
    preds = preds.squeeze(1).to("cpu")

    for i in range(num_examples):
        # GT + overlay with angles and intensity
        axes[0, i].imshow(gt_lr[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("GT", fontsize=10)
        overlay_txt = (
            f"{int(round(n_angles_np[i]))} angles | I0={intensity_vals[i]:.3g}"
        )
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

        # Overlay text (top-left)
        overlay_txt = (
            f"{int(round(n_angles_np[i]))} angles | I0={intensity_vals[i]:.3g}"
        )
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
        axes[1, i].imshow(fbp[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
        axes[1, i].set_title(
            f"FBP  PSNR {fbp_metrics[i]['PSNR']:.2f}  SSIM {fbp_metrics[i]['SS']:.3f}",
            fontsize=8,
        )

        # Pred + metrics
        axes[2, i].imshow(preds[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, i].axis("off")
        axes[2, i].set_title(
            f"Pred PSNR {pred_metrics[i]['PSNR']:.2f}  SSIM {pred_metrics[i]['SS']:.3f}",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
