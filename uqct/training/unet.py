import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from uqct.ct import (AstraParallelOp3D, forward_and_fbp_2d,
                     get_astra_geometry_3d, iradon_astra, poisson, sinogram_ct)
from uqct.datasets.utils import get_dataset
from uqct.debugging import plot_img

L = 5
N_ANGLES = 200
MIN_EXPOSURE = 1e4
MAX_EXPOSURE = 1e9


def sample_exposure(
    n: int,
    low: float = MIN_EXPOSURE,
    high: float = MAX_EXPOSURE,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    u = torch.rand(n, device=device)
    return torch.exp(
        u * (torch.log(torch.as_tensor(high)) - torch.log(torch.as_tensor(low)))
        + torch.log(torch.as_tensor(low))
    )


def sample_fbp_dense(
    x: torch.Tensor,
    op: AstraParallelOp3D,
    proj_geom_lr: dict[str, Any],
    vol_geom_lr: dict[str, dict],
    device: torch.device,
):
    I_0 = sample_exposure(op.nx, device=device).view(-1, 1, 1) / N_ANGLES
    scale = L / x.shape[-1]
    radon = op.forward(x.squeeze(1))
    counts = poisson(I_0 * torch.exp(-scale * radon))  # (B, 200, 256)
    counts_lr = counts.view(counts.shape[0], counts.shape[1], 128, 2).sum(
        -1
    )  # (B, 200, 128)
    I_0_lr = I_0 * 2
    sino = sinogram_ct(counts_lr, I_0_lr, L).clip(0)
    fbp = iradon_astra(sino.transpose(1, 2), vol_geom_lr, proj_geom_lr).clip(0, 1)
    return fbp[: x.shape[0]], I_0_lr[: x.shape[0]]


def sample_fbp_sparse(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exposures = sample_exposure(len(images))
    n_angles = np.random.randint(1, N_ANGLES + 1, (len(images),))
    split_indices = np.cumsum(n_angles[:-1])
    total = int(n_angles.sum())
    angle_sets = np.split(np.random.rand(total) * 360, split_indices)
    fbp, I_0 = forward_and_fbp_2d(images, angle_sets, exposures.tolist(), l=L)
    return (
        fbp,
        I_0,
        torch.tensor(n_angles, device=images.device),
    )


def build_unet(sparse: bool = False, dropout: float = 0.37) -> UNet2DModel:
    # Same architecture as training (uqct.training.unet: main)
    channels = (128, 128, 256, 256, 512, 512)
    down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    return UNet2DModel(
        sample_size=128,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        dropout=dropout,
        block_out_channels=channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        class_embed_type="timestep" if sparse else None,
        num_class_embeds=200 if sparse else None,
    )


def save_ckpt(
    unet: UNet2DModel,
    epoch: int,
    val_loss: float,
    run_dir: Path,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    scaler: torch.GradScaler,
    best_val: float,
    global_step: int,
    epoch_ckpt: bool,
    pbar: tqdm,
) -> None:
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    if epoch_ckpt:
        pbar.write(f"Saving checkpoint: {epoch=}, {val_loss=}, {ckpt_dir=}")
    to_save = unet._orig_mod if hasattr(unet, "_orig_mod") else unet
    payload = {
        "unet": to_save.state_dict(),
        "epoch": epoch,  # next epoch will start from this value
        "val_loss": val_loss,
        "best_val": best_val,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if epoch_ckpt:
        name = f"epoch_{epoch:04}.pt"
    else:
        name = "best.pt"
    torch.save(payload, ckpt_dir / name)


def loss_fn(
    x: torch.Tensor,
    fbp: torch.Tensor,
    I_0: torch.Tensor,
    unet: UNet2DModel,
    n_angles: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x * 2.0 - 1.0
    fbp = fbp * 2.0 - 1.0
    if fbp.ndim == 3:
        fbp.unsqueeze_(1)
    if x.ndim == 3:
        x.unsqueeze_(1)

    _n_angles = n_angles if n_angles is not None else N_ANGLES
    exposure_norm = (
        (I_0 * _n_angles - MIN_EXPOSURE) / (MAX_EXPOSURE - MIN_EXPOSURE) * 999
    )  # [0, 999]

    if n_angles is None:
        pred = unet(
            fbp,
            timestep=exposure_norm.flatten(),
            return_dict=False,
        )[0]
    else:
        pred = unet(fbp, exposure_norm, class_labels=_n_angles - 1)[0]
    x_lr = F.interpolate(x, size=pred.shape[-2:], mode="area")  # 256x256 -> 128x128
    return F.mse_loss(pred, x_lr)


def load_model_ckpt(
    ckpt_path: Path,
    unet: UNet2DModel,
    device: torch.device,
) -> None:
    start_epoch = 0
    best_val = float("inf")
    global_step = 0

    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["unet"]

        target = unet._orig_mod if hasattr(unet, "_orig_mod") else unet
        target.load_state_dict(sd, strict=True)  # type: ignore

        print(
            f"Resumed from '{ckpt_path}': epoch={start_epoch}, best_val={best_val:.6f}, global_step={global_step}"
        )


@click.command()
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lamino", "lung", "composite"]),
    help="dataset name",
)
@click.option(
    "--batch-size", default=32, type=int, help="Number of examples per batch."
)
@click.option("--epochs", default=50, type=int, help="Total number of training epochs.")
@click.option(
    "--learning-rate", default=0.0001, type=float, help="Initial learning rate."
)
@click.option("--dropout", default=0.37, type=float, help="Dropout")
@click.option(
    "--weight-decay",
    default=0.0043,
    type=float,
    help="Weight decay (L2 regularization).",
)
@click.option(
    "--load-model-ckpt",
    default="",
    type=click.Path(path_type=Path),
    help="Path to a checkpoint to resume from.",
)
@click.option(
    "--seed",
    default=0,
    type=int,
    help="Random seed",
)
@click.option(
    "--sparse",
    is_flag=True,
    default=False,
    help="Train for the sparse setting (dense if omitted).",
)
def main(**kwargs):
    if kwargs["sparse"]:
        print(f"Running SPARSE training")
    else:
        print(f"Running DENSE training")

    # Seeding
    torch.random.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])

    # Device & perf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))
    torch.backends.cudnn.benchmark = True

    # Load dataset
    train_set, _ = get_dataset(kwargs["dataset"], True)

    # Create forward projector
    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))
    side_length = train_set[0].shape[-1]
    proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(
        angles, side_length, kwargs["batch_size"]
    )
    proj_geom_lr, vol_geom_lr = get_astra_geometry_3d(angles, 128, kwargs["batch_size"])
    op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)

    # Set up directories
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")  # e.g. 2025-09-13_14-27

    on_cluster = Path("/cluster").exists()
    root_dir = Path("/cluster/scratch/mgaetzner/uqct") if on_cluster else Path(".")
    sparse_or_dense = "sparse" if kwargs["sparse"] else "dense"
    run_dir = (
        root_dir
        / "runs"
        / f"unet_{sparse_or_dense}"
        / f"{ts}_{kwargs['dataset']}_{kwargs['batch_size']}_{kwargs['epochs']}_{kwargs['learning_rate']}_{kwargs['dropout']}_{kwargs['weight_decay']}_{kwargs['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: '{run_dir}'")

    # Create U-Net
    unet = build_unet(kwargs["sparse"], kwargs["dropout"])
    unet = unet.to(device)  # type: ignore
    try:
        unet: UNet2DModel = torch.compile(unet)  # type: ignore
    except Exception:
        print(f"Failed to compile U-Net")

    # Split train set into training and validation subset
    dataset_size = len(train_set)
    val_size = int(0.05 * dataset_size)
    train_size = dataset_size - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    num_workers = max(2, min(8, os.cpu_count() or 2))
    train_loader = DataLoader(
        train_subset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=kwargs["batch_size"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=max(2, num_workers // 2),
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    # Set up training
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=kwargs["learning_rate"],
        weight_decay=kwargs["weight_decay"],
        fused=(device.type == "cuda"),
    )
    num_update_steps_per_epoch = len(train_loader)
    num_train_steps = num_update_steps_per_epoch * kwargs["epochs"]
    if kwargs["load_model_ckpt"].is_file():
        num_warmup_steps = 0
    else:
        num_warmup_steps = int(0.1 * num_train_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    writer = SummaryWriter(log_dir=run_dir / "tb")

    load_model_ckpt(kwargs["load_model_ckpt"], unet, device)
    global_step = 0
    best_val = float("inf")

    # Run training
    for epoch in (pbar := tqdm(range(kwargs["epochs"]))):
        unet.train()
        for x in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            optimizer.zero_grad(set_to_none=True)
            x: torch.Tensor = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                if kwargs["sparse"]:
                    fbp, I_0, n_angles = sample_fbp_sparse(x)
                    loss = loss_fn(x, fbp, I_0, unet, n_angles)
                else:
                    fbp, I_0 = sample_fbp_dense(
                        x, op, proj_geom_lr, vol_geom_lr, device
                    )
                    loss = loss_fn(x, fbp, I_0, unet)

            if not torch.isfinite(loss):
                print("Non-finite loss, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        unet.eval()
        val_losses = []
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    if kwargs["sparse"]:
                        fbp, I_0, n_angles = sample_fbp_sparse(x)
                        vloss = loss_fn(x, fbp, I_0, unet, n_angles)
                    else:
                        fbp, I_0 = sample_fbp_dense(
                            x, op, proj_geom_lr, vol_geom_lr, device
                        )
                        vloss = loss_fn(x, fbp, I_0, unet)
                val_losses.append(vloss.item())
        mean_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        print(f"Mean val loss: {mean_val_loss}")
        writer.add_scalar("val/loss_epoch", mean_val_loss, epoch)

        if mean_val_loss < best_val:
            best_val = mean_val_loss
            save_ckpt(
                unet,
                epoch,
                mean_val_loss,
                run_dir,
                optimizer,
                lr_scheduler,
                scaler,
                best_val,
                global_step,
                False,
                pbar,
            )


if __name__ == "__main__":
    main()
