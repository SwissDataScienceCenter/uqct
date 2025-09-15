import os
import random
from datetime import datetime
from pathlib import Path

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

from uqct.ct import (AstraParallelOp3D, get_astra_geometry_3d, iradon_astra,
                     sinogram_ct)
from uqct.datasets.utils import (KWARGS_COMPOSITE, KWARGS_LAMINO, KWARGS_LUNG,
                                 get_dataset)
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


def sample_fbp(x: torch.Tensor, op: AstraParallelOp3D, device: torch.device):
    I_0 = sample_exposure(op.nx, device=device).view(-1, 1, 1) / N_ANGLES
    scale = L / x.shape[-1]
    radon = op.forward(x.squeeze(1))
    counts = torch.poisson(I_0 * torch.exp(-scale * radon))
    sino = sinogram_ct(counts, I_0, L).clip(0)
    fbp = iradon_astra(sino.transpose(1, 2), op.vol_geom, op.proj_geom).clip(0, 1)
    return fbp[: x.shape[0]], I_0[: x.shape[0]]


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
    x: torch.Tensor, fbp: torch.Tensor, I_0: torch.Tensor, unet: UNet2DModel
) -> torch.Tensor:
    x = x.to(unet.device)
    x = x * 2.0 - 1.0
    fbp = fbp * 2.0 - 1.0
    if len(fbp.shape) == 3:
        fbp = fbp.unsqueeze(1)
    exposure_norm = (
        (I_0 * N_ANGLES - MIN_EXPOSURE) / (MAX_EXPOSURE - MIN_EXPOSURE) * 999
    )  # [0, 999]
    pred = unet(fbp, timestep=exposure_norm.flatten(), return_dict=False)[0]
    return F.mse_loss(pred, fbp)


def maybe_resume(
    ckpt_path: Path,
    unet: UNet2DModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler: torch.GradScaler,
    device: torch.device,
):
    start_epoch = 0
    best_val = float("inf")
    global_step = 0

    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["unet"]

        target = unet._orig_mod if hasattr(unet, "_orig_mod") else unet
        target.load_state_dict(sd, strict=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt and isinstance(scaler, torch.GradScaler):
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        global_step = int(
            ckpt.get(
                "global_step", start_epoch * len(lr_scheduler.optimizer.param_groups)
            )
        )  # fallback

        print(
            f"Resumed from '{ckpt_path}': epoch={start_epoch}, best_val={best_val:.6f}, global_step={global_step}"
        )

    return start_epoch, best_val, global_step


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
    "--learning-rate", default=0.0025, type=float, help="Initial learning rate."
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
def main(**kwargs):
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
    settings = {
        "composite": {"kwargs": KWARGS_COMPOSITE, "filetype": "nii"},
        "lamino": {"kwargs": KWARGS_LAMINO, "filetype": "tiff"},
        "lung": {"kwargs": KWARGS_LUNG, "filetype": "h5"},
    }
    train_set, _ = get_dataset(
        settings[kwargs["dataset"]]["kwargs"], settings[kwargs["dataset"]]["filetype"]
    )

    # Create forward projector
    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))
    side_length = train_set[0].shape[-1]
    proj_geom, vol_geom = get_astra_geometry_3d(
        angles, side_length, kwargs["batch_size"]
    )
    op = AstraParallelOp3D(proj_geom, vol_geom)

    # Set up directories
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")  # e.g. 2025-09-13_14-27

    on_cluster = Path("/cluster").exists()
    root_dir = Path("/cluster/scratch/mgaetzner/uqct") if on_cluster else Path(".")
    run_dir = (
        root_dir
        / "runs"
        / "unet"
        / f"{ts}_{kwargs['dataset']}_{kwargs['batch_size']}_{kwargs['epochs']}_{kwargs['learning_rate']}_{kwargs['dropout']}_{kwargs['weight_decay']}_{kwargs['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: '{run_dir}'")

    # Create U-Net
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
    unet = UNet2DModel(
        sample_size=128,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        dropout=kwargs["dropout"],
        block_out_channels=channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    unet = unet.to(device)  # type: ignore
    # if not on_cluster:
    #     unet.enable_gradient_checkpointing()
    try:
        unet: UNet2DModel = torch.compile(unet)  # type: ignore
    except Exception:
        pass

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
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )
    writer = SummaryWriter(log_dir=run_dir / "tb")

    start_epoch, best_val, global_step = maybe_resume(
        kwargs["load_model_ckpt"], unet, optimizer, lr_scheduler, scaler, device
    )

    # Run training
    for epoch in (pbar := tqdm(range(start_epoch, kwargs["epochs"]))):
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
                fbp, I_0 = sample_fbp(x, op, device)
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
                x = x.to(device, non_blocking=True) * 2.0 - 1.0
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    fbp, I_0 = sample_fbp(x, op, device)
                    vloss = loss_fn(x, fbp, I_0, unet)
                val_losses.append(vloss.item())
        mean_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
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
