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

from uqct.ct import fbp, forward_and_fbp_2d, sample_observations, sinogram_from_counts
from uqct.datasets.utils import get_dataset

try:
    import lovely_tensors as lt

    lt.monkey_patch()
except Exception as _:
    pass

L = 5
N_ANGLES = 200
ANGULAR_RANGE = 180
N_BINS_HR = 256
N_BINS_LR = 128
MIN_TOTAL_INTENSITY = 1e4
MAX_TOTAL_INTENSITY = 1e9


def sample_intensities(
    n: int,
    low: float = MIN_TOTAL_INTENSITY,
    high: float = MAX_TOTAL_INTENSITY,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    u = torch.rand(n, device=device)
    return torch.exp(
        u * (torch.log(torch.as_tensor(high)) - torch.log(torch.as_tensor(low)))
        + torch.log(torch.as_tensor(low))
    )


def sample_fbp_dense(
    x: torch.Tensor,
    angles: torch.Tensor,
    device: torch.device,
):
    total_intensities = sample_intensities(x.shape[0], device=device)
    intensities = total_intensities / N_ANGLES / N_BINS_HR
    intensities = intensities.reshape(-1, 1, 1, 1).expand(-1, -1, len(angles), -1)
    counts_lr = sample_observations(x, intensities, angles)
    intensities_lr = intensities * 2
    sino = sinogram_from_counts(counts_lr, intensities_lr, L).clip(0)
    out = fbp(sino, angles).clip(0, 1).to(device)
    return out, total_intensities


def sample_fbp_sparse(
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_intensities = sample_intensities(len(images))
    n_angles = np.random.randint(1, N_ANGLES + 1, (len(images),))
    split_indices = np.cumsum(n_angles[:-1])
    total = int(n_angles.sum())
    angle_sets = np.split(np.random.rand(total) * ANGULAR_RANGE, split_indices)
    fbp = forward_and_fbp_2d(images, angle_sets, total_intensities.tolist(), l=L)
    return (
        fbp.to(device),
        total_intensities.to(device),
        torch.tensor(n_angles, device=device),
    )


def build_unet(sparse: bool = False, dropout: float = 0.37) -> UNet2DModel:
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
    fbps: torch.Tensor,
    total_intensities: torch.Tensor,
    unet: UNet2DModel,
    n_angles: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x * 2.0 - 1.0
    fbps = fbps * 2.0 - 1.0
    if fbps.ndim == 3:
        fbps.unsqueeze_(1)
    if x.ndim == 3:
        x.unsqueeze_(1)

    _n_angles = n_angles if n_angles is not None else N_ANGLES
    total_intensities_norm = (
        (total_intensities - MIN_TOTAL_INTENSITY)
        / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY)
        * 999
    )

    if n_angles is None:
        pred = unet(
            fbps,
            timestep=total_intensities_norm.flatten(),
            return_dict=False,
        )[0]
    else:
        pred = unet(fbps, total_intensities_norm, class_labels=_n_angles - 1)[0]
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
        print("Running SPARSE training")
    else:
        print("Running DENSE training")

    # Device & perf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))
    torch.backends.cudnn.benchmark = True

    # Load dataset
    train_set, _ = get_dataset(kwargs["dataset"], True)

    # Seeding
    torch.random.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])

    # Create forward projector
    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False))

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
    # try:
    #     unet: UNet2DModel = torch.compile(unet)  # type: ignore
    # except Exception:
    #     print("Failed to compile U-Net")

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
                    fbp, intensities, n_angles = sample_fbp_sparse(x, device)
                    loss = loss_fn(x, fbp, intensities, unet, n_angles)
                else:
                    fbp, intensities = sample_fbp_dense(x, angles, device)
                    loss = loss_fn(x, fbp, intensities, unet)

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
                        fbp, intensities, n_angles = sample_fbp_sparse(x)
                        vloss = loss_fn(x, fbp, intensities, unet, n_angles)
                    else:
                        fbp, intensities = sample_fbp_dense(x, angles, device)
                        vloss = loss_fn(x, fbp, intensities, unet)
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
