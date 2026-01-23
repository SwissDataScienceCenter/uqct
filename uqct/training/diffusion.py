import os
import random
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from uqct.datasets.utils import get_dataset
from uqct.training.unet import N_ANGLES, norm_intensities, sample_fbp_sparse


class UNet2DModelAux(nn.Module):
    def __init__(
        self,
        emb_dim: int = 2,
        dropout: float = 0.37,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.proj = nn.Linear(2, emb_dim)
        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        up_block_types = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
        channels = (128, 128, 256, 256, 512, 512)
        self.unet = UNet2DConditionModel(
            sample_size=128,
            in_channels=2,
            out_channels=1,
            layers_per_block=2,
            dropout=dropout,
            block_out_channels=channels,  # type: ignore
            down_block_types=down_block_types,  # type: ignore
            up_block_types=up_block_types,  # type: ignore
            cross_attention_dim=emb_dim,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        fbp: torch.Tensor,
        timestep: torch.Tensor,
        intensity_norm: torch.Tensor,
        n_angles_norm: torch.Tensor,
    ) -> torch.Tensor:
        proj = self.proj(torch.stack([intensity_norm, n_angles_norm], dim=1)).unsqueeze(
            -2
        )
        if x_t.ndim != fbp.ndim:
            fbp = fbp.unsqueeze(-3)
        x_t_fbp = torch.cat([x_t, fbp], dim=-3)
        return self.unet.forward(
            x_t_fbp, timestep, encoder_hidden_states=proj, return_dict=False
        )[0]


def save_ckpt(
    unet: UNet2DModel | UNet2DModelAux,
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
        "unet": to_save.state_dict(),  # type: ignore
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


def loss_fn_either(
    x_0: torch.Tensor,
    unet: UNet2DModel | UNet2DModelAux,
    noise_scheduler: DDPMScheduler,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if hasattr(unet, "unet"):
        fbp, intensities, n_angles = sample_fbp_sparse(x_0 * 0.5 + 0.5, device)
        x_0 = F.interpolate(x_0.to(unet.device), size=(128, 128), mode="area")
        loss = loss_fn_cond(x_0, fbp, intensities, n_angles, unet, noise_scheduler)  # type: ignore
    else:
        x_0 = F.interpolate(x_0.to(unet.device), size=(128, 128), mode="area")
        loss = loss_fn_uncond(x_0, unet, noise_scheduler)  # type: ignore
    return loss


def loss_fn_uncond(
    x_0: torch.Tensor, unet: UNet2DModel, noise_scheduler: DDPMScheduler
) -> torch.Tensor:
    x_0 = x_0.to(unet.device)
    noise = torch.randn_like(x_0).to(unet.device)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,  # type: ignore
        (x_0.shape[0],),
        device=x_0.device,
    ).long()
    x_t = noise_scheduler.add_noise(x_0, noise, timesteps)  # type: ignore
    noise_pred = unet(x_t, timestep=timesteps, return_dict=False)[0]
    return F.mse_loss(noise_pred, noise)


def loss_fn_cond(
    x_0: torch.Tensor,
    fbp: torch.Tensor,
    intensities: torch.Tensor,
    n_angles: torch.Tensor,
    unet: UNet2DModelAux,
    noise_scheduler: DDPMScheduler,
) -> torch.Tensor:
    fbp = (fbp - 0.5) * 2
    x_0 = x_0.to(unet.device)
    noise = torch.randn_like(x_0).to(unet.device)
    intensity_norm = 2 * ((norm_intensities(intensities) / 999) - 0.5)
    n_angles_norm = (n_angles - N_ANGLES / 2) / (N_ANGLES / 2)
    timestep = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,  # type: ignore
        (x_0.shape[0],),
        device=x_0.device,
    ).long()
    x_t = noise_scheduler.add_noise(x_0, noise, timestep)  # type: ignore
    noise_pred = unet.forward(
        x_t,
        fbp,
        timestep=timestep,
        intensity_norm=intensity_norm,
        n_angles_norm=n_angles_norm,
    )
    return F.mse_loss(noise_pred, noise)


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
    "--cond",
    default=False,
    type=bool,
    help="Whether to train a conditional diffusion model",
)
def main(**kwargs):
    # Seeding
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Device & perf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))
    torch.backends.cudnn.benchmark = True

    # Load dataset
    train_set, _ = get_dataset(kwargs["dataset"], True)

    # Set up directories
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")  # e.g. 2025-09-13_14-27

    root_dir = Path(os.getenv("UQCT_ROOT_DIR", "."))
    if not root_dir.exists():
        root_dir = Path(".")
    run_dir = (
        root_dir
        / "runs"
        / ("diffusion_cond" if kwargs["cond"] else "diffusion")
        / f"{ts}_{kwargs['dataset']}_{kwargs['batch_size']}_{kwargs['epochs']}_{kwargs['learning_rate']}_{kwargs['dropout']}_{kwargs['weight_decay']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: '{run_dir}'")

    # Create U-Net
    if kwargs["cond"]:
        unet = UNet2DModelAux(2, kwargs["dropout"], device)
    else:
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
    if not Path("/cluster").exists():
        if isinstance(unet, UNet2DModel):
            unet.enable_gradient_checkpointing()
        else:
            unet.unet.enable_gradient_checkpointing()
    try:
        unet: UNet2DModel | UNet2DModelAux = torch.compile(unet)  # type: ignore
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
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
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
    global_step = 0
    best_val = float("inf")

    # Run training
    for epoch in (pbar := tqdm(range(0, kwargs["epochs"]))):
        unet.train()
        for x_0 in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            optimizer.zero_grad(set_to_none=True)
            x_0 = x_0.to(device, non_blocking=True) * 2.0 - 1.0
            if device.type == "cuda":
                x_0 = x_0.to(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                loss = loss_fn_either(x_0, unet, noise_scheduler, device)

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
            for x_0 in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                x_0 = x_0.to(device, non_blocking=True) * 2.0 - 1.0
                if device.type == "cuda":
                    x_0 = x_0.to(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    loss = loss_fn_either(x_0, unet, noise_scheduler, device)
                val_losses.append(loss.item())
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
