import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import astra
import click
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from uqct.datasets.utils import (KWARGS_COMPOSITE, KWARGS_LAMINO, KWARGS_LUNG,
                                 get_dataset)


def generate_random_fbp_astra(
    imgs: torch.Tensor,  # [B,1,H,W] or [B,H,W], float in [0,1]
    angles_choices: Sequence[int] = (20, 200),  # uniform choice
    exposure_min: float = 1e4,  # uniform in [min, max]
    exposure_max: float = 1e8,
    filter_name: str = "ram-lak",
    log_eps: float = 1.0,  # -log((k+log_eps)/I0)
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        recons:   [B,1,H,W] torch tensor (same dtype/device as imgs)
        angles:   [B] torch.long
        exposures:[B] torch.float
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ---- validate input ----
    assert imgs.ndim in (3, 4), "imgs must be [B,H,W] or [B,1,H,W]"
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)
    B, C, H, W = imgs.shape
    assert C == 1, "single-channel expected"
    assert H == W, "square images only"
    assert imgs.dtype in (torch.float32, torch.float64), "imgs must be float"
    assert (
        imgs.min().item() >= 0.0 and imgs.max().item() <= 1.0
    ), "imgs should be in [0,1]"

    in_device = imgs.device
    imgs_np = imgs.detach().cpu().numpy().astype(np.float32, copy=False)

    # ---- sample angles & exposures (uniform) ----
    angles_choices = np.array(list(angles_choices), dtype=np.int32)  # type: ignore
    angles = np.random.choice(angles_choices, size=B).astype(np.int32)
    exposures = np.random.uniform(exposure_min, exposure_max, size=B).astype(np.float64)

    # ---- helper: detector width to cover the square FOV ----
    def _detector_count_for_parallel(N: int) -> int:
        return int(math.ceil(np.sqrt(2) * N))

    det_count = _detector_count_for_parallel(H)
    breakpoint()
    vol_geom = astra.create_vol_geom(H, W)

    # ---- cache per-angle ASTRA objects (no projector needed) ----
    cache = {}
    for A in np.unique(angles):
        theta = np.linspace(0.0, math.pi, num=int(A), endpoint=False).astype(np.float32)
        proj_geom = astra.create_proj_geom("parallel", 1.0, det_count, theta)

        vol_id = astra.data2d.create("-vol", vol_geom, 0)
        sino_id = astra.data2d.create("-sino", proj_geom, 0)
        rec_id = astra.data2d.create("-vol", vol_geom, 0)

        # FP_CUDA without ProjectorId: pass geometries directly
        cfg_fp = astra.astra_dict("FP_CUDA")
        cfg_fp["VolumeDataId"] = vol_id
        cfg_fp["ProjectionDataId"] = sino_id
        alg_fp = astra.algorithm.create(cfg_fp)

        # FBP_CUDA without ProjectorId
        cfg_fbp = astra.astra_dict("FBP_CUDA")
        cfg_fbp["ProjectionDataId"] = sino_id
        cfg_fbp["ReconstructionDataId"] = rec_id
        cfg_fbp["FilterType"] = filter_name
        alg_fbp = astra.algorithm.create(cfg_fbp)

        cache[int(A)] = dict(
            vol_id=vol_id,
            sino_id=sino_id,
            rec_id=rec_id,
            alg_fp=alg_fp,
            alg_fbp=alg_fbp,
        )

    # ---- process each image (reuse cached ids) ----
    recons_np = np.empty((B, 1, H, W), dtype=np.float32)
    for i in range(B):
        A = int(angles[i])
        E = float(exposures[i])
        I0 = max(1.0, E / max(1, A))  # photons per ray at zero attenuation

        ids = cache[A]

        # forward projection (GPU)
        astra.data2d.store(ids["vol_id"], imgs_np[i, 0])  # [H,W]
        astra.algorithm.run(ids["alg_fp"])
        p = astra.data2d.get(ids["sino_id"]).astype(np.float32, copy=False)  # [D, A]

        # Poisson noise on host
        I_mean = I0 * np.exp(-p / H, dtype=np.float64)  # rate as float64
        k = np.random.poisson(I_mean).astype(np.float32, copy=False)
        p_hat = -np.log((k + log_eps) / I0, dtype=np.float32)

        # FBP (GPU)
        astra.data2d.store(ids["sino_id"], p_hat)
        astra.algorithm.run(ids["alg_fbp"])
        rec = astra.data2d.get(ids["rec_id"]).astype(np.float32, copy=False)  # [H,W]
        recons_np[i, 0] = rec

    # ---- cleanup ----
    for ids in cache.values():
        astra.algorithm.delete(ids["alg_fp"])
        astra.algorithm.delete(ids["alg_fbp"])
        astra.data2d.delete(ids["vol_id"])
        astra.data2d.delete(ids["sino_id"])
        astra.data2d.delete(ids["rec_id"])

    # ---- back to torch ----
    recons = torch.from_numpy(recons_np).to(dtype=imgs.dtype, device=in_device)
    angles_t = torch.from_numpy(angles.astype(np.int64)).to(device=in_device)
    exposure_t = torch.from_numpy(exposures.astype(np.float32)).to(device=in_device)
    return recons, angles_t, exposure_t


def save_ckpt(
    unet: UNet2DConditionModel,
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
    x_0: torch.Tensor,
    fbp: torch.Tensor,
    angles: torch.Tensor,  # shape [B], integer in [1, 200]
    exposure: torch.Tensor,  # shape [B], float in [1e4, 1e8]
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
) -> torch.Tensor:
    device = unet.device
    x_0 = x_0.to(device)

    # Forward diffusion
    noise = torch.randn_like(x_0).to(unet.device)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,  # type: ignore
        (x_0.shape[0],),
        device=x_0.device,
    ).long()
    x_t = noise_scheduler.add_noise(x_0, noise, timesteps)  # type: ignore

    # Prep U-Net inputs
    fbp = fbp.to(device)
    x_t_cond = torch.cat([x_t, fbp])

    angles = angles.to(device).float()
    angles_norm = (((angles - 1.0) * 2) - 199) / 199
    exposure = exposure.to(device)
    exposure_norm = (torch.log(exposure) - math.log(1e4)) / math.log(1e4)
    class_labels = torch.stack([angles_norm, exposure_norm], dim=-1)

    # Predict noise
    noise_pred = unet(
        x_t_cond,
        timestep=timesteps,
        class_labels=class_labels,
        return_dict=False,
    )[0]

    return F.mse_loss(noise_pred, noise)


def maybe_resume(
    ckpt_path: Path,
    unet: UNet2DConditionModel,
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
    settings = {
        "composite": {"kwargs": KWARGS_COMPOSITE, "filetype": "nii"},
        "lamino": {"kwargs": KWARGS_LAMINO, "filetype": "tiff"},
        "lung": {"kwargs": KWARGS_LUNG, "filetype": "h5"},
    }
    train_set, test_set = get_dataset(
        settings[kwargs["dataset"]]["kwargs"], settings[kwargs["dataset"]]["filetype"]
    )

    # Set up directories
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")  # e.g. 2025-09-13_14-27

    on_cluster = Path("/cluster").exists()
    root_dir = Path("/cluster/scratch/mgaetzner/uqct") if on_cluster else Path(".")
    run_dir = (
        root_dir
        / "runs"
        / "diffusion"
        / f"{ts}_{kwargs['dataset']}_{kwargs['batch_size']}_{kwargs['epochs']}_{kwargs['learning_rate']}_{kwargs['dropout']}_{kwargs['weight_decay']}"
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
    unet = UNet2DConditionModel(
        sample_size=128,
        in_channels=2,  # x_t + FBP
        out_channels=1,
        layers_per_block=2,
        dropout=kwargs["dropout"],
        block_out_channels=channels,  # type: ignore
        down_block_types=down_block_types,  # type: ignore
        up_block_types=up_block_types,  # type: ignore
        class_embed_type="projection",
        projection_class_embeddings_input_dim=2,
    )
    unet = unet.to(device)  # type: ignore
    if not on_cluster:
        unet.enable_gradient_checkpointing()
    try:
        unet: UNet2DConditionModel = torch.compile(unet)  # type: ignore
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

    start_epoch, best_val, global_step = maybe_resume(
        kwargs["load_model_ckpt"], unet, optimizer, lr_scheduler, scaler, device
    )

    # Run training
    for epoch in (pbar := tqdm(range(start_epoch, kwargs["epochs"]))):
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
                loss = loss_fn(x_0, unet, noise_scheduler)

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
                    vloss = loss_fn(x_0, unet, noise_scheduler)
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
