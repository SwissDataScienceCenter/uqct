from pathlib import Path
from typing import Optional

import click
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm


def generate_samples(
    unet: UNet2DModel,
    num_samples: int,
    noise_scheduler: DDPMScheduler,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    unet.eval()
    channels = unet.config["in_channels"]
    size = unet.config["sample_size"]
    sample = torch.randn((num_samples, channels, size, size), device=unet.device)

    if n_steps is None:
        timesteps = list(reversed(range(noise_scheduler.config["num_train_timesteps"])))
    else:
        timesteps = (
            torch.linspace(
                0, noise_scheduler.config["num_train_timesteps"] - 1, n_steps
            )
            .flip(0)
            .int()
            .tolist()
        )
        noise_scheduler.set_timesteps(timesteps=timesteps)

    for t in tqdm(timesteps, total=len(timesteps), desc="denoising"):
        t_ten = torch.full((num_samples,), t, device=unet.device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = unet(sample, t_ten, return_dict=False)[0]
        step = noise_scheduler.step(noise_pred, t, sample)
        sample = step.prev_sample  # type: ignore

    return sample.clip(0, 1)


def load_unet(ckpt_path: Path) -> UNet2DModel:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        block_out_channels=channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["unet"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    unet.load_state_dict(sd, strict=True)
    unet = unet.to(device)  # type: ignore
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']}")
    return unet


@click.command()
@click.option("--ckpt-path", type=click.Path(path_type=Path), help="Path to checkpoint")
def main(**kwargs):
    unet = load_unet(kwargs["ckpt_path"])
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    samples = generate_samples(unet, 4, scheduler, n_steps=1000).cpu().numpy()
    samples = ((samples + 1.0) / 2).clip(0.0, 1.0)
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(2, 2)

    for i in range(2):
        for j in range(2):
            image_ij = samples[i * 2 + j].reshape(128, 128)
            axes[i, j].imshow(image_ij, cmap="grey")
    plt.show()


if __name__ == "__main__":
    main()
