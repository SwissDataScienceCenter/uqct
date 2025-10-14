from pathlib import Path
from typing import Callable, Literal

import click
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import optim
from tqdm.auto import tqdm

from uqct.ct import Experiment, nll

DatasetName = Literal["lung", "composite", "lamino"]


class Diffusion:
    def __init__(
        self,
        dataset: DatasetName,
        experiment: Experiment,
        num_steps: int = 50,
        buffer: int = 5,
        sgd_steps: int = 50,
        lr: float = 0.1,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.experiment = experiment
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # U-Net
        ckpt_path = find_ckpt(dataset)
        self.unet = load_unet(ckpt_path)
        self.unet.eval()
        for param in self.unet.parameters():  # type: ignore
            param.requires_grad = False

        # Diffusion
        self.num_steps = num_steps
        self.buffer = buffer
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, clip_sample=False
        )

        # Guidance
        self.loss_fct = get_guidance_loss_fn(experiment)
        self.sgd_steps = sgd_steps
        self.lr = lr

    def predict_x_0(
        self, t: int, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_t.device
        timesteps = torch.LongTensor([t]).to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            noise_pred = self.unet(x_t, timesteps, return_dict=False)[0]

        x_0_pred = self.noise_scheduler.step(
            noise_pred, int(timesteps.item()), x_t.reshape(noise_pred.shape)
        ).pred_original_sample  # type: ignore
        assert isinstance(x_0_pred, torch.Tensor)

        x_t_previous = self.noise_scheduler.step(
            noise_pred, int(timesteps.item()), x_t.reshape(noise_pred.shape)
        ).prev_sample  # type: ignore

        return noise_pred, x_0_pred, x_t_previous

    def step(self, t: torch.Tensor, target_t: int, x_t: torch.Tensor):
        device = x_t.device
        noise_pred, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
        new_timestep = torch.LongTensor([target_t]).to(device)
        new_x_t = self.noise_scheduler.add_noise(
            x_0_pred, torch.randn_like(x_0_pred), new_timestep  # type: ignore
        ).to(device)
        return new_x_t, noise_pred

    @torch.inference_mode()
    def reverse(
        self,
        x_t_start: torch.Tensor,
        t_start: int,
        t_end: int,
        num_steps=50,
    ) -> torch.Tensor:
        with torch.no_grad():
            x_t = x_t_start.clone()
            timesteps = torch.linspace(t_start, t_end, num_steps + 1).int()
            for i in tqdm(range(1, len(timesteps)), disable=not self.verbose):
                t = timesteps[i - 1]
                target_t = timesteps[i]
                x_t, _ = self.step(t, int(target_t.item()), x_t)
            return x_t

    @torch.inference_mode()
    def sample(self, num_samples: int = 10) -> torch.Tensor:
        side_length = self.experiment.counts.shape[-1]
        x_t = torch.randn(
            *self.experiment.batch_dims,
            num_samples,
            side_length,
            side_length,
            device=self.device,
        )
        timesteps = torch.linspace(0, 999 + self.buffer, self.num_steps + 1).int()
        it = tqdm(range(len(timesteps)), disable=not self.verbose)
        for i in it:
            t = timesteps[i - 1]
            target_t = timesteps[i]
            new_timestep = torch.LongTensor([target_t]).to(self.device)
            noise_scheduler.previous_timestep = lambda _: target_t  # type: ignore

            _, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
            x_t = x_0_pred

            x_t, guidance_loss = guide(
                x_t, self.loss_fct, sgd_steps=self.sgd_steps, lr=self.lr, verbose=False
            )
            it.set_postfix({"loss": f"{guidance_loss:.3f}"})

            x_t = self.noise_scheduler.add_noise(
                x_t,
                torch.randn_like(x_t),
                new_timestep,  # type: ignore
            )

        x_t = self.reverse(x_t, self.buffer, 0, num_steps=self.buffer)
        _, x_t, _ = self.predict_x_0(0, x_t)
        return denorm_image(x_t)


def denorm_image(image: torch.Tensor) -> torch.Tensor:
    return ((image + 1) / 2).clip(0, 1)


def find_ckpt(dataset: DatasetName) -> Path:
    ckpt_dir_candidates = [
        Path(x) / f"ddpm_unconditional_128_{dataset}.pt"
        for x in (
            "/mydata/chip/shared/checkpoints/diffusion",
            "checkpoints/diffusion",
            "../checkpoints/diffusion",
        )
    ]
    ckpt_path = ckpt_dir_candidates[0]
    for ckpt_path in ckpt_dir_candidates[1:]:
        if ckpt_path.exists():
            break
    if not ckpt_path.exists():
        raise ValueError(f"Could not find diffusion checkpoint for dataset {dataset}")
    return ckpt_path


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


def get_guidance_loss_fn(
    experiment: Experiment,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a loss function that takes as input a batch of images and returns the Poisson NLL loss."""
    if experiment.sparse:

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            breakpoint()
            loss = nll(
                image, experiment.counts, experiment.intensities, experiment.angles
            )
            return loss

    else:

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            return nll(
                image, experiment.counts, experiment.intensities, experiment.angles
            ).mean()

    return loss_fn


def guide(
    x_t: torch.Tensor,
    loss_fct: Callable[..., torch.Tensor],
    sgd_steps: int | list[int] = 50,
    lr: float = 0.1,
    verbose: bool = False,
) -> tuple[torch.Tensor, float]:
    lr_list, sgd_steps_list = lr, sgd_steps
    if not isinstance(lr, list):
        lr_list = [lr]
        sgd_steps_list = [sgd_steps]

    circle_mask = torch.ones(*x_t.shape[-2:], device=x_t.device)
    radius = x_t.shape[-1] // 2
    y, x = torch.meshgrid(
        torch.arange(x_t.shape[-2], device=x_t.device),
        torch.arange(x_t.shape[-1], device=x_t.device),
        indexing="ij",
    )
    mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    circle_mask[~mask] = 0
    loss = torch.tensor([float("inf")])
    x_t = torch.nn.Parameter(x_t)

    for lr_instance, sgd_steps_instance in zip(lr_list, sgd_steps_list):  # type: ignore
        optimizer = optim.Adam([x_t], lr=lr_instance)
        it = tqdm(range(sgd_steps_instance), disable=not verbose)
        for _ in it:
            loss = loss_fct(denorm_image(x_t))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_t.data[..., ~mask] = -1
            it.set_postfix({"loss": f"{loss.item():.3f}"})
    return x_t, loss.item()


@click.command()
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lung", "composite", "lamino"]),
    help="Which dataset to generate samples for",
)
@click.option(
    "--sparse",
    default=False,
    type=bool,
    help="Whether to generate samples for the sparse setting",
)
def main(dataset: DatasetName, sparse: bool):
    import numpy as np

    from uqct.ct import sample_observations
    from uqct.datasets.utils import get_dataset
    from uqct.debugging import plot_img

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    _, test_set = get_dataset(dataset, True)

    # ---- Quick viz on a few test examples (kept from your script) ----
    num_gt = min(5, len(test_set))
    gt = torch.stack([test_set[i] for i in range(num_gt)], dim=0).to(
        device
    )  # (N,1,256,256)

    angles = torch.from_numpy(np.linspace(0, 180, 200, endpoint=False))
    if sparse:
        raise ValueError()
    else:
        intensities = torch.logspace(1e5, 1e8, 10)
        counts = sample_observations(gt, intensities, angles)

    experiment = Experiment(counts, intensities, angles, False)
    diffusion = Diffusion(dataset, experiment, sgd_steps=0)
    sample = diffusion.sample()
    plot_img(sample)


if __name__ == "__main__":
    main()
