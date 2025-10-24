from pathlib import Path
from typing import Callable, Literal

import click
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import optim
from tqdm.auto import tqdm

from uqct.ct import (
    Experiment,
    fbp,
    fbp_2d,
    nll,
    norm_l1,
    pearson_chi_square,
    sinogram_from_counts,
)
from uqct.debugging import plot_img

DatasetName = Literal["lung", "composite", "lamino"]


class Diffusion:
    def __init__(
        self,
        dataset: DatasetName,
        num_steps: int = 50,
        buffer: int = 20,
        sgd_steps: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # U-Net
        self.batch_size = batch_size
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
        self.sgd_steps = sgd_steps
        self.lr = lr

    def predict_x_0(
        self, t: int, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_t.device
        timesteps = torch.LongTensor([t]).to(device)

        noise_preds, x0_preds, x_t_prevs = [], [], []
        in_shape = x_t.shape
        x_t_flat = x_t.view(-1, 1, x_t.shape[-1], x_t.shape[-1])

        # Split into batches of size <= self.batch_size
        for batch in x_t_flat.split(self.batch_size):
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    noise_pred = self.unet(
                        batch,
                        timesteps,
                        return_dict=False,
                    )[0]

            step_result = self.noise_scheduler.step(
                noise_pred, int(timesteps.item()), batch.reshape(noise_pred.shape)
            )
            x_0_pred = step_result.pred_original_sample  # type: ignore
            x_t_previous = step_result.prev_sample  # type: ignore

            noise_preds.append(noise_pred)
            x0_preds.append(x_0_pred)
            x_t_prevs.append(x_t_previous)

        # Concatenate all batch results
        noise_pred = torch.cat(noise_preds, dim=0).view(in_shape)
        x_0_pred = torch.cat(x0_preds, dim=0).view(in_shape)
        x_t_previous = torch.cat(x_t_prevs, dim=0).view(in_shape)

        return noise_pred, x_0_pred, x_t_previous

    def step(self, t: torch.Tensor, target_t: int, x_t: torch.Tensor):
        device = x_t.device
        noise_pred, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
        new_timestep = torch.LongTensor([target_t]).to(device)
        new_x_t = self.noise_scheduler.add_noise(
            x_0_pred,
            torch.randn_like(x_0_pred),
            new_timestep,  # type: ignore
        ).to(device)
        return new_x_t, noise_pred

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

    def sample(
        self,
        experiment: Experiment,
        replicates: int = 10,
        guidance_loss_fn: Callable | None = None,
        l2_ball_radius: float = float("inf"),
    ) -> torch.Tensor:
        """
        Returns
            torch.Tensor: `(..., n_angles, replicates, 1, side_length, side_length)` (sparse) or `(..., n_rounds, replicates, 1, side_length, side_length)` (dense).
        """
        # TODO: Try restricting guidance output to a l2-ball around the current mean.

        side_length = experiment.counts.shape[-1]
        if experiment.sparse:
            # (rep, ..., n_angles, 1, side_length, side_length)
            x_t = torch.randn(
                replicates,
                *experiment.batch_dims,
                experiment.counts.shape[-2],
                side_length,
                side_length,
                device=self.device,
            )
        else:
            x_t = torch.randn(
                replicates,
                *experiment.batch_dims,
                experiment.counts.shape[-3],
                side_length,
                side_length,
                device=self.device,
            )
        rep_first_shape = x_t.shape

        timesteps = torch.linspace(0, 500, self.num_steps + 1).int()
        it = tqdm(range(self.num_steps), disable=not self.verbose)
        optimizer = None

        for i in it:
            t = timesteps[-i - 1]
            target_t = timesteps[-i - 2]
            new_timestep = torch.LongTensor([target_t]).to(self.device)
            self.noise_scheduler.previous_timestep = lambda _: target_t  # type: ignore

            _, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
            x_t = x_0_pred.view(rep_first_shape)

            if guidance_loss_fn is not None:
                lr = self.lr
                if self.num_steps - i < 20:
                    lr = self.lr * (self.num_steps - i) / 25
                x_t, guidance_loss, optimizer = guide(
                    x_t,
                    guidance_loss_fn,
                    sgd_steps=self.sgd_steps,
                    lr=lr,
                    optimizer=optimizer,
                    l2_ball_radius=l2_ball_radius,
                    verbose=False,
                )
                it.set_postfix({"loss": f"{guidance_loss:.3f}"})

            x_t = self.noise_scheduler.add_noise(
                x_t,
                torch.randn_like(x_t),
                new_timestep,  # type: ignore
            )

        x_t = self.reverse(x_t, self.buffer, 0, num_steps=self.buffer)
        _, x_t, _ = self.predict_x_0(0, x_t)
        out = denorm_image(x_t).reshape(rep_first_shape)

        # Massage from
        #    (replicates, ..., n_angles or n_rounds, side_length, side_length)
        # to (..., n_angles or n_rounds, replicates, 1, side_length, side_length),
        n_batch_dims = len(experiment.batch_dims)
        out_perm = (
            *tuple(range(1, n_batch_dims + 1)),
            n_batch_dims + 1,
            0,
            x_t.ndim - 2,
            x_t.ndim - 1,
        )
        return out.permute(out_perm).unsqueeze(-3)


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
        n_angles = experiment.counts.shape[-2]
        mask = torch.tril(
            torch.ones(
                n_angles, n_angles, dtype=torch.bool, device=experiment.counts.device
            )
        )

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            squares = nll(
                image,
                experiment.counts.unsqueeze(-3),
                experiment.intensities.unsqueeze(-3),
                experiment.angles,
            )
            return squares[..., mask, :].mean()

    else:
        counts_csum = experiment.counts.cumsum(-3).unsqueeze(0)
        intensities_csum = experiment.intensities.cumsum(-3).unsqueeze(0)

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            return nll(image, counts_csum, intensities_csum, experiment.angles).mean()

    return loss_fn


def guide(
    x_t: torch.Tensor,
    loss_fct: Callable[..., torch.Tensor],
    sgd_steps: int = 50,
    lr: float = 0.1,
    optimizer: None | optim.Adam = None,
    l2_ball_radius: float = float("inf"),
    verbose: bool = False,
) -> tuple[torch.Tensor, float, optim.Adam]:
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

    y_0 = denorm_image(x_t)

    if optimizer is None:
        y = torch.nn.Parameter(y_0)
        optimizer = optim.Adam([y], lr=lr)
    else:
        y = optimizer.param_groups[0]["params"][0]
        with torch.no_grad():
            y.copy_(y_0)
    it = tqdm(range(sgd_steps), disable=not verbose)
    for _ in it:
        optimizer.zero_grad()
        yp = y * mask
        loss = loss_fct(yp.clip(0))

        # Punish out of range
        low = yp[yp < 0]
        if len(low) > 0:
            loss += low.abs().mean()
        high = yp[yp > 1]
        if len(high) > 0:
            loss += (high - 1).abs().mean()

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y.data = project_to_l2_ball(y, y_0, l2_ball_radius)
            y.data[..., ~mask] = 0.0

        it.set_postfix({"loss": f"{loss.item():.3f}"})
    optimizer.zero_grad(set_to_none=True)
    x_t_guided = norm_image(y).clip(-1, 1)
    return x_t_guided, loss.item(), optimizer


def norm_image(image: torch.Tensor) -> torch.Tensor:
    return (image - 0.5) * 2


def denorm_image(
    image: torch.Tensor, min_v: float = 0.0, max_v: float = 1.0
) -> torch.Tensor:
    return ((image + 1) / 2).clip(min_v, max_v)


def project_to_l2_ball(
    y: torch.Tensor,
    y_0: torch.Tensor,
    l2_ball_radius: float | torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Project y onto the L2 ball of radius l2_ball_radius centered at y_0,
    where each sample is a (r, r) block in the last three dimensions.

    Shapes:
        y, y_0: (..., r, r)  (same shape, broadcast okay if compatible)
        l2_ball_radius: float or tensor broadcastable to (..., 1, 1)

    Returns:
        y_proj with the same shape as y.
    """
    diff = y - y_0
    norms = torch.linalg.vector_norm(diff, ord=2, dim=(-2, -1), keepdims=True)
    scale = torch.clamp(l2_ball_radius / (norms + eps), max=1.0)
    y_proj = y_0 + diff * scale
    return y_proj


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
    import lovely_tensors as lt
    import numpy as np

    from uqct.ct import sample_observations
    from uqct.datasets.utils import get_dataset

    lt.monkey_patch()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _, test_set = get_dataset(dataset, True)

    n_gt = min(2, len(test_set))
    gt = torch.stack([test_set[i] for i in range(n_gt)], dim=0).to(device)

    n_angles = 50
    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)
    total_intensity = 1e5
    n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)
    if sparse:
        intensities = intensities.view(1, 1, 1).expand(-1, n_angles, -1) / (
            n_angles * n_detectors_hr
        )
    else:
        n_rounds = 5
        intensities = intensities.view(1, 1, 1, 1).expand(
            -1, n_rounds, n_angles, -1
        ) / (n_angles * n_detectors_hr * n_rounds)
    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts, intensities_lr, angles, sparse)
    diffusion = Diffusion(
        dataset,
        num_steps=50,
        buffer=10,
        sgd_steps=100,
        lr=0.001,
        verbose=True,
    )

    guidance_loss_fn = get_guidance_loss_fn(experiment)
    sample = diffusion.sample(experiment, 3, guidance_loss_fn, 256 * 0.1)
    print(sample)
    plot_img(*gt, *sample.reshape(-1, 128, 128), share_range=True)


if __name__ == "__main__":
    main()
