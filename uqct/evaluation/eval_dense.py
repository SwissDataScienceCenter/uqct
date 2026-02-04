import argparse
import collections
import datetime
import glob
import hashlib
import json
import math
import os
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import xarray as xr
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from uqct.ct import (
    Tomogram,
    circular_mask,
    fbp,
    nll,
    sample_observations,
    sinogram_from_counts,
)
from uqct.datasets.utils import get_dataset
from uqct.metrics import psnr, rmse, ssim
from uqct.models.diffusion import load_unet as load_diffusion_unet
from uqct.models.guided_diffusion import GradientGuidance, GuidedDiffusionPipeline
from uqct.training.unet import (
    MAX_TOTAL_INTENSITY,
    MIN_TOTAL_INTENSITY,
    N_ANGLES,
    N_BINS_HR,
    build_unet,
    norm_intensities,
)
from uqct.uq import (
    basic_ci,
    coverage,
    error_correlation,
    error_r2,
    gaussian_ci,
    gaussian_conservative_ci,
    percentile_ci,
    simultaneous_ci,
    simultaneous_coverage,
    sparsification_error,
)

try:
    import git
except ImportError:
    git = None


class ObservationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        seeds: list[int],
        intensities: torch.Tensor,
        angles: torch.tensor,
    ):
        self.dataset = dataset
        self.seeds = seeds
        self.intensities = intensities
        self.angles = angles
        self.device = intensities.device
        self._dataset_cache: dict[int, Any] = dict()

    def __len__(self):
        return len(cast(Sized, self.dataset)) * len(self.seeds)

    def __getitem__(self, item):
        seed = self.seeds[item // len(cast(Sized, self.dataset))]
        idx = item % len(cast(Sized, self.dataset))
        return self.get(idx, seed)

    def get(self, idx: int, seed: int):
        if idx not in self._dataset_cache:
            self._dataset_cache[idx] = self.dataset[idx]
        image = self._dataset_cache[idx].to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(
            seed + 100000 * int(idx)
        )
        # TODO: divide by 2 is important!!!
        # It seems how it currently works
        # - we sample at half the target intensity
        # - pixels are binned from 256 to 128, but this does not change total intensity
        # - the model computes total intensity by assuming 128 dectector pixels, but intensity per observation is normalized by 256 bins, this is where the factor 2 comes from.
        # TLDR: The model effectively assumes a lower total intensity by a factor of 2, so we sample the observations by dividing total intensity by the same factor.
        # seems like not the right place to add this factor, because the effect intensity is always off by a factor 2 ...
        data = sample_observations(
            image, self.intensities / 2, self.angles, generator=generator
        )
        return idx, seed, image, data


def fbp_recon(counts, intensities, angles):
    """Simple FBP reconstruction from an Experiment object."""
    sinogram = sinogram_from_counts(counts, intensities).clamp_min(0.0)
    return fbp(sinogram, angles).clamp(0.0, 1.0)


def tv_loss(image: torch.Tensor) -> torch.Tensor:
    """Compute Total Variation (TV) prior."""
    diff_h = image[..., 1:, :] - image[..., :-1, :]
    diff_w = image[..., :, 1:] - image[..., :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


class IterativeRecon:
    def __init__(
        self,
        steps: int = 200,
        init_method: str = "fbp",
        use_sigmoid: bool = False,
        lr: float = 1e-2,
        prior: torch.Tensor | None = None,
        loss: str = "nll",
        tv_weight: float = 0.0,  # added TV weight
        device: torch.device = None,
    ):
        self.steps = steps
        self.init_method = init_method
        self.use_sigmoid = use_sigmoid
        self.lr = lr
        self._prior = prior
        self.loss = loss
        self.tv_weight = tv_weight
        self.device = device

    def _build_prior(self, counts, angles, intensities):
        if self.init_method == "fbp":
            sinogram = sinogram_from_counts(counts, intensities).clamp_min(0.0)
            prior = fbp(sinogram, angles).clamp(0.0, 1.0)
        # elif self.init_method == "fbp_weighted":
        #     prior = fbp(counts, angles, intensities, weighted=True)
        elif self.init_method == "zeros":
            bs = counts.shape[0:-2]
            prior = torch.zeros(
                (*bs, counts.shape[-1], counts.shape[-1]), device=counts.device
            )
        elif self.init_method == "const":
            bs = counts.shape[0:-2]
            prior = (
                torch.ones(
                    (*bs, counts.shape[-1], counts.shape[-1]), device=counts.device
                )
                * 0.5
            )
        elif self.init_method == "random":
            bs = counts.shape[0:-2]
            prior = torch.randn(
                (*bs, counts.shape[-1], counts.shape[-1]), device=counts.device
            ).clip(0, 1)
        elif self.init_method == "prior" and self._prior is not None:
            prior = self._prior.clone().to(counts.device)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        return prior

    def __call__(
        self,
        counts: torch.Tensor,
        intensities: torch.Tensor,
        angles: torch.Tensor,
        verbose=False,
    ) -> torch.Tensor:
        """
        Perform iterative reconstruction given counts, intensities, and angles.
        Returns reconstructed image tensor (B, H, W).
        """
        with torch.enable_grad():
            prior_img = self._build_prior(counts, angles, intensities)
            tomogram = Tomogram(
                prior=prior_img.detach(), use_sigmoid=self.use_sigmoid, circle=True
            )

            optimizer = torch.optim.Adam(tomogram.parameters(), lr=self.lr)
            # if self.loss == "nll":
            #     loss_fn = nll
            # else:
            #     # mse_ct must be available in scope if using this branch
            #     loss_fn = lambda recon, meas, angs, alloc: mse_ct(recon, meas, angs, alloc, vst=anscombe_transform)

            circle_mask = circular_mask(
                prior_img.shape[-1], device=tomogram.image.device
            )
            it = tqdm(
                range(self.steps), desc="Iterative Reconstruction", disable=not verbose
            )
            for step in it:
                optimizer.zero_grad()

                loss = (
                    nll(tomogram(), counts, intensities, angles)
                    .mean(dim=(-1, -2))
                    .sum()
                )

                if self.tv_weight:
                    _tv_loss = tv_loss(tomogram()).mean()
                    loss += self.tv_weight * _tv_loss

                loss.backward()
                optimizer.step()

                # circle mask and clamp
                with torch.no_grad():
                    tomogram.image.clamp_(min=0.0, max=1.0)
                    tomogram.image.mul_(circle_mask)

                it.set_postfix(
                    loss=f"{loss.item():.10f}, tv={_tv_loss.item() if self.tv_weight else 0.0:.10f}"
                )

        with torch.no_grad():
            recon = tomogram()

        return recon.detach()


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


class UNetRecon:
    def __init__(self, unet: UNet2DModel):
        self.unet = unet

    def __call__(self, counts, intensities, angles):
        sinogram = sinogram_from_counts(counts, intensities).clamp_min(0.0)
        fbp_recon = fbp(sinogram, angles).clamp(0.0, 1.0)

        # ToDo: added a factor 2 to account for HR normalization
        total_intensities = 2 * intensities.expand(counts.shape).sum(dim=(-1, -2))

        x_in = fbp_recon * 2.0 - 1.0
        exposure_norm = (
            (total_intensities - MIN_TOTAL_INTENSITY)
            / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY)
            * 999
        )
        exposure_norm = exposure_norm.clamp(0.0, 999.0)

        batch_dims = x_in.size()[:-2]
        img_shape = x_in.shape[-2:]

        x_in = x_in.view(-1, *img_shape)
        while exposure_norm.dim() < len(batch_dims):
            exposure_norm = exposure_norm.unsqueeze(0)

        exposure_norm = exposure_norm.expand(batch_dims).reshape(-1)

        y = self.unet(x_in.unsqueeze(1), timestep=exposure_norm, return_dict=False)[0]
        y = ((y + 1.0) / 2.0).clamp(0.0, 1.0)  # (B,128,128)

        # circular mask
        mask = circular_mask(y.shape[-1], device=y.device)
        y = y * mask
        y = y.view(*batch_dims, *img_shape)
        return y


class UnetEnsembleRecon:
    def __init__(self, unets: list[UNet2DModel]):
        self.unets = unets
        self.recons = [UNetRecon(unet) for unet in unets]

    def __call__(self, counts, intensities, angles):
        recons = [recon(counts, intensities, angles) for recon in self.recons]
        return torch.stack(recons, dim=0)


class BootstrapRecon:
    def __init__(self, reconstructor, num_samples=5):
        self.reconstructor = reconstructor
        self.num_samples = num_samples

    def __call__(self, counts, intensities, angles):
        recons = []
        for _ in range(self.num_samples):
            # sample with replacement in angle dimension (dim=-2)
            indices = torch.randint(
                0, counts.shape[-2], (counts.shape[-2],), device=counts.device
            )
            counts_bs = counts.index_select(-2, indices)
            intensities_bs = intensities.index_select(-2, indices)
            angles_bs = angles.index_select(0, indices)
            recon_bs = self.reconstructor(counts_bs, intensities_bs, angles_bs)
            recons.append(recon_bs)

        return torch.stack(recons, dim=0)


def guidance_loss(counts, intensities, angles, length_scale=5.0):
    """
    Define a loss function for the diffusion model.
    This can be used to guide the diffusion process.
    """
    data_shape = counts.shape[:-2]
    circle_mask = circular_mask(counts.shape[-1], device=counts.device)

    def loss_fn(image):
        img_shape = image.shape[-2:]
        image = image.view(-1, *data_shape, *img_shape)
        image = ((image + 1.0) / 2).clip(0, 1)
        image = image * circle_mask
        loss = nll(image, counts, intensities, angles, length_scale=length_scale)
        return loss.sum()

    return loss_fn


class DiffusionRecon:
    def __init__(
        self,
        unet,
        scheduler,
        num_samples=5,
        num_inference_steps=100,
        timesteps=None,
        guidance_start=500,
        guidance_end=20,
        guidance_num_gradient_steps=10,
        guidance_lr=1e-1,
        guidance_lr_decay=False,
        guidance_loss=guidance_loss,
        seed=None,
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.guidance_num_gradient_steps = guidance_num_gradient_steps
        self.guidance_lr = guidance_lr
        self.guidance_lr_decay = guidance_lr_decay
        self.guidance_loss = guidance_loss

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        counts,
        intensities,
        angles,
        num_samples=None,
        verbose=False,
        **guidance_loss_kwargs,
    ):
        if num_samples is None:
            num_samples = self.num_samples
        guided_diffusion = GuidedDiffusionPipeline(self.unet, self.scheduler)

        loss_fct = self.guidance_loss(
            counts, intensities, angles, **guidance_loss_kwargs
        )
        guidance = GradientGuidance(
            loss_fct=loss_fct,
            num_gradient_steps=self.guidance_num_gradient_steps,
            guidance_start=self.guidance_start,
            guidance_end=self.guidance_end,
            lr=self.guidance_lr,
            learning_rate_decay=self.guidance_lr_decay,
        )

        samples = guided_diffusion(
            batch_size=len(counts) * num_samples,
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance=guidance,
            verbose=verbose,
            generator=self.generator,
        )
        # circular mask
        mask = circular_mask(samples.shape[-1], device=samples.device)
        samples = samples * mask
        return samples.view(num_samples, *counts.shape[:-2], 128, 128)


class CondDiffusionRecon:
    def __init__(
        self,
        unet,
        scheduler,
        num_samples=5,
        num_inference_steps=100,
        timesteps=None,
        guidance_start=500,
        guidance_end=20,
        guidance_num_gradient_steps=10,
        guidance_lr=1e-1,
        guidance_lr_decay=False,
        guidance_loss=guidance_loss,
        seed=None,
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.guidance_num_gradient_steps = guidance_num_gradient_steps
        self.guidance_lr = guidance_lr
        self.guidance_lr_decay = guidance_lr_decay
        self.guidance_loss = guidance_loss

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        counts,
        intensities,
        angles,
        num_samples=None,
        verbose=False,
        **guidance_loss_kwargs,
    ):
        if num_samples is None:
            num_samples = self.num_samples

        guided_diffusion = GuidedDiffusionPipeline(self.unet, self.scheduler)

        image_shape = (
            len(counts) * num_samples,
            1,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )

        loss_fct = self.guidance_loss(
            counts, intensities, angles, **guidance_loss_kwargs
        )
        guidance = GradientGuidance(
            loss_fct=loss_fct,
            num_gradient_steps=self.guidance_num_gradient_steps,
            guidance_start=self.guidance_start,
            guidance_end=self.guidance_end,
            lr=self.guidance_lr,
            learning_rate_decay=self.guidance_lr_decay,
        )

        # compute fbps
        sinogram = sinogram_from_counts(counts, intensities).clamp_min(0.0)
        fbps = fbp(sinogram, angles).clamp(0.0, 1.0)
        # ToDo: added a factor 2 to account for HR normalization
        total_intensities = 2 * intensities.expand(counts.shape).sum(
            dim=(-1, -2)
        ).squeeze(0)

        # print(fbps.shape, counts.shape, intensities.shape)
        fbps_norm = fbps * 2.0 - 1.0
        fbps_norm = (
            fbps_norm.unsqueeze(0)
            .expand(num_samples, -1, -1, -1, -1)
            .reshape(-1, 1, fbps.shape[-2], fbps.shape[-1])
        )
        # print(fbps_norm.shape)

        # compute intensity
        intensities_norm = (
            2 * ((norm_intensities(total_intensities) / 999) - 0.5)
        ).clip(-1, 1)
        intensities_norm = (
            intensities_norm.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1)
        )

        n_angles = torch.tensor(len(angles), device=angles.device).float()
        n_angles_norm = ((n_angles - N_ANGLES / 2) / (N_ANGLES / 2)).clip(-1, 1)

        n_angles_norm = (
            n_angles_norm.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(num_samples, len(counts), 1)
            .reshape(-1)
        )

        cond_kwargs = dict(
            fbp=fbps_norm, intensity_norm=intensities_norm, n_angles_norm=n_angles_norm
        )

        samples = guided_diffusion(
            batch_size=len(counts) * num_samples,
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance=guidance,
            verbose=verbose,
            cond_kwargs=cond_kwargs,
            image_shape=image_shape,
            generator=self.generator,
        )

        # circular mask
        mask = circular_mask(samples.shape[-1], device=samples.device)
        samples = samples * mask

        return samples.view(num_samples, *counts.shape[:-2], 128, 128)


def schedule_uniform(total_intensity, n_steps, init_fraction=None, device=None):
    if init_fraction:
        alloc = [init_fraction]
    else:
        init_fraction = 0.0
        alloc = []
        n_steps += 1
    per_step_fraction = (1.0 - init_fraction) / n_steps
    alloc += [per_step_fraction] * n_steps
    return torch.tensor(alloc, device=device) * total_intensity


def schedule_exponential(
    total_intensity,
    num_steps,
    initial_intensity,
    device=None,
):
    # Log-uniform cumulative targets from initial_intensity to total_intensity
    cumsum = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(initial_intensity, device=device)),
            torch.log(torch.tensor(total_intensity, device=device)),
            num_steps + 1,
            device=device,
        )
    )

    # Convert cumulative schedule to per-step intensities
    intensities = torch.diff(torch.cat([torch.zeros(1, device=device), cumsum]))

    return intensities


def sample_observations_reproducible(
    images: torch.Tensor,
    indices: torch.Tensor,
    schedule: torch.Tensor,
    angles: torch.Tensor,
    seed_offset: int = 0,
):
    """
    Sample Poisson observations from high-res images according to the given schedule and angles.
    Args:
        images (torch.Tensor): (B, 1, H, W) High-res image tensor.
        schedule (torch.Tensor): (T,) Tensor specifying the intensity allocation per step.
        angles (torch.Tensor): (N,) Tensor of projection angles in degrees.
        seed_offset (int): Offset to add to the random seed for reproducibility.
    Returns:
        torch.Tensor: (B, T, N, W) Tensor of sampled Poisson counts.
    """

    data = []
    for idx, image in zip(indices, images):
        generator = torch.Generator(device=device).manual_seed(
            seed_offset + 100000 * int(idx)
        )
        _data = sample_observations(
            image.unsqueeze(0), schedule / 2, angles, generator=generator
        )  # TODO: divide by 2 is important!!!
        data.append(_data)
    return torch.cat(data, dim=0)


def nll_mixture(
    images: torch.Tensor,
    counts: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    length_scale: int = 5,
) -> torch.Tensor:
    """
    Arguments:
        images (`torch.Tensor`): (..., n_preds, H, W)
        counts (`torch.Tensor`): (..., n_angles, n_detectors)
        intensities (`torch.Tensor`): (..., n_angles, 1)
        angles (`torch.Tensor`): (n_angles,)
        l: (`int`)
    Returns:
        `torch.Tensor`: (...)
    """

    n_pred = images.shape[0]

    # (..., n_pred, n_angles, side_length)
    nlls = -nll(images, counts, intensities, angles, length_scale=length_scale).double()
    nlls = nlls.sum((-1, -2))  # (n+pred, ...)
    nlls -= math.log(n_pred)
    mix = -torch.logsumexp(nlls, dim=0)  # (...)
    return mix.float()


def guidance_loss_beta(
    counts, intensities, angles, beta, data_steps, schedule_steps, length_scale=5.0
):
    """
    Define a loss function for the diffusion model.
    This can be used to guide the diffusion process.
    """
    data_shape = counts.shape[:-2]
    circle_mask = circular_mask(counts.shape[-1], device=counts.device)

    def loss_fn(image):
        img_shape = image.shape[-2:]
        image = image.view(-1, *data_shape, *img_shape)
        image = ((image + 1.0) / 2).clip(0, 1)
        image = image * circle_mask
        image = image.unsqueeze(-4)  # add step dimension

        step_nll = nll(
            image, data_steps, schedule_steps, angles, length_scale=length_scale
        ).sum(dim=[-1, -2, -3])
        remaining_step_nll = step_nll[..., 1:]
        beta_loss = remaining_step_nll.sum(dim=-1)

        loss = torch.abs(beta_loss - beta)  # + first_step_nll
        return loss.sum()  # remaining dimensions (samples, batch, steps)

    return loss_fn


def guidance_loss_diverse(
    counts,
    intensities,
    angles,
    beta,
    data_steps,
    schedule_steps,
    length_scale=5.0,
    dist_loss_fac=1000.0,
):
    """
    Define a loss function for the diffusion model.
    This can be used to guide the diffusion process.
    """
    data_shape = counts.shape[:-2]
    circle_mask = circular_mask(counts.shape[-1], device=counts.device)

    def loss_fn(image):
        img_shape = image.shape[-2:]
        image = image.view(-1, *data_shape, *img_shape)
        image = ((image + 1.0) / 2).clip(0, 1)
        image = image * circle_mask
        image = image.unsqueeze(-4)  # add step dimension

        step_nll = nll(
            image, data_steps, schedule_steps, angles, length_scale=length_scale
        ).sum(dim=[-1, -2, -3])
        remaining_step_nll = step_nll[..., 1:]

        cond = remaining_step_nll.sum(dim=-1) <= beta
        dists = torch.square(image[:, None, ...] - image[None, :, ...]).sum(
            dim=(-1, -2, -3)
        )
        dist_loss_vals = dists.mean(dim=1).squeeze(-1)  # squeeze step dimension

        # print all shapes with variables names
        # print(f"step_nll: {step_nll.shape}, remaining_step_nll: {remaining_step_nll.shape}, cond: {cond.shape}, dists: {dists.shape}, dist_loss_vals: {dist_loss_vals.shape}", f"beta: {beta}")
        loss_dist_term = dist_loss_fac * cond.float() * (-dist_loss_vals)

        # beta_loss = remaining_step_nll.sum(dim=-1)

        loss = step_nll.sum(dim=-1) + loss_dist_term
        return loss.sum()  # remaining dimensions (samples, batch, steps)

    return loss_fn


def interpolate(images, size, mode="area"):
    batch_dims = images.size()[:-3]
    img_shape = images.shape[-3:]
    x_in = images.view(-1, *img_shape)
    y = F.interpolate(x_in, size=size, mode=mode)
    return y.view(*batch_dims, *y.shape[-3:])


def rotate_images(images, degree):
    return torch.vmap(lambda img: TF.rotate(img, degree))(images)


def experiment_id(params):
    """Create a unique ID string from experiment parameters."""
    relevant = {k: v for k, v in params.items() if v is not None}
    id_str = json.dumps(relevant, sort_keys=True)
    return hashlib.md5(id_str.encode()).hexdigest()[:8]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image quality metrics.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["composite", "lamino", "lung"],
        default="lung",
        help="Dataset to use for evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "fbp",
            "unet",
            "cond_diffusion",
            "beta_cond_diffusion",
            "diverse_cond_diffusion",
            "diffusion",
            "gt",
            "unet_ensemble",
            "mle",
            "map",
            "fbp_bootstrap",
            "unet_bootstrap",
        ],
        default="fbp",
        help="Model to use for evaluation.",
    )

    def parse_seeds(arg):
        # Accepts comma-separated ints or ranges like 0-100
        seeds = []
        for part in arg.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                seeds.extend(range(int(start), int(end)))
            elif part:
                seeds.append(int(part))
        return seeds

    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=[0],
        help="Random seed(s) for observation sampling. Accepts comma-separated values or ranges like 0-100.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    # parser.add_argument("--batch_steps", action="store_true", default=True, help="Whether to batch over observation steps.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results.",
        required=True,
    )
    parser.add_argument(
        "--num_images", type=int, default=50, help="Number of images to evaluate."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate per image.",
    )
    parser.add_argument(
        "--total_intensity",
        type=float,
        default=1e8,
        help="Total intensity for observation schedule.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of steps in the observation schedule.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["uniform", "exponential"],
        default="uniform",
        help="Type of observation schedule.",
    )
    parser.add_argument(
        "--init_fraction",
        type=float,
        default=None,
        help="Initial fraction of total intensity in the schedule.",
    )
    parser.add_argument(
        "--initial_intensity",
        type=float,
        default=1e6,
        help="Initial intensity for exponential schedule.",
    )
    parser.add_argument("--rotation", type=float, default=None)
    parser.add_argument(
        "--iterative_num_gradient_steps",
        type=int,
        default=100,
        help="Number of gradient steps for iterative reconstruction.",
    )
    parser.add_argument(
        "--iterative_lr",
        type=float,
        default=1e-1,
        help="Learning rate for iterative reconstruction.",
    )
    parser.add_argument(
        "--diffusion_num_inference_steps",
        type=int,
        default=100,
        help="Number of inference steps for diffusion models.",
    )
    parser.add_argument(
        "--diffusion_seed",
        type=int,
        default=0,
        help="Random seed for diffusion sampling.",
    )
    parser.add_argument(
        "--guidance_num_gradient_steps",
        type=int,
        default=10,
        help="Number of gradient steps for diffusion guidance.",
    )
    parser.add_argument(
        "--guidance_lr",
        type=float,
        default=1e-3,
        help="Learning rate for diffusion guidance.",
    )
    parser.add_argument(
        "--guidance_lr_decay",
        action="store_true",
        help="Whether to use learning rate decay for diffusion guidance.",
    )
    parser.add_argument(
        "--guidance_end",
        type=int,
        default=10,
        help="Guidance end timestep for diffusion models.",
    )
    parser.add_argument(
        "--num_bootstrap_samples",
        type=int,
        default=10,
        help="Number of bootstrap samples for uncertainty estimation.",
    )
    parser.add_argument(
        "--existing",
        default=None,
        choices=["skip", "overwrite"],
        help="Behavior if output file exists.",
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        default=1e3,
        help="Total Variation weight for iterative reconstruction.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs="+",
        default=[0.05],
        help="Confidence level for metrics.",
    )
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}, Model: {args.model}")

    mixing_models = [
        "diffusion",
        "cond_diffusion",
        "unet_ensemble",
        "fbp_bootstrap",
        "unet_bootstrap",
        "beta_cond_diffusion",
        "diverse_cond_diffusion",
    ]
    per_sample_models = ["gt", "cond_diffusion", "diffusion", "diverse_cond_diffusion"]

    angles_rotation_deg = torch.tensor(
        [0.0] + np.logspace(np.log(0.1), np.log(90), 10, base=np.e).tolist()
    )

    # List of argument names to include as metadata
    excluded_keys = ["existing"]

    # Collect selected parameters
    experiment_params = {
        k: getattr(args, k, None)
        for k, v in vars(args).items()
        if k not in excluded_keys
    }

    output_file_name = experiment_id(experiment_params)
    # Add extra metadata
    experiment_params["experiment_id"] = output_file_name
    experiment_params["datetime"] = datetime.datetime.now().isoformat()

    # get env variables that start with RUNAI
    for k, v in os.environ.items():
        if k.startswith("RUNAI_"):
            experiment_params[k] = v

    # check git state and warn if not clean, ask for confirmation to proceed
    if git is None:
        print("gitpython not installed, skipping git state check.")
    else:
        # check for uncommitted changes excluding notebooks
        repo = git.Repo(search_parent_directories=True)
        experiment_params["git_commit"] = repo.head.object.hexsha
        experiment_params["git_branch"] = repo.active_branch.name
        experiment_params["git_diff"] = repo.git.diff("--", ".", ":(exclude)notebooks")

        # Combined dirty check
        print(f"Git commit: {experiment_params['git_commit']}")
        print(f"Git branch: {experiment_params['git_branch']}")
        # print(f"Git diff: {experiment_params['git_diff']}")

    # Output dir
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory at {args.output_dir}")
        os.makedirs(args.output_dir)

    # File name and existence/overwrite check
    model_name = args.model
    if args.model == "gt" and args.rotation is not None:
        model_name += f"_rot{args.rotation}"

    # output_file = Path(args.output_dir) / f"eval_{args.dataset}_{model_name}.nc"
    output_file = os.path.join(args.output_dir, output_file_name + ".nc")
    if os.path.exists(output_file):
        if args.existing == "skip":
            print(
                f"Results file {output_file} exists. Skipping evaluation as per --skip flag."
            )
            exit(0)
        elif args.existing == "overwrite":
            print(
                f"Results file {output_file} exists. Overwriting as per --overwrite flag."
            )
            os.remove(output_file)
        else:
            resp = (
                input(f"File {output_file} exists. Delete and start over? [y/N]: ")
                .strip()
                .lower()
            )
            if resp == "y":
                os.remove(output_file)
            else:
                print("Exiting without overwriting existing results.")
                exit(0)

    # get dataset
    _, test_set = get_dataset(args.dataset, True)
    if args.num_images is not None:
        test_set = torch.utils.data.Subset(
            test_set, list(range(10, args.num_images + 10))
        )

    # set up observation parameters
    num_angles = N_ANGLES
    angles = (
        torch.from_numpy(np.linspace(0, 180, num_angles, endpoint=False))
        .float()
        .to(device)
    )
    if args.schedule == "uniform":
        schedule = schedule_uniform(
            args.total_intensity,
            args.num_steps,
            init_fraction=args.init_fraction,
            device=device,
        )
    elif args.schedule == "exponential":
        schedule = schedule_exponential(
            args.total_intensity,
            args.num_steps,
            initial_intensity=args.initial_intensity,
            device=device,
        )
    total_intensities = schedule.clone()
    schedule = (
        schedule.reshape(-1, 1, 1, 1).expand(-1, 1, num_angles, 1)
        / num_angles
        / N_BINS_HR
    )

    # set up observation dataset and dataloader
    obs_dataset = ObservationDataset(
        test_set, seeds=args.seeds, intensities=schedule, angles=angles
    )
    obs_dataloader = torch.utils.data.DataLoader(
        obs_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # load model
    if args.model == "gt":
        recon = None
    elif args.model in ["fbp", "fbp_bootstrap"]:
        recon = fbp_recon
    elif args.model == "mle":
        recon = IterativeRecon(
            steps=args.iterative_num_gradient_steps,
            init_method="zeros",
            use_sigmoid=False,
            lr=args.iterative_lr,
            loss="nll",
            device=device,
        )
    elif args.model == "map":
        recon = IterativeRecon(
            steps=args.iterative_num_gradient_steps,
            init_method="zeros",
            use_sigmoid=False,
            lr=args.iterative_lr,
            loss="nll",
            tv_weight=args.tv_weight,
            device=device,
        )
    elif args.model in ["unet", "unet_bootstrap"]:
        ckpt_path = Path(
            f"/mydata/chip/shared/checkpoints/uqct/unet_dense/unet_dense_128_{args.dataset}_0.pt"
        )
        print(f"Loading UNet checkpoint from {ckpt_path}")
        unet = load_unet(ckpt_path, sparse=False).to(device).eval()
        recon = UNetRecon(unet)
    elif args.model == "unet_ensemble":
        unets = []
        for i in range(10):
            ckpt_path = Path(
                f"/mydata/chip/shared/checkpoints/uqct/unet_dense/unet_dense_128_{args.dataset}_{i}.pt"
            )
            print(f"Loading UNet checkpoint from {ckpt_path}")
            unet = load_unet(ckpt_path, sparse=False).to(device).eval()
            unets.append(unet)
        recon = UnetEnsembleRecon(unets)
    elif args.model == "diffusion":
        ckpt_path = Path(
            f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_unconditional_128_{args.dataset}.pt"
        )
        unet = load_diffusion_unet(ckpt_path, cond=False)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        recon = DiffusionRecon(
            unet,
            scheduler,
            num_samples=args.num_samples,
            num_inference_steps=args.diffusion_num_inference_steps,
            guidance_start=1000,
            guidance_end=args.guidance_end,
            guidance_num_gradient_steps=args.guidance_num_gradient_steps,
            guidance_lr=args.guidance_lr,
            guidance_lr_decay=args.guidance_lr_decay,
            seed=args.diffusion_seed if args.diffusion_seed != -1 else None,
        )
    elif args.model in [
        "cond_diffusion",
        "beta_cond_diffusion",
        "diverse_cond_diffusion",
    ]:
        ckpt_path = Path(
            f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_conditional_128_{args.dataset}.pt"
        )
        unet = load_diffusion_unet(ckpt_path, cond=True)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        if args.model in ["beta_cond_diffusion", "diverse_cond_diffusion"]:
            # find beta dataset
            # list all *.nc files in the checkpoint directory

            nc_files = [
                f
                for f in sorted(glob.glob(f"{args.output_dir}/*.nc", recursive=False))
                if not f.endswith("samples.nc")
            ]
            beta_file = None
            beta_ds = None
            for f in nc_files:
                ds = xr.open_dataset(f)
                match_kwargs = [
                    "model",
                    "dataset",
                    "diffusion_num_inference_steps",
                    "guidance_end",
                    "guidance_num_gradient_steps",
                    "guidance_lr",
                    "guidance_lr_decay",
                    "diffusion_seed",
                    "initial_intensity",
                    "total_intensity",
                    "num_steps",
                    "num_samples",
                    "seeds",
                    "num_images",
                ]
                target_str_args = {k: str(v) for k, v in vars(args).items()}
                # hard code parameters to match
                target_str_args["model"] = "cond_diffusion"  # match cond_diffusion
                target_str_args["guidance_lr"] = str(1e-3)
                target_str_args["guidance_lr_decay"] = str(False)
                target_str_args["guidance_num_gradient_steps"] = str(5)
                target_str_args["diffusion_num_inference_steps"] = str(50)

                # check if all match
                if all(target_str_args[k] == ds.attrs[k] for k in match_kwargs):
                    if beta_file is not None:
                        raise ValueError(
                            f"Multiple matching beta datasets found: {beta_file} and {f}"
                        )

                    print(f"Found matching beta dataset: {f}")
                    beta_file = f
                    beta_ds = ds

            if beta_ds is None:
                raise ValueError("No matching beta dataset found for guidance.")
            beta_ds["beta"] = beta_ds["seq_nll"].cumsum(dim="step")
            beta_ds["beta_mix"] = beta_ds["seq_nll_mix"].cumsum(dim="step")
            if args.model == "diverse_cond_diffusion":
                guidance_loss_fn = guidance_loss_diverse  # type: ignore
            elif args.model == "beta_cond_diffusion":
                guidance_loss_fn = guidance_loss_beta  # type: ignore
        else:
            guidance_loss_fn = guidance_loss  # type: ignore
        recon = CondDiffusionRecon(
            unet,
            scheduler,
            num_samples=args.num_samples,
            num_inference_steps=args.diffusion_num_inference_steps,
            guidance_start=1000,
            guidance_end=args.guidance_end,
            guidance_num_gradient_steps=args.guidance_num_gradient_steps,
            guidance_lr=args.guidance_lr,
            guidance_lr_decay=args.guidance_lr_decay,
            seed=args.diffusion_seed if args.diffusion_seed != -1 else None,
            guidance_loss=guidance_loss_fn,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if model_name.endswith("_bootstrap"):
        print(
            f"Wrapping reconstructor with BootstrapRecon using {args.num_bootstrap_samples} samples."
        )
        recon = BootstrapRecon(recon, num_samples=args.num_bootstrap_samples)

    # Prepare to collect batched results
    res = collections.defaultdict(list)
    res_samples = collections.defaultdict(list)

    t0 = time.time()
    for indices, seed, images, data in tqdm(obs_dataloader):
        # free up memory
        torch.cuda.empty_cache()

        images = images.to(device)
        images_lr = F.interpolate(images, size=(128, 128), mode="area")
        data = data.to(device)

        # Shape conventions:
        # Images (Batch, Step, 1, H, W)
        # Sampled Predictions (Sample, Batch, Step, 1, H, W) or (Batch, Step, 1, H, W) for non-mixing models
        if args.model == "gt":
            # prediction is ground truth image across steps
            pred = (
                images_lr.clone()
                .unsqueeze(1)
                .expand(-1, schedule.shape[0], -1, -1, -1)
                .contiguous()
            )

            # samples are rotated versions of ground truth
            images_rotated = torch.stack(
                [
                    rotate_images(images, rot_deg.item())
                    for rot_deg in angles_rotation_deg
                ],
                dim=0,
            )
            images_rotated = interpolate(images_rotated, size=(128, 128), mode="area")
            samples = (
                images_rotated.unsqueeze(2)
                .expand(-1, -1, schedule.shape[0], -1, -1, -1)
                .contiguous()
            )
            num_samples = len(angles_rotation_deg)
            sample_coords = angles_rotation_deg
        else:
            data_cumsum = data.cumsum(dim=1)
            schedule_cumsum = schedule.unsqueeze(0).cumsum(dim=1)
            with torch.no_grad():
                if args.model in ["beta_cond_diffusion", "diverse_cond_diffusion"]:
                    # print(indices, seed)
                    indices_np = indices.cpu().numpy()
                    seeds_np = seed.cpu().numpy()

                    # Build tuples for selection
                    pairs = list(zip(indices_np, seeds_np))

                    # Select using .sel with a list of tuples
                    beta = (
                        cast(Any, beta_ds)
                        .sel(model="cond_diffusion")["beta_mix"]
                        .stack(sample=("index", "seed"))
                        .sel(sample=pairs)
                        .squeeze()
                        .transpose("sample", "step")
                    )

                    beta = torch.from_numpy(beta.values).to(device).float()

                    data_steps = data
                    schedule_steps = schedule.unsqueeze(0)

                    beta_delta = beta + torch.log(1 / torch.tensor(0.05))
                    pred = [
                        recon(  # type: ignore
                            data_cumsum[:, i],
                            schedule_cumsum[:, i],
                            angles,
                            beta=beta_delta[:, i],
                            data_steps=data_steps[:, : i + 1],
                            schedule_steps=schedule_steps[:, : i + 1],
                        ).detach()  # type: ignore
                        for i in range(data.shape[1])
                    ]

                else:
                    pred = [
                        recon(data_cumsum[:, i], schedule_cumsum[:, i], angles).detach()  # type: ignore
                        for i in range(data.shape[1])
                    ]
            pred = torch.stack(pred, dim=-4)
        # free up memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            if args.model in mixing_models:
                if (
                    args.model
                    in [
                        "diffusion",
                        "cond_diffusion",
                        "unet_ensemble",
                        "diverse_cond_diffusion",
                    ]
                    or args.num_bootstrap_samples <= 20
                ):
                    _seq_nll_mix = nll_mixture(
                        pred[:, :, :-1].contiguous(), data[:, 1:], schedule[1:], angles
                    ).squeeze(-1)

                    res["seq_nll_mix"].append(_seq_nll_mix.detach().cpu())
                    res["beta_mix"].append(_seq_nll_mix.cumsum(dim=1).detach().cpu())

                mean_pred = pred.mean(dim=0)
                abs_error = torch.abs(mean_pred - images_lr.unsqueeze(1))
                for _delta in args.delta:
                    for ci_fun, ci_name in [
                        (percentile_ci, "percentile"),
                        (gaussian_ci, "gaussian"),
                        (basic_ci, "basic"),
                        (gaussian_conservative_ci, "gaussian_conservative_ci"),
                        (simultaneous_ci, "simultaneous_ci"),
                    ]:
                        lo, hi = ci_fun(pred, _delta)  # type: ignore

                        _coverage = coverage(lo, hi, images_lr.unsqueeze(1)).squeeze(-1)  # type: ignore
                        _simultaneous_coverage = simultaneous_coverage(
                            lo, hi, images_lr.unsqueeze(1)
                        ).squeeze(-1)  # type: ignore
                        _error_correlation = error_correlation(
                            hi - lo, abs_error
                        ).squeeze(-1)
                        _error_r2 = error_r2(hi - lo, abs_error).squeeze(-1)
                        _width = (hi - lo).mean(dim=(-1, -2, -3))
                        _ause = sparsification_error(hi - lo, abs_error).squeeze(-1)

                        res[f"uq_width_{ci_name}_{_delta}"].append(
                            _width.detach().cpu()
                        )
                        res[f"uq_coverage_{ci_name}_{_delta}"].append(
                            _coverage.detach().cpu()
                        )
                        res[f"uq_simultaneous_coverage_{ci_name}_{_delta}"].append(
                            _simultaneous_coverage.detach().cpu()
                        )
                        res[f"uq_error_correlation_{ci_name}_{_delta}"].append(
                            _error_correlation.detach().cpu()
                        )
                        res[f"uq_error_r2_{ci_name}_{_delta}"].append(
                            _error_r2.detach().cpu()
                        )
                        res[f"uq_ause_{ci_name}_{_delta}"].append(_ause.detach().cpu())

                # all subsequent metrics use point prediction
                samples = pred
                pred = mean_pred
                num_samples = args.num_samples
                sample_coords = torch.arange(num_samples)

            # compute metrics for point prediction
            res["rmse"].append(
                rmse(pred, images_lr.unsqueeze(1), circle_mask=True)
                .squeeze(-1)
                .detach()
                .cpu()
            )
            res["psnr"].append(
                psnr(pred, images_lr.unsqueeze(1), circle_mask=True, data_range=1.0)
                .squeeze(-1)
                .detach()
                .cpu()
            )
            res["ssim"].append(
                ssim(pred, images_lr.unsqueeze(1), circle_mask=True, data_range=1.0)
                .squeeze(-1)
                .detach()
                .cpu()
            )
            res["seq_nll"].append(
                nll(pred[:, :-1].contiguous(), data[:, 1:], schedule[1:], angles)
                .sum((-1, -2, -3))
                .detach()
                .cpu()
            )
            res["beta"].append(
                nll(pred[:, :-1].contiguous(), data[:, 1:], schedule[1:], angles)
                .sum((-1, -2, -3))
                .cumsum(dim=1)
                .detach()
                .cpu()
            )
            res["nll"].append(
                nll(pred.unsqueeze(2), data.unsqueeze(1), schedule, angles)
                .sum((-1, -2, -3))
                .cumsum(dim=2)
                .diagonal(dim1=1, dim2=2)
                .detach()
                .cpu()
            )
            res["seed"].append(torch.as_tensor(seed).view(-1).cpu())
            res["index"].append(torch.as_tensor(indices).view(-1).cpu())
            # nll starting at step 1
            nll_1 = (
                nll(
                    pred[:, 1:].unsqueeze(2),
                    data[:, 1:].unsqueeze(1),
                    schedule[1:],
                    angles,
                )
                .sum((-1, -2, -3))
                .cumsum(dim=2)
                .diagonal(dim1=1, dim2=2)
                .detach()
                .cpu()
            )
            res["nll_1"].append(nll_1)

            # compute per sample metrics
            if model_name in per_sample_models:
                sample_indices = (
                    indices.unsqueeze(0).expand(num_samples, -1).reshape(-1)
                )
                sample_seed = seed.unsqueeze(0).expand(num_samples, -1).reshape(-1)
                samples = samples.reshape(-1, *samples.shape[2:])
                sample_images_lr = (
                    images_lr.unsqueeze(0)
                    .expand(num_samples, -1, -1, -1, -1)
                    .reshape(-1, *images_lr.shape[1:])
                    .unsqueeze(1)
                )
                sample_data = (
                    data.unsqueeze(0)
                    .expand(num_samples, -1, -1, -1, -1, -1)
                    .reshape(
                        -1,
                        *data.shape[1:],
                    )
                )

                # metrics for all samples
                res_samples["rmse"].append(
                    rmse(samples, sample_images_lr, circle_mask=True)
                    .squeeze(-1)
                    .detach()
                    .cpu()
                )
                res_samples["psnr"].append(
                    psnr(samples, sample_images_lr, circle_mask=True, data_range=1.0)
                    .squeeze(-1)
                    .detach()
                    .cpu()
                )
                res_samples["ssim"].append(
                    ssim(samples, sample_images_lr, circle_mask=True, data_range=1.0)
                    .squeeze(-1)
                    .detach()
                    .cpu()
                )
                res_samples["seq_nll"].append(
                    nll(
                        samples[:, :-1].contiguous(),
                        sample_data[:, 1:],
                        schedule[1:],
                        angles,
                    )
                    .sum((-1, -2, -3))
                    .detach()
                    .cpu()
                )
                res_samples["beta"].append(
                    nll(
                        samples[:, :-1].contiguous(),
                        sample_data[:, 1:],
                        schedule[1:],
                        angles,
                    )
                    .sum((-1, -2, -3))
                    .cumsum(dim=1)
                    .detach()
                    .cpu()
                )
                res_samples["seed"].append(torch.as_tensor(sample_seed).view(-1).cpu())
                res_samples["index"].append(
                    torch.as_tensor(sample_indices).view(-1).cpu()
                )
                res_samples["sample_coords"].append(
                    sample_coords[:, None].expand(-1, len(images)).reshape(-1)
                )

                # memory intensive
                # _nll = nll(samples.unsqueeze(2), sample_data.unsqueeze(1), schedule, angles).sum((-1,-2, -3)).cumsum(dim=2).diagonal(dim1=1, dim2=2).detach().cpu()

                _nll = (
                    torch.cat(
                        [
                            nll(
                                samples[:, step, None, None].contiguous(),
                                sample_data[:, None, : step + 1],
                                schedule[: step + 1],
                                angles,
                            )
                            .sum((-1, -2, -3))
                            .sum(dim=2)
                            for step in range(0, schedule.shape[0])
                        ],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                )

                res_samples["nll"].append(_nll)
                # nll starting at step 1, memory intensive
                # nll_1 = nll(samples[:, 1:].unsqueeze(2), sample_data[:, 1:].unsqueeze(1), schedule[1:], angles).sum((-1,-2, -3)).cumsum(dim=2).diagonal(dim1=1, dim2=2).detach().cpu()

                nll_1 = (
                    torch.cat(
                        [
                            nll(
                                samples[:, step, None, None].contiguous(),
                                sample_data[:, None, 1 : step + 1],
                                schedule[1 : step + 1],
                                angles,
                            )
                            .sum((-1, -2, -3))
                            .sum(dim=2)
                            for step in range(1, schedule.shape[0])
                        ],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                )

                res_samples["nll_1"].append(nll_1)

    total_time = time.time() - t0
    experiment_params["total_time_sec"] = total_time
    # stack all results
    for k in res:
        res[k] = torch.cat(res[k], dim=0)  # type: ignore

    # Insert NaN values in front to match shape (N_samples, N_steps)
    N_samples, N_steps = res["psnr"].shape  # type: ignore
    res["seq_nll"] = torch.cat([torch.full((N_samples, 1), 0.0), res["seq_nll"]], dim=1)  # type: ignore
    res["beta"] = torch.cat([torch.full((N_samples, 1), 0.0), res["beta"]], dim=1)  # type: ignore
    res["nll_1"] = torch.cat([torch.full((N_samples, 1), 0.0), res["nll_1"]], dim=1)  # type: ignore

    if "seq_nll_mix" in res:
        # seq_nll_mix_tensor = torch.cat(res['seq_nll_mix'], dim=0)  # expected (N_samples, N_steps-1)
        res["seq_nll_mix"] = torch.cat(
            [torch.full((N_samples, 1), 0.0), res["seq_nll_mix"]], dim=1
        )  # type: ignore
        res["beta_mix"] = torch.cat(
            [torch.full((N_samples, 1), 0.0), res["beta_mix"]], dim=1
        )  # type: ignore

    data_vars = {
        k: (["sample", "step"], v.numpy())  # type: ignore
        for k, v in res.items()
        if k
        in [
            "psnr",
            "rmse",
            "ssim",
            "nll",
            "nll_1",
            "seq_nll",
            "seq_nll_mix",
            "beta",
            "beta_mix",
        ]
        or k.startswith("uq_")
    }

    # First build a sample/step dataset, then reshape to (dataset, index, seed, step, model)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": np.arange(N_samples),
            "step": np.arange(N_steps),
            "index": ("sample", res["index"].numpy()),  # type: ignore
            "seed": ("sample", res["seed"].numpy()),  # type: ignore
        },
    )
    # Turn (sample) into a MultiIndex of (index, seed) and unstack -> dims (index, seed, step)
    ds = ds.set_index(sample=("index", "seed")).unstack("sample")
    ds = ds.expand_dims(dataset=[args.dataset], model=[model_name])
    ds = ds.transpose("dataset", "index", "seed", "step", "model")

    # Add schedule intensity as value, indexed by step
    ds["intensity"] = ("step", total_intensities.cpu().numpy())

    # Attach to xarray Dataset
    ds.attrs.update({k: str(v) for k, v in experiment_params.items()})
    ds.attrs["samples"] = False

    ds.to_netcdf(output_file)
    print(f"Results written to {output_file}")

    if model_name in per_sample_models:
        for k in res_samples:
            res_samples[k] = torch.cat(res_samples[k], dim=0)  # type: ignore

        # Insert NaN values in front to match shape (N_samples, N_steps)
        N_samples, N_steps = res_samples["psnr"].shape  # type: ignore
        res_samples["seq_nll"] = torch.cat(
            [torch.full((N_samples, 1), 0.0), res_samples["seq_nll"]], dim=1
        )  # type: ignore
        res_samples["beta"] = torch.cat(
            [torch.full((N_samples, 1), 0.0), res_samples["beta"]], dim=1
        )  # type: ignore
        res_samples["nll_1"] = torch.cat(
            [torch.full((N_samples, 1), 0.0), res_samples["nll_1"]], dim=1
        )  # type: ignore
        data_vars = {
            k: (["sample", "step"], v.numpy())  # type: ignore
            for k, v in res_samples.items()
            if k in ["psnr", "rmse", "ssim", "nll", "nll_1", "seq_nll"]
        }
        # First build a sample/step dataset, then reshape to (dataset, index, seed, step, model)
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "sample": np.arange(N_samples),
                "step": np.arange(N_steps),
                "index": ("sample", res_samples["index"].numpy()),  # type: ignore
                "sample_coords": ("sample", res_samples["sample_coords"].numpy()),  # type: ignore
                "seed": ("sample", res_samples["seed"].numpy()),  # type: ignore
            },
        )
        # Turn (sample) into a MultiIndex of (index, seed) and unstack -> dims (index, seed, step)
        ds = ds.set_index(sample=("index", "sample_coords", "seed")).unstack("sample")
        ds = ds.expand_dims(dataset=[args.dataset], model=[model_name])
        ds = ds.transpose("dataset", "index", "sample_coords", "seed", "step", "model")

        ds = ds.rename(
            {"sample_coords": "rotation" if model_name == "gt" else "sample"}
        )
        # Add schedule intensity as value, indexed by step
        ds["intensity"] = ("step", total_intensities.cpu().numpy())

        # Attach to xarray Dataset
        ds.attrs.update({k: str(v) for k, v in experiment_params.items()})
        ds.attrs["experiment_id"] = ds.attrs["experiment_id"] + "_samples"
        ds.attrs["samples"] = True

        ds.to_netcdf(output_file[:-3] + "_samples.nc")
        print(f"Results written to {output_file[:-3] + '_samples.nc'}")
