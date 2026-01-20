import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any
import argparse
import os

from uqct.datasets.utils import get_dataset
from uqct.metrics import get_metrics
from uqct.ct import fbp, nll, Experiment, Tomogram, anscombe_transform, radon, sinogram_from_counts, poisson, sample_observations, circular_mask

import torch.nn.functional as F
import astra

from uqct.models.diffusion import load_unet as load_diffusion_unet
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from uqct.models.guided_diffusion import GradientGuidance, GuidedDiffusionPipeline
from uqct.metrics import psnr, rmse, ssim
import xarray as xr

from uqct.training.unet import N_BINS_HR
import datetime
import hashlib
import json

try:
    import git
except ImportError:
    git = None

class ObservationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, seeds: list[int], intensities: torch.Tensor, angles: torch.Tensor):
        self.dataset = dataset
        self.seeds = seeds
        self.intensities = intensities
        self.angles = angles
        self.device = intensities.device
        self._dataset_cache = dict()

    def __len__(self):
        return len(self.dataset) * len(self.seeds)

    def __getitem__(self, item):
        seed = self.seeds[item // len(self.dataset)]
        idx = item % len(self.dataset)
        return self.get(idx, seed)
    
    def get(self, idx: int, seed: int):
        if not idx in self._dataset_cache:
            self._dataset_cache[idx] = self.dataset[idx]
        image = self._dataset_cache[idx].to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed + 100000*int(idx))
        # TODO: divide by 2 is important!!!
        # It seems how it currently works
        # - we sample at half the target intensity
        # - pixels are binned from 256 to 128, but this does not change total intensity
        # - the model computes total intensity by assuming 128 dectector pixels, but intensity per observation is normalized by 256 bins, this is where the factor 2 comes from.
        # TLDR: The model effectively assumes a lower total intensity by a factor of 2, so we sample the observations by dividing total intensity by the same factor.
        # seems like not the right place to add this factor, because the effect intensity is always off by a factor 2 ...
        data = sample_observations(image, self.intensities / 2, self.angles, generator=generator)
        return idx, seed, image, data


def fbp_recon(counts, intensities, angles):
    """Simple FBP reconstruction from an Experiment object."""
    sinogram = sinogram_from_counts(counts, intensities)
    return fbp(sinogram, angles)


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
        device: torch.device = None
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
            sinogram = sinogram_from_counts(counts, intensities)
            prior = fbp(sinogram, angles)
        # elif self.init_method == "fbp_weighted":
        #     prior = fbp(counts, angles, intensities, weighted=True)
        elif self.init_method == "zeros":
            bs = counts.shape[0:-2]
            prior = torch.zeros((*bs, counts.shape[-1], counts.shape[-1]), device=counts.device)
        elif self.init_method == "const":
            bs = counts.shape[0:-2]
            prior = torch.ones((*bs, counts.shape[-1], counts.shape[-1]), device=counts.device) * 0.5
        elif self.init_method == "random":
            bs = counts.shape[0:-2]
            prior = torch.randn((*bs, counts.shape[-1], counts.shape[-1]), device=counts.device).clip(0, 1)
        elif self.init_method == "prior" and self._prior is not None:
            prior = self._prior.clone().to(counts.device)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        return prior

    def __call__(self, counts: torch.Tensor, intensities: torch.Tensor, angles: torch.Tensor, verbose=False) -> torch.Tensor:
        """
        Perform iterative reconstruction given counts, intensities, and angles.
        Returns reconstructed image tensor (B, H, W).
        """
        with torch.enable_grad():
            prior_img = self._build_prior(counts, angles, intensities)
            tomogram = Tomogram(prior=prior_img.detach(), use_sigmoid=self.use_sigmoid, circle=True)

            optimizer = torch.optim.Adam(tomogram.parameters(), lr=self.lr)
            # if self.loss == "nll":
            #     loss_fn = nll
            # else:
            #     # mse_ct must be available in scope if using this branch
            #     loss_fn = lambda recon, meas, angs, alloc: mse_ct(recon, meas, angs, alloc, vst=anscombe_transform)

            circle_mask = circular_mask(prior_img.shape[-1], device=tomogram.image.device)
            it = tqdm(range(self.steps), desc="Iterative Reconstruction", disable=not verbose)
            for step in it:
                optimizer.zero_grad()

                loss = nll(tomogram(), counts, intensities, angles).mean(dim=(-1, -2)).sum()

                if self.tv_weight:
                    _tv_loss = tv_loss(tomogram()).mean()
                    loss += self.tv_weight * _tv_loss

                loss.backward()
                optimizer.step()
    
                # circle mask and clamp
                with torch.no_grad():
                    tomogram.image.clamp_(min=0.0, max=1.0)
                    tomogram.image.mul_(circle_mask)

                it.set_postfix(loss=f"{loss.item():.10f}, tv={_tv_loss.item() if self.tv_weight else 0.0:.10f}")

        with torch.no_grad():
            recon = tomogram()

        return recon.detach()
    

from pathlib import Path
from uqct.training.unet import ( MIN_TOTAL_INTENSITY, MAX_TOTAL_INTENSITY, N_ANGLES, build_unet, sample_fbp_sparse)
from diffusers.models.unets.unet_2d import UNet2DModel


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
        sinogram = sinogram_from_counts(counts, intensities)
        fbp_recon = fbp(sinogram, angles)

        total_intensities = intensities.expand(counts.shape).sum(dim=(-1, -2))

        x_in = (fbp_recon * 2.0 - 1.0)
        exposure_norm = (
            (total_intensities - MIN_TOTAL_INTENSITY) / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY) * 999
        )


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
            indices = torch.randint(0, counts.shape[-2], (counts.shape[-2],), device=counts.device)
            counts_bs = counts.index_select(-2, indices)
            intensities_bs = intensities.index_select(-2, indices)
            angles_bs = angles.index_select(0, indices)
            recon_bs = self.reconstructor(counts_bs, intensities_bs, angles_bs)
            recons.append(recon_bs)

        return torch.stack(recons, dim=0)

def guidance_loss(counts, intensities, angles, l=5.):
    """
    Define a loss function for the diffusion model.
    This can be used to guide the diffusion process.
    """
    data_shape = counts.shape[:-2]
    circle_mask = circular_mask(counts.shape[-1], device=counts.device)
    def loss_fn(image):
        img_shape = image.shape[-2:]
        image = image.view(-1, *data_shape, *img_shape)
        image = ((image + 1.0)/2).clip(0, 1)
        image = image * circle_mask
        loss =  nll(image, counts, intensities, angles, l=l)
        return loss.sum()
    return loss_fn

class DiffusionRecon:

    def __init__(self, unet, scheduler, num_samples=5, num_inference_steps=100, timesteps=None, guidance_start=500, guidance_end=20, guidance_num_gradient_steps=10, guidance_lr=1e-1, guidance_lr_decay=False, guidance_loss=guidance_loss, seed=None):
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

    def __call__(self, counts, intensities, angles, num_samples=None, verbose=False, **guidance_loss_kwargs):
        if num_samples is None:
            num_samples = self.num_samples
        guided_diffusion = GuidedDiffusionPipeline(self.unet, self.scheduler)

        loss_fct = self.guidance_loss(counts, intensities, angles, **guidance_loss_kwargs)
        guidance = GradientGuidance(
            loss_fct=loss_fct, 
            num_gradient_steps=self.guidance_num_gradient_steps,
            guidance_start=self.guidance_start,
            guidance_end=self.guidance_end, 
            lr=self.guidance_lr,
            learning_rate_decay=self.guidance_lr_decay
        )

        samples = guided_diffusion(
            batch_size=len(counts) * num_samples,
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance=guidance,
            verbose=verbose,
            generator=self.generator
        )
        # circular mask
        mask = circular_mask(samples.shape[-1], device=samples.device)
        samples = samples * mask
        return samples.view(num_samples, *counts.shape[:-2], 128, 128)

from uqct.training.unet import norm_intensities
import xarray as xr
import collections
import pandas as pd

class CondDiffusionRecon:

    def __init__(self, unet, scheduler, num_samples=5, num_inference_steps=100, timesteps=None, guidance_start=500, guidance_end=20, guidance_num_gradient_steps=10, guidance_lr=1e-1, guidance_lr_decay=False, guidance_loss=guidance_loss, seed=None):
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

    def __call__(self, counts, intensities, angles, num_samples=None, verbose=False, **guidance_loss_kwargs):
        if num_samples is None:
            num_samples = self.num_samples

        guided_diffusion = GuidedDiffusionPipeline(self.unet, self.scheduler)

        image_shape = (len(counts) * num_samples, 1, self.unet.config.sample_size, self.unet.config.sample_size)

        loss_fct = self.guidance_loss(counts, intensities, angles, **guidance_loss_kwargs)
        guidance = GradientGuidance(
            loss_fct=loss_fct, 
            num_gradient_steps=self.guidance_num_gradient_steps,
            guidance_start=self.guidance_start,
            guidance_end=self.guidance_end, 
            lr=self.guidance_lr,
            learning_rate_decay=self.guidance_lr_decay
        )

        # compute fbps
        sinogram = sinogram_from_counts(counts, intensities)
        fbps = fbp(sinogram, angles)
        intensities = intensities.expand(counts.shape).sum(dim=(-1, -2)).squeeze(0)

        # print(fbps.shape, counts.shape, intensities.shape)
        fbps_norm = (fbps * 2.0 - 1.0)
        fbps_norm = fbps_norm.unsqueeze(0).expand(num_samples, -1, -1, -1, -1).reshape(-1, 1, fbps.shape[-2], fbps.shape[-1])
        # print(fbps_norm.shape)
        

        # compute intensity

        intensities_norm = (2 * ((norm_intensities(intensities) / 999) - 0.5)).clip(
            -1, 1
        )
        intensities_norm = intensities_norm.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1)

        n_angles = torch.tensor(len(angles), device=angles.device).float()
        n_angles_norm = ((n_angles - N_ANGLES / 2) / (N_ANGLES / 2)).clip(-1, 1)

        n_angles_norm = n_angles_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(num_samples, len(counts), 1).reshape(-1)

        cond_kwargs = dict(
            fbp=fbps_norm,
            intensity_norm=intensities_norm,
            n_angles_norm=n_angles_norm
        )


        samples = guided_diffusion(
            batch_size=len(counts) * num_samples,
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance=guidance,
            verbose=verbose,
            cond_kwargs=cond_kwargs,
            image_shape=image_shape,
            generator=self.generator
        )

        # circular mask
        mask = circular_mask(samples.shape[-1], device=samples.device)
        samples = samples * mask

        return samples.view(num_samples, *counts.shape[:-2], 128, 128)


def schedule_uniform(total_intensity, n_steps, init_fraction=None, device=None):
    if init_fraction:
        alloc = [init_fraction]
    else:
        init_fraction = 0.
        alloc = []
        n_steps += 1
    per_step_fraction = (1. - init_fraction) / n_steps
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
    intensities = torch.diff(
        torch.cat([torch.zeros(1, device=device), cumsum])
    )

    return intensities


def sample_observations_reproducible(
    images: torch.Tensor,
    indices: torch.Tensor,
    schedule: torch.Tensor,
    angles: torch.Tensor,
    seed_offset: int = 0):
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
        generator = torch.Generator(device=device).manual_seed(seed_offset + 100000*int(idx))
        _data = sample_observations(image.unsqueeze(0), schedule / 2, angles, generator=generator) # TODO: divide by 2 is important!!!
        data.append(_data)
    return torch.cat(data, dim=0)

import math

def nll_mixture(
    images: torch.Tensor,
    counts: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    l: int = 5,
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
    nlls = -nll(images, counts, intensities, angles, l).double()
    nlls = nlls.sum((-1, -2))  # (n+pred, ...)
    nlls -= math.log(n_pred)
    mix = -torch.logsumexp(nlls, dim=0)  # (...)
    return mix.float()

import torchvision.transforms.functional as TF

def rotate_images(images, degree):
    return torch.vmap(lambda img: TF.rotate(img, degree))(images)

def experiment_id(params):
    """Create a unique ID string from experiment parameters."""
    relevant = {k: v for k, v in params.items() if v is not None}
    id_str = json.dumps(relevant, sort_keys=True)
    return hashlib.md5(id_str.encode()).hexdigest()[:8]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image quality metrics.")
    parser.add_argument("--dataset", type=str, choices=["composite", "lamino", "lung"], default="lung", help="Dataset to use for evaluation.")
    parser.add_argument("--model", type=str, choices=["fbp", "unet", "cond_diffusion", "diffusion", "gt", "unet_ensemble", "mle", "map"], default="fbp", help="Model to use for evaluation.")
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    # parser.add_argument("--batch_steps", action="store_true", default=True, help="Whether to batch over observation steps.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results.", required=True)
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to evaluate.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate per image.")
    parser.add_argument("--total_intensity", type=float, default=1e8, help="Total intensity for observation schedule.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps in the observation schedule.")
    parser.add_argument("--schedule", type=str, choices=["uniform", "exponential"], default="uniform", help="Type of observation schedule.")
    parser.add_argument("--init_fraction", type=float, default=None, help="Initial fraction of total intensity in the schedule.")
    parser.add_argument("--initial_intensity", type=float, default=1e6, help="Initial intensity for exponential schedule.")
    parser.add_argument("--rotation", type=float, default=None)
    parser.add_argument("--iterative_num_gradient_steps", type=int, default=100, help="Number of gradient steps for iterative reconstruction.")
    parser.add_argument("--iterative_lr", type=float, default=1e-1, help="Learning rate for iterative reconstruction.")
    parser.add_argument("--diffusion_num_inference_steps", type=int, default=100, help="Number of inference steps for diffusion models.")
    parser.add_argument("--diffusion_seed", type=int, default=0, help="Random seed for diffusion sampling.")
    parser.add_argument("--guidance_num_gradient_steps", type=int, default=10, help="Number of gradient steps for diffusion guidance.")
    parser.add_argument("--guidance_lr", type=float, default=1e-3, help="Learning rate for diffusion guidance.")
    parser.add_argument("--guidance_lr_decay", action='store_true', help="Whether to use learning rate decay for diffusion guidance.")
    parser.add_argument("--guidance_end", type=int, default=10, help="Guidance end timestep for diffusion models.")
    parser.add_argument("--num_bootstrap_samples", type=int, default=None, help="Number of bootstrap samples for uncertainty estimation.")
    parser.add_argument("--existing", default=None, choices=["skip", "overwrite"], 
    help="Behavior if output file exists.")
    parser.add_argument("--tv_weight", type=float, default=1e3, help="Total Variation weight for iterative reconstruction.")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}, Model: {args.model}")

    mixing_models = ["diffusion", "cond_diffusion", "unet_ensemble"]

     # List of argument names to include as metadata
    excluded_keys = ['existing']

    # Collect selected parameters
    experiment_params = {k: getattr(args, k, None) for k, v in vars(args).items() if k not in excluded_keys}

    output_file_name = experiment_id(experiment_params)
    # Add extra metadata
    experiment_params["experiment_id"] = output_file_name
    experiment_params["datetime"] = datetime.datetime.now().isoformat()

    
    # check git state and warn if not clean, ask for confirmation to proceed
    if git is None:
        print("gitpython not installed, skipping git state check.")
    else:
        # check for uncommitted changes excluding notebooks
        repo = git.Repo(search_parent_directories=True)
        experiment_params['git_commit'] = repo.head.object.hexsha
        experiment_params['git_branch'] = repo.active_branch.name
        experiment_params['git_diff'] = repo.git.diff('--', '.', ':(exclude)notebooks')

        # Combined dirty check
        print(f"Git commit: {experiment_params['git_commit']}")
        print(f"Git branch: {experiment_params['git_branch']}")
        print(f"Git diff: {experiment_params['git_diff']}")

    # Output dir
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory at {args.output_dir}")
        os.makedirs(args.output_dir)

    # File name and existence/overwrite check
    model_name = args.model
    if args.model == 'gt' and args.rotation is not None:
        model_name += f"_rot{args.rotation}"

    if args.num_bootstrap_samples is not None:
        model_name += f"_bootstrap"
    # output_file = Path(args.output_dir) / f"eval_{args.dataset}_{model_name}.nc"
    output_file = os.path.join(args.output_dir, output_file_name + ".nc")
    if os.path.exists(output_file):
        if args.existing == "skip":
            print(f"Results file {output_file} exists. Skipping evaluation as per --skip flag.")
            exit(0)
        elif args.existing == "overwrite":
            print(f"Results file {output_file} exists. Overwriting as per --overwrite flag.")
            os.remove(output_file)
        else:
            resp = input(f"File {output_file} exists. Delete and start over? [y/N]: ").strip().lower()
            if resp == "y":
                os.remove(output_file)
            else:
                print("Exiting without overwriting existing results.")
                exit(0)


    # get dataset
    _ , test_set = get_dataset(args.dataset, True)
    if args.num_images is not None:
        test_set = torch.utils.data.Subset(test_set, list(range(args.num_images)))
 
    # set up observation parameters
    num_angles = N_ANGLES
    angles = torch.from_numpy(np.linspace(0, 180, num_angles, endpoint=False)).float().to(device)
    if args.schedule == "uniform":
        schedule = schedule_uniform(
            args.total_intensity, args.num_steps, init_fraction=args.init_fraction, device=device
        )
    elif args.schedule == "exponential":
        schedule = schedule_exponential(
            args.total_intensity, args.num_steps, initial_intensity=args.initial_intensity, device=device
        )
    total_intensities = schedule.clone()
    schedule = schedule.reshape(-1, 1, 1, 1).expand(-1, 1, num_angles, 1) / num_angles / N_BINS_HR
    
    # set up observation dataset and dataloader
    obs_dataset = ObservationDataset(test_set, seeds=args.seeds, intensities=schedule, angles=angles)
    obs_dataloader = torch.utils.data.DataLoader(
        obs_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # load model
    if args.model == "gt":
        recon = None
    elif args.model == "fbp":
        recon = fbp_recon
    elif args.model == "mle":
        recon = IterativeRecon(
            steps=args.iterative_num_gradient_steps,
            init_method="zeros",
            use_sigmoid=False,
            lr=args.iterative_lr,
            loss="nll",
            device=device
        )
    elif args.model == "map":
        recon = IterativeRecon(
            steps=args.iterative_num_gradient_steps,
            init_method="zeros",
            use_sigmoid=False,
            lr=args.iterative_lr,
            loss="nll",
            tv_weight=args.tv_weight,
            device=device
        )
    elif args.model == "unet":
        ckpt_path = Path(f'/mydata/chip/shared/checkpoints/uqct/unet_dense/unet_dense_128_{args.dataset}_0.pt')
        print(f"Loading UNet checkpoint from {ckpt_path}")
        unet = load_unet(ckpt_path, sparse=False).to(device).eval()
        recon = UNetRecon(unet)
    elif args.model == "unet_ensemble":
        unets = []
        for i in range(10):
            ckpt_path = Path(f'/mydata/chip/shared/checkpoints/uqct/unet_dense/unet_dense_128_{args.dataset}_{i}.pt')
            print(f"Loading UNet checkpoint from {ckpt_path}")
            unet = load_unet(ckpt_path, sparse=False).to(device).eval()
            unets.append(unet)
        recon = UnetEnsembleRecon(unets)
    elif args.model == "diffusion":
        ckpt_path = Path(f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_unconditional_128_{args.dataset}.pt")
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
            seed=args.diffusion_seed if args.diffusion_seed != -1 else None
        )
    elif args.model == "cond_diffusion":
        ckpt_path = Path(f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_conditional_128_{args.dataset}.pt")
        unet = load_diffusion_unet(ckpt_path, cond=True)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
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
            seed=args.diffusion_seed if args.diffusion_seed != -1 else None
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    

    if args.num_bootstrap_samples is not None and args.model in ["mle", "unet", "fbp"]:
        print(f"Wrapping reconstructor with BootstrapRecon using {args.num_bootstrap_samples} samples.")
        recon = BootstrapRecon(recon, num_samples=args.num_bootstrap_samples)
    
    # Prepare to collect batched results
    all_psnr = []
    all_rmse = []
    all_ssim = []
    all_index = []
    all_seed = []
    all_seq_nll = []
    all_nll = []
    all_seq_nll_mix = []

    for indices, seed, images, data in tqdm(obs_dataloader):
        images = images.to(device)
        images_lr = F.interpolate(images, size=(128, 128), mode="area")
        data = data.to(device)

        # Shape conventions:
        # Images (Batch, Step, 1, H, W)
        # Sampled Predictions (Sample, Batch, Step, 1, H, W) or (Batch, Step, 1, H, W) for non-mixing models
        if args.model == "gt":
            if args.rotation is not None:
                images_rotated = rotate_images(images, args.rotation)
                pred = F.interpolate(images_rotated, size=(128, 128), mode="area").unsqueeze(1).expand(-1, schedule.shape[0], -1, -1, -1).contiguous()
            else:
                pred = images_lr.clone().unsqueeze(1).expand(-1, schedule.shape[0], -1, -1, -1).contiguous()
        else:
            data_cumsum = data.cumsum(dim=1)
            schedule_cumsum = schedule.unsqueeze(0).cumsum(dim=1)
            with torch.no_grad():
                pred = [recon(data_cumsum[:, i], schedule_cumsum[:, i], angles) for i in range(data.shape[1])]
            pred = torch.stack(pred, dim=-4)

        if args.model in mixing_models:
            _seq_nll_mix = nll_mixture(
                pred[:, :, :-1].contiguous(), data[:, 1:], schedule[1:], angles
            ).squeeze(-1)
            all_seq_nll_mix.append(_seq_nll_mix.detach().cpu())
            pred = pred.mean(dim=0)  # average over samples for remaining metrics

        _rmse = rmse(pred, images_lr.unsqueeze(1), circle_mask=True).squeeze(-1)
        _psnr = psnr(pred, images_lr.unsqueeze(1), circle_mask=True, data_range=1.0).squeeze(-1)
        _ssim = ssim(pred, images_lr.unsqueeze(1), circle_mask=True, data_range=1.0).squeeze(-1)
        _seq_nll = nll(pred[:, :-1].contiguous(), data[:, 1:], schedule[1:], angles).sum((-1,-2)).squeeze(-1)
        _nll = nll(pred.unsqueeze(2), data.unsqueeze(1), schedule, angles).sum((-1,-2, -3)).cumsum(dim=2).diagonal(dim1=1, dim2=2) 

        all_psnr.append(_psnr.detach().cpu())
        all_rmse.append(_rmse.detach().cpu())
        all_ssim.append(_ssim.detach().cpu())
        all_index.append(torch.as_tensor(indices).view(-1).cpu())
        all_seed.append(torch.as_tensor(seed).view(-1).cpu())
        
        all_seq_nll.append(_seq_nll.detach().cpu())
        all_nll.append(_nll.detach().cpu())

    # stack all batches -> (N_samples, N_steps)
    psnr_tensor = torch.cat(all_psnr, dim=0)   # (N_samples, N_steps)
    rmse_tensor = torch.cat(all_rmse, dim=0)   # (N_samples, N_steps)
    ssim_tensor = torch.cat(all_ssim, dim=0)   # (N_samples, N_steps)
    index_tensor = torch.cat(all_index, dim=0) # (N_samples,)
    seed_tensor = torch.cat(all_seed, dim=0)   # (N_samples,)
    N_samples, N_steps = psnr_tensor.shape
    seq_nll_tensor = torch.cat(all_seq_nll, dim=0)     # (N_samples - 1, N_steps)
    nll_tensor = torch.cat(all_nll, dim=0)         # (N_samples, N_steps)

    # Insert NaN values in front to match shape (N_samples, N_steps)
    seq_nll_tensor = torch.cat([torch.full((N_samples, 1), 0.), seq_nll_tensor], dim=1)

    data_vars = {
            "psnr": (["sample", "step"], psnr_tensor.numpy()),
            "rmse": (["sample", "step"], rmse_tensor.numpy()),
            "ssim": (["sample", "step"], ssim_tensor.numpy()),
            "nll": (["sample", "step"], nll_tensor.numpy()),
            "seq_nll": (["sample", "step"], seq_nll_tensor.numpy()),
    }

    # First build a sample/step dataset, then reshape to (dataset, index, seed, step, model)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": np.arange(N_samples),
            "step": np.arange(N_steps),
            "index": ("sample", index_tensor.numpy()),
            "seed": ("sample", seed_tensor.numpy()),
        },
    )
    # Turn (sample) into a MultiIndex of (index, seed) and unstack -> dims (index, seed, step)
    ds = ds.set_index(sample=("index", "seed")).unstack("sample")
    ds = ds.expand_dims(dataset=[args.dataset], model=[model_name])
    ds = ds.transpose("dataset", "index", "seed", "step", "model")

    # Optional: add mixture NLL as a separate "model" entry (model=f"{args.model}_mix")
    if all_seq_nll_mix:
        seq_nll_mix_tensor = torch.cat(all_seq_nll_mix, dim=0)  # expected (N_samples, N_steps-1)
        seq_nll_mix_tensor = torch.cat(
            [torch.full((N_samples, 1), 0., dtype=seq_nll_mix_tensor.dtype), seq_nll_mix_tensor],
            dim=1,
        )

        ds_mix = xr.Dataset(
            data_vars={"seq_nll": (["sample", "step"], seq_nll_mix_tensor.numpy())},
            coords={
                "sample": np.arange(N_samples),
                "step": np.arange(N_steps),
                "index": ("sample", index_tensor.numpy()),
                "seed": ("sample", seed_tensor.numpy()),
            },
        )
        ds_mix = ds_mix.set_index(sample=("index", "seed")).unstack("sample")
        ds_mix = ds_mix.expand_dims(dataset=[args.dataset], model=[f"{args.model}_mix"])
        ds_mix = ds_mix.transpose("dataset", "index", "seed", "step", "model")

        ds = xr.concat([ds, ds_mix], dim="model")

    # Add schedule intensity as value, indexed by step
    schedule_values = schedule.reshape(-1).cpu().numpy()
    ds["intensity"] = ("step", total_intensities.cpu().numpy())

    # Attach to xarray Dataset
    ds.attrs.update({k : str(v) for k, v in experiment_params.items()})

    ds.to_netcdf(output_file)
    print(f"Results written to {output_file}")
