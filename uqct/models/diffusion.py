from pathlib import Path
from typing import Callable, Literal
import math

import click
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from torch import optim
from tqdm.auto import tqdm

from uqct.ct import Experiment, apply_circular_mask, circular_mask, nll
from uqct.debugging import plot_img
from uqct.models.unet import FBPUNet
from uqct.training.diffusion import UNet2DModelAux
from uqct.training.unet import N_ANGLES, norm_intensities

DatasetName = Literal["lung", "composite", "lamino"]


class Diffusion:
    def __init__(
        self,
        dataset: DatasetName,
        num_steps: int = 50,
        sgd_steps: int = 100,
        lr: float | None = None,
        batch_size: int = 64,
        cond: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.cond = cond

        # U-Net
        self.batch_size = batch_size
        ckpt_path = find_ckpt(dataset, self.cond)
        self.unet = load_unet(ckpt_path, cond)
        self.unet.eval()
        for param in self.unet.parameters():  # type: ignore
            param.requires_grad = False

        # Diffusion
        self.num_steps = num_steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.noise_scheduler.set_timesteps(num_inference_steps=num_steps)

        # Guidance
        self.sgd_steps = sgd_steps
        self.lr = lr

    def set_lr_from_experiment(self, experiment: Experiment) -> float:
        """Computes and returns a best guess of the optimal learning rate.
        The internal `lr` parameter gets set to this value."""
        sparse2coef = {True: (-0.1655, 0.0188), False: {-0.0786, 0.01}}
        beta_0, beta_1 = sparse2coef[experiment.sparse]
        total_intensity = experiment.intensities.view(
            math.prod(experiment.batch_dims), -1, 1
        )[0].sum()
        total_intensity *= experiment.counts.shape[-1]
        self.lr = beta_1 * math.log(total_intensity) + beta_0
        return self.lr

    def predict_noise_cond(
        self,
        t: int,
        x_t: torch.Tensor,
        fbps_norm: torch.Tensor,
        intensities_norm: torch.Tensor,
        n_angles_norm: torch.Tensor,
    ) -> torch.Tensor:
        device = x_t.device

        noise_preds = []
        in_shape = x_t.shape
        hw = x_t.shape[-2:]
        x_t_flat = x_t.view(-1, 1, *hw)
        fbps_norm = fbps_norm.reshape(-1, 1, *hw)
        intensities_norm = intensities_norm.flatten()
        n_angles_norm = n_angles_norm.flatten()
        timesteps = torch.LongTensor([t]).expand(len(x_t_flat)).to(device)

        # Split into batches of size <= self.batch_size
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                for batch_idx in range(0, len(x_t_flat), self.batch_size):
                    x_t_b = x_t_flat[batch_idx : batch_idx + self.batch_size]
                    fbps_b = fbps_norm[batch_idx : batch_idx + self.batch_size]
                    timesteps_b = timesteps[batch_idx : batch_idx + self.batch_size]
                    intensities_norm_b = intensities_norm[
                        batch_idx : batch_idx + self.batch_size
                    ]
                    n_angles_norm_b = n_angles_norm[
                        batch_idx : batch_idx + self.batch_size
                    ]
                    noise_pred = self.unet(
                        x_t_b,
                        fbps_b,
                        timesteps_b,
                        intensities_norm_b,
                        n_angles_norm_b,
                    )
                    noise_preds.append(noise_pred)
        noise_pred = torch.cat(noise_preds, dim=0).view(in_shape)
        return noise_pred

    def predict_noise(
        self,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        device = x_t.device
        timesteps = torch.LongTensor([t]).to(device)

        noise_preds = []
        in_shape = x_t.shape
        x_t_flat = x_t.view(-1, 1, x_t.shape[-1], x_t.shape[-1])

        # Split into batches of size <= self.batch_size
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                for batch in x_t_flat.split(self.batch_size):
                    noise_pred = self.unet(
                        batch,
                        timesteps,
                        return_dict=False,
                    )[0]
                    noise_preds.append(noise_pred)
        noise_pred = torch.cat(noise_preds, dim=0).view(in_shape)
        return noise_pred

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        guidance_loss_fn: Callable | None = None,
        optimizer: optim.Adam | None = None,
        generator=None,
    ) -> DDPMSchedulerOutput:
        """
        This is the DDPM reference implementation of the reverse diffusion step.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`]
        """
        t = timestep

        prev_t = self.noise_scheduler.previous_timestep(t)

        predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.noise_scheduler.alphas_cumprod[prev_t]
            if prev_t >= 0
            else self.noise_scheduler.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)
        mask = circular_mask(
            pred_original_sample.shape[-1],
            device=pred_original_sample.device,
            dtype=torch.bool,
        )
        pred_original_sample[..., ~mask] = -1.0

        if guidance_loss_fn is not None:
            pred_original_sample = guide(
                pred_original_sample,
                guidance_loss_fn,
                sgd_steps=self.sgd_steps,
                lr=self.lr * (timestep / 1000),
                optimizer=optimizer,
                verbose=False,
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (
                self.noise_scheduler._get_variance(  # type: ignore
                    t, predicted_variance=predicted_variance
                )
                ** 0.5
            ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )

    def reverse(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, disable=not self.verbose):
                noise_pred, _, _ = self.predict_noise(t, image)
                image = self.step(noise_pred, t, image).prev_sample
            return denorm_image(image)

    def sample(
        self,
        experiment: Experiment,
        replicates: int = 10,
        schedule: torch.Tensor | None = None,
        guidance_loss_fn: Callable | None = None,
    ) -> torch.Tensor:
        """
        Returns
            torch.Tensor: `(..., n_angles, replicates, 1, side_length, side_length)` (sparse) or `(..., n_rounds, replicates, 1, side_length, side_length)` (dense).
        """
        if self.lr is None:
            self.set_lr_from_experiment(experiment)

        side_length = experiment.counts.shape[-1]
        if experiment.sparse:
            # (rep, ..., n_angles, 1, side_length, side_length)
            n_angles_schedule = (
                len(schedule) if schedule is not None else experiment.counts.shape[-2]
            )
            x_t = torch.randn(
                replicates,
                *experiment.batch_dims,
                n_angles_schedule,
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

        it = tqdm(self.noise_scheduler.timesteps, disable=not self.verbose)

        fbps, intensities, class_labels = FBPUNet._prepare_inputs_from_experiment(
            experiment, schedule
        )
        fbps = fbps.to(self.device)
        intensities = intensities.to(self.device)
        if class_labels is None:  # Dense
            n_angles = torch.full(experiment.batch_dims, N_ANGLES, device=self.device)
            fbps = fbps.squeeze(-3)
        else:  # Sparse
            class_labels = class_labels.to(self.device)
            n_angles = class_labels + 1
        fbps_norm = ((fbps - 0.5) * 2).expand_as(x_t)
        intensities_norm = (2 * ((norm_intensities(intensities) / 999) - 0.5)).clip(
            -1, 1
        )
        n_angles_norm = ((n_angles - N_ANGLES / 2) / (N_ANGLES / 2)).clip(-1, 1)
        intensities_norm = intensities_norm.view(1, *intensities_norm.shape).expand(
            replicates, *(-1 for _ in range(intensities_norm.ndim))
        )
        n_angles_norm = n_angles_norm.view(1, *n_angles_norm.shape).expand(
            replicates, *(-1 for _ in range(n_angles_norm.ndim))
        )
        for t in it:
            if self.cond:
                noise_pred = self.predict_noise_cond(
                    t, x_t, fbps_norm, intensities_norm, n_angles_norm
                )
            else:
                noise_pred = self.predict_noise(t, x_t)

            guidance_loss_fn_ = guidance_loss_fn if (20 < t < 1000) else None
            x_t = self.step(noise_pred, t, x_t, guidance_loss_fn_).prev_sample
        out = denorm_image(x_t).reshape(rep_first_shape)
        out = apply_circular_mask(out)

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


def find_ckpt(dataset: DatasetName, cond: bool) -> Path:
    filename = (
        f"ddpm_conditional_128_{dataset}.pt"
        if cond
        else f"ddpm_unconditional_128_{dataset}.pt"
    )
    ckpt_dir_candidates = [
        Path(x) / filename
        for x in (
            "/mydata/chip/shared/checkpoints/uqct/diffusion",
            "checkpoints/diffusion",
            "../checkpoints/diffusion",
        )
    ]

    for ckpt_path in ckpt_dir_candidates:
        if ckpt_path.exists():
            return ckpt_path
    raise ValueError(f"Could not find diffusion checkpoint for dataset {dataset}")


def load_unet(
    ckpt_path: Path, cond: bool, verbose: bool = False
) -> UNet2DModel | UNet2DModelAux:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    channels = (128, 128, 256, 256, 512, 512)
    if cond:
        unet = UNet2DModelAux(2, 0.0, device)
    else:
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
    if verbose:
        print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']}")
    return unet


def get_guidance_loss_fn(
    experiment: Experiment, schedule: torch.Tensor | None = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a loss function that takes as input a batch of images and returns the Poisson NLL loss."""
    if experiment.sparse:
        n_angles = experiment.counts.shape[-2]
        device = experiment.counts.device
        if schedule is None:
            schedule = torch.arange(1, n_angles + 1, device=device)
        mask = torch.arange(n_angles, device=device).expand(
            len(schedule), -1
        ) < schedule.to(device).unsqueeze(1)

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            nlls = nll(
                image,
                experiment.counts.unsqueeze(-3),
                experiment.intensities.unsqueeze(-3),
                experiment.angles,
            )
            return nlls[..., mask, :].mean((-2, -1)).sum()

    else:
        assert schedule is None, (
            "Schedules are currently unsupported for the dense setting."
        )
        counts_csum = experiment.counts.cumsum(-3).unsqueeze(0)
        intensities_csum = experiment.intensities.cumsum(-3).unsqueeze(0)

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            nlls = nll(image, counts_csum, intensities_csum, experiment.angles)
            return nlls.mean((-2, -1)).sum()

    return loss_fn


def guide(
    x_t: torch.Tensor,
    loss_fct: Callable[..., torch.Tensor],
    sgd_steps: int = 50,
    lr: float = 0.1,
    optimizer: None | optim.Adam = None,
    verbose: bool = False,
) -> torch.Tensor:
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
            y.data[..., ~mask] = 0.0

        it.set_postfix({"loss": f"{loss.item():.3f}"})
    optimizer.zero_grad(set_to_none=True)
    x_t_guided = norm_image(y).clip(-1, 1)
    return x_t_guided


def norm_image(image: torch.Tensor) -> torch.Tensor:
    return (image - 0.5) * 2


def denorm_image(
    image: torch.Tensor, min_v: float = 0.0, max_v: float = 1.0
) -> torch.Tensor:
    return ((image + 1) / 2).clip(min_v, max_v)


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
@click.option(
    "--cond",
    default=False,
    type=bool,
    help="Whether to use a conditional diffusion model",
)
@click.option("--total-intensity", default=1e7, type=float, help="Total intensity")
def main(dataset: DatasetName, sparse: bool, cond: bool, total_intensity):
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

    train_set, test_set = get_dataset(dataset, True)
    n_gt = min(2, len(test_set))
    gt = torch.stack([test_set[i] for i in range(n_gt)], dim=0).to(device)

    n_angles = 200

    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)
    n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)
    if sparse:
        intensities = intensities.view(1, 1, 1, 1).expand(n_gt, -1, n_angles, -1) / (
            n_angles * n_detectors_hr
        )
        schedule = torch.tensor([1, 25, 50, 75, 100, 125, 150, 175, 200])
    else:
        n_rounds = 1
        intensities = intensities.view(1, 1, 1, 1).expand(
            n_gt, n_rounds, n_angles, -1
        ) / (n_angles * n_detectors_hr * n_rounds)
        schedule = None
    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts, intensities_lr, angles, sparse)
    diffusion = Diffusion(
        dataset,
        num_steps=100,
        sgd_steps=10,
        batch_size=16,
        cond=cond,
        verbose=True,
    )

    guidance_loss_fn = get_guidance_loss_fn(experiment, schedule)
    sample = diffusion.sample(experiment, 1, schedule, guidance_loss_fn)
    # sample = diffusion.reverse(torch.randn(5, 1, 128, 128, device=device))
    print(sample)
    gt_lr = F.interpolate(gt, (128, 128), mode="area")
    plot_img(*gt_lr, *sample.reshape(-1, 128, 128), name="diffusion", share_range=True)


if __name__ == "__main__":
    main()
