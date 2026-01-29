from typing import Literal

import click
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm


class GradientGuidance:
    """
    Applies gradient-based guidance to an image tensor during diffusion model sampling.
    """

    def __init__(
        self,
        loss_fct,
        num_gradient_steps=10,
        guidance_start=1000,
        guidance_end=0,
        lr=1e-1,
    ):
        """
        Initialize the GradientGuidance class.

        Args:
            loss_fct (callable): Loss function used for guidance optimization.
            num_gradient_steps (int, optional): Number of gradient descent steps per guidance application.
            guidance_start (int, optional): Timestep to begin applying guidance.
            guidance_end (int, optional): Timestep to stop applying guidance.
            lr (float, optional): Learning rate for gradient optimization.
        """
        self.loss_fct = loss_fct
        self.num_gradient_steps = num_gradient_steps
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.lr = lr

    def __call__(self, pred_original_sample, t):
        """
        Apply gradient-based guidance to the predicted original sample.
        Args:
            pred_original_sample (torch.Tensor): The predicted original sample from the diffusion model.
            t (int): Current timestep in the diffusion process.
        Returns:
            torch.Tensor: The guided original sample after applying gradient-based optimization.
        """
        if t < self.guidance_end or t > self.guidance_start:
            return pred_original_sample
        with torch.enable_grad():
            image = torch.nn.Parameter(pred_original_sample.detach().clone())
            optimizer = torch.optim.Adam([image], lr=self.lr)

            for _ in range(self.num_gradient_steps):
                optimizer.zero_grad()
                loss = self.loss_fct(image)
                loss.backward()
                optimizer.step()
        return image.detach()


class GuidedDiffusionPipeline:
    """
    A pipeline for guided diffusion image generation using a UNet and scheduler.
    Follows the DDPM reference implementation.
    """

    def __init__(self, unet, scheduler: DDPMScheduler):
        """
        Initializes the GuidedDiffusion model with the specified UNet and scheduler.
        Args:
            unet: The UNet model used for image generation or processing.
            scheduler: The scheduler that manages the diffusion process steps.
        """

        self.unet = unet
        self.scheduler = scheduler

    def __call__(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        num_inference_steps: int = 1000,
        guidance=None,
        verbose=False,
    ):
        """
        Generates images using the guided diffusion process.
        Args:
            batch_size (int, optional): Number of images to generate in a batch.
            generator (Optional[torch.Generator], optional): PyTorch random number generator for reproducibility.
            num_inference_steps (int, optional): Number of diffusion steps to perform during inference.
            guidance (optional): Optional guidance information to condition the diffusion process.
            verbose (bool, optional): If True, displays a progress bar during inference.
        Returns:
            torch.Tensor: Generated images as a tensor with values scaled to [0, 1].
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        device = self.unet.device
        if device.type == "mps":
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(device)
        else:
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, device=device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps, disable=not verbose):
            # 1. predict noise model_output
            with torch.no_grad():
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.step(
                model_output, t, image, guidance=guidance, generator=generator
            ).prev_sample  # type: ignore

        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        guidance: GradientGuidance | None = None,
        generator=None,
        return_dict: bool = True,
    ) -> DDPMSchedulerOutput | tuple:
        """
        This is the DDPM reference implementation of the reverse diffusion step, with an additional guidance step.

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
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.scheduler.previous_timestep(t)

        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and self.scheduler.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.scheduler.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)  # type: ignore
        elif self.scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.scheduler.config.clip_sample_range,
                self.scheduler.config.clip_sample_range,
            )

        # extra step: guidance
        if guidance is not None:
            pred_original_sample = guidance(pred_original_sample, t)

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
            if self.scheduler.variance_type == "fixed_small_log":
                variance = (
                    self.scheduler._get_variance(
                        t, predicted_variance=predicted_variance
                    )
                    * variance_noise
                )
            elif self.scheduler.variance_type == "learned_range":
                variance = self.scheduler._get_variance(
                    t, predicted_variance=predicted_variance
                )
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (
                    self.scheduler._get_variance(
                        t, predicted_variance=predicted_variance
                    )
                    ** 0.5
                ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )


@click.command()
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lung", "composite", "lamino"]),
    help="Which dataset to generate samples for",
)
def main(dataset: Literal["lung", "composite", "lamino"]):
    import numpy as np

    from uqct.ct import nll, sample_observations
    from uqct.datasets.utils import get_dataset
    from uqct.debugging import plot_img
    from uqct.models.diffusion import find_ckpt, load_unet

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

    n_angles = 180
    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)
    total_intensity = 1e5
    # n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)

    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2

    def guidance_loss(counts, intensities, angles, length_scale=5.0, circle=True):
        """
        Define a loss function for the diffusion model.
        This can be used to guide the diffusion process.
        """
        data_shape = counts.shape[:-2]
        img_shape = (counts.shape[-1], counts.shape[-1])

        if circle:
            # generate circle mask
            circle_mask = torch.ones(*img_shape, device=counts.device)
            radius = img_shape[-1] // 2
            y, x = torch.meshgrid(
                torch.arange(img_shape[-2], device=counts.device),
                torch.arange(img_shape[-1], device=counts.device),
                indexing="ij",
            )
            mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
            circle_mask[~mask] = 0

        def loss_fn(image):
            img_shape = image.shape[-2:]
            image = image.view(-1, *data_shape, *img_shape)
            image = ((image + 1.0) / 2).clip(0, 1)
            if circle:
                image = image * circle_mask
            loss = nll(image, counts, intensities, angles, length_scale=length_scale)
            return loss.sum()

        return loss_fn

    ckpt_path = find_ckpt(dataset, cond=False)
    print(f"Loading unet from {ckpt_path}")
    unet = load_unet(ckpt_path, cond=False)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    guided_diffusion = GuidedDiffusionPipeline(unet, scheduler)

    loss_fct = guidance_loss(counts, intensities_lr, angles)
    guidance = GradientGuidance(
        loss_fct=loss_fct,
        num_gradient_steps=5,
        guidance_start=1000,
        guidance_end=20,
        lr=1e-1,
    )

    num_samples = 3

    print("Generating samples...")
    samples = guided_diffusion(
        batch_size=n_gt * num_samples,
        num_inference_steps=100,
        guidance=guidance,
        verbose=True,
    )
    samples = samples.view(num_samples, *counts.shape[:-2], 128, 128)

    print("Plotting results...")
    plot_img(*gt, *samples.reshape(-1, 128, 128), share_range=True)


if __name__ == "__main__":
    main()
