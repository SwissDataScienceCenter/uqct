import os
import warnings
from pathlib import Path
from typing import Callable, Literal
import einops

import numpy as np
import onnxruntime as ort
import tensorrt
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from torch import optim
from tqdm.auto import tqdm

from uqct.ct import (
    Experiment,
    apply_circular_mask,
    circular_mask,
    lr_from_experiment,
    nll,
    prepare_inputs_from_experiment,
)
from uqct.debugging import plot_img
from uqct.training.diffusion import UNet2DModelAux
from uqct.training.unet import N_ANGLES, norm_intensities
from uqct.utils import get_checkpoint_dir, get_hardware_specific_engine_path

DatasetName = Literal["lung", "composite", "lamino"]


class Diffusion:
    def __init__(
        self,
        dataset: DatasetName,
        num_steps: int = 100,
        gradient_steps: int = 1,
        lr: float | None = None,
        batch_size: int = 32,
        cond: bool = False,
        onnx: bool = False,
        verbose: bool = False,
        anneal_lr: bool = True,
    ):
        assert (
            not onnx or cond
        ), "ONNX-based inference is not supported for unconditional models."

        self.verbose = verbose
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.cond = cond
        self.anneal_lr = anneal_lr

        # U-Net
        self.batch_size = batch_size

        if onnx:
            onnx_fp = (
                get_checkpoint_dir()
                / "diffusion"
                / f"ddpm_conditional_128_{dataset}.onnx"
            )
            try:
                trt_lib_path = os.path.join(
                    os.path.dirname(tensorrt.__file__) + "_libs"
                )
                os.environ["LD_LIBRARY_PATH"] = (
                    trt_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                )
            except Exception as e:
                warnings.warn(str(e))

            min_shapes = "x_t:1x1x128x128,fbps:1x1x128x128,timesteps:1,intensities_norm:1,n_angles_norm:1"
            opt_shapes = f"x_t:{batch_size}x1x128x128,fbps:{batch_size}x1x128x128,timesteps:{batch_size},intensities_norm:{batch_size},n_angles_norm:{batch_size}"
            max_shapes = f"x_t:{batch_size}x1x128x128,fbps:{batch_size}x1x128x128,timesteps:{batch_size},intensities_norm:{batch_size},n_angles_norm:{batch_size}"

            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": get_hardware_specific_engine_path(
                            dataset
                        ),
                        "trt_profile_min_shapes": min_shapes,
                        "trt_profile_opt_shapes": opt_shapes,
                        "trt_profile_max_shapes": max_shapes,
                    },
                ),
                "CUDAExecutionProvider",
            ]
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = int(
                os.environ.get("OMP_NUM_THREADS", 8)
            )
            self.ort_session = ort.InferenceSession(
                onnx_fp, sess_options=sess_options, providers=providers
            )
            self.io_binding = self.ort_session.io_binding()
        else:
            self.ort_session = None
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
        self.gradient_steps = gradient_steps
        self.lr = lr

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
        timesteps = (
            torch.LongTensor([t]).expand(len(x_t_flat)).to(device, dtype=torch.int32)
        )

        def to_batch(batch_idx: int) -> dict[str, torch.Tensor]:
            return {
                "x_t": x_t_flat[batch_idx : batch_idx + self.batch_size],
                "fbps": fbps_norm[batch_idx : batch_idx + self.batch_size],
                "timesteps": timesteps[batch_idx : batch_idx + self.batch_size],
                "intensities_norm": intensities_norm[
                    batch_idx : batch_idx + self.batch_size
                ],
                "n_angles_norm": n_angles_norm[batch_idx : batch_idx + self.batch_size],
            }

        if self.ort_session:
            for batch_idx in range(0, len(x_t_flat), self.batch_size):
                inputs = to_batch(batch_idx)
                for name, tensor in inputs.items():
                    if tensor.dtype == torch.float32:
                        np_type = np.float32
                    else:
                        np_type = np.int32

                    self.io_binding.bind_input(
                        name=name,
                        device_type="cuda",
                        device_id=0,
                        element_type=np_type,
                        shape=tuple(tensor.shape),
                        buffer_ptr=tensor.data_ptr(),
                    )
                noise_pred = torch.empty(
                    inputs["x_t"].shape, device=device, dtype=torch.float32
                )
                self.io_binding.bind_output(
                    "pred",
                    "cuda",
                    0,
                    np.float32,
                    tuple(noise_pred.shape),
                    noise_pred.data_ptr(),
                )
                self.ort_session.run_with_iobinding(self.io_binding)
                noise_preds.append(noise_pred)
        else:
            # Split into batches of size <= self.batch_size
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    for batch_idx in range(0, len(x_t_flat), self.batch_size):
                        inputs = to_batch(batch_idx)
                        noise_pred = self.unet(
                            inputs["x_t"],
                            inputs["fbps"],
                            inputs["timesteps"],
                            inputs["intensities_norm"],
                            inputs["n_angles_norm"],
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
            assert self.lr is not None, "Learning rate uninitialized"
            pred_original_sample = guide(
                pred_original_sample,
                guidance_loss_fn,
                gradient_steps=self.gradient_steps,
                lr=self.lr * (timestep / 1000) if self.anneal_lr else self.lr,
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
                noise_pred, _, _ = self.predict_noise(t, image)  # type: ignore
                image = self.step(noise_pred, t, image).prev_sample  # type: ignore
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
            self.lr = lr_from_experiment(experiment)

        side_length = experiment.counts.shape[-1]
        if experiment.sparse:
            # (rep, ..., n_angles, 1, side_length, side_length)
            n_angles_schedule = (
                len(schedule) if schedule is not None else experiment.counts.shape[-2]
            )
            x_t_shape = (
                replicates,
                *experiment.batch_dims,
                n_angles_schedule,
                side_length,
                side_length,
            )
        else:
            x_t_shape = (
                replicates,
                *experiment.batch_dims,
                experiment.counts.shape[-3],
                side_length,
                side_length,
            )

        fbps, intensities, class_labels = prepare_inputs_from_experiment(
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

        # Use x_t_shape to expand normalized inputs
        fbps_norm = ((fbps - 0.5) * 2).expand(x_t_shape)
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

        max_retries = 5
        out = None

        for attempt in range(max_retries):
            # Init x_t
            x_t = torch.randn(x_t_shape, device=self.device)

            it = tqdm(self.noise_scheduler.timesteps, disable=not self.verbose)
            failed = False

            for t in it:
                if self.cond:
                    noise_pred = self.predict_noise_cond(
                        t,
                        x_t,
                        fbps_norm,
                        intensities_norm,
                        n_angles_norm,  # type: ignore
                    )
                else:
                    noise_pred = self.predict_noise(t, x_t)  # type: ignore

                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    if self.verbose:
                        warnings.warn(
                            f"NaN/Inf in noise_pred (Attempt {attempt + 1}/{max_retries})"
                        )
                    failed = True
                    break

                guidance_loss_fn_ = guidance_loss_fn if (20 < t < 1000) else None
                step_out = self.step(noise_pred, t, x_t, guidance_loss_fn_)
                x_t = step_out.prev_sample

                if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                    if self.verbose:
                        warnings.warn(
                            f"NaN/Inf in x_t (Attempt {attempt + 1}/{max_retries})"
                        )
                    failed = True
                    break

            if not failed:
                out = denorm_image(x_t)
                break
            elif attempt == max_retries - 1:
                warnings.warn("Max retries reached. Returning FBP fallback.")
                out = fbps.expand(x_t_shape)

        if out is None:
            out = fbps.expand(x_t_shape)

        out = out.reshape(x_t_shape)
        out = apply_circular_mask(out)

        # Massage from
        #    (replicates, ..., n_angles or n_rounds, side_length, side_length)
        # to (..., n_angles or n_rounds, replicates, 1, side_length, side_length),
        n_batch_dims = len(experiment.batch_dims)
        ndim = len(x_t_shape)
        out_perm = (
            *tuple(range(1, n_batch_dims + 1)),
            n_batch_dims + 1,
            0,
            ndim - 2,
            ndim - 1,
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
            """
            Arguments:
                image: (replicates, ..., schedule_length, side_length, side_length)

            Returns:
                torch.Tensor: (1,)
            """
            counts_unsq = einops.repeat(
                experiment.counts,
                "... a w -> r ... t a w",
                r=image.shape[0],
                t=len(schedule),
            )
            intensities_unsq = einops.repeat(
                experiment.intensities,
                "... a 1 -> r ... t a 1",
                r=image.shape[0],
                t=len(schedule),
            )
            nlls = nll(
                image,
                counts_unsq,
                intensities_unsq,
                experiment.angles,
            )
            nlls[..., ~mask, :] = 0.0
            # return nlls.mean(-1).sum()
            return nlls.sum()

    else:
        assert (
            schedule is None
        ), "Schedules are currently unsupported for the dense setting."
        counts_csum = experiment.counts.cumsum(-3).unsqueeze(0)
        intensities_csum = experiment.intensities.cumsum(-3).unsqueeze(0)

        def loss_fn(image: torch.Tensor) -> torch.Tensor:
            nlls = nll(image, counts_csum, intensities_csum, experiment.angles)
            return nlls.mean((-2, -1)).sum()

    return loss_fn


def guide(
    x_t: torch.Tensor,
    loss_fct: Callable[..., torch.Tensor],
    gradient_steps: int = 50,
    lr: float = 0.1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        x_t (`torch.Tensor`): Shape (..., H, W) images with pixel values in range [-1, 1]

    Returns:
        `torch.Tensor`: Guided shape (..., H, W) images with pixel values in range [-1, 1]
    """
    radius = x_t.shape[-1] // 2
    y, x = torch.meshgrid(
        torch.arange(x_t.shape[-2], device=x_t.device),
        torch.arange(x_t.shape[-1], device=x_t.device),
        indexing="ij",
    )
    mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    loss = torch.tensor([float("inf")])

    # Init Optimizer
    y_0 = denorm_image(x_t).clone()
    y = torch.nn.Parameter(y_0).requires_grad_()
    optimizer = optim.Adam([y], lr=float(lr))

    it = tqdm(range(gradient_steps), disable=not verbose, fused=True)
    lowest_loss = float("inf")
    best_y = y.data.clone()
    for _ in it:
        optimizer.zero_grad()
        yp = (y * mask).clip(0)
        loss = loss_fct(yp)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y.data[..., ~mask] = 0.0
            y.data[y.data < 0] = 0.0
            y.data[y.data > 1] = 1.0

        loss = loss.item()
        if loss < lowest_loss:
            lowest_loss = loss
            best_y = y.data.clone()

        it.set_postfix({"loss": f"{loss:1.2e}", "lowest_loss": f"{lowest_loss:1.2e}"})

    x_t_guided = norm_image(best_y.data.detach()).clip(-1, 1)
    return x_t_guided


def norm_image(image: torch.Tensor) -> torch.Tensor:
    return (image - 0.5) * 2


def denorm_image(
    image: torch.Tensor, min_v: float = 0.0, max_v: float = 1.0
) -> torch.Tensor:
    return ((image + 1) / 2).clip(min_v, max_v)
