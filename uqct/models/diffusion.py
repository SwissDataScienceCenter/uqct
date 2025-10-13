from pathlib import Path
from typing import Callable, Optional

import click
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import optim
from tqdm.auto import tqdm


def get_diffusion_samples(
    ckpt_path,
    target,
    loss_fct=None,
    device=None,
    verbose=False,
    buffer=5,
    t_start=999,
    sgd_steps=None,
    lr=None,
    num_steps=100,
):
    if lr is None:
        lr = [0.01, 0.005]
    if sgd_steps is None:
        sgd_steps = [100, 50]

    td = TomographicDiffusion(ckpt_path, target.shape, buffer=buffer).to(device)
    print(x_t.min(), x_t.max())

    images = td.guided_diffusion_pipeline(
        num_samples,
        t_start,
        0,
        num_steps,
        verbose=verbose,
        sgd_steps=sgd_steps,
        lr=lr,
    )

    return images


# Find checkpoint
CKPT_DIR_CANDIDATES = (
    Path(x) / "ddpm_unconditional_128_lung.pt"
    for x in ("/mydata/chip/shared/checkpoints/diffusion", "../checkpoints/diffusion")
)
while not (ckpt_path := next(CKPT_DIR_CANDIDATES)).exists():
    print(f"Model path: '{ckpt_path}' doesn't exists")


def guidance_loss(measurements, exposure, angles):
    """
    Define a loss function for the diffusion model.
    This can be used to guide the diffusion process.
    """

    def loss_fn(image):
        loss = nll(image, measurements, exposure, angles)
        return loss

    return loss_fn


class DiffusionModel:
    def __init__(
        self,
        ckpt_path,
        experiment,
        num_samples=5,
        t_start=999,
        sgd_steps=None,
        lr=None,
        num_steps=50,
        buffer=5,
    ):
        self.ckpt_path = ckpt_path
        self.experiment = experiment
        self.num_samples = num_samples
        self.t_start = t_start
        self.sgd_steps = sgd_steps
        self.lr = lr
        self.num_steps = num_steps
        self.buffer = buffer

    def sample(self, extra_guidance_loss=None, num_steps=None, sgd_steps=None, lr=None):
        num_samples = self.num_samples
        t_start = self.t_start
        if sgd_steps is None:
            sgd_steps = self.sgd_steps
        if lr is None:
            lr = self.lr
        if num_steps is None:
            num_steps = self.num_steps

        target = fbp_ct(
            self.experiment.measurements,
            self.experiment.angles,
            self.experiment.exposure,
            weighted=True,
        )
        print(target.shape)

        # expand second dimension to match num_samples
        target = target.expand(-1, num_samples, -1, -1, -1)
        batch_dims, data_dims = target.shape[:-3], target.shape[-3:]
        target = target.reshape(-1, *data_dims)
        print(f"Final target: {target.shape}")

        def guidance_loss(measurements, exposure, angles):
            """
            Define a loss function for the diffusion model.
            This can be used to guide the diffusion process.
            """

            def loss_fn(image):
                # print(image.shape)
                image = image.view((*batch_dims, *data_dims))
                loss = nll(image, measurements, exposure, angles).mean()

                if extra_guidance_loss is not None:
                    loss += extra_guidance_loss(image)
                return loss

            return loss_fn

        return get_diffusion_samples(
            self.ckpt_path,
            target,
            loss_fct=(
                guidance_loss(
                    self.experiment.measurements,
                    self.experiment.exposure,
                    self.experiment.angles,
                )
                if self.sgd_steps
                else None
            ),
            device=self.experiment.measurements.device,
            verbose=True,
            num_samples=num_samples,
            buffer=self.buffer,
            t_start=t_start,
            sgd_steps=sgd_steps,
            lr=lr,
            num_steps=num_steps,
        ).view((*batch_dims, *data_dims))


class TomographicDiffusion(nn.Module):
    def __init__(
        self,
        ckpt_path: Path,
        image_shape: tuple[int, int],
        buffer=5,
    ):
        super(TomographicDiffusion, self).__init__()
        self.unet = load_unet(ckpt_path)
        self.unet.eval()  # type: ignore
        for param in self.unet.parameters():  # type: ignore
            param.requires_grad = False
        self.x_t = nn.Parameter(torch.zeros(image_shape))
        self.buffer = buffer
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, clip_sample=False
        )

    def get_img(self):
        return ((self.x_t + 1) / 2).clip(0, 1)

    def predict_x_0(
        self, t: int, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.x_t.device
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
        device = self.x_t.device
        noise_pred, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
        new_timestep = torch.LongTensor([target_t]).to(device)
        new_x_t = self.noise_scheduler.add_noise(
            x_0_pred, torch.randn_like(x_0_pred), new_timestep  # type: ignore
        ).to(device)

        return new_x_t, noise_pred

    def diffusion_pipeline(
        self,
        x_t_start: torch.Tensor,
        t_start: int,
        t_end: int,
        num_steps=50,
        verbose=False,
    ) -> torch.Tensor:
        with torch.no_grad():
            x_t = x_t_start.clone()
            timesteps = torch.linspace(t_start, t_end, num_steps + 1).int()
            for i in tqdm(range(1, len(timesteps)), disable=not verbose):
                t = timesteps[i - 1]
                target_t = timesteps[i]
                x_t, _ = self.step(t, int(target_t.item()), x_t)
            return x_t

    def guided_diffusion_pipeline(
        self,
        t_start: int,
        t_end: int,
        num_steps: int = 50,
        verbose: bool = False,
        sgd_steps: int = 50,
        lr: float = 0.1,
    ) -> torch.Tensor:
        device = self.x_t.device

        x_t = self.noise_scheduler.add_noise(
            torch.zeros_like(self.x_t),
            torch.randn_like(self.x_t),
            torch.LongTensor([t_start]).to(device),  # type: ignore
        ).to(device)

        timesteps = torch.linspace(t_start, t_end + self.buffer, num_steps + 1).int()
        it = tqdm(range(len(timesteps)), disable=not verbose)
        for i in it:
            t = timesteps[i - 1]
            target_t = timesteps[i]
            new_timestep = torch.LongTensor([target_t]).to(device)
            noise_scheduler.previous_timestep = lambda _: target_t  # type: ignore

            with torch.no_grad():
                _, x_0_pred, _ = self.predict_x_0(int(t.item()), x_t)
                self.x_t *= 0
                self.x_t += x_0_pred

            guidance_loss = self.loss_guidance(
                loss_fct, sgd_steps=sgd_steps, lr=lr, verbose=False
            )
            it.set_postfix({"loss": f"{guidance_loss:.3f}"})

            x_t = self.noise_scheduler.add_noise(
                self.x_t,
                torch.randn_like(self.x_t),
                new_timestep,  # type: ignore
            ).to(device)

        x_t = self.diffusion_pipeline(
            x_t,
            t_end + self.buffer,
            t_end,
            num_steps=self.buffer,
            verbose=verbose,
        )

        _, self.x_t.data, _ = self.predict_x_0(t_end, x_t)
        return self.get_img()

    def loss_guidance(
        self,
        loss_fct: Callable[..., torch.Tensor],
        sgd_steps: int | list[int] = 50,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> float:
        lr_list, sgd_steps_list = lr, sgd_steps
        if not isinstance(lr, list):
            lr_list = [lr]
            sgd_steps_list = [sgd_steps]

        circle_mask = torch.ones(*self.x_t.shape[-2:], device=self.x_t.device)
        radius = self.x_t.shape[-1] // 2
        y, x = torch.meshgrid(
            torch.arange(self.x_t.shape[-2], device=self.x_t.device),
            torch.arange(self.x_t.shape[-1], device=self.x_t.device),
            indexing="ij",
        )
        mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
        circle_mask[~mask] = 0
        loss = torch.tensor([float("inf")])

        for lr_instance, sgd_steps_instance in zip(lr_list, sgd_steps_list):  # type: ignore
            parameters = list(self.parameters())
            optimizer = optim.Adam(parameters, lr=lr_instance)
            it = tqdm(range(sgd_steps_instance), disable=not verbose)
            for _ in it:
                loss = loss_fct(self.get_img())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self.x_t.data[..., ~mask] = -1
                it.set_postfix({"loss": f"{loss.item():.3f}"})
        return loss.item()


@torch.inference_mode()
def generate_samples(
    unet: UNet2DModel,
    num_samples: int,
    noise_scheduler: DDPMScheduler,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    unet.eval()
    device = unet.device
    channels = unet.config["in_channels"]
    size = unet.config["sample_size"]
    sample = torch.randn(
        (num_samples, channels, size, size),
        device=unet.device,
        dtype=next(unet.parameters()).dtype,
    )

    if n_steps is None:
        n_steps = noise_scheduler.config["num_train_timesteps"]  # 1000
    noise_scheduler.set_timesteps(n_steps, device=device)

    for t in tqdm(
        noise_scheduler.timesteps,
        total=len(noise_scheduler.timesteps),
        desc="denoising",
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            noise_pred = unet(sample, t, return_dict=False)[0]
        out = noise_scheduler.step(model_output=noise_pred, timestep=t, sample=sample)  # type: ignore
        sample = out.prev_sample

    return sample


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
