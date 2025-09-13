import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch import optim
from tqdm.auto import tqdm


class TomographicDiffusion(nn.Module):
    def __init__(
        self,
        image_shape,
        unet: UNet2DModel,
        use_sigmoid: bool = True,
        buffer=5,
        fourier_magnitude=None,
        shifts=None,
    ):
        super(TomographicDiffusion, self).__init__()
        self.computed_image = None
        self.unet = unet
        for param in self.unet.parameters():
            param.requires_grad = False
        self.x_t = nn.Parameter(torch.zeros(image_shape))
        self.use_sigmoid = use_sigmoid
        self.buffer = buffer
        self.fourier_magnitude = fourier_magnitude
        self.shifts = shifts

    def get_img(self):
        if self.use_sigmoid:
            image = 5 * (self.x_t - 0.5)
            image = torch.sigmoid(image)
        else:
            image = self.x_t

        self.computed_image = image
        return image

    def predict_x_0(self, t, x_t, noise_scheduler, conditioning=None):
        device = self.x_t.device
        timesteps = torch.LongTensor([t]).to(device)

        # Handle conditional diffusion
        if conditioning is not None:
            # Ensure conditioning has the right shape and add channel dimension if needed
            if conditioning.dim() == 3:  # [bs, H, W]
                conditioning = conditioning.unsqueeze(1)  # [bs, 1, H, W]
            elif conditioning.dim() == 2:  # [H, W]
                conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # Concatenate conditioning with noisy image
            conditioning_input = torch.cat([conditioning, x_t], dim=1)
            noise_pred = self.unet(conditioning_input, timesteps, return_dict=False)[0]
        else:
            noise_pred = self.unet(x_t, timesteps, return_dict=False)[0]

        x_0_pred = noise_scheduler.step(
            noise_pred, timesteps.item(), x_t.reshape(noise_pred.shape)
        ).pred_original_sample

        x_t_previous = noise_scheduler.step(
            noise_pred, timesteps.item(), x_t.reshape(noise_pred.shape)
        ).prev_sample

        return noise_pred, x_0_pred, x_t_previous

    def step(self, t, target_t, x_t, noise_scheduler, conditioning=None):
        device = self.x_t.device
        noise_pred, x_0_pred, x_t_previous = self.predict_x_0(
            t, x_t, noise_scheduler, conditioning
        )
        new_timestep = torch.LongTensor([target_t]).to(device)
        new_x_t = noise_scheduler.add_noise(
            x_0_pred, torch.randn_like(x_0_pred), new_timestep
        ).to(device)

        return new_x_t, noise_pred

    def diffusion_pipeline(
        self,
        x_t_start,
        t_start,
        t_end,
        noise_scheduler,
        num_steps=50,
        verbose=False,
        conditioning=None,
    ):
        device = self.x_t.device
        with torch.no_grad():
            x_t = x_t_start.clone()

            timesteps = torch.linspace(t_start, t_end, num_steps + 1).int()
            for i in tqdm(range(1, len(timesteps)), disable=not verbose):
                t = timesteps[i - 1]
                target_t = timesteps[i]
                x_t, _ = self.step(t, target_t, x_t, noise_scheduler, conditioning)
            return x_t

    def guided_diffusion_pipeline(
        self,
        x_t_start,
        t_start,
        t_end,
        noise_scheduler,
        num_steps=50,
        loss_fct=None,
        inpainting_fct=None,
        batch_size=10,
        verbose=False,
        sgd_steps=50,
        lr=0.1,
        with_finetuning: bool = False,
        dps: bool = False,
        conditioning=None,
        # fourier_inpainting: bool = False,
        # inpainting_range=0
    ):
        device = self.x_t.device

        x_t = noise_scheduler.add_noise(
            x_t_start,
            torch.randn_like(x_t_start),
            torch.LongTensor([t_start]).to(device),
        ).to(device)

        timesteps = torch.linspace(t_start, t_end + self.buffer, num_steps + 1).int()
        it = tqdm(range(len(timesteps)), disable=not verbose)
        for i in it:
            t = timesteps[i - 1]
            target_t = timesteps[i]
            new_timestep = torch.LongTensor([target_t]).to(device)
            noise_scheduler.previous_timestep = lambda x: target_t

            with torch.no_grad():
                noise_pred, x_0_pred, x_t_previous = self.predict_x_0(
                    t, x_t, noise_scheduler, conditioning
                )
                self.x_t *= 0
                if dps:
                    self.x_t += x_t_previous
                else:
                    self.x_t += x_0_pred

            if loss_fct is not None:
                guidance_loss = self.loss_guidance(
                    t, loss_fct, sgd_steps=sgd_steps, lr=lr, verbose=False
                )
                it.set_postfix({"loss": f"{guidance_loss:.3f}"})

            if inpainting_fct is not None:
                self.x_t.data = inpainting_fct(self.x_t).data

            if not dps:
                x_t = noise_scheduler.add_noise(
                    self.get_img(), torch.randn_like(self.get_img()), new_timestep
                ).to(device)
            else:
                x_t = self.get_img()

        # if with_finetuning:
        #     with torch.no_grad():
        #         self.x_t *= 0
        #         self.x_t += x_t[:, 0]
        #
        #     guidance(
        #         t, hr_sinogram, lr_sinogram, sgd_steps=[500, 200, 200], batch_size=batch_size, lr=[0.1, 0.01, 0.001], verbose=verbose
        #     )
        #     x_t = self.get_img().unsqueeze(1)

        x_t = self.diffusion_pipeline(
            x_t,
            t_end + self.buffer,
            t_end,
            noise_scheduler,
            num_steps=self.buffer,
            verbose=verbose,
            conditioning=conditioning,
        )

        _, x_t, _ = self.predict_x_0(t_end, x_t, noise_scheduler, conditioning)
        if inpainting_fct is not None:
            x_t = inpainting_fct(x_t)

        return x_t

    def loss_guidance(self, t, loss_fct, sgd_steps=50, lr=0.1, verbose=False):
        if type(lr) != list:
            lr = [lr]
            sgd_steps = [sgd_steps]

        circle_mask = torch.ones(*self.x_t.shape[-2:], device=self.x_t.device)
        radius = self.x_t.shape[-1] // 2
        y, x = torch.meshgrid(
            torch.arange(self.x_t.shape[-2], device=self.x_t.device),
            torch.arange(self.x_t.shape[-1], device=self.x_t.device),
            indexing="ij",
        )
        mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
        circle_mask[~mask] = 0

        for lr_instance, sgd_steps_instance in zip(lr, sgd_steps):
            parameters = list(self.parameters())
            if self.shifts is not None:
                parameters.append(self.shifts)
            optimizer = optim.AdamW(parameters, lr=lr_instance)
            it = tqdm(range(sgd_steps_instance), disable=not verbose)
            for _ in it:
                if self.shifts is not None:
                    loss = loss_fct(self.x_t, shift=self.shifts)
                else:
                    loss = loss_fct(self.x_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # circle mask and clamp
                with torch.no_grad():
                    self.x_t.clamp_(min=0, max=1)
                    self.x_t.mul_(circle_mask)

                it.set_postfix({"loss": f"{loss.item():.3f}"})

        return loss.item()

    def forward(
        self, sinogram_angles, filter=None, filter_in_sinogram_space: bool = False
    ):
        device = self.x_t.device
        if not filter_in_sinogram_space:
            images = self.get_img()
        else:
            images = self.get_img()

        rotation_matrix = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(torch.deg2rad(sinogram_angles)),
                        -torch.sin(torch.deg2rad(sinogram_angles)),
                        torch.zeros_like(sinogram_angles, device=device),
                    ],
                    1,
                ),
                torch.stack(
                    [
                        torch.sin(torch.deg2rad(sinogram_angles)),
                        torch.cos(torch.deg2rad(sinogram_angles)),
                        torch.zeros_like(sinogram_angles, device=device),
                    ],
                    1,
                ),
            ],
            1,
        )
        current_grid = F.affine_grid(
            rotation_matrix.to(images.device),
            images.repeat(len(sinogram_angles), 1, 1, 1).size(),
            align_corners=False,
        )

        rotated = F.grid_sample(
            images.repeat(len(sinogram_angles), 1, 1, 1).float(),
            current_grid.repeat(1, 1, 1, 1),
            align_corners=False,
        )
        rotated = rotated.transpose(0, 1)
        # Sum over one of the dimensions to compute the projection
        sinogram = rotated.sum(axis=-2).squeeze(2)
        sinogram = sinogram if len(sinogram) > 1 else sinogram[0]
        if not filter_in_sinogram_space:
            return sinogram
        else:
            return filter(sinogram)


if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    model = UNet2DModel(
        sample_size=512,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            64,
            64,
            128,
            128,
            256,
            256,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    bs = 2
    td = TomographicDiffusion((bs, 512, 512), model, True, buffer=0)

    print(td.forward(torch.linspace(0, 180, 10)))

    from chip.models.forward_models import fourier_filtering
    from chip.utils.utils import create_circle_filter, create_gaussian_filter
    from diffusers import DDPMScheduler

    side_length = 512
    frequency_cut_out_radius = 15
    circle_filter = create_circle_filter(frequency_cut_out_radius, side_length)
    gaussian_filter = create_gaussian_filter(sigma=10, size=side_length)

    current_filter = circle_filter

    lr_forward_function = lambda x: fourier_filtering(x, current_filter)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # img = td.diffusion_pipeline(torch.randn(bs, 512, 512).unsqueeze(1), 999, 0, noise_scheduler, 2)
    img = td.guided_diffusion_pipeline(
        torch.randn(bs, 512, 512).unsqueeze(1),
        999,
        0,
        noise_scheduler,
        2,
        lr_forward_function=lr_forward_function,
        batch_size=10,
        verbose=True,
        sgd_steps=[2, 10],
        lr=[0.1, 0.01],
    )
    print(img)
