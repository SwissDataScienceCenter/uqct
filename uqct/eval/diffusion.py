from typing import Literal, Optional

import click
import einops

from uqct.ct import Experiment
import torch
from uqct.eval.run import run_evaluation
from uqct.models.diffusion import Diffusion, get_guidance_loss_fn
from uqct.eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


def run_diffusion(
    dataset: DatasetName,
    sparse: bool,
    cond: bool,
    total_intensity: float,
    schedule_length: int,
    gradient_steps: int,
    guidance_lr: float | None,
    image_range: tuple[int, int],
    seed: int,
    replicates: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    max_angle: int,
):
    diffusion = Diffusion(
        dataset,
        batch_size=32,
        lr=guidance_lr,
        gradient_steps=gradient_steps,
        cond=cond,
        onnx=True,
        verbose=True,
    )

    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        guidance_loss_fn = get_guidance_loss_fn(experiment, schedule)
        # (..., T, replicates, 1, side_length, side_length)
        sample = diffusion.sample(
            experiment,
            replicates,
            schedule,
            guidance_loss_fn,
        )
        # Rearrange to (N, T, H, W)
        return einops.rearrange(sample, "n t r 1 w h -> n t r w h")

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="diffusion",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=schedule_start,
        schedule_type=schedule_type,
        schedule_length=schedule_length,
        max_angle=max_angle,
        extra_metadata=dict(
            cond=cond, guidance_lr=guidance_lr, gradient_steps=gradient_steps
        ),
    )


@click.command()
@click.option(
    "--cond",
    default=False,
    type=bool,
    help="Whether to use a conditional diffusion model",
)
@click.option(
    "--guidance-lr", default=None, type=Optional[float], help="Guidance learning rate"
)
@click.option(
    "--gradient-steps",
    default=20,
    type=int,
    help="Number of guidance steps per denoising step",
)
@click.option(
    "--replicates",
    default=1,
    type=int,
    help="Number of replicates to sample",
)
@common_options
def main(
    dataset: DatasetName,
    sparse: bool,
    cond: bool,
    total_intensity: float,
    schedule_length: int,
    gradient_steps: int,
    guidance_lr: float | None,
    image_range: tuple[int, int],
    seed: int,
    replicates: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    max_angle: int,
):
    run_diffusion(
        dataset,
        sparse,
        cond,
        total_intensity,
        schedule_length,
        gradient_steps,
        guidance_lr,
        image_range,
        seed,
        replicates,
        n_angles,
        schedule_start,
        schedule_type,
        max_angle,
    )


if __name__ == "__main__":
    main()
