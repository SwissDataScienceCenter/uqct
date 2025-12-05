from typing import Literal, Optional

import click
import einops

from uqct.model_eval.experiment import (
    CTSettings,
    setup_experiment,
    evaluate_and_save,
)
from uqct.models.diffusion import Diffusion, get_guidance_loss_fn
from uqct.model_eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


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
    gradient_steps: int,
    guidance_lr: float | None,
    image_range: tuple[int, int],
    seed: int,
    replicates: int,
):
    gt, experiment, schedule = setup_experiment(
        dataset, image_range, total_intensity, sparse, seed
    )
    diffusion = Diffusion(
        dataset,
        batch_size=32,
        lr=guidance_lr,
        gradient_steps=gradient_steps,
        cond=cond,
        onnx=True,
        verbose=True,
    )

    guidance_loss_fn = get_guidance_loss_fn(experiment, schedule)
    # (..., T, replicates, 1, side_length, side_length)
    sample = diffusion.sample(
        experiment,
        replicates,
        schedule - 1 if schedule is not None else None,
        guidance_loss_fn,
    )
    # Rearrange to (N, T, H, W) - wait, evaluate_and_save handles (N, T, H, W) or (N, T, 1, H, W)
    # diffusion.py originally did: sample = einops.rearrange(sample, "n t 1 1 w h -> n t w h")
    # Let's keep it consistent.
    sample = einops.rearrange(sample, "n t 1 1 w h -> n t w h")

    ct_settings = CTSettings(
        dataset=dataset,
        total_intensity=total_intensity,
        sparse=sparse,
        image_start_index=image_range[0],
        image_end_index=image_range[1],
    )

    evaluate_and_save(
        preds=sample,
        gt=gt,
        experiment=experiment,
        schedule=schedule,
        ct_settings=ct_settings,
        model_name="diffusion",
        seed=seed,
        extra_metadata=dict(
            cond=cond, guidance_lr=guidance_lr, gradient_steps=gradient_steps
        ),
    )


if __name__ == "__main__":
    main()
