from collections import defaultdict
from typing import Literal, Optional

import click
import einops
import torch.nn.functional as F

from uqct.ct import nll_mixture_angle_schedule
from uqct.metrics import get_metrics
from uqct.model_eval.experiment import CTSettings, Metrics, Run, setup_experiment
from uqct.models.diffusion import Diffusion, get_guidance_loss_fn

DatasetName = Literal["lung", "composite", "lamino"]


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
    "--image-range",
    default=(0, 10),
    type=int,
    nargs=2,
    help="Test set images range (exclusive)",
)
@click.option("--seed", default=0, type=int, help="Random seed")
def main(
    dataset: DatasetName,
    sparse: bool,
    cond: bool,
    total_intensity: float,
    gradient_steps: int,
    guidance_lr: float | None,
    image_range: tuple[int, int],
    seed: int,
):
    gt, experiment, schedule = setup_experiment(
        dataset, image_range, total_intensity, sparse, seed
    )
    n_gt = len(gt)
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
        experiment, 1, schedule - 1 if schedule is not None else None, guidance_loss_fn
    )
    sample = einops.rearrange(sample, "n t 1 1 w h -> n t w h")
    gt_lr = F.interpolate(
        einops.rearrange(gt, "n w h -> n 1 w h"), (128, 128), mode="area"
    )
    gt_lr = einops.rearrange(gt_lr, "n 1 w h -> n w h")

    metric2lists = defaultdict(list)
    for image_index in range(n_gt):
        for t in range(sample.shape[-3]):
            image_gt = gt_lr[image_index]
            image_pred = sample[image_index][t]
            for k, v in get_metrics(image_gt, image_pred).items():
                if image_index + 1 > len(metric2lists[k]):
                    metric2lists[k].append(list())
                metric2lists[k][image_index].append(v)
    metric2lists = dict(metric2lists)

    if sparse:
        assert schedule is not None, "Expecting schedule to not be None."
        nlls_pred = nll_mixture_angle_schedule(
            einops.rearrange(sample, "n t w h -> n t 1 w h"),
            experiment.counts,
            experiment.intensities,
            experiment.angles,
            schedule,
            sparse=False,
        )
        nlls_gt = nll_mixture_angle_schedule(
            einops.repeat(gt_lr, "n w h -> n t 1 w h", t=len(schedule)),
            experiment.counts,
            experiment.intensities,
            experiment.angles,
            schedule,
            sparse=False,
        )
    else:
        raise NotImplementedError()
    ct_settings = CTSettings(
        dataset=dataset,
        total_intensity=total_intensity,
        sparse=sparse,
        image_start_index=image_range[0],
        image_end_index=image_range[1],
    )
    metrics = Metrics(
        psnr=metric2lists["PSNR"],
        ssim=metric2lists["SS"],
        rmse=metric2lists["RMSE"],
        zeroone=metric2lists["ZeroOne"],
        l1=metric2lists["L1"],
        nll_pred=nlls_pred.tolist(),
        nll_gt=nlls_gt.tolist(),
    )
    run = Run(
        ct_settings=ct_settings,
        model="diffusion",
        metrics=metrics,
        seed=seed,
        preds=sample.cpu().numpy(),
        extra=dict(cond=cond, guidance_lr=guidance_lr, gradient_steps=gradient_steps),
    )
    run.dump_parquet()


if __name__ == "__main__":
    main()
