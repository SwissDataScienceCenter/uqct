from typing import Literal

import click
import torch

from uqct.ct import Experiment
from uqct.eval.options import common_options
from uqct.eval.run import run_evaluation
from uqct.models.unet import FBPUNet

DatasetName = Literal["lung", "composite", "lamino"]


def run_unet(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    member: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    schedule_length: int,
    max_angle: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FBPUNet(
        dataset=dataset,
        member=member,
        sparse=sparse,
        batch_size=32,
        model_device=device,
        verbose=True,
    )

    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        # (N, T, H, W)
        preds = model.predict(experiment, schedule)
        return preds

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="unet",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=schedule_start,
        schedule_type=schedule_type,
        schedule_length=schedule_length,
        max_angle=max_angle,
        extra_metadata=dict(member=member),
    )


@click.command()
@click.option(
    "--member",
    default=0,
    type=int,
    help="Ensemble member to load (0-9)",
)
@common_options
def main(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    member: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    schedule_length: int,
    max_angle: int,
):
    run_unet(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        member,
        n_angles,
        schedule_start,
        schedule_type,
        schedule_length,
        max_angle,
    )


if __name__ == "__main__":
    main()
