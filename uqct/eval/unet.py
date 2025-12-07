from typing import Literal

import click
import torch

from uqct.eval.run import (
    CTSettings,
    setup_experiment,
    evaluate_and_save,
    run_evaluation,
)
from uqct.models.unet import FBPUNet
from uqct.eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


def run_unet(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    member: int,
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

    def predictor_fn(experiment, schedule):
        # (N, T, H, W) -> (N, T, 1, H, W)
        return model.predict(experiment, schedule).unsqueeze(2)

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="unet",
        predictor_fn=predictor_fn,
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
):
    run_unet(dataset, sparse, total_intensity, image_range, seed, member)


if __name__ == "__main__":
    main()
