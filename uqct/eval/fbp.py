from typing import Literal

import click
import torch

from uqct.ct import prepare_inputs_from_experiment, Experiment
from uqct.eval.run import run_evaluation
from uqct.eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


def run_fbp(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    schedule_length: int,
    max_angle: int,
):
    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        # (N, T, H, W)
        preds, _, _ = prepare_inputs_from_experiment(experiment, schedule)
        return preds

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="fbp",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=schedule_start,
        schedule_type=schedule_type,
        schedule_length=schedule_length,
        max_angle=max_angle,
    )


@click.command()
@common_options
def main(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["linear", "exp"],
    schedule_length: int,
    max_angle: int,
):
    run_fbp(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        n_angles,
        schedule_start,
        schedule_type,
        schedule_length,
        max_angle,
    )


if __name__ == "__main__":
    main()
