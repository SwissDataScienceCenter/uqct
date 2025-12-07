from typing import Literal

import click

from uqct.ct import prepare_inputs_from_experiment
from uqct.eval.run import (
    run_evaluation,
)
from uqct.eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


def run_fbp(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
):
    def predictor_fn(experiment, schedule):
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
    )


@click.command()
@common_options
def main(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
):
    run_fbp(dataset, sparse, total_intensity, image_range, seed)


if __name__ == "__main__":
    main()
