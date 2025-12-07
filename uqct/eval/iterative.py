from uqct.models.iterative import reconstruct, ReconstructionMethod
from typing import Literal

import click
from uqct.eval.run import (
    run_evaluation,
)
from uqct.eval.options import common_options

DatasetName = Literal["lung", "composite", "lamino"]


def run_iterative(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    method: ReconstructionMethod,
    lr: float,
    patience: int,
    tv_weight: float,
    max_steps: int,
):
    def predictor_fn(experiment, schedule):
        # Output shape: (N, 1, H, W)
        return reconstruct(
            experiment, schedule, method, lr, patience, tv_weight, max_steps
        )

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name=method,
        predictor_fn=predictor_fn,
        extra_metadata=dict(
            lr=lr,
            patience=patience,
            tv_weight=tv_weight if method == "map" else None,
            max_steps=max_steps,
        ),
    )


@click.command()
@common_options
@click.option(
    "--method",
    default="mle",
    type=click.Choice(["mle", "map"]),
    help="Reconstruction method",
)
@click.option(
    "--lr",
    default=1e-2,
    type=float,
    help="Learning rate (default: 1e-2)",
)
@click.option(
    "--patience",
    default=50,
    type=int,
    help="Patience for early stopping",
)
@click.option(
    "--tv-weight",
    default=-1.0,
    type=float,
    help="Weight for TV prior (only for MAP, default: 0.01)",
)
@click.option(
    "--max-steps",
    default=20000,
    type=int,
    help="Maximum number of optimization steps",
)
def main(
    dataset: DatasetName,
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    method: ReconstructionMethod,
    lr: float,
    patience: int,
    tv_weight: float,
    max_steps: int,
):
    run_iterative(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        method,
        lr,
        patience,
        tv_weight,
        max_steps,
    )


if __name__ == "__main__":
    main()
