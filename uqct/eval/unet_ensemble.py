from typing import Literal

import click
import torch

from uqct.ct import Experiment
from uqct.eval.options import common_options
from uqct.eval.run import run_evaluation
from uqct.models.unet import FBPUNetEnsemble, DatasetName


def run_unet_ensemble(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FBPUNetEnsemble loads all 10 members internally
    model = FBPUNetEnsemble(
        dataset=dataset,
        sparse=sparse,
        batch_size=32,
        device=device,
    )

    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        # (N, M, T, H, W) -> (N, T, M, H, W)
        # We treat ensemble members as replicates (R)
        preds = model.predict(experiment, schedule, aggregate="none")
        if preds.ndim == 5:
            # Swap M and T dimensions to match (N, T, R, H, W) expected by evaluate_and_save
            preds = preds.permute(0, 2, 1, 3, 4)
        return preds

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="unet_ensemble",
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
    run_unet_ensemble(
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
