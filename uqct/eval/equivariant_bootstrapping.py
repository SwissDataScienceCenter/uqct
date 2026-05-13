import click

from uqct.eval.options import common_options
from uqct.other_methods.equivariant_bootstrapping import run_equivariant_bootstrapping


@click.command()
@common_options
@click.option("--n-bootstraps", default=20, help="Number of bootstrap samples")
@click.option(
    "--rotation-std-deg",
    default=8.0,
    type=float,
    help="Standard deviation (in degrees) of the Gaussian rotation angle. "
    "Defaults to 8 to match the tomography setting in Tachella & Pereyra (2024).",
)
@click.option(
    "--flip/--no-flip",
    default=False,
    help="Augment with random horizontal/vertical flips.",
)
def main(
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
    n_bootstraps,
    rotation_std_deg,
    flip,
):
    # Schedule options come from `common_options` but are unused — the
    # equivariant bootstrap always uses the full angle set at a single time step.
    del schedule_start, schedule_type, schedule_length
    run_equivariant_bootstrapping(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        n_angles,
        max_angle,
        n_bootstraps,
        rotation_std_deg=rotation_std_deg,
        flip=flip,
    )


if __name__ == "__main__":
    main()
