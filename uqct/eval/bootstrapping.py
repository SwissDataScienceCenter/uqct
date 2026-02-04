import click

from uqct.eval.options import common_options
from uqct.other_methods.bootstrapping import run_bootstrapping


@click.command()
@common_options
@click.option("--n-bootstraps", default=20, help="Number of bootstrap samples")
@click.option("--method", default="fbp", help="Method to bootstrap (fbp, unet)")
def main(
    dataset,
    sparse,
    total_intensity,
    image_range,
    seed,
    n_angles,
    max_angle,
    n_bootstraps,
    method,
):
    run_bootstrapping(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        n_angles,
        max_angle,
        n_bootstraps,
        method=method,
    )


if __name__ == "__main__":
    main()
