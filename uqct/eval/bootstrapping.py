import click
from uqct.eval.options import common_options
from uqct.other_methods.bootstrapping import run_bootstrapping


@click.command()
@common_options
@click.option("--n-bootstraps", default=20, help="Number of bootstrap samples")
@click.option(
    "--comparison", is_flag=True, default=False, help="Run in comparison mode"
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
    comparison,
):
    run_bootstrapping(
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
        comparison,
    )


if __name__ == "__main__":
    main()
