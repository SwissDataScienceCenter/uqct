import click

from uqct.eval.options import common_options
from uqct.other_methods.skrock import run_skrock


@click.command()
@common_options
@click.option("--n-burnin", default=1000, type=int, help="Burn-in iterations")
@click.option(
    "--n-samples", default=1000, type=int, help="Post-burn-in samples (replicate axis)"
)
@click.option("--n-stages", default=10, type=int, help="SK-ROCK Chebyshev stages (s)")
@click.option("--eta", default=0.05, type=float, help="SK-ROCK damping eta")
@click.option(
    "--dt-perc",
    default=0.95,
    type=float,
    help="Step size as a fraction of the SK-ROCK stability bound l_s / L.",
)
@click.option(
    "--prior",
    default="tv",
    type=click.Choice(["tv", "none"]),
    help="Prior: 'tv' (Moreau-Yosida TV, recommended) or 'none' (MLE Langevin).",
)
@click.option(
    "--tv-weight",
    default=-1.0,
    type=float,
    help="TV prior weight beta. -1 -> auto: tv-weight-calibration x heuristic(intensity).",
)
@click.option(
    "--tv-weight-calibration",
    default=2.0,
    type=float,
    help="Multiplier on the heuristic beta when --tv-weight is -1 (2.0 = ECE-calibrated; "
    "1.0 = bare MAP heuristic / paper-spirit value).",
)
@click.option(
    "--my-lambda-factor",
    default=1.0,
    type=float,
    help="Moreau-Yosida parameter as a multiple of 1/L_f (Durmus 2018 recommends 1).",
)
@click.option(
    "--chambolle-iters",
    default=25,
    type=int,
    help="Inner iterations for the Chambolle TV prox.",
)
@click.option(
    "--lipschitz-iters",
    default=40,
    type=int,
    help="Power iterations for the ||R||^2 estimate.",
)
@click.option(
    "--sampler-seed", default=0, type=int, help="Seed for the SK-ROCK Markov chain."
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
    n_burnin,
    n_samples,
    n_stages,
    eta,
    dt_perc,
    prior,
    tv_weight,
    tv_weight_calibration,
    my_lambda_factor,
    chambolle_iters,
    lipschitz_iters,
    sampler_seed,
):
    # SK-ROCK uses the full angle set at a single time step; schedule unused.
    del schedule_start, schedule_type, schedule_length

    run_skrock(
        dataset,
        sparse,
        total_intensity,
        image_range,
        seed,
        n_angles,
        max_angle,
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_stages=n_stages,
        eta=eta,
        dt_perc=dt_perc,
        prior=prior,
        tv_weight=tv_weight,
        tv_weight_calibration=tv_weight_calibration,
        my_lambda_factor=my_lambda_factor,
        chambolle_iters=chambolle_iters,
        lipschitz_iters=lipschitz_iters,
        sampler_seed=sampler_seed,
    )


if __name__ == "__main__":
    main()
