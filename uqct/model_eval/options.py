import click
from typing import Optional


def common_options(func):
    """Decorator to add common CLI options."""
    func = click.option(
        "--dataset",
        default="lamino",
        type=click.Choice(["lung", "composite", "lamino"]),
        help="Which dataset to generate samples for",
    )(func)
    func = click.option(
        "--sparse",
        default=False,
        type=bool,
        help="Whether to generate samples for the sparse setting",
    )(func)
    func = click.option(
        "--total-intensity", default=1e7, type=float, help="Total intensity"
    )(func)
    func = click.option(
        "--image-range",
        default=(0, 10),
        type=int,
        nargs=2,
        help="Test set images range (exclusive)",
    )(func)
    func = click.option("--seed", default=0, type=int, help="Random seed")(func)
    return func
