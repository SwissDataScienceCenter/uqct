import click
import tomllib
import itertools
import sys
from typing import Literal

from uqct.datasets.utils import DatasetName
from uqct.eval.fbp import run_fbp
from uqct.eval.iterative import run_iterative
from uqct.eval.diffusion import run_diffusion
from uqct.eval.unet import run_unet
from uqct.utils import get_root_dir


@click.group()
def cli():
    """Unified evaluation CLI."""
    pass


@cli.command()
@click.option(
    "--model",
    required=True,
    type=click.Choice(["fbp", "mle", "map", "unet", "diffusion"]),
    help="Model name (fbp, mle, map, unet, diffusion)",
)
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
def run(
    model: Literal["fbp", "mle", "map", "unet", "diffusion"],
    job_id: int | None,
    sparse: bool,
):
    """Run evaluation for a specific model and configuration."""

    # Load settings
    # Assuming running from project root or uqct is in pythonpath
    # We look for settings.toml relative to uqct package or project root
    root = get_root_dir()
    settings_path = root / "uqct" / "settings.toml"

    if not settings_path.exists():
        click.echo(f"Settings file not found at {settings_path}")
        sys.exit(1)

    with open(settings_path, "rb") as f:
        settings = tomllib.load(f)["eval"]

    datasets = settings["datasets"]
    intensities = settings["total_intensity_values"]
    image_range = tuple(settings["image_range"])

    # Create grid of configurations
    # Order: Iterate datasets, then intensities? Or vice versa?
    # Let's match typical array order: Dataset outer, Intensity inner, or flat.
    # itertools.product(A, B) yields (a0, b0), (a0, b1)...
    grid = list(itertools.product(datasets, intensities))

    if job_id is not None:
        if job_id < 0 or job_id >= len(grid):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid)-1})")
            sys.exit(1)

        dataset, intensity = grid[job_id]
        click.echo(f"Running evaluation for {model}:")
        click.echo(f"  Dataset: {dataset}")
        click.echo(f"  Intensity: {intensity}")
        click.echo(f"  Sparse: {sparse}")
        click.echo(f"  Image Range: {image_range}")

        _dispatch(model, dataset, intensity, sparse, image_range)
    else:
        # If no job_id, run all locally
        click.echo(f"Running all {len(grid)} configurations for {model}...")
        for i, (dataset, intensity) in enumerate(grid):
            click.echo(f"\n--- Config {i+1}/{len(grid)}: {dataset}, {intensity} ---")
            _dispatch(model, dataset, intensity, sparse, image_range)


def _dispatch(
    model: Literal["fbp", "mle", "map", "unet", "diffusion"],
    dataset: DatasetName,
    intensity: float,
    sparse: bool,
    image_range: tuple[int, int],
):
    seed = 0  # Default seed

    if model == "fbp":
        run_fbp(dataset, sparse, intensity, image_range, seed)

    elif model in ["mle", "map"]:
        # Defaults from iterative.py
        lr = 1e-2
        patience = 50
        max_steps = 20000
        tv_weight = -1.0  # Default

        run_iterative(
            dataset=dataset,
            sparse=sparse,
            total_intensity=intensity,
            image_range=image_range,
            seed=seed,
            method=model,  # type: ignore
            lr=lr,
            patience=patience,
            tv_weight=tv_weight,
            max_steps=max_steps,
        )

    elif model == "unet":
        # Default member 0
        run_unet(dataset, sparse, intensity, image_range, seed, member=0)

    elif model == "diffusion":
        # Defaults from diffusion.py
        run_diffusion(
            dataset=dataset,
            sparse=sparse,
            cond=True,
            total_intensity=intensity,
            gradient_steps=20,
            guidance_lr=1e-2,
            image_range=image_range,
            seed=seed,
            replicates=16,
        )
    else:
        click.echo(
            f"Unknown model '{model}'. Supported: fbp, mle, map, unet, diffusion."
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
