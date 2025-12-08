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
    required=False,
    type=click.Choice(["fbp", "mle", "map", "unet", "diffusion"]),
    default=None,
    help="Model name (fbp, mle, map, unet, diffusion). Required only if running locally without job-id to filter.",
)
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
def run(
    model: Literal["fbp", "mle", "map", "unet", "diffusion"] | None,
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

    models = settings.get("models", ["fbp", "mle", "map", "unet", "diffusion"])
    datasets = settings["datasets"]
    intensities = settings["total_intensity_values"]
    schedule_length = settings.get("schedule_length", 32)

    full_image_range = tuple(settings["image_range"])
    start, end = full_image_range
    # Create chunks of 10
    image_ranges = [(i, min(i + 10, end)) for i in range(start, end, 10)]

    # Create grid of configurations
    # Order: Dataset, Intensity, Image Range, Model (Model inner-most to fail fast)
    grid = list(itertools.product(datasets, intensities, image_ranges, models))

    if job_id is not None:
        if job_id < 0 or job_id >= len(grid):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid)-1})")
            sys.exit(1)

        dataset, intensity, current_image_range, model = grid[job_id]  # type: ignore
        if model is None:
            raise ValueError("Model in grid cannot be None")

        click.echo(f"Running evaluation for Job ID {job_id}:")
        click.echo(f"  Model: {model}")
        click.echo(f"  Dataset: {dataset}")
        click.echo(f"  Intensity: {intensity}")
        click.echo(f"  Sparse: {sparse}")
        click.echo(f"  Image Range: {current_image_range}")

        _dispatch(model, dataset, intensity, sparse, current_image_range, schedule_length, settings)  # type: ignore
    else:
        # If no job_id, run all locally
        if model:
            # Filter grid for specific model if provided
            grid = [(d, i, r, m) for d, i, r, m in grid if m == model]

        click.echo(f"Running {len(grid)} configurations...")
        for i, (d, inten, r, m) in enumerate(grid):
            click.echo(f"\n--- Config {i+1}/{len(grid)}: {m}, {d}, {inten}, {r} ---")
            _dispatch(m, d, inten, sparse, r, schedule_length, settings)  # type: ignore


def _dispatch(
    model: Literal["fbp", "mle", "map", "unet", "diffusion"],
    dataset: DatasetName,
    intensity: float,
    sparse: bool,
    image_range: tuple[int, int],
    schedule_length: int,
    settings: dict,
):
    seed = 0  # Default seed

    if model == "fbp":
        run_fbp(dataset, sparse, intensity, image_range, seed, schedule_length)

    elif model in ["mle", "map"]:
        # Load from settings
        cfg = settings.get(model, {})
        # Fallback defaults if not in settings (though settings.toml should have them)
        lr = cfg.get("lr", 1e-2)
        patience = cfg.get("patience", 50)
        max_steps = cfg.get("max_steps", 20000)
        tv_weight = cfg.get("tv_weight", -1.0)

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
            schedule_length=schedule_length,
        )

    elif model == "unet":
        cfg = settings.get("unet", {})
        member = cfg.get("member", 0)
        run_unet(dataset, sparse, intensity, image_range, seed, member, schedule_length)

    elif model == "diffusion":
        cfg = settings.get("diffusion", {})
        gradient_steps = cfg.get("gradient_steps", 20)
        guidance_lr = cfg.get("guidance_lr", 1e-2)
        replicates = cfg.get("replicates", 16)
        cond = cfg.get("cond", True)

        run_diffusion(
            dataset=dataset,
            sparse=sparse,
            cond=cond,
            total_intensity=intensity,
            gradient_steps=gradient_steps,
            guidance_lr=guidance_lr,
            image_range=image_range,
            seed=seed,
            replicates=replicates,
            schedule_length=schedule_length,
        )
    else:
        click.echo(
            f"Unknown model '{model}'. Supported: fbp, mle, map, unet, diffusion."
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
