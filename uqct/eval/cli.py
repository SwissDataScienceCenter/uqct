import sys
import tomllib
from typing import Any, Literal

import click

import torch
from uqct.datasets.utils import DatasetName
from uqct.eval.bootstrapping import run_bootstrapping
from uqct.eval.diffusion import run_diffusion
from uqct.eval.fbp import run_fbp
from uqct.eval.iterative import run_iterative
from uqct.eval.unet import run_unet
from uqct.eval.unet_ensemble import run_unet_ensemble
from uqct.utils import get_root_dir


def build_grid(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the job grid based on settings."""
    all_models = settings.get(
        "models",
        ["fbp", "mle", "map", "unet", "unet_ensemble", "diffusion"],
    )
    datasets = settings["datasets"]
    intensities = settings["total_intensity_values"]
    seed_range = settings.get("seed_range", [0, 1])
    seeds = list(range(seed_range[0], seed_range[1]))
    full_image_range = tuple(settings["image_range"])
    start, end = full_image_range

    grid = []

    # 1. FBP: 1 Global Job. Loops Datasets, Intensities, Seeds. Full range.
    if "fbp" in all_models:
        grid.append(
            {
                "model": "fbp",
                "datasets": datasets,  # List -> Loop
                "intensities": intensities,  # List -> Loop
                "image_range": full_image_range,
                "seeds": seeds,  # List -> Loop
            }
        )

    # 2. MLE/MAP: 1 Job per (Model, Dataset, Intensity). Loops seeds, chunk-20.
    iterative = ["mle"]
    chunks_20 = [(i, min(i + 20, end)) for i in range(start, end, 20)]
    for m in iterative:
        if m in all_models:
            for d in datasets:
                for i in intensities:
                    grid.append(
                        {
                            "model": m,
                            "dataset": d,
                            "intensity": i,  # Scalar -> One Job
                            "seeds": seeds,  # List -> Loop
                            "image_ranges": chunks_20,  # List -> Loop
                        }
                    )

    # 3. U-Net: 1 Job per (Dataset, Intensity). Loops seeds. Full range.
    if "unet" in all_models:
        for d in datasets:
            for i in intensities:
                grid.append(
                    {
                        "model": "unet",
                        "dataset": d,  # Scalar -> One Job
                        "intensity": i,  # Scalar -> One Job
                        "seeds": seeds,  # List -> Loop
                        "image_range": full_image_range,
                    }
                )

    # 4. U-Net Ensemble: 1 Job per (Dataset, Intensity, Seed). Full range.
    if "unet_ensemble" in all_models:
        for d in datasets:
            for i in intensities:
                for s in seeds:
                    grid.append(
                        {
                            "model": "unet_ensemble",
                            "dataset": d,
                            "intensity": i,
                            "seed": s,  # Scalar -> Scalar
                            "image_range": full_image_range,
                        }
                    )

    # 5. Diffusion: Granular. 1 Job per (Dataset, Intensity, Seed, Chunk-10).
    if "diffusion" in all_models:
        chunks_10 = [(i, min(i + 10, end)) for i in range(start, end, 10)]
        for d in datasets:
            for i in intensities:
                for s in seeds:
                    for r in chunks_10:
                        grid.append(
                            {
                                "model": "diffusion",
                                "dataset": d,
                                "intensity": i,
                                "seed": s,
                                "image_range": r,
                            }
                        )

    return grid


def build_bootstrapping_grid(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the bootstrapping job grid (Unchunked)."""
    datasets = settings["datasets"]
    intensities = settings["total_intensity_values"]
    full_image_range = tuple(settings["image_range"])

    bs_cfg = settings.get("bootstrapping", {})
    methods = bs_cfg.get("methods", ["fbp"])
    if "intensities" in bs_cfg:
        intensities = bs_cfg["intensities"]
    seed_range = bs_cfg.get("seed_range", [0, 1])
    seeds = list(range(seed_range[0], seed_range[1]))

    grid = []
    # Order: Dataset -> Intensity -> Seeds -> Method
    for d in datasets:
        for i_lvl in intensities:
            for s in seeds:
                for m in methods:
                    grid.append(
                        {
                            "method": m,
                            "dataset": d,
                            "intensity": i_lvl,
                            "seed": s,
                            "image_range": full_image_range,
                        }
                    )
    return grid


@click.group()
def cli():
    """Unified evaluation CLI."""
    pass


@cli.command()
@click.option(
    "--model",
    required=False,
    type=click.Choice(["fbp", "mle", "map", "unet", "unet_ensemble", "diffusion"]),
    default=None,
    help="Model name (fbp, mle, map, unet, unet_ensemble, diffusion). Required only if running locally without job-id to filter.",
)
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
@click.option(
    "--duplicate/--no-duplicate",
    default=False,
    show_default=True,
    help="Allow duplicate runs. If False, skips if run exists.",
)
@click.option(
    "--recompute-metrics",
    is_flag=True,
    default=False,
    help="Recompute metrics using existing predictions.",
)
def run(
    model: Literal["fbp", "mle", "map", "unet", "unet_ensemble", "diffusion"] | None,
    job_id: int | None,
    sparse: bool,
    duplicate: bool,
    recompute_metrics: bool,
):
    """Run evaluation for a specific model and configuration."""

    root = get_root_dir()
    settings_path = root / "uqct" / "settings.toml"

    if not settings_path.exists():
        click.echo(f"Settings file not found at {settings_path}")
        sys.exit(1)

    section = "eval-sparse" if sparse else "eval-dense"
    with open(settings_path, "rb") as f:
        full_config = tomllib.load(f)
        if section not in full_config:
            click.echo(f"Section [{section}] not found in settings.toml")
            sys.exit(1)
        settings = full_config[section].copy()

        # Merge model-specific configs (mle, map, unet, diffusion) which live in [eval]
        if "eval" in full_config:
            settings.update(full_config["eval"])

    schedule_length = settings.get("schedule_length", 32)

    # ---------------------------------------------------------
    # Build Heterogeneous Job Grid
    # ---------------------------------------------------------
    grid = build_grid(settings)

    # ---------------------------------------------------------
    # Execution Logic
    # ---------------------------------------------------------

    if job_id is not None:
        if job_id < 0 or job_id >= len(grid):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid) - 1})")
            sys.exit(1)

        task = grid[job_id]
        if model and task["model"] != model:
            # If user supplied --model, we could check consistency, but --job-id is absolute.
            # Just warn or ignore.
            pass

        click.echo(f"Running Job ID {job_id} (Model: {task['model']})")
        execute_task(
            task, sparse, schedule_length, settings, duplicate, recompute_metrics
        )

    else:
        # Local execution - run all or filter
        tasks_to_run = grid
        if model:
            tasks_to_run = [t for t in grid if t["model"] == model]

        click.echo(f"Running {len(tasks_to_run)} jobs locally...")
        for i, task in enumerate(tasks_to_run):
            click.echo(
                f"\n--- Local Job {i + 1}/{len(tasks_to_run)}: {task['model']} ---"
            )
            execute_task(
                task, sparse, schedule_length, settings, duplicate, recompute_metrics
            )


def execute_task(
    task: dict,
    sparse: bool,
    schedule_length: int,
    settings: dict[str, Any],
    duplicate: bool,
    recompute_metrics: bool,
):
    """
    Recursively unfolds lists in `task` dict to call _dispatch with scalar values.
    Order of unpacking: Datasets -> Intensities -> Seeds -> ImageRanges.
    """
    # Keys that might be lists
    # Note: 'model' is always scalar here (defined in grid generation)

    # 1. Datasets
    ds_arg = task.get("datasets", task.get("dataset"))
    if isinstance(ds_arg, list):
        for d in ds_arg:
            # Create sub-task with scalar
            sub = task.copy()
            sub["dataset"] = d
            sub.pop("datasets", None)
            execute_task(
                sub, sparse, schedule_length, settings, duplicate, recompute_metrics
            )
        return
    else:
        dataset = ds_arg

    # 2. Intensities
    int_arg = task.get("intensities", task.get("intensity"))
    if isinstance(int_arg, list):
        for i in int_arg:
            sub = task.copy()
            sub["intensity"] = i
            sub.pop("intensities", None)
            execute_task(
                sub, sparse, schedule_length, settings, duplicate, recompute_metrics
            )
        return
    else:
        intensity = int_arg

    # 3. Seeds
    seed_arg = task.get("seeds", task.get("seed"))
    if isinstance(seed_arg, list):
        for s in seed_arg:
            sub = task.copy()
            sub["seed"] = s
            sub.pop("seeds", None)
            execute_task(
                sub, sparse, schedule_length, settings, duplicate, recompute_metrics
            )
        return
    else:
        seed = seed_arg

    # 4. Image Ranges
    rng_arg = task.get("image_ranges", task.get("image_range"))
    if isinstance(rng_arg, list):
        for r in rng_arg:
            sub = task.copy()
            sub["image_range"] = r
            sub.pop("image_ranges", None)
            execute_task(
                sub, sparse, schedule_length, settings, duplicate, recompute_metrics
            )
        return
    else:
        image_range = rng_arg

    # Base case: All scalars
    model = task["model"]

    print(
        f"  -> Dispatching: {model}, {dataset}, {intensity}, {image_range}, seed={seed}"
    )

    try:
        _dispatch(
            model=model,
            dataset=dataset,  # type: ignore
            intensity=intensity,  # type: ignore
            sparse=sparse,
            image_range=image_range,  # type: ignore
            schedule_length=schedule_length,
            settings=settings,
            duplicate=duplicate,
            recompute_metrics=recompute_metrics,
            seed=seed,  # type: ignore
        )
    except Exception as e:
        # Log error but try to continue if this is part of a larger loop?
        # Standard behavior: if one sub-task fails, print traceback.
        # Ideally we want to continue other seeds if one seed fails.
        import traceback

        click.echo(f"ERROR processing {model}, {dataset}, {intensity}, {seed}: {e}")
        traceback.print_exc()
        # Ensure we mark job as failed (non-zero exit) at the end if strict?
        # For now, print error allows monitor_jobs.py to catch "Error" string.
        pass


def _dispatch(
    model: str,
    dataset: DatasetName,
    intensity: float,
    sparse: bool,
    image_range: tuple[int, int],
    schedule_length: int,
    settings: dict[str, Any],
    duplicate: bool,
    recompute_metrics: bool,
    seed: int,
):
    import h5py

    from uqct.eval.run import evaluate_and_save, setup_experiment
    from uqct.utils import get_results_dir

    # Check for existing run
    results_dir = get_results_dir() / "runs"
    # Format: model:dataset:intensity:sparse:start-end:seed:timestamp.parquet
    prefix = f"{model}:{dataset}:{intensity}:{sparse}:{image_range[0]}-{image_range[1]}:{seed}:"
    pattern = f"{prefix}*.parquet"
    files = list(results_dir.glob(pattern))

    if files:
        if recompute_metrics:
            # Pick the latest one just in case there are multiple (though duplicate=False prevents that usually)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            target_metric_file = files[0]
            timestamp_str = target_metric_file.name[len(prefix) : -len(".parquet")]
            h5_file = target_metric_file.with_suffix(".h5")

            if not h5_file.exists():
                click.echo(
                    f"Warning: Found metrics file {target_metric_file} but no matching h5 file {h5_file}. Skipping recompute."
                )
                return

            click.echo(
                f"Recomputing metrics for {model} (Using existing predictions from {h5_file})"
            )

            # Load predictions
            with h5py.File(h5_file, "r") as f:
                preds = torch.from_numpy(f["preds"][:])

            # Setup experiment (needed for GT and Experiment object)
            # Common parameters from settings
            n_angles = settings.get("n_angles", 200)
            schedule_start = settings.get("schedule_start", 10)
            schedule_type = settings.get("schedule_type", "exp")
            max_angle = settings.get("max_angle", 180)

            # For recompute, we need to load preds to device if we want to be consistent, but evaluate_and_save expects tensor.
            # Usually preds are on CPU when loaded. evaluate_and_save handles it?
            # evaluate_and_save calls get_metrics which expects tensors.
            # verify device.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            preds = preds.to(device)

            # Re-run setup to get GT and Experiment
            gt, experiment, schedule = setup_experiment(
                dataset,
                image_range,
                intensity,
                sparse,
                seed,
                schedule_length,
                schedule_start,
                schedule_type,
                n_angles,
                max_angle,
            )

            ct_settings_kwargs = {
                "dataset": dataset,
                "total_intensity": intensity,
                "sparse": sparse,
                "image_start_index": image_range[0],
                "image_end_index": image_range[1],
            }
            if schedule is not None:
                ct_settings_kwargs["angle_schedule"] = schedule.tolist()

            from uqct.eval.run import CTSettings

            ct_settings = CTSettings(**ct_settings_kwargs)

            # Evaluate and save (overwriting logic is inside dump_parquet if we reuse timestamp)
            evaluate_and_save(
                preds=preds,
                gt=gt,
                experiment=experiment,
                schedule=schedule,
                ct_settings=ct_settings,
                model_name=model,
                seed=seed,
                extra_metadata=None,  # We lose extra metadata if we don't load it from old parquet...
                timestamp=timestamp_str,
            )
            return

        elif not duplicate:
            click.echo(f"Skipping {model} run (found {len(files)} existing files).")
            return

    # Common parameters from settings
    n_angles = settings.get("n_angles", 200)
    schedule_start = settings.get("schedule_start", 10)
    schedule_type = settings.get("schedule_type", "exp")
    max_angle = settings.get("max_angle", 180)

    if model == "fbp":
        run_fbp(
            dataset,
            sparse,
            intensity,
            image_range,
            seed,
            n_angles,
            schedule_start,
            schedule_type,
            schedule_length,
            max_angle,
        )

    elif model in ["mle", "map"]:
        # Load from settings
        cfg = settings.get(model, {})
        # Fallback defaults if not in settings (though settings.toml should have them)
        lr = cfg.get("lr", 1e-1)
        patience = cfg.get("patience", 50)
        max_steps = cfg.get("max_steps", 100)
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
            n_angles=n_angles,
            schedule_start=schedule_start,
            schedule_type=schedule_type,
            schedule_length=schedule_length,
            max_angle=max_angle,
        )

    elif model == "unet":
        cfg = settings.get("unet", {})
        member = cfg.get("member", 0)
        run_unet(
            dataset,
            sparse,
            intensity,
            image_range,
            seed,
            member,
            n_angles,
            schedule_start,
            schedule_type,
            schedule_length,
            max_angle,
        )

    elif model == "unet_ensemble":
        run_unet_ensemble(
            dataset,
            sparse,
            intensity,
            image_range,
            seed,
            n_angles,
            schedule_start,
            schedule_type,
            schedule_length,
            max_angle,
        )

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
            n_angles=n_angles,
            schedule_start=schedule_start,
            schedule_type=schedule_type,
            schedule_length=schedule_length,
            max_angle=max_angle,
        )

    else:
        click.echo(
            f"Unknown model '{model}'. Supported: fbp, mle, map, unet, unet_ensemble, diffusion."
        )
        sys.exit(1)


@cli.command(name="bootstrapping")
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
def bootstrapping(job_id: int | None, sparse: bool):
    """Run bootstrapping evaluation."""
    root = get_root_dir()
    settings_path = root / "uqct" / "settings.toml"

    if not settings_path.exists():
        click.echo(f"Settings file not found at {settings_path}")
        sys.exit(1)

    section = "eval-sparse" if sparse else "eval-dense"
    with open(settings_path, "rb") as f:
        full_config = tomllib.load(f)
        if section not in full_config:
            click.echo(f"Section [{section}] not found in settings.toml")
            sys.exit(1)
        settings = full_config[section]

    # Check if bootstrapping config exists
    if "bootstrapping" not in settings:
        click.echo(f"No [bootstrapping] config found in {section}")
        sys.exit(1)

    bs_cfg = settings["bootstrapping"]
    n_bootstraps = bs_cfg.get("n_bootstraps", 10)

    # Build Grid
    grid = build_bootstrapping_grid(settings)

    # Execution
    if job_id is not None:
        if job_id < 0 or job_id >= len(grid):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid) - 1})")
            sys.exit(1)

        task = grid[job_id]
        click.echo(f"Running Bootstrapping Job ID {job_id} (Method: {task['method']})")
        execute_bootstrapping_task(task, sparse, settings, n_bootstraps)
    else:
        click.echo(f"Running {len(grid)} bootstrapping jobs locally...")
        for i, task in enumerate(grid):
            click.echo(f"\n--- Job {i + 1}/{len(grid)}: {task['method']} ---")
            execute_bootstrapping_task(task, sparse, settings, n_bootstraps)


def execute_bootstrapping_task(task, sparse, settings, n_bootstraps):
    # Common parameters
    n_angles = settings.get("n_angles", 200)
    max_angle = settings.get("max_angle", 180)

    run_bootstrapping(
        dataset=task["dataset"],
        sparse=sparse,
        total_intensity=task["intensity"],
        image_range=task["image_range"],
        seed=task["seed"],
        n_angles=n_angles,
        max_angle=max_angle,
        n_bootstraps=n_bootstraps,
        method=task["method"],
    )


if __name__ == "__main__":
    cli()
