import sys
import tomllib
from typing import Any, Literal

import click
import torch

from uqct.datasets.utils import DatasetName
from uqct.eval.bootstrapping import run_bootstrapping
from uqct.eval.diffusion import run_diffusion
from uqct.eval.equivariant_bootstrapping import run_equivariant_bootstrapping
from uqct.eval.fbp import run_fbp
from uqct.eval.iterative import run_iterative
from uqct.eval.unet import run_unet
from uqct.eval.unet_ensemble import run_unet_ensemble
from uqct.other_methods.skrock import run_skrock
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

    # Chunk into batches of 10
    start, end = full_image_range
    chunks_10 = [(i, min(i + 10, end)) for i in range(start, end, 10)]

    bs_cfg = settings.get("bootstrapping", {})
    methods = bs_cfg.get("methods", ["fbp"])
    if "intensities" in bs_cfg:
        intensities = bs_cfg["intensities"]
    seed_range = bs_cfg.get("seed_range", [0, 1])
    seeds = list(range(seed_range[0], seed_range[1]))

    grid = []
    # Order: Dataset -> Intensity -> Seeds -> Method -> Image Ranges
    for d in datasets:
        for i_lvl in intensities:
            for s in seeds:
                for m in methods:
                    for r in chunks_10:
                        grid.append(
                            {
                                "method": m,
                                "dataset": d,
                                "intensity": i_lvl,
                                "seed": s,
                                "image_range": r,
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

            ct_settings = CTSettings(**ct_settings_kwargs)  # type: ignore

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


# ---------------------------------------------------------------------------
# Shared plumbing for the per-method subcommands (bootstrapping / equivariant-
# bootstrapping / skrock): settings loading, the (dataset x intensity x seed x
# image-chunk) grid, the --no-duplicate skip check, and calibrated-cell lookup.
# ---------------------------------------------------------------------------
def _load_eval_section(
    sparse: bool, method_key: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the ``eval-sparse`` / ``eval-dense`` section of ``settings.toml`` and
    the ``method_key`` sub-table. Echoes and ``sys.exit(1)`` on any missing piece.
    Returns ``(section_settings, method_cfg)``."""
    settings_path = get_root_dir() / "uqct" / "settings.toml"
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
    if method_key not in settings:
        click.echo(f"No [{method_key}] config found in {section}")
        sys.exit(1)
    return settings, settings[method_key]


def _build_chunked_grid(
    settings: dict[str, Any], method_key: str, chunk: int = 10
) -> list[dict[str, Any]]:
    """Cartesian grid (dataset x intensity x seed x image-chunk) for the methods
    that chunk the image range into blocks of ``chunk`` images (equivariant
    bootstrapping, SK-ROCK). ``settings[method_key]`` may override ``intensities``
    (default ``total_intensity_values``) and ``seed_range`` (default ``[0, 1]``)."""
    cfg = settings.get(method_key, {})
    intensities = cfg.get("intensities", settings["total_intensity_values"])
    seed_lo, seed_hi = cfg.get("seed_range", [0, 1])
    start, end = settings["image_range"]
    chunks = [(i, min(i + chunk, end)) for i in range(start, end, chunk)]
    return [
        {"dataset": d, "intensity": i_lvl, "seed": s, "image_range": r}
        for d in settings["datasets"]
        for i_lvl in intensities
        for s in range(seed_lo, seed_hi)
        for r in chunks
    ]


def _run_exists(
    model: str,
    dataset: str,
    intensity: float,
    sparse: bool,
    image_range: tuple[int, int],
    seed: int,
) -> bool:
    """Whether a results parquet already exists for this cell (same prefix the
    ``run`` command's ``--no-duplicate`` check uses: ``model:dataset:intensity:
    sparse:start-end:seed:timestamp.parquet``)."""
    from uqct.utils import get_results_dir

    prefix = f"{model}:{dataset}:{intensity}:{sparse}:{image_range[0]}-{image_range[1]}:{seed}:"
    return bool(list((get_results_dir() / "runs").glob(f"{prefix}*.parquet")))


def _skip_if_done(
    model: str, task: dict[str, Any], sparse: bool, duplicate: bool
) -> bool:
    """``True`` (and logs a "Skipping ..." line) when ``--no-duplicate`` is in
    effect and a result for this cell already exists; ``False`` otherwise."""
    if duplicate or not _run_exists(
        model,
        task["dataset"],
        task["intensity"],
        sparse,
        task["image_range"],
        task["seed"],
    ):
        return False
    click.echo(
        f"Skipping {model} {task['dataset']} I={task['intensity']:.0e} "
        f"{task['image_range']} seed={task['seed']} (results already exist)."
    )
    return True


def _find_calibrated(
    cfg: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any] | None:
    """The ``[[...method....calibrated]]`` entry matching this task's
    ``(dataset, intensity)``, if one was recorded; else ``None``."""
    for entry in cfg.get("calibrated", []):
        if entry.get("dataset") == task["dataset"] and float(
            entry.get("intensity", -1)
        ) == float(task["intensity"]):
            return entry
    return None


def _dispatch_jobs(
    grid: list[dict[str, Any]], job_id: int | None, label: str, run_one
) -> None:
    """Run ``run_one(task)`` for one selected ``job_id`` (a SLURM array element) or,
    if ``job_id is None``, every task in ``grid``. Echoes progress; exits on a bad id."""
    if job_id is not None:
        if not (0 <= job_id < len(grid)):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid) - 1})")
            sys.exit(1)
        click.echo(f"Running {label} job {job_id}/{len(grid) - 1}.")
        run_one(grid[job_id])
        return
    click.echo(f"Running {len(grid)} {label} jobs locally...")
    for i, task in enumerate(grid):
        click.echo(f"\n--- Job {i + 1}/{len(grid)} ---")
        run_one(task)


@cli.command(name="bootstrapping")
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
@click.option(
    "--duplicate/--no-duplicate",
    default=False,
    show_default=True,
    help="Allow duplicate runs. If False (default), skips cells whose results already exist.",
)
def bootstrapping(job_id: int | None, sparse: bool, duplicate: bool):
    """Run bootstrapping evaluation."""
    settings, bs_cfg = _load_eval_section(sparse, "bootstrapping")
    n_bootstraps = bs_cfg.get("n_bootstraps", 1000)
    grid = build_bootstrapping_grid(settings)

    if job_id is not None:
        if not (0 <= job_id < len(grid)):
            click.echo(f"Job ID {job_id} out of range (0-{len(grid) - 1})")
            sys.exit(1)
        task = grid[job_id]
        click.echo(f"Running Bootstrapping Job ID {job_id} (Method: {task['method']})")
        execute_bootstrapping_task(task, sparse, settings, n_bootstraps, duplicate)
    else:
        click.echo(f"Running {len(grid)} bootstrapping jobs locally...")
        for i, task in enumerate(grid):
            click.echo(f"\n--- Job {i + 1}/{len(grid)}: {task['method']} ---")
            execute_bootstrapping_task(task, sparse, settings, n_bootstraps, duplicate)


def execute_bootstrapping_task(
    task, sparse, settings, n_bootstraps, duplicate: bool = False
):
    if _skip_if_done(f"bootstrapping_{task['method']}", task, sparse, duplicate):
        return

    run_bootstrapping(
        dataset=task["dataset"],
        sparse=sparse,
        total_intensity=task["intensity"],
        image_range=task["image_range"],
        seed=task["seed"],
        n_angles=settings.get("n_angles", 200),
        max_angle=settings.get("max_angle", 180),
        n_bootstraps=n_bootstraps,
        method=task["method"],
    )


@cli.command(name="equivariant-bootstrapping")
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
@click.option(
    "--duplicate/--no-duplicate",
    default=False,
    show_default=True,
    help="Allow duplicate runs. If False (default), skips cells whose results already exist.",
)
def equivariant_bootstrapping(job_id: int | None, sparse: bool, duplicate: bool):
    """Run equivariant bootstrapping evaluation (estimator: FBPUNet)."""
    settings, eb_cfg = _load_eval_section(sparse, "equivariant_bootstrapping")
    grid = _build_chunked_grid(settings, "equivariant_bootstrapping")
    n_angles = settings.get("n_angles", 200)
    max_angle = settings.get("max_angle", 180)
    n_bootstraps = eb_cfg.get("n_bootstraps", 100)

    def run_one(task: dict[str, Any]) -> None:
        if _skip_if_done("equivariant_bootstrapping", task, sparse, duplicate):
            return
        entry = _find_calibrated(eb_cfg, task)
        rotation_std_deg = (
            float(entry["rotation_std_deg"])
            if entry
            else float(eb_cfg.get("rotation_std_deg", 8.0))
        )
        flip = bool(entry["flip"]) if entry else bool(eb_cfg.get("flip", False))
        run_equivariant_bootstrapping(
            dataset=task["dataset"],
            sparse=sparse,
            total_intensity=task["intensity"],
            image_range=task["image_range"],
            seed=task["seed"],
            n_angles=n_angles,
            max_angle=max_angle,
            n_bootstraps=n_bootstraps,
            rotation_std_deg=rotation_std_deg,
            flip=flip,
        )

    _dispatch_jobs(grid, job_id, "equivariant bootstrapping", run_one)


@cli.command(name="skrock")
@click.option(
    "--job-id", type=int, default=None, help="SLURM array job ID to select config"
)
@click.option("--sparse", is_flag=True, default=True, help="Use sparse setting")
@click.option(
    "--duplicate/--no-duplicate",
    default=False,
    show_default=True,
    help="Allow duplicate runs. If False (default), skips cells whose results already exist.",
)
def skrock(job_id: int | None, sparse: bool, duplicate: bool):
    """Run SK-ROCK Langevin sampler evaluation."""
    settings, sk_cfg = _load_eval_section(sparse, "skrock")
    grid = _build_chunked_grid(settings, "skrock")
    n_angles = settings.get("n_angles", 200)
    max_angle = settings.get("max_angle", 180)

    def run_one(task: dict[str, Any]) -> None:
        if _skip_if_done("skrock", task, sparse, duplicate):
            return
        entry = _find_calibrated(sk_cfg, task)
        # ECE-calibrated TV weight for this (dataset, intensity) if one was recorded
        # (scripts/calibrate_skrock.py), else sk_cfg["tv_weight"] (normally -1 = auto).
        tv_weight = (
            float(entry["tv_weight"]) if entry else float(sk_cfg.get("tv_weight", -1.0))
        )
        run_skrock(
            dataset=task["dataset"],
            sparse=sparse,
            total_intensity=task["intensity"],
            image_range=task["image_range"],
            seed=task["seed"],
            n_angles=n_angles,
            max_angle=max_angle,
            n_burnin=sk_cfg.get("n_burnin", 1000),
            n_samples=sk_cfg.get("n_samples", 1000),
            n_stages=sk_cfg.get("n_stages", 10),
            eta=sk_cfg.get("eta", 0.05),
            dt_perc=sk_cfg.get("dt_perc", 0.95),
            prior=sk_cfg.get("prior", "tv"),
            tv_weight=tv_weight,
            tv_weight_calibration=sk_cfg.get("tv_weight_calibration", 2.0),
            my_lambda_factor=sk_cfg.get("my_lambda_factor", 1.0),
            chambolle_iters=sk_cfg.get("chambolle_iters", 25),
            lipschitz_iters=sk_cfg.get("lipschitz_iters", 40),
            sampler_seed=sk_cfg.get("sampler_seed", 0),
        )

    _dispatch_jobs(grid, job_id, "SK-ROCK", run_one)


if __name__ == "__main__":
    cli()
