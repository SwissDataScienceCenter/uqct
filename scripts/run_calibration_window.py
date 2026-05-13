"""Generate predictions on the alpha-calibration window (image_range=(0,10)).

For one (method, dataset, intensity) cell at seed 0. Reads per-cell calibrated
hyperparams from uqct/settings.toml so the *hyperparam-only* calibration that
already shapes the test-window predictions is also baked into the calibration
window -- the alpha-cal layer then sits cleanly on top of that.

Submitted by ``scripts/submit_calibration_window.sh``; each SLURM task may
process several ``--cell-idx`` values back-to-back to amortise startup overhead.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import click

DATASETS = ["lung", "composite", "lamino"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
CELLS = [(d, i) for d in DATASETS for i in INTENSITIES]  # 18 cells

CAL_IMAGE_RANGE = (0, 10)
CAL_SEED = 0


def _calibrated_entry(cfg_section: dict, dataset: str, intensity: float) -> dict | None:
    for entry in cfg_section.get("calibrated", []):
        if entry.get("dataset") == dataset and float(entry.get("intensity", -1)) == intensity:
            return entry
    return None


def _run_skrock(dataset: str, intensity: float, cfg: dict) -> None:
    from uqct.other_methods.skrock import run_skrock
    sk = cfg["skrock"]
    cal = _calibrated_entry(sk, dataset, intensity)
    tv_weight = float(cal["tv_weight"]) if cal else float(sk.get("tv_weight", -1.0))
    run_skrock(
        dataset=dataset, sparse=True, total_intensity=intensity,
        image_range=CAL_IMAGE_RANGE, seed=CAL_SEED,
        n_angles=cfg.get("n_angles", 200), max_angle=cfg.get("max_angle", 180),
        n_burnin=sk.get("n_burnin", 1000),
        n_samples=sk.get("n_samples", 1000),
        n_stages=sk.get("n_stages", 10),
        eta=sk.get("eta", 0.05),
        dt_perc=sk.get("dt_perc", 0.95),
        prior=sk.get("prior", "tv"),
        tv_weight=tv_weight,
        tv_weight_calibration=sk.get("tv_weight_calibration", 2.0),
        my_lambda_factor=sk.get("my_lambda_factor", 1.0),
        chambolle_iters=sk.get("chambolle_iters", 25),
        lipschitz_iters=sk.get("lipschitz_iters", 40),
        sampler_seed=sk.get("sampler_seed", 0),
    )


def _run_eb(dataset: str, intensity: float, cfg: dict) -> None:
    from uqct.other_methods.equivariant_bootstrapping import run_equivariant_bootstrapping
    eb = cfg["equivariant_bootstrapping"]
    cal = _calibrated_entry(eb, dataset, intensity)
    rotation_std_deg = float(cal["rotation_std_deg"]) if cal else float(eb.get("rotation_std_deg", 8.0))
    flip = bool(cal["flip"]) if cal else bool(eb.get("flip", False))
    run_equivariant_bootstrapping(
        dataset=dataset, sparse=True, total_intensity=intensity,
        image_range=CAL_IMAGE_RANGE, seed=CAL_SEED,
        n_angles=cfg.get("n_angles", 200), max_angle=cfg.get("max_angle", 180),
        n_bootstraps=eb.get("n_bootstraps", 100),
        rotation_std_deg=rotation_std_deg, flip=flip,
    )


def _run_bootstrapping(dataset: str, intensity: float, cfg: dict, estimator: str) -> None:
    from uqct.other_methods.bootstrapping import run_bootstrapping
    bs = cfg["bootstrapping"]
    run_bootstrapping(
        dataset=dataset, sparse=True, total_intensity=intensity,
        image_range=CAL_IMAGE_RANGE, seed=CAL_SEED,
        n_angles=cfg.get("n_angles", 200), max_angle=cfg.get("max_angle", 180),
        n_bootstraps=bs.get("n_bootstraps", 1000),
        method=estimator,
    )


def _run_boundary(dataset: str, intensity: float, cfg: dict, full_cfg: dict) -> None:
    """Diffusion -> diffusion-boundary on the resulting parquet."""
    from uqct.eval.diffusion import run_diffusion
    from uqct.utils import get_results_dir
    df_cfg = full_cfg.get("eval", {}).get("diffusion", {})
    run_diffusion(
        dataset=dataset, sparse=True, cond=df_cfg.get("cond", True),
        total_intensity=intensity,
        schedule_length=cfg.get("schedule_length", 32),
        gradient_steps=df_cfg.get("gradient_steps", 20),
        guidance_lr=df_cfg.get("guidance_lr", 1e-2),
        image_range=CAL_IMAGE_RANGE, seed=CAL_SEED,
        replicates=df_cfg.get("replicates", 16),
        n_angles=cfg.get("n_angles", 200),
        schedule_start=cfg.get("schedule_start", 10),
        schedule_type=cfg.get("schedule_type", "exp"),
        max_angle=cfg.get("max_angle", 180),
    )
    runs = get_results_dir() / "runs"
    intensity_str = f"{intensity:g}" if intensity == int(intensity) else str(intensity)
    # match either '1e+04' or '10000.0' style filenames
    patterns = [
        f"diffusion:{dataset}:{intensity}:True:0-10:0:*.parquet",
        f"diffusion:{dataset}:{intensity_str}:True:0-10:0:*.parquet",
    ]
    found: list[Path] = []
    for pat in patterns:
        found += list(runs.glob(pat))
    if not found:
        sys.exit(f"diffusion parquet not found for {dataset} {intensity}: tried {patterns}")
    latest = max(found, key=lambda p: p.stat().st_mtime)
    print(f"diffusion_boundary input: {latest}")
    subprocess.run(
        [sys.executable, "-m", "uqct.eval.diffusion_boundary",
         str(latest), "--idx-range", "0", "10", "--replicates", "10"],
        check=True,
    )


@click.command()
@click.option(
    "--method", required=True,
    type=click.Choice([
        "skrock", "equivariant_bootstrapping",
        "bootstrapping_fbp", "bootstrapping_unet", "boundary",
    ]),
)
@click.option("--cell-idx", required=True, type=int,
              help="0..17 over (dataset, intensity) pairs in lexicographic order.")
def main(method: str, cell_idx: int) -> None:
    if not (0 <= cell_idx < len(CELLS)):
        sys.exit(f"cell-idx {cell_idx} out of range [0, {len(CELLS)})")
    dataset, intensity = CELLS[cell_idx]
    print(f"=== calibration window: {method} {dataset} I={intensity:.0e} ===")

    settings_path = Path(__file__).parent.parent / "uqct" / "settings.toml"
    full_cfg = tomllib.loads(settings_path.read_text())
    cfg = full_cfg["eval-sparse"]

    if method == "skrock":
        _run_skrock(dataset, intensity, cfg)
    elif method == "equivariant_bootstrapping":
        _run_eb(dataset, intensity, cfg)
    elif method == "bootstrapping_fbp":
        _run_bootstrapping(dataset, intensity, cfg, "fbp")
    elif method == "bootstrapping_unet":
        _run_bootstrapping(dataset, intensity, cfg, "unet")
    elif method == "boundary":
        _run_boundary(dataset, intensity, cfg, full_cfg)


if __name__ == "__main__":
    main()
