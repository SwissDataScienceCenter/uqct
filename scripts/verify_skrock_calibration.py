"""Re-run the seed-0 SK-ROCK eval on cells whose TV weight was recalibrated.

Reads ``[[eval-sparse.skrock.calibrated]]`` from ``uqct/settings.toml``, and for
every listed (dataset, intensity) cell × every chunk of ``image_range`` (default
``[10, 110)`` in chunks of 10), deletes any stale
``results/runs/skrock:DS:I:True:LO-HI:SEED:*.{parquet,h5}`` and re-runs SK-ROCK at
the verification budget below so the post-recalibration coverage plots
(``scripts/plot_skrock_uq.py``) reflect the new betas. Cells not in the
``calibrated`` block keep their existing seed-0 results untouched.

This is the "short run, one seed" verification step: ~``len(calibrated) * 10`` chunk
runs, several hours on a single GPU. ``--keep-existing`` skips chunks that already
have a (presumably already-verified) result.
"""

from __future__ import annotations

import tomllib

import click

from uqct.other_methods.skrock import run_skrock
from uqct.utils import get_results_dir, get_root_dir

# Verification budget = the budget the original seed-0 sweep used (matches the
# existing results/runs/skrock:*.parquet files; see their `n_burnin`/`n_samples`
# columns). The settings.toml comment lists 1000/1000 as the paper budget, but the
# in-tree sweep used 500/500 -- keep parity for an apples-to-apples comparison.
N_BURNIN = 500
N_SAMPLES = 500


@click.command()
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Which seed to verify (only seed 0 has existing results to overwrite).",
)
@click.option(
    "--chunk",
    type=int,
    default=10,
    show_default=True,
    help="Image-chunk size; matches the production sweep.",
)
@click.option(
    "--keep-existing/--delete-existing",
    default=False,
    show_default=True,
    help="If a chunk's parquet already exists, skip it (use for resumes).",
)
@click.option(
    "--datasets",
    default="",
    show_default=True,
    help="Restrict to these datasets, comma-separated (default: all calibrated).",
)
@click.option(
    "--intensities",
    default="",
    show_default=True,
    help="Restrict to these intensities, comma-separated (default: all calibrated).",
)
def main(seed: int, chunk: int, keep_existing: bool, datasets: str, intensities: str):
    root = get_root_dir()
    with open(root / "uqct" / "settings.toml", "rb") as f:
        cfg = tomllib.load(f)["eval-sparse"]
    sk = cfg["skrock"]
    cells = list(sk.get("calibrated", []))
    if datasets:
        ds_filter = set(datasets.split(","))
        cells = [c for c in cells if c["dataset"] in ds_filter]
    if intensities:
        i_filter = {float(x) for x in intensities.split(",")}
        cells = [c for c in cells if float(c["intensity"]) in i_filter]
    if not cells:
        raise SystemExit("No calibrated cells match the filter.")

    start, end = cfg["image_range"]
    chunks = [(i, min(i + chunk, end)) for i in range(start, end, chunk)]
    runs_dir = get_results_dir() / "runs"

    n_total = len(cells) * len(chunks)
    n_angles = cfg.get("n_angles", 200)
    max_angle = cfg.get("max_angle", 180)

    print(
        f"Verifying {len(cells)} recalibrated cells x {len(chunks)} chunks "
        f"= {n_total} runs at budget {N_BURNIN}+{N_SAMPLES} (seed={seed})."
    )

    done = 0
    for c in cells:
        ds = c["dataset"]
        intensity = float(c["intensity"])
        tv_weight = float(c["tv_weight"])
        mult = c.get("tv_weight_multiplier", "?")
        for lo, hi in chunks:
            done += 1
            prefix = f"skrock:{ds}:{intensity}:True:{lo}-{hi}:{seed}:"
            existing = sorted(runs_dir.glob(f"{prefix}*"))
            if existing and keep_existing:
                print(f"[{done}/{n_total}] SKIP (exists) {prefix}*")
                continue
            for p in existing:
                p.unlink()
            print(
                f"[{done}/{n_total}] {ds} I={intensity:.0e} [{lo},{hi}) "
                f"tv_weight=x{mult} ({tv_weight:.3e})  burnin={N_BURNIN} samples={N_SAMPLES}",
                flush=True,
            )
            run_skrock(
                dataset=ds,
                sparse=True,
                total_intensity=intensity,
                image_range=(lo, hi),
                seed=seed,
                n_angles=n_angles,
                max_angle=max_angle,
                n_burnin=N_BURNIN,
                n_samples=N_SAMPLES,
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
    print(f"\nDone: {n_total} runs.")


if __name__ == "__main__":
    main()
