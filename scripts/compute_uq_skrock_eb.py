"""Per-image CI width + empirical coverage for SK-ROCK and equivariant bootstrap.

Mirrors the definitions in ``uqct.vis.plot_uq.compute_stats_from_samples`` exactly
(clamp lo/hi to [0, 1] before coverage and width; *no* circular mask on width or
coverage) so the numbers can be plotted alongside the cached
``results/uq_comparison.json`` entries for boundary / fbp-bootstrap / unet-bootstrap.

Filters
-------
* seed = 0 only (the only seed any non-diffusion method has on disk anyway).
* image_range start >= 10 -- skips the calibration window [0, 10) used by
  ``scripts/calibrate_*`` runs.
* dedup by (model, dataset, intensity, image_range), keeping the most recent
  timestamp (same prefix the CLI's ``--no-duplicate`` uses).

Output
------
``results/plots/uq_widths_coverage_<method>.parquet`` with columns
``method, dataset, intensity, image, ci_method, width, ind_cov, sim_cov``.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.uq import (
    basic_ci,
    gaussian_ci,
    gaussian_conservative_ci,
    percentile_ci,
    simultaneous_ci,
    student_t_bonferroni_ci,
    student_t_ci,
    studentized_ci,
)
from uqct.utils import get_results_dir
from uqct.vis.plot_uq import get_ground_truth

# Project-wide error level.
DELTA = 0.05

CI_FNS = {
    "gaussian": gaussian_ci,
    "gaussian_cons": gaussian_conservative_ci,
    "percentile": percentile_ci,
    "basic": basic_ci,
    "studentized": studentized_ci,
    "simultaneous": simultaneous_ci,
    "student_t": student_t_ci,
    "student_t_bonferroni": student_t_bonferroni_ci,
}

# Filename layout from `evaluate_and_save`:
#   <model>:<dataset>:<intensity>:<sparse>:<start>-<end>:<seed>:<timestamp>.h5
NAME_RE = re.compile(
    r"^(?P<model>[\w]+):(?P<ds>[a-z]+):(?P<intensity>[\d.eE+-]+):"
    r"(?P<sparse>True|False):(?P<start>\d+)-(?P<end>\d+):(?P<seed>\d+):"
)


def parse_name(path: Path) -> dict | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    return {
        "model": m["model"],
        "dataset": m["ds"],
        "intensity": float(m["intensity"]),
        "sparse": m["sparse"] == "True",
        "start": int(m["start"]),
        "end": int(m["end"]),
        "seed": int(m["seed"]),
    }


def latest_per_cell(files: list[Path]) -> list[Path]:
    """Group by (model, dataset, intensity, start, end, seed); keep latest mtime."""
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in files:
        meta = parse_name(f)
        if meta is None:
            continue
        key = (
            meta["model"],
            meta["dataset"],
            meta["intensity"],
            meta["start"],
            meta["end"],
            meta["seed"],
        )
        groups[key].append(f)
    out: list[Path] = []
    for key, fs in groups.items():
        out.append(max(fs, key=lambda p: p.stat().st_mtime))
    return sorted(out)


def compute_for_image(samples: torch.Tensor, target: torch.Tensor) -> dict:
    """samples: (R, H, W); target: (H, W). Returns per-CI-method (width, ind_cov, sim_cov)."""
    row = {}
    for ci_name, ci_fn in CI_FNS.items():
        try:
            lo, hi = ci_fn(samples, delta=DELTA, bdim=0)
        except Exception:  # e.g. studentized w/ degenerate samples
            row[f"{ci_name}_width"] = float("nan")
            row[f"{ci_name}_ind_cov"] = float("nan")
            row[f"{ci_name}_sim_cov"] = float("nan")
            continue
        lo = lo.clamp(0, 1)
        hi = hi.clamp(0, 1)
        in_bounds = (target >= lo) & (target <= hi)
        row[f"{ci_name}_width"] = (hi - lo).abs().clamp(0, 1).mean().item()
        row[f"{ci_name}_ind_cov"] = in_bounds.float().mean().item()
        row[f"{ci_name}_sim_cov"] = float(in_bounds.all().item())
    return row


def process_method(
    method: str,
    glob_pattern: str,
    chosen_ci: str,
    device: torch.device,
    image_start_min: int = 10,
    runs_subdir: str = "runs",
    h5_key: str = "preds",
    sample_axis_slice: tuple | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Walk all ``glob_pattern`` files, return one row per (image, ci_method).

    ``runs_subdir`` defaults to ``results/runs/`` but can be set to e.g.
    ``boundary_sampling`` for the diffusion-boundary outputs. ``h5_key`` and
    ``sample_axis_slice`` let callers extract the right tensor shape per method
    (default = ``preds[:, -1]`` -> ``(N, R, H, W)``). ``seeds`` filters to a
    specific seed list; ``None`` means seed=0 only (back-compat).
    """
    if seeds is None:
        seeds = [0]
    runs_dir = get_results_dir() / runs_subdir
    files = [Path(p) for p in glob(str(runs_dir / glob_pattern))]
    print(f"{method}: {len(files)} files matched {glob_pattern}")
    parsed = [(f, parse_name(f)) for f in files]
    kept = [
        (f, m)
        for f, m in parsed
        if m is not None and m["seed"] in seeds and m["start"] >= image_start_min
    ]
    print(f"{method}: {len(kept)} after seeds={seeds} + start>={image_start_min}")
    kept_files = latest_per_cell([f for f, _ in kept])
    print(f"{method}: {len(kept_files)} after dedup")

    rows: list[dict] = []
    for k, f in enumerate(kept_files):
        meta = parse_name(f)
        assert meta is not None
        t0 = time.time()
        with h5py.File(f, "r") as h:
            arr = h[h5_key]
            if sample_axis_slice is not None:
                # Used by boundary: (N, S, R, 1, H, W) -> (N, R, H, W).
                arr = arr[sample_axis_slice]
            elif arr.ndim == 5:
                # Default: (N, T, R, H, W) -> last schedule step -> (N, R, H, W).
                arr = arr[:, -1, :, :, :]
            else:
                arr = arr[:]
        samples = torch.from_numpy(arr).to(device).float()
        gt = get_ground_truth(meta["dataset"], (meta["start"], meta["end"]))
        if gt.shape[-2:] != samples.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(1), size=samples.shape[-2:], mode="area"
            ).squeeze(1)
        n = min(len(gt), len(samples))
        gt, samples = gt[:n], samples[:n]
        for i in range(n):
            r = compute_for_image(samples[i], gt[i])
            r.update(
                method=method,
                dataset=meta["dataset"],
                intensity=meta["intensity"],
                seed=meta["seed"],
                image=meta["start"] + i,
                chosen_ci=chosen_ci,
            )
            rows.append(r)
        print(
            f"  [{k + 1:>3}/{len(kept_files)}] {f.name[:80]}... "
            f"n={n} dt={time.time() - t0:.1f}s"
        )
        del samples, gt
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "skrock",
            "equivariant_bootstrapping_fbp",
            "equivariant_bootstrapping",
            "bootstrapping_fbp",
            "bootstrapping_unet",
            "boundary",
        ],
        help="Which method to process.",
    )
    parser.add_argument(
        "--image-start-min",
        type=int,
        default=10,
        help="Drop cells whose `start` < this. Use 0 to keep the calibration window.",
    )
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[0],
        help="Comma-separated seed list (default: 0). Use 0,1,2,...,9 for all.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    if args.method == "skrock":
        df = process_method(
            method="skrock",
            glob_pattern="skrock:*.h5",
            chosen_ci="percentile",  # SK-ROCK posterior samples; percentile is the natural choice.
            device=dev,
            image_start_min=args.image_start_min,
            seeds=args.seeds,
        )
    elif args.method in ("bootstrapping_fbp", "bootstrapping_unet"):
        df = process_method(
            method=args.method,
            glob_pattern=f"{args.method}:*.h5",
            chosen_ci="percentile",
            device=dev,
            image_start_min=args.image_start_min,
            seeds=args.seeds,
        )
    elif args.method == "boundary":
        # Boundary lives under results/boundary_sampling/, with h5 key "sampled_images"
        # shaped (N, S, R, 1, H, W). We slice the last sampling step + drop the singleton.
        df = process_method(
            method="boundary",
            glob_pattern="boundary_diffusion:*.h5",
            chosen_ci="student_t",  # 10 replicates is too few for a percentile CI.
            device=dev,
            image_start_min=args.image_start_min,
            runs_subdir="boundary_sampling",
            h5_key="sampled_images",
            sample_axis_slice=(slice(None), -1, slice(None), 0, slice(None), slice(None)),
            seeds=args.seeds,
        )
    else:
        # The current code names files `equivariant_bootstrapping:*` (estimator is
        # always FBPUNet now). We keep the *parquet label* as
        # `equivariant_bootstrapping_fbp` so the plot script and the cached
        # uq_comparison.json stay aligned -- but read whichever prefix is on disk.
        glob_pattern = "equivariant_bootstrapping:*.h5"
        if args.method == "equivariant_bootstrapping_fbp":
            # Back-compat: also accept the old _fbp suffix on disk.
            from glob import glob
            runs_dir = get_results_dir() / "runs"
            if not glob(str(runs_dir / glob_pattern)) and glob(
                str(runs_dir / "equivariant_bootstrapping_fbp:*.h5")
            ):
                glob_pattern = "equivariant_bootstrapping_fbp:*.h5"
        df = process_method(
            method="equivariant_bootstrapping_fbp",
            glob_pattern=glob_pattern,
            chosen_ci="percentile",
            device=dev,
            image_start_min=args.image_start_min,
            seeds=args.seeds,
        )

    out = args.out or (
        get_results_dir() / "plots" / f"uq_widths_coverage_{args.method}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(
        f"wrote {out}  ({len(df)} rows, "
        f"{df.groupby(['dataset','intensity']).size().shape[0]} cells)"
    )


if __name__ == "__main__":
    main()
