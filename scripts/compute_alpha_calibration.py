"""ECE-based post-hoc alpha calibration of UQ methods.

Two-stage pipeline so it works across machines (cluster has SK-ROCK/EB cal h5s;
local has bootstrap test h5s; boundary has cal on cluster + test on local):

Stage A (--stage cal):
    For each (dataset, intensity) cell at seed 0, image_range=(0,10):
      * load samples h5
      * compute method-appropriate CI bounds at a grid of alphas
      * sweep a scalar width-scale c; pick c* minimizing
          ECE(c) = (1/|A|) * sum_a |emp_cov(c, a) - (1 - a)|
        over the alpha grid. emp_cov measured on the 10 calibration images.
      * also record raw ECE (c=1) for comparison.
    Output: results/plots/alpha_cal_<method>_c.parquet with columns
        method, dataset, intensity, c_star, ece_raw, ece_cal.

Stage B (--stage test):
    For each cell, given c_star, load test h5 (image_range=10-110, seed 0),
    compute the calibrated 95% CI (alpha=0.05 scaled by c_star), report
    coverage and width on test images. Also report raw (c=1).
    Output: results/plots/alpha_cal_<method>_metrics.parquet with columns
        method, dataset, intensity, c_star, ind_cov_raw, ind_cov_cal,
        width_raw, width_cal, ece_raw, ece_cal (carried over).

Stage both: runs A then B in one process (only works when cal and test h5s
are on the same filesystem).

CI choice per method matches scripts/plot_uq_comparison.py:
    percentile for {skrock, equivariant_bootstrapping, bootstrapping_fbp,
    bootstrapping_unet}; student_t for boundary (10 replicates is too few
    for percentile to be stable).

Note: not the paper protocol -- the paper applies hyperparam-only calibration.
Outputs are clearly named ``alpha_cal_*`` so they cannot be confused with the
canonical ``uq_widths_coverage_*`` parquets.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.uq import percentile_ci, student_t_ci
from uqct.utils import get_results_dir
from uqct.vis.plot_uq import get_ground_truth

# Method specs: (h5 glob (under runs_subdir), runs_subdir, CI fn, h5 key, slice fn)
METHOD_CFG = {
    "skrock": dict(
        prefix="skrock", subdir="runs", ci=percentile_ci, h5_key="preds",
        slicer=lambda a: a[:, -1, :, :, :],  # (N, T, R, H, W) -> last step
    ),
    "equivariant_bootstrapping": dict(
        prefix="equivariant_bootstrapping", subdir="runs", ci=percentile_ci,
        h5_key="preds", slicer=lambda a: a[:, -1, :, :, :],
    ),
    "bootstrapping_fbp": dict(
        prefix="bootstrapping_fbp", subdir="runs", ci=percentile_ci,
        h5_key="preds", slicer=lambda a: a[:, -1, :, :, :],
    ),
    "bootstrapping_unet": dict(
        prefix="bootstrapping_unet", subdir="runs", ci=percentile_ci,
        h5_key="preds", slicer=lambda a: a[:, -1, :, :, :],
    ),
    "boundary": dict(
        prefix="boundary_diffusion", subdir="boundary_sampling", ci=student_t_ci,
        h5_key="sampled_images",
        slicer=lambda a: a[:, -1, :, 0, :, :],  # (N, S, R, 1, H, W) -> (N, R, H, W)
    ),
}

# Sweep grids. ALPHA_GRID is what enters ECE; C_GRID is the width-scale sweep.
# C_GRID spans 0.1..50 -- the previous 0.1..4 cap clipped cells whose raw CI is
# so tight that 4x expansion still doesn't reach nominal coverage (notably
# bootstrap_unet at 1e4 and bootstrap_fbp at 1e4). Finer resolution near 1
# (where most methods sit), coarser further out.
ALPHA_GRID = [0.01, 0.05, 0.10, 0.20, 0.50]
C_GRID = np.unique(np.concatenate([
    np.linspace(0.05, 1.0, 20),       # shrinkage candidates (0.05 step)
    np.linspace(1.05, 5.0, 80),       # near 1 (0.05 step)
    np.geomspace(5.0, 50.0, 25)[1:],  # log-spaced expansion up to 50
]))

NAME_RE = re.compile(
    r"^[\w_]+:(?P<ds>[a-z]+):(?P<intensity>[\d.eE+-]+):"
    r"True:(?P<start>\d+)-(?P<end>\d+):(?P<seed>\d+):"
)


def parse_meta(path: Path) -> dict | None:
    m = NAME_RE.match(path.name)
    return None if m is None else {
        "dataset": m["ds"], "intensity": float(m["intensity"]),
        "start": int(m["start"]), "end": int(m["end"]), "seed": int(m["seed"]),
    }


def latest_per_cell(files: list[Path]) -> list[Path]:
    """Group files by (dataset, intensity, start, end, seed), keep latest mtime."""
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in files:
        meta = parse_meta(f)
        if meta is None:
            continue
        key = (meta["dataset"], meta["intensity"], meta["start"], meta["end"], meta["seed"])
        groups[key].append(f)
    return sorted(max(fs, key=lambda p: p.stat().st_mtime) for fs in groups.values())


def find_files(method: str, image_start: int, image_end: int) -> list[Path]:
    cfg = METHOD_CFG[method]
    base = get_results_dir() / cfg["subdir"]
    files = [Path(p) for p in glob(str(base / f"{cfg['prefix']}:*.h5"))]
    kept: list[Path] = []
    for f in files:
        m = parse_meta(f)
        if m is None or m["seed"] != 0:
            continue
        if m["start"] == image_start and m["end"] == image_end:
            kept.append(f)
    return latest_per_cell(kept)


def load_samples(path: Path, method: str, device: torch.device) -> torch.Tensor:
    cfg = METHOD_CFG[method]
    with h5py.File(path, "r") as h:
        arr = cfg["slicer"](h[cfg["h5_key"]])
    return torch.from_numpy(arr).to(device).float()


def ci_at_alpha(samples: torch.Tensor, method: str, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image (lo, hi) at level alpha. samples: (R, H, W)."""
    cfg = METHOD_CFG[method]
    return cfg["ci"](samples, delta=alpha, bdim=0)


def coverage_width_at_c(
    lo: torch.Tensor, hi: torch.Tensor, target: torch.Tensor, c: float,
) -> tuple[float, float]:
    """Scale half-width by c, clamp to [0,1], return (per-pixel coverage, mean width)."""
    half = (hi - lo) * 0.5
    ctr = (hi + lo) * 0.5
    lo_c = (ctr - c * half).clamp(0, 1)
    hi_c = (ctr + c * half).clamp(0, 1)
    in_bounds = (target >= lo_c) & (target <= hi_c)
    cov = in_bounds.float().mean().item()
    width = (hi_c - lo_c).abs().clamp(0, 1).mean().item()
    return cov, width


def calibrate_cell(
    samples: torch.Tensor, gt: torch.Tensor, method: str,
) -> tuple[float, float, float]:
    """Find c* minimizing ECE on this cell. Returns (c_star, ece_cal, ece_raw)."""
    bounds = {a: ci_at_alpha(samples, method, a) for a in ALPHA_GRID}

    def ece_for(c: float) -> float:
        return float(np.mean([
            abs(coverage_width_at_c(*bounds[a], gt, c)[0] - (1 - a))
            for a in ALPHA_GRID
        ]))

    eces = [ece_for(c) for c in C_GRID]
    i_star = int(np.argmin(eces))
    c_star = float(C_GRID[i_star])
    ece_cal = float(eces[i_star])
    # raw ECE (c=1, the nearest grid point)
    i_one = int(np.argmin(np.abs(C_GRID - 1.0)))
    ece_raw = float(eces[i_one])
    return c_star, ece_cal, ece_raw


def stage_cal(method: str, device: torch.device) -> pd.DataFrame:
    """Compute c* per (dataset, intensity) from calibration h5s (image_range=0-10)."""
    files = find_files(method, 0, 10)
    print(f"{method}: {len(files)} cal files (image_range=0-10)")
    rows: list[dict] = []
    for k, f in enumerate(files):
        meta = parse_meta(f)
        assert meta is not None
        t0 = time.time()
        samples = load_samples(f, method, device)  # (10, R, H, W)
        gt = get_ground_truth(meta["dataset"], (0, 10))
        if gt.shape[-2:] != samples.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1), size=samples.shape[-2:], mode="area").squeeze(1)
        n = min(len(gt), len(samples))
        gt, samples = gt[:n], samples[:n]
        # Aggregate across the 10 calibration images: pool samples per-pixel.
        # Treat each (image, pixel) as an independent calibration point.
        # Equivalent to averaging ECE across images.
        c_stars: list[float] = []
        ece_cals: list[float] = []
        ece_raws: list[float] = []
        for i in range(n):
            c, ec, er = calibrate_cell(samples[i], gt[i], method)
            c_stars.append(c); ece_cals.append(ec); ece_raws.append(er)
        # Image-averaged ECE; pick c* that minimizes the average ECE across the cal images.
        bounds_all = [{a: ci_at_alpha(samples[i], method, a) for a in ALPHA_GRID} for i in range(n)]
        def ece_avg(c: float) -> float:
            vals = []
            for i in range(n):
                for a in ALPHA_GRID:
                    cov, _ = coverage_width_at_c(*bounds_all[i][a], gt[i], c)
                    vals.append(abs(cov - (1 - a)))
            return float(np.mean(vals))
        eces = [ece_avg(c) for c in C_GRID]
        i_star = int(np.argmin(eces))
        c_star = float(C_GRID[i_star])
        ece_cal = float(eces[i_star])
        ece_raw = float(eces[int(np.argmin(np.abs(C_GRID - 1.0)))])
        rows.append(dict(
            method=method,
            dataset=meta["dataset"],
            intensity=meta["intensity"],
            c_star=c_star,
            ece_cal=ece_cal,
            ece_raw=ece_raw,
        ))
        print(f"  [{k+1}/{len(files)}] {meta['dataset']} I={meta['intensity']:.0e} "
              f"c*={c_star:.3f} ECE_raw={ece_raw:.3f} ECE_cal={ece_cal:.3f} dt={time.time()-t0:.1f}s")
        del samples, gt
    return pd.DataFrame(rows)


def stage_test(method: str, c_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Apply c* to test h5s (image_range=10-110), report coverage+width per image at alpha=0.05."""
    # Test files are 10 chunks of (image_range start..start+10).
    cfg = METHOD_CFG[method]
    base = get_results_dir() / cfg["subdir"]
    all_test = []
    for f in [Path(p) for p in glob(str(base / f"{cfg['prefix']}:*.h5"))]:
        m = parse_meta(f)
        if m is None or m["seed"] != 0 or m["start"] < 10:
            continue
        all_test.append(f)
    all_test = latest_per_cell(all_test)
    print(f"{method}: {len(all_test)} test files (image_range>=10)")
    c_lookup = {(r.dataset, r.intensity): r.c_star for r in c_df.itertuples()}
    ece_lookup = {(r.dataset, r.intensity): (r.ece_raw, r.ece_cal) for r in c_df.itertuples()}

    rows: list[dict] = []
    for k, f in enumerate(all_test):
        meta = parse_meta(f)
        assert meta is not None
        c_star = c_lookup.get((meta["dataset"], meta["intensity"]))
        if c_star is None:
            print(f"  skip {f.name}: no c* for ({meta['dataset']}, {meta['intensity']:.0e})")
            continue
        ece_raw, ece_cal = ece_lookup[(meta["dataset"], meta["intensity"])]
        t0 = time.time()
        samples = load_samples(f, method, device)
        gt = get_ground_truth(meta["dataset"], (meta["start"], meta["end"]))
        if gt.shape[-2:] != samples.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1), size=samples.shape[-2:], mode="area").squeeze(1)
        n = min(len(gt), len(samples))
        gt, samples = gt[:n], samples[:n]
        for i in range(n):
            lo, hi = ci_at_alpha(samples[i], method, 0.05)
            cov_raw, w_raw = coverage_width_at_c(lo, hi, gt[i], 1.0)
            cov_cal, w_cal = coverage_width_at_c(lo, hi, gt[i], c_star)
            rows.append(dict(
                method=method,
                dataset=meta["dataset"],
                intensity=meta["intensity"],
                seed=meta["seed"],
                image=meta["start"] + i,
                c_star=c_star,
                ind_cov_raw=cov_raw,
                ind_cov_cal=cov_cal,
                width_raw=w_raw,
                width_cal=w_cal,
                ece_raw=ece_raw,
                ece_cal=ece_cal,
            ))
        print(f"  [{k+1}/{len(all_test)}] {f.name[:60]}... n={n} c*={c_star:.3f} dt={time.time()-t0:.1f}s")
        del samples, gt
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHOD_CFG))
    parser.add_argument("--stage", required=True, choices=["cal", "test", "both"])
    parser.add_argument("--c-star-file", type=Path, default=None,
                        help="for --stage test: input parquet with c_star per cell.")
    parser.add_argument("--out-dir", type=Path,
                        default=get_results_dir() / "plots")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    if args.stage in ("cal", "both"):
        cal_df = stage_cal(args.method, dev)
        cal_out = args.out_dir / f"alpha_cal_{args.method}_c.parquet"
        cal_df.to_parquet(cal_out)
        print(f"wrote {cal_out} ({len(cal_df)} cells)")
    else:
        if args.c_star_file is None:
            parser.error("--stage test requires --c-star-file")
        cal_df = pd.read_parquet(args.c_star_file)

    if args.stage in ("test", "both"):
        test_df = stage_test(args.method, cal_df, dev)
        test_out = args.out_dir / f"alpha_cal_{args.method}_metrics.parquet"
        test_df.to_parquet(test_out)
        print(f"wrote {test_out} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
