"""Compare CI width and empirical coverage of UQ methods on the test window.

Reads per-image parquets produced by ``scripts/compute_uq_skrock_eb.py`` for
each of the five methods, validates that every (dataset, intensity, image, seed)
cell is present, and aborts if any method has gaps. Then aggregates per
(dataset, intensity) and plots CI width and per-pixel coverage vs. intensity.

Inputs (per-image rows, all required):
    results/plots/uq_widths_coverage_skrock.parquet
    results/plots/uq_widths_coverage_equivariant_bootstrapping_fbp.parquet
    results/plots/uq_widths_coverage_bootstrapping_fbp.parquet
    results/plots/uq_widths_coverage_bootstrapping_unet.parquet
    results/plots/uq_widths_coverage_boundary.parquet

Outputs:
    results/plots/uq_methods_summary.parquet
    results/plots/uq_methods_ci_width.{pdf,png}
    results/plots/uq_methods_coverage.{pdf,png}

Filter: seed=0, image in [10, 110); no alpha-calibration applied -- this is the
paper-faithful, hyperparameter-only comparison.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.utils import get_results_dir

DELTA = 0.05  # 95% nominal
DATASETS = ["lung", "composite", "lamino"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
IMAGES = list(range(10, 110))
SEEDS = [0]

# Method -> (display label, CI variant key in the per-image parquet, color).
METHOD_SPEC = {
    "boundary":                      ("Diffusion (boundary)", "student_t",  "tab:purple"),
    "bootstrapping_fbp":             ("FBP bootstrap",        "percentile", "tab:orange"),
    "bootstrapping_unet":            ("U-Net bootstrap",      "percentile", "tab:green"),
    "equivariant_bootstrapping_fbp": ("Equivariant bootstrap", "percentile", "tab:red"),
    "skrock":                        ("SK-ROCK",              "percentile", "tab:blue"),
}
METHOD_ORDER = list(METHOD_SPEC)

EXPECTED_TUPLES = set(product(DATASETS, INTENSITIES, IMAGES, SEEDS))  # 5400 cells per method


def _load_and_validate(method: str, plots_dir: Path) -> pd.DataFrame:
    path = plots_dir / f"uq_widths_coverage_{method}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{method}: parquet missing at {path}")
    df = pd.read_parquet(path)
    df = df[df.seed.isin(SEEDS)]
    df = df[(df.image >= IMAGES[0]) & (df.image <= IMAGES[-1])]
    # Normalize intensities: cached files sometimes mix 1e+04 vs 10000.0.
    df = df.copy()
    df["intensity"] = df["intensity"].astype(float)
    df["dataset"] = df["dataset"].astype(str)
    df["image"] = df["image"].astype(int)
    df["seed"] = df["seed"].astype(int)

    # No duplicate rows for the same (dataset, intensity, image, seed) cell.
    key = ["dataset", "intensity", "image", "seed"]
    dup = df[df.duplicated(subset=key, keep=False)]
    if len(dup):
        raise ValueError(
            f"{method}: {len(dup)} duplicate rows after filter; e.g. {dup.head(3).to_dict('records')}"
        )

    have = set(zip(df.dataset, df.intensity.round(0).astype(int).astype(float), df.image, df.seed))
    expected = {(d, i, im, s) for (d, i, im, s) in EXPECTED_TUPLES}
    missing = expected - have
    if missing:
        # Summarize misses
        by_cell = pd.DataFrame(list(missing), columns=["dataset", "intensity", "image", "seed"]) \
            .groupby(["dataset", "intensity"]).size().reset_index(name="missing")
        raise ValueError(
            f"{method}: {len(missing)} required cells missing out of {len(expected)}. "
            f"Summary by (dataset, intensity):\n{by_cell.to_string(index=False)}"
        )
    print(f"  {method}: validated {len(df)} rows over {len(have)} unique cells.")
    return df


def _aggregate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Aggregate per-image rows -> per-(dataset, intensity) mean and SEM."""
    _, ci, _ = METHOD_SPEC[method]
    agg = (
        df.groupby(["dataset", "intensity"])
        .agg(
            width=(f"{ci}_width", "mean"),
            width_std=(f"{ci}_width", "std"),
            ind_cov=(f"{ci}_ind_cov", "mean"),
            ind_cov_std=(f"{ci}_ind_cov", "std"),
            sim_cov=(f"{ci}_sim_cov", "mean"),
            sim_cov_std=(f"{ci}_sim_cov", "std"),
            n=(f"{ci}_width", "count"),
        )
        .reset_index()
    )
    agg.insert(0, "method", method)
    if not (agg["n"] == len(IMAGES) * len(SEEDS)).all():
        bad = agg[agg["n"] != len(IMAGES) * len(SEEDS)]
        raise ValueError(f"{method}: cells with wrong image count after aggregate:\n{bad}")
    return agg.drop(columns="n")


def main() -> None:
    results_dir = get_results_dir()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- load + validate every method (fails loud) ---
    print("Validating per-image data for every method...")
    aggregated: list[pd.DataFrame] = []
    for m in METHOD_ORDER:
        df = _load_and_validate(m, plots_dir)
        aggregated.append(_aggregate(df, m))
    out = pd.concat(aggregated, ignore_index=True)
    out.to_parquet(plots_dir / "uq_methods_summary.parquet")
    print(f"wrote {plots_dir / 'uq_methods_summary.parquet'} ({len(out)} rows)")

    for metric, ylabel, fname in [
        ("width", "Mean 95% CI width (FOV incl., clamped [0,1])", "uq_methods_ci_width"),
        ("ind_cov", "Per-pixel 95% coverage", "uq_methods_coverage"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=(metric != "width"))
        for col, ds in enumerate(DATASETS):
            ax = axes[col]
            for method in METHOD_ORDER:
                label, _ci, color = METHOD_SPEC[method]
                sub = out[(out.method == method) & (out.dataset == ds)].sort_values("intensity")
                x = sub.intensity.to_numpy()
                y = sub[metric].to_numpy()
                yerr = sub[f"{metric}_std"].to_numpy() / np.sqrt(len(IMAGES))  # SEM
                ax.errorbar(x, y, yerr=yerr, marker="o", ms=4, lw=1.4,
                            capsize=2, color=color, label=label)
            ax.set_xscale("log")
            if metric == "width":
                ax.set_yscale("log")
            if metric == "ind_cov":
                ax.axhline(1 - DELTA, color="k", lw=0.6, ls="--",
                           label="nominal 95%" if col == 0 else None)
                ax.set_ylim(0.4, 1.02)
            ax.set_title(ds)
            ax.set_xlabel("Total intensity $N_0$")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=8)
        fig.suptitle(
            f"UQ methods comparison "
            f"({'CI width' if metric == 'width' else 'empirical coverage'}, "
            f"seed 0, images 10-110, no alpha calibration)", y=1.03,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(plots_dir / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"wrote {plots_dir / f'{fname}.{ext}'}")
        plt.close(fig)

    # Compact terminal tables
    print("\n=== mean 95% CI width by (method, dataset, intensity) ===")
    print(out.set_index(["method", "dataset", "intensity"])["width"]
          .unstack("intensity").round(4))
    print("\n=== per-pixel coverage by (method, dataset, intensity) ===")
    print(out.set_index(["method", "dataset", "intensity"])["ind_cov"]
          .unstack("intensity").round(3))


if __name__ == "__main__":
    main()
