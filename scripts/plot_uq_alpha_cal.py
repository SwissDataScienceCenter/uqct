"""Alpha-calibration comparison plot (post-hoc, NOT the paper protocol).

Reads the per-image alpha-cal metrics parquets written by
``scripts/compute_alpha_calibration.py``:
    results/plots/alpha_cal_<method>_metrics.parquet
    (columns: method, dataset, intensity, seed, image, c_star,
              ind_cov_raw, ind_cov_cal, width_raw, width_cal, ece_raw, ece_cal)

Aggregates per (method, dataset, intensity) to compare:
  * paper protocol (hyperparam-only)          -> ``ind_cov_raw`` / ``width_raw``
  * paper protocol + alpha calibration (this) -> ``ind_cov_cal`` / ``width_cal``

Two figures, side-by-side (raw / calibrated) so the post-hoc fix is obvious:
    results/plots/uq_methods_alpha_calibrated_coverage.{pdf,png}
    results/plots/uq_methods_alpha_calibrated_ci_width.{pdf,png}

This is clearly labeled "post-hoc / not in paper" in the suptitle and is
output to a separate filename so it cannot be confused with the canonical
``uq_methods_paper_*`` plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.utils import get_results_dir
from uqct.vis.style import MODEL_COLORS, MODEL_NAMES

DATASETS = ["lung", "composite", "lamino"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
DELTA = 0.05  # 95% nominal

# All 5 methods alpha-calibrated. Display order top-to-bottom.
METHODS = [
    "boundary",
    "bootstrapping_fbp",
    "bootstrapping_unet",
    "equivariant_bootstrapping_fbp",
    "skrock",
]


def _load_method(method: str, plots_dir: Path) -> pd.DataFrame:
    path = plots_dir / f"alpha_cal_{method}_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{method}: parquet missing at {path}")
    df = pd.read_parquet(path)
    df["intensity"] = df["intensity"].astype(float)
    df["dataset"] = df["dataset"].astype(str)
    df["image"] = df["image"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df


def _aggregate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-image rows -> per-(dataset, intensity) mean + std."""
    agg = (
        df.groupby(["dataset", "intensity"])
        .agg(
            ind_cov_raw=("ind_cov_raw", "mean"),
            ind_cov_raw_std=("ind_cov_raw", "std"),
            ind_cov_cal=("ind_cov_cal", "mean"),
            ind_cov_cal_std=("ind_cov_cal", "std"),
            width_raw=("width_raw", "mean"),
            width_raw_std=("width_raw", "std"),
            width_cal=("width_cal", "mean"),
            width_cal_std=("width_cal", "std"),
            c_star=("c_star", "first"),  # constant per cell
            ece_raw=("ece_raw", "first"),
            ece_cal=("ece_cal", "first"),
            n=("ind_cov_raw", "count"),
        )
        .reset_index()
    )
    agg.insert(0, "method", method)
    return agg


def main() -> None:
    plots_dir = get_results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frames = [_aggregate(_load_method(m, plots_dir), m) for m in METHODS]
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(plots_dir / "uq_methods_alpha_calibrated_summary.parquet")
    print(f"wrote {plots_dir / 'uq_methods_alpha_calibrated_summary.parquet'} ({len(out)} rows)")

    for metric, ylabel, ylim, log_y, fname in [
        ("ind_cov", "Per-pixel 95% coverage", (0.3, 1.02), False,
         "uq_methods_alpha_calibrated_coverage"),
        ("width",   "Mean 95% CI width",      None,         True,
         "uq_methods_alpha_calibrated_ci_width"),
    ]:
        # Single-line plot per method (calibrated values only). Same layout
        # as plots/uq_comparsion/sparse_combined_chosen_metrics so the reader
        # can place this side-by-side with the paper baseline.
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
        for col, ds in enumerate(DATASETS):
            ax = axes[col]
            for m in METHODS:
                sub = out[(out.method == m) & (out.dataset == ds)].sort_values("intensity")
                if sub.empty:
                    continue
                color = MODEL_COLORS.get(m, "gray")
                label = MODEL_NAMES.get(m, m)
                x = sub.intensity.to_numpy()
                y = sub[f"{metric}_cal"].to_numpy()
                yerr = sub[f"{metric}_cal_std"].to_numpy() / np.sqrt(100)  # SEM over 100 imgs
                ax.errorbar(x, y, yerr=yerr, marker="x", ms=5, lw=1.4,
                            capsize=2, color=color, label=label)
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if metric == "ind_cov":
                ax.axhline(1 - DELTA, color="k", lw=0.6, ls=":",
                           label="nominal 95%" if col == 0 else None)
            ax.set_title(ds)
            ax.set_xlabel("Total intensity $N_0$")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=max(len(handles), 5), frameon=False, fontsize=8,
                   columnspacing=1.2)
        fig.suptitle(
            f"$\\alpha$-calibration applied (seed 0, post-hoc, NOT the paper protocol) "
            f"-- {metric}",
            y=1.03,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(plots_dir / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"wrote {plots_dir / f'{fname}.{ext}'}")
        plt.close(fig)

    # Terminal tables
    print("\n=== c* per (method, dataset, intensity) ===")
    print(out.set_index(["method", "dataset", "intensity"])["c_star"]
          .unstack("intensity").round(2).to_string())
    print("\n=== coverage: raw -> alpha-cal ===")
    for m in METHODS:
        sub = out[out.method == m]
        print(f"\n--- {m} ---")
        raw = sub.set_index(["dataset", "intensity"])["ind_cov_raw"].unstack("intensity").round(3)
        cal = sub.set_index(["dataset", "intensity"])["ind_cov_cal"].unstack("intensity").round(3)
        print(f"raw:\n{raw.to_string()}\ncal:\n{cal.to_string()}")
    print("\n=== ECE: raw -> alpha-cal (calibration set) ===")
    print(out.set_index(["method", "dataset", "intensity"])[["ece_raw", "ece_cal"]]
          .unstack("intensity").round(3).to_string())


if __name__ == "__main__":
    main()
