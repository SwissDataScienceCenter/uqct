"""Paper-style UQ comparison plot (10 seeds where available).

Older methods (diffusion ct-boundary, FBP bootstrap, U-Net bootstrap):
    use the cached ``results/uq_comparison.json`` produced by
    ``uqct.vis.plot_uq`` -- these are the paper numbers, aggregated over the
    files that existed at JSON generation time (10 seeds for the paper).

Newer methods (SK-ROCK, equivariant bootstrap):
    use the freshly-computed multi-seed per-image parquets at
    ``results/plots/uq_widths_coverage_{skrock,equivariant_bootstrapping_fbp}.parquet``
    (computed by ``scripts/compute_uq_skrock_eb.py`` with ``--seeds 0,1,...,9``).

Each method uses its appropriate CI variant (percentile for the bootstraps and
SK-ROCK, Student-t for boundary). NO alpha calibration applied -- this is the
hyperparameter-only comparison that matches the paper protocol.

The original ``results/uq_comparison.json`` is read-only here. Outputs:
    results/uq_comparison_merged.json          # JSON with old + new methods
    results/plots/uq_methods_paper_summary.parquet
    results/plots/uq_methods_paper_ci_width.{pdf,png}
    results/plots/uq_methods_paper_coverage.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.utils import get_results_dir

DELTA = 0.05
DATASETS = ["lung", "composite", "lamino"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
SEEDS_EXPECTED = list(range(10))
IMAGES_EXPECTED = list(range(10, 110))  # 100 test images, 10–110

# (display label, CI variant key, color, data source, JSON key if applicable)
METHOD_SPEC = {
    "boundary": dict(
        label="Diffusion (boundary)", ci="student_t",  color="tab:purple",
        source="json", json_key="boundary",
    ),
    "bootstrapping_fbp": dict(
        label="FBP bootstrap",        ci="percentile", color="tab:orange",
        source="json", json_key="fbp",
    ),
    "bootstrapping_unet": dict(
        label="U-Net bootstrap",      ci="percentile", color="tab:green",
        source="json", json_key="unet",
    ),
    "equivariant_bootstrapping_fbp": dict(
        label="Equivariant bootstrap", ci="percentile", color="tab:red",
        source="parquet",
        parquet="uq_widths_coverage_equivariant_bootstrapping_fbp.parquet",
    ),
    "skrock": dict(
        label="SK-ROCK",              ci="percentile", color="tab:blue",
        source="parquet",
        parquet="uq_widths_coverage_skrock.parquet",
    ),
}
METHOD_ORDER = list(METHOD_SPEC)


def _from_json(json_data: dict, method: str) -> pd.DataFrame:
    """Pull (dataset, intensity, width/cov, std) rows from the cached JSON.

    The JSON stores `<ci>_width`, `<ci>_width_std`, `<ci>_ind_cov`,
    `<ci>_ind_cov_std`, etc. as parallel 6-long lists, in `INTENSITIES` order.
    The `std` is over images (and any seeds aggregated at JSON build time).
    """
    spec = METHOD_SPEC[method]
    rows: list[dict] = []
    for ds in DATASETS:
        per_method = json_data.get(ds, {}).get(spec["json_key"], {})
        ints = per_method.get("intensity", [])
        if not ints:
            print(f"  ?? {method} / {ds}: empty in JSON")
            continue
        for k, i_lvl in enumerate(ints):
            rows.append(dict(
                method=method, dataset=ds, intensity=float(i_lvl),
                width=per_method.get(f"{spec['ci']}_width", [np.nan]*(k+1))[k],
                width_std=per_method.get(f"{spec['ci']}_width_std", [np.nan]*(k+1))[k],
                ind_cov=per_method.get(f"{spec['ci']}_ind_cov", [np.nan]*(k+1))[k],
                ind_cov_std=per_method.get(f"{spec['ci']}_ind_cov_std", [np.nan]*(k+1))[k],
                sim_cov=per_method.get(f"{spec['ci']}_sim_cov", [np.nan]*(k+1))[k],
                sim_cov_std=per_method.get(f"{spec['ci']}_sim_cov_std", [np.nan]*(k+1))[k],
                source="json",
                n_seeds=None,  # unknown from JSON
            ))
    return pd.DataFrame(rows)


def _from_parquet(plots_dir: Path, method: str) -> pd.DataFrame:
    """Aggregate per-image rows -> per-(dataset, intensity) over all available seeds."""
    spec = METHOD_SPEC[method]
    path = plots_dir / spec["parquet"]
    if not path.exists():
        raise FileNotFoundError(f"{method}: parquet missing at {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["intensity"] = df["intensity"].astype(float)
    df["dataset"] = df["dataset"].astype(str)
    df["image"] = df["image"].astype(int)
    df["seed"] = df["seed"].astype(int)
    n_seeds = df.seed.nunique()
    ci = spec["ci"]
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
    expected = len(IMAGES_EXPECTED) * n_seeds
    bad = agg[agg["n"] != expected]
    if len(bad):
        print(f"  ⚠ {method}: {len(bad)} cells with wrong row count "
              f"(expected {expected} = 100 imgs × {n_seeds} seeds)")
        print(bad.to_string(index=False))
    agg = agg.drop(columns="n")
    agg.insert(0, "method", method)
    agg["source"] = "parquet"
    agg["n_seeds"] = n_seeds
    return agg


def _aggregate_parquet_to_json_entry(df_pq: pd.DataFrame) -> dict:
    """Per-image parquet (single method, single dataset, multi-seed rows) ->
    nested dict in the same shape as ``uq_comparison.json``'s per-method block.

    For each CI variant present in the parquet (gaussian/percentile/basic/etc.),
    emit per-intensity lists of mean and std (over all seeds × all images) for
    width, ind_cov, sim_cov. Plus ``intensity`` and (where present) ``ause``.
    """
    by_intensity = df_pq.groupby("intensity").agg(list)
    # CI variant names = parquet columns matching `<name>_width` etc.
    metric_suffixes = ("_width", "_ind_cov", "_sim_cov")
    ci_variants = sorted({
        c.rsplit("_", 1)[0].rsplit("_", 1)[0]  # ``percentile_ind_cov`` -> ``percentile``
        for c in df_pq.columns
        if any(c.endswith(suf) for suf in metric_suffixes)
        and "cov_" not in c.rsplit("_", 1)[-1]  # exclude width_std-style
    })
    # Build CI variant set by matching base names directly.
    ci_variants = set()
    for c in df_pq.columns:
        for suf in ("_width", "_ind_cov", "_sim_cov"):
            if c.endswith(suf):
                ci_variants.add(c[: -len(suf)])
    entry: dict[str, list] = {}
    intensities_sorted = sorted(df_pq.intensity.unique())
    entry["intensity"] = intensities_sorted
    for ci in sorted(ci_variants):
        for metric in ("width", "ind_cov", "sim_cov"):
            col = f"{ci}_{metric}"
            if col not in df_pq.columns:
                continue
            means = []
            stds = []
            for i_lvl in intensities_sorted:
                sub = df_pq[df_pq.intensity == i_lvl][col].to_numpy()
                means.append(float(np.nanmean(sub)))
                stds.append(float(np.nanstd(sub)))
            entry[col] = means
            entry[f"{col}_std"] = stds
    return entry


def main() -> None:
    results_dir = get_results_dir()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sources...")
    json_path = results_dir / "uq_comparison.json"
    json_data = json.load(open(json_path)) if json_path.exists() else {}

    # Build merged JSON: deep-copy the existing data and add the new-method entries.
    import copy
    merged_json: dict = copy.deepcopy(json_data)
    # Ensure each dataset key exists in merged JSON.
    for ds in DATASETS:
        merged_json.setdefault(ds, {})

    frames: list[pd.DataFrame] = []
    for m in METHOD_ORDER:
        spec = METHOD_SPEC[m]
        if spec["source"] == "json":
            df = _from_json(json_data, m)
            print(f"  {m}: {len(df)} rows from JSON ({spec['json_key']})")
        else:
            df = _from_parquet(plots_dir, m)
            print(f"  {m}: {len(df)} rows from parquet (n_seeds={df.n_seeds.iloc[0]})")
            # Inject into merged JSON: per-dataset aggregation, same shape as legacy.
            full_pq = pd.read_parquet(plots_dir / spec["parquet"])
            for ds in DATASETS:
                sub = full_pq[full_pq.dataset == ds]
                if sub.empty:
                    continue
                merged_json[ds][m] = _aggregate_parquet_to_json_entry(sub)
        frames.append(df)

    # Write merged JSON.
    merged_path = results_dir / "uq_comparison_merged.json"
    json.dump(merged_json, open(merged_path, "w"))
    print(f"wrote {merged_path}")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(plots_dir / "uq_methods_paper_summary.parquet")
    print(f"wrote {plots_dir / 'uq_methods_paper_summary.parquet'} ({len(out)} rows)")

    for metric, ylabel, fname in [
        ("width", "Mean 95% CI width", "uq_methods_paper_ci_width"),
        ("ind_cov", "Per-pixel 95% coverage", "uq_methods_paper_coverage"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=(metric != "width"))
        for col, ds in enumerate(DATASETS):
            ax = axes[col]
            for m in METHOD_ORDER:
                spec = METHOD_SPEC[m]
                sub = out[(out.method == m) & (out.dataset == ds)].sort_values("intensity")
                if sub.empty:
                    continue
                x = sub.intensity.to_numpy()
                y = sub[metric].to_numpy()
                yerr = sub[f"{metric}_std"].to_numpy()
                ax.errorbar(x, y, yerr=yerr, marker="o", ms=4, lw=1.4,
                            capsize=2, color=spec["color"], label=spec["label"])
            ax.set_xscale("log")
            if metric == "width":
                ax.set_yscale("log")
            if metric == "ind_cov":
                ax.axhline(1 - DELTA, color="k", lw=0.6, ls="--",
                           label="nominal 95%" if col == 0 else None)
                ax.set_ylim(0.3, 1.02)
            ax.set_title(ds)
            ax.set_xlabel("Total intensity $N_0$")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=8)
        fig.suptitle(
            f"UQ methods comparison (paper protocol, "
            f"{'CI width' if metric == 'width' else 'coverage'}, no alpha calibration)",
            y=1.03,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(plots_dir / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"wrote {plots_dir / f'{fname}.{ext}'}")
        plt.close(fig)

    # Terminal tables
    print("\n=== mean 95% CI width by (method, dataset, intensity) ===")
    print(out.set_index(["method", "dataset", "intensity"])["width"]
          .unstack("intensity").round(4).to_string())
    print("\n=== per-pixel coverage by (method, dataset, intensity) ===")
    print(out.set_index(["method", "dataset", "intensity"])["ind_cov"]
          .unstack("intensity").round(3).to_string())


if __name__ == "__main__":
    main()
