import json
from collections import defaultdict
from collections.abc import Callable
from glob import glob
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from uqct.uq import (
    basic_ci,
    error_correlation,
    error_r2,
    gaussian_ci,
    gaussian_conservative_ci,
    percentile_ci,
    simultaneous_ci,
    sparsification_error,
    student_t_bonferroni_ci,
    student_t_ci,
    studentized_ci,
)
from uqct.vis.style import (
    ICML_COLUMN_WIDTH,
    ICML_TEXT_WIDTH,
    MODEL_NAMES,
    get_model_colors,
)

# Configuration matching the user's setup
TOTAL_INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
DATASETS = ["lamino", "composite", "lung"]
# SK-ROCK + equivariant bootstrap are sample-based posterior methods like the
# unet_ensemble path -- treated as `is_bootstrapping=False` with percentile CI.
METHODS = [
    "fbp",
    "unet",
    "unet_ensemble",
    "boundary",
    "distance_maximization",
    "skrock",
    "equivariant_bootstrapping_fbp",
]
# Methods that live in results/runs/ but aren't bootstrap-style file naming.
SAMPLE_BASED_METHODS = {
    "unet_ensemble",
    "skrock",
    "equivariant_bootstrapping_fbp",
}
# Map analysis-level method name -> on-disk file prefix.
DISK_PREFIX = {
    "equivariant_bootstrapping_fbp": "equivariant_bootstrapping",
}


def load_h5_data(file_path: str, key: str = "preds") -> np.ndarray:
    """Loads prediction data from the corresponding H5 file."""
    if file_path.endswith(".parquet"):
        h5_path = file_path.replace(".parquet", ".h5")
    else:
        h5_path = file_path

    with h5py.File(h5_path, "r") as f:
        if key not in f:
            if "sampled_images" in f:
                return f["sampled_images"][:]
            raise KeyError(f"Key {key} not found in {h5_path}")
        return f[key][:]


def get_ground_truth(dataset: str, image_range: tuple[int, int]) -> torch.Tensor:
    """Loads ground truth images for a given dataset and range."""
    from typing import cast

    from uqct.datasets.utils import DatasetName, get_dataset

    _, test_set = get_dataset(cast(DatasetName, dataset), True)

    start, end = image_range
    indices = range(start, end)

    gt_list = []
    for i in indices:
        gt_list.append(test_set[i])

    gt = torch.stack(gt_list, dim=0)
    return gt.squeeze(1).cuda()  # (N, H, W)


def compute_stats_from_samples(
    samples: torch.Tensor, gt: torch.Tensor, chosen_ci_fn: Callable
) -> dict[str, list[float]]:
    """
    Computes all UQ statistics for a batch of samples and ground truth.
    Args:
        samples: (N, Samples, H, W)
        gt: (N, H, W)
    Returns:
        Dictionary of lists of metric values (one per item in batch).
    """
    metrics = defaultdict(list)

    # Pre-compute Mean and Std for Global Metrics
    pred_mean = samples.mean(dim=1)
    pred_std = samples.std(dim=1, unbiased=True)
    abs_error = (pred_mean - gt).abs()

    # Iterate over batch
    for i in range(len(gt)):
        item_samples = samples[i]  # (S, H, W)
        item_target = gt[i]  # (H, W)
        item_std = pred_std[i]  # (H, W)
        item_abs_err = abs_error[i]  # (H, W)

        # 1. CI-based Metrics
        ci_methods = {
            "gaussian": gaussian_ci,
            "gaussian_cons": gaussian_conservative_ci,
            "percentile": percentile_ci,
            "basic": basic_ci,
            "studentized": studentized_ci,
            "simultaneous": simultaneous_ci,
            "student_t": student_t_ci,
            "student_t_bonferroni": student_t_bonferroni_ci,
            "chosen": chosen_ci_fn,
        }

        for name, func in ci_methods.items():
            kwargs = {"bdim": 0, "delta": 0.05}
            lower, upper = func(item_samples, **kwargs)  # type: ignore
            lower = lower.clamp(0, 1)
            upper = upper.clamp(0, 1)

            # Coverage
            in_bounds = (item_target >= lower) & (item_target <= upper)
            sim_cov = in_bounds.all().float().item()
            ind_cov = in_bounds.float().mean().item()
            width_map = (upper - lower).abs().clamp(0, 1)
            width = width_map.mean().item()

            metrics[f"{name}_sim_cov"].append(sim_cov)
            metrics[f"{name}_ind_cov"].append(ind_cov)
            metrics[f"{name}_width"].append(width)

            # Use Gaussian Width for Correlation/R2 as standard
            if name == "gaussian":
                # Check for NaNs or Infs
                if (
                    torch.isfinite(width_map).all()
                    and torch.isfinite(item_abs_err).all()
                ):
                    metrics["error_corr"].append(
                        error_correlation(width_map, item_abs_err).item()
                    )
                    metrics["error_r2"].append(error_r2(width_map, item_abs_err).item())
                else:
                    # Append NaNs if invalid
                    metrics["error_corr"].append(float("nan"))
                    metrics["error_r2"].append(float("nan"))

        # 2. AUSE (Sparsification Error)
        ause = sparsification_error(item_std, item_abs_err).item()
        metrics["ause"].append(ause)

    return metrics


def find_files(
    dataset: str,
    intensity: float,
    model: str,
    is_bootstrapping: bool,
    n_bootstraps: int | None = None,
) -> list[str]:
    """Finds run files matching the criteria."""

    if model == "boundary":
        search_dir = "results/boundary_sampling"
        pattern = f"{search_dir}/boundary_diffusion:{dataset}:{intensity}*:*.h5"
        files = glob(pattern)
        out = sorted(files)
        return out
    elif model == "distance_maximization":
        search_dir = "results/uncertainty_distance"
        pattern = f"{search_dir}/diffusion:{dataset}:{intensity}:*.h5"
        files = glob(pattern)
        return sorted(files)

    if is_bootstrapping:
        search_model = f"bootstrapping_{model}"
    else:
        # Allow analysis-level names to differ from on-disk prefixes (e.g.
        # equivariant_bootstrapping_fbp -> equivariant_bootstrapping).
        search_model = DISK_PREFIX.get(model, model)
    pattern = f"results/runs/{search_model}:{dataset}:{intensity}:True:*.parquet"
    files = glob(pattern)
    valid_files = []
    for f in files:
        h5_path = f.replace(".parquet", ".h5")
        if Path(h5_path).exists():
            # Check n_bootstraps if required
            if is_bootstrapping and n_bootstraps is not None:
                try:
                    df = pd.read_parquet(f)
                    if "n_bootstraps" not in df.columns:
                        continue
                    if int(df["n_bootstraps"].iloc[0]) != n_bootstraps:
                        continue
                except Exception:
                    continue

            valid_files.append(f)

    def get_ts(x):
        try:
            val = pd.read_parquet(x)["timestamp"].iloc[0]
            return pd.to_datetime(val)
        except Exception:
            return pd.Timestamp.min

    valid_files.sort(key=get_ts, reverse=True)
    return valid_files


def _load_precomputed_metrics(method: str) -> pd.DataFrame | None:
    """Look up the per-image UQ parquet a previous ``compute_uq_skrock_eb.py``
    run produced. Returns ``None`` if absent.

    Mapping is by the analysis-level method name (``method`` here matches
    METHODS). Per-image rows carry every CI variant's width/coverage already.
    """
    parquet = Path("results/plots") / f"uq_widths_coverage_{method}.parquet"
    if not parquet.exists():
        return None
    return pd.read_parquet(parquet)


# Method -> CI variant exposed as the 'chosen_*' alias by calculate_metrics.
# Matches the chosen_ci_fn branching in the h5-sample path further below.
CHOSEN_CI_BY_METHOD = {
    "boundary": "student_t",
    "unet_ensemble": "student_t",
    "skrock": "percentile",
    "equivariant_bootstrapping_fbp": "percentile",
    "bootstrapping_fbp": "percentile",
    "bootstrapping_unet": "percentile",
    # default below: "percentile"
}


def _metrics_from_precomputed(
    df: pd.DataFrame, dataset: str, intensity: float, method: str
) -> dict[str, float]:
    """Aggregate per-image rows (already computed elsewhere) to mean+std at a
    given (dataset, intensity). Returns the same keys ``calculate_metrics`` does,
    including method-appropriate ``chosen_*`` aliases."""
    sub = df[(df["dataset"] == dataset) & (np.isclose(df["intensity"], intensity))]
    if len(sub) == 0:
        return {}
    ret: dict[str, float] = {}
    metric_cols = [
        c for c in sub.columns
        if c.endswith(("_width", "_ind_cov", "_sim_cov"))
    ]
    for c in metric_cols:
        vals = sub[c].to_numpy()
        ret[c] = float(np.nanmean(vals))
        ret[f"{c}_std"] = float(np.nanstd(vals))
    # Alias the method-appropriate CI variant as 'chosen_*' so the combined
    # plot (which only renders entries that have 'chosen_ind_cov') picks this
    # method up.
    chosen_ci = CHOSEN_CI_BY_METHOD.get(method, "percentile")
    for metric in ("width", "ind_cov", "sim_cov"):
        src = f"{chosen_ci}_{metric}"
        if src in ret:
            ret[f"chosen_{metric}"] = ret[src]
            ret[f"chosen_{metric}_std"] = ret[f"{src}_std"]
    return ret


def calculate_metrics(
    dataset: str, intensity: float, method: str, n_bootstraps: int | None = None
) -> dict[str, float]:
    """Computes all UQ statistics for a given setting."""

    # Fast path: per-image parquet already computed (multi-seed for the newer
    # methods). Aggregates without re-reading h5 samples.
    precomp = _load_precomputed_metrics(method)
    if precomp is not None:
        m = _metrics_from_precomputed(precomp, dataset, intensity, method)
        if m:
            return m

    # helper to aggregate results
    all_metrics = defaultdict(list)

    if method == "boundary":
        boundary_files = find_files(dataset, intensity, method, is_bootstrapping=False)
        if not boundary_files:
            return {}

        for b_file in boundary_files:
            # Load samples (N, S, R, 1, H, W)
            samples_all = load_h5_data(b_file, key="sampled_images")

            # Extract samples
            if samples_all.ndim == 6:
                # (N, S, R, 1, H, W) -> (N, R, H, W)
                samples = samples_all[:, -1, :, 0, :, :]
            elif samples_all.ndim == 5:
                # (N, S, R, H, W) -> (N, R, H, W)
                samples = samples_all[:, -1, :, :, :]
            else:
                print(f"Skipping {b_file}: shape {samples_all.shape}")
                continue

            # Parse Range
            parts = Path(b_file).name.split(":")
            range_part = None
            for p in parts:
                if "-" in p and p.replace("-", "").isdigit():
                    range_part = p
                    break

            if not range_part:
                continue

            start, end = map(int, range_part.split("-"))
            gt = get_ground_truth(dataset, (start, end))
            samples = torch.tensor(samples).cuda()

            if gt.shape[-2:] != samples.shape[-2:]:
                gt = F.interpolate(
                    gt.unsqueeze(1), size=samples.shape[-2:], mode="area"
                ).squeeze(1)

            n = min(len(gt), len(samples))
            gt = gt[:n]
            samples = samples[:n]

            # Compute Stats
            chunk_metrics = compute_stats_from_samples(samples, gt, student_t_ci)

            for k, v in chunk_metrics.items():
                all_metrics[k].extend(v)

    elif method == "distance_maximization":
        files = find_files(dataset, intensity, method, is_bootstrapping=False)
        for file in files:
            start, end = list(map(int, file.split(":")[4].split("-")))

            gt = get_ground_truth(dataset, (start, end))
            gt_lr = F.interpolate(
                gt.unsqueeze(1), size=gt.shape[-1] // 2, mode="area"
            ).squeeze(1)

            samples_all = torch.from_numpy(load_h5_data(file, key="maximizers")).to(
                gt_lr.device
            )
            lb, ub = samples_all.min(1).values, samples_all.max(1).values
            in_ci = (lb <= gt_lr) & (gt_lr <= ub)

            # Compute per-image statistics (N,)
            sim_cov = in_ci.flatten(1).all(1).float()
            ind_cov = in_ci.float().mean((1, 2))
            width = (ub - lb).abs().clamp(0, 1).mean((1, 2))

            all_metrics["distance_maximization_sim_cov"].extend(sim_cov.cpu().tolist())
            all_metrics["distance_maximization_ind_cov"].extend(ind_cov.cpu().tolist())
            all_metrics["distance_maximization_width"].extend(width.cpu().tolist())
            all_metrics["chosen_sim_cov"].extend(sim_cov.cpu().tolist())
            all_metrics["chosen_ind_cov"].extend(ind_cov.cpu().tolist())
            all_metrics["chosen_width"].extend(width.cpu().tolist())
    else:
        # Bootstrapping / Ensemble / posterior-sampler methods.
        # Sample-based methods (unet_ensemble, skrock, equivariant_bootstrapping_fbp)
        # use is_bootstrapping=False and look up the on-disk prefix via DISK_PREFIX.
        if method in SAMPLE_BASED_METHODS:
            boot_files = find_files(dataset, intensity, method, is_bootstrapping=False)
        else:
            boot_files = find_files(
                dataset,
                intensity,
                method,
                is_bootstrapping=True,
                n_bootstraps=n_bootstraps,
            )

        if not boot_files:
            return {}

        files_to_process = []

        processed_ranges: set[tuple[int, int]] = set()

        for f in boot_files:
            parts = Path(f).name.split(":")
            range_part = None
            for p in parts:
                if "-" in p and p.replace("-", "").isdigit():
                    range_part = p
                    break

            if not range_part:
                continue

            start, end = map(int, range_part.split("-"))
            r_tuple = (start, end)
            if r_tuple in processed_ranges:
                continue
            is_overlap = False
            for pr in processed_ranges:
                if max(start, pr[0]) < min(end, pr[1]):
                    is_overlap = True
                    break

            if is_overlap:
                continue

            processed_ranges.add(r_tuple)
            files_to_process.append(f)

        if not files_to_process:
            files_to_process = [boot_files[0]]

        for boot_file in files_to_process:
            df_boot = pd.read_parquet(boot_file)

            start = int(df_boot["image_start_index"].iloc[0])
            end = int(df_boot["image_end_index"].iloc[0])

            preds_boot_np = load_h5_data(boot_file)
            preds_boot = torch.tensor(preds_boot_np).cuda()

            if method == "unet_ensemble":
                chosen_ci_fn = student_t_ci
                if preds_boot.ndim == 5:
                    preds_boot = preds_boot[:, -1, ...]
            elif method == "skrock":
                # SK-ROCK posterior samples; many samples -> percentile is OK.
                # Shape is (N, T=1, R, H, W) -> last (only) schedule step.
                chosen_ci_fn = percentile_ci
                if preds_boot.ndim == 5:
                    preds_boot = preds_boot[:, -1, ...]
            elif method == "equivariant_bootstrapping_fbp":
                # Equivariant bootstrap also uses sample-based percentile CI.
                chosen_ci_fn = percentile_ci
                if preds_boot.ndim == 5:
                    preds_boot = preds_boot[:, -1, ...]
            else:
                chosen_ci_fn = percentile_ci
                if preds_boot.ndim == 5:
                    preds_boot = preds_boot[:, 0, ...]

            gt = get_ground_truth(dataset, (start, end))

            if gt.shape[-2:] != preds_boot.shape[-2:]:
                gt = F.interpolate(
                    gt.unsqueeze(1), size=preds_boot.shape[-2:], mode="area"
                ).squeeze(1)

            n = min(len(gt), len(preds_boot))
            gt = gt[:n]
            preds_boot = preds_boot[:n]

            chunk_metrics = compute_stats_from_samples(preds_boot, gt, chosen_ci_fn)

            for k, v in chunk_metrics.items():
                all_metrics[k].extend(v)

    if not all_metrics:
        return {}

    ret = {}
    for k, v in all_metrics.items():
        ret[k] = float(np.nanmean(v))
        ret[f"{k}_std"] = float(np.nanstd(v))
    return ret


@click.command()
@click.option(
    "--n-bootstraps", type=int, default=None, help="Filter by number of bootstraps"
)
@click.option(
    "--cache",
    type=click.Path(),
    default="results/uq_comparison.json",
    show_default=True,
    help="Path to read cached results from. Original 'uq_comparison.json' is "
    "treated as read-only -- missing-method entries are computed fresh and the "
    "combined result is written to --out instead.",
)
@click.option(
    "--out",
    type=click.Path(),
    default="results/uq_comparison_merged.json",
    show_default=True,
    help="Where to write the merged results JSON. Defaults to a NEW file so the "
    "paper-canonical uq_comparison.json is never overwritten.",
)
def main(n_bootstraps, cache, out):
    # Gather Data
    results = {d: {m: defaultdict(list) for m in METHODS} for d in DATASETS}

    files_found = False

    # Load cached results if present and reuse per-method entries. Missing
    # methods are computed and added; the cache file itself is never written.
    cache_data = {}
    if Path(cache).exists():
        cache_data = json.load(open(cache))
        print(f"Loaded cached results from {cache}")

    # Determine which (dataset, method) pairs still need computation.
    todo: list[tuple[str, str]] = []
    for d in DATASETS:
        for m in METHODS:
            cached_entry = cache_data.get(d, {}).get(m, {})
            if cached_entry and cached_entry.get("intensity"):
                # Reuse cache.
                results[d][m] = cached_entry
                files_found = True
            else:
                todo.append((d, m))

    if todo:
        print(f"Computing {len(todo)} missing (dataset, method) cells...")
        for dataset, method in tqdm(todo, desc="Missing cells", leave=True):
            # Single fresh accumulator for this (dataset, method) -- do not reset
            # on each intensity.
            acc: dict = defaultdict(list)
            for intensity in tqdm(TOTAL_INTENSITIES, desc=f"{dataset}/{method}", leave=False):
                metrics = calculate_metrics(
                    dataset, intensity, method, n_bootstraps=n_bootstraps
                )
                if metrics:
                    for k, v in metrics.items():
                        acc[k].append(v)
                    acc["intensity"].append(intensity)
                    files_found = True
            results[dataset][method] = dict(acc)

    # Convert defaultdicts to plain dicts for JSON.
    serializable = {
        d: {m: dict(results[d][m]) if not isinstance(results[d][m], dict) else results[d][m]
            for m in results[d]}
        for d in results
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(serializable, open(out, "w"))
    print(f"Wrote merged results to {out}")

    if not files_found:
        print("No valid data found to plot.")
        return

    # Prepare Metrics List
    ci_methods = [
        "gaussian",
        "gaussian_cons",
        "percentile",
        "basic",
        "studentized",
        "simultaneous",
        "student_t",
        "student_t_bonferroni",
        "chosen",
    ]
    ci_metrics = ["sim_cov", "ind_cov", "width"]

    metrics_to_plot = []
    titles = {}

    for method_name in ci_methods:
        for metric in ci_metrics:
            key = f"{method_name}_{metric}"
            metrics_to_plot.append(key)

            metric_title = metric.replace("_", " ").title()
            if metric == "sim_cov":
                metric_title = r"Coverage (\%)"
            if metric == "ind_cov":
                metric_title = r"Coverage (\%)"
            if metric == "width":
                metric_title = r"CI Width"

            titles[key] = f"{metric_title}"

    metrics_to_plot.extend(["error_corr", "error_r2", "ause"])
    titles["error_corr"] = "Error-Width Correlation"
    titles["error_r2"] = "Error-Width R2"
    titles["ause"] = "AUSE (Sparsification Error)"

    # Output Dir
    out_dir = Path("plots/uq_comparsion")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop over metrics -> Create one figure per metric
    for metric_key in metrics_to_plot:
        # Check if we have data for this metric
        has_data = False
        for d in DATASETS:
            for m in METHODS:
                if results[d][m].get(metric_key):
                    has_data = True
                    break
            if has_data:
                break

        if not has_data:
            continue

        fig, axes = plt.subplots(
            len(DATASETS),
            1,
            figsize=(ICML_COLUMN_WIDTH, 4.2),
            sharey=True,
            sharex=True,
        )
        # axes is (3,) array

        metric_title = titles.get(metric_key, metric_key)
        # fig.supylabel(metric_title, x=0.03)

        for row_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx]

            # Loop methods
            for method in METHODS:
                data = results[dataset][method]
                if not data or not data["intensity"]:
                    continue

                # Sort by intensity
                ints = np.array(data["intensity"])

                if method == "distance_maximization":
                    if "sim_cov" in metric_key:
                        vals = np.array(data["distance_maximization_sim_cov"])
                        search_key = "distance_maximization_sim_cov"
                    elif "ind_cov" in metric_key:
                        vals = np.array(data["distance_maximization_ind_cov"])
                        search_key = "distance_maximization_ind_cov"
                    elif "width" in metric_key:
                        vals = np.array(data["distance_maximization_width"])
                        search_key = "distance_maximization_width"
                    else:
                        search_key = (
                            None  # Ensure search_key is defined for the next if
                        )

                    if search_key and search_key in data:
                        vals = np.array(data[search_key])
                        std_vals = np.array(data[f"{search_key}_std"])
                    else:
                        continue  # Skip if metric type not relevant (e.g. ause)
                else:
                    if metric_key not in data:
                        continue
                    vals = np.array(data[metric_key])
                    std_vals = np.array(data[f"{metric_key}_std"])

                if len(ints) != len(vals):
                    print(
                        f"Warning: Length mismatch for {dataset} {method} {metric_key}"
                    )
                    continue

                sort_idx = np.argsort(ints)

                model_label = MODEL_NAMES.get(method, method)

                if model_label in ("FBP", "U-Net"):
                    label = f"{model_label} Bootstr."
                else:
                    label = f"{model_label}"

                if "_cov" in metric_key.lower():
                    vals_plot = vals[sort_idx] * 100
                    std_plot = std_vals[sort_idx] * 100
                else:
                    vals_plot = vals[sort_idx]
                    std_plot = std_vals[sort_idx]

                if method == "boundary":
                    color = get_model_colors().get("diffusion")
                else:
                    color = get_model_colors().get(method)

                ax.plot(
                    ints[sort_idx],
                    vals_plot,
                    label=label,
                    color=color,
                    marker="x",
                )
                ax.fill_between(
                    ints[sort_idx],
                    vals_plot - std_plot,
                    vals_plot + std_plot,
                    color=color,
                    alpha=0.2,
                )

            ax.set_xscale("log")
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Total Intensity")
            ax.set_ylabel(metric_title)

            ax.set_title(f"{dataset.title()} Dataset")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Legend logic
        handles, labels = [], []
        seen_labels = set()

        def add_handle_label(h, lbl):
            if lbl not in seen_labels and lbl is not None:
                handles.append(h)
                labels.append(lbl)
                seen_labels.add(lbl)

        # 1) Loop over axes to collect handles/labels
        for ax in axes:
            if ax.lines:
                h_list, l_list = ax.get_legend_handles_labels()
                for h, lbl in zip(h_list, l_list):
                    add_handle_label(h, lbl)

        fig.tight_layout(rect=(0, 0.09, 1, 1))

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                frameon=False,
            )

        # Save
        out_path = out_dir / f"sparse_{metric_key}.pdf"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close(fig)

    # --- Combined Chosen Metrics Plot ---
    # Rows: Chosen Coverage, Chosen Width
    # Cols: Lamino, Composite, Lung
    combined_metrics = ["chosen_ind_cov", "chosen_width"]
    combined_titles = [r"Coverage (\%)", "CI Width"]

    # Check if we have data for these
    has_combined_data = False
    for d in DATASETS:
        for m in METHODS:
            if results[d][m].get("chosen_ind_cov"):
                has_combined_data = True
                break
        if has_combined_data:
            break

    if has_combined_data:
        # Compressed height: about half -> 3.0 inches (vs 5 or 6)
        # Standard column width * 2 approx text width.
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(ICML_TEXT_WIDTH, 2.5),  # Compact height
            sharex=True,
            sharey="row",
        )

        # Row 1 Titles: Datasets
        for c_idx, ds in enumerate(DATASETS):
            axes[0, c_idx].set_title(ds.title())

        handles, labels = [], []
        seen_labels = set()

        def add_handle_label(h, lbl):
            if lbl not in seen_labels and lbl is not None:
                handles.append(h)
                labels.append(lbl)
                seen_labels.add(lbl)

        for r_idx, metric_key in enumerate(combined_metrics):
            axes[r_idx, 0].set_ylabel(combined_titles[r_idx])

            for c_idx, dataset in enumerate(DATASETS):
                ax = axes[r_idx, c_idx]

                # Plot all methods
                for method in METHODS:
                    data = results[dataset][method]
                    if not data or not data["intensity"]:
                        continue

                    # Retrieve data logic (similar to above but specific to chosen metrics)
                    # For distance maximization, we need explicit mapping
                    if method == "distance_maximization":
                        if metric_key == "chosen_ind_cov":
                            vals = np.array(data.get("chosen_ind_cov", []))
                            std_vals = np.array(data.get("chosen_ind_cov_std", []))
                        elif metric_key == "chosen_width":
                            vals = np.array(data.get("chosen_width", []))
                            std_vals = np.array(data.get("chosen_width_std", []))
                        else:
                            continue
                    else:
                        # Standard methods
                        vals = np.array(data.get(metric_key, []))
                        std_vals = np.array(data.get(f"{metric_key}_std", []))

                    if len(vals) == 0:
                        continue

                    ints = np.array(data["intensity"])
                    if len(ints) != len(vals):
                        continue

                    sort_idx = np.argsort(ints)

                    model_label = MODEL_NAMES.get(method, method)
                    if model_label in ("FBP", "U-Net"):
                        label = f"{model_label} Bootstr."
                    else:
                        label = f"{model_label}"

                    # Remove "Mix." or similar if it appears (though plot_uq doesn't usually add it)
                    # User request: "don't write 'Mix' there"
                    # Just in case, explicit check?
                    # The labels here are "FBP Bootstr.", "Diffusion", "Worst-Case", etc.
                    # "Mix" came from plot_scaling.py.

                    color = get_model_colors().get(
                        method if method != "boundary" else "diffusion"
                    )

                    if "_cov" in metric_key:
                        vals_plot = vals[sort_idx] * 100
                        std_plot = std_vals[sort_idx] * 100
                    else:
                        vals_plot = vals[sort_idx]
                        std_plot = std_vals[sort_idx]

                    line = ax.plot(
                        ints[sort_idx],
                        vals_plot,
                        label=label,
                        color=color,
                        marker="x",
                        markersize=4,
                        linewidth=1.0,
                    )

                    ax.fill_between(
                        ints[sort_idx],
                        vals_plot - std_plot,
                        vals_plot + std_plot,
                        color=color,
                        alpha=0.2,
                    )

                    # Collect handles (only need to do this once per method really)
                    # But doing it per plot ensures we catch all visible methods
                    # We'll filter duplicates with add_handle_label
                    add_handle_label(line[0], label)

                ax.set_xscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Shared Footer
        for ax in axes[-1, :]:
            ax.set_xlabel("Total Intensity")

        # Legend layout
        # "Reduce space between legend and lower plot"
        # Adjust rect or bbox
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=5,
                frameon=False,
                borderpad=0.2,
                labelspacing=0.2,
            )

        out_path = out_dir / "sparse_combined_chosen_metrics.pdf"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
