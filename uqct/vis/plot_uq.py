from collections import defaultdict
from glob import glob
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.uq import (
    basic_ci,
    error_correlation,
    error_r2,
    gaussian_ci,
    gaussian_conservative_ci,
    percentile_ci,
    simultaneous_ci,
    sparsification_error,
    studentized_ci,
)
from uqct.vis.style import MODEL_NAMES, get_model_colors

# Configuration matching the user's setup
TOTAL_INTENSITIES = [1e6, 1e7, 1e8, 1e9]
DATASETS = ["lung", "composite", "lamino"]
METHODS = ["fbp", "unet", "unet_ensemble", "boundary"]

# Plotting style
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (6.75, 5.0),  # Adjust for 3x3
    }
)


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
    from uqct.datasets.utils import get_dataset

    _, test_set = get_dataset(dataset, True)

    start, end = image_range
    indices = range(start, end)

    gt_list = []
    for i in indices:
        gt_list.append(test_set[i])

    gt = torch.stack(gt_list, dim=0)
    return gt.squeeze(1).cuda()  # (N, H, W)


def compute_stats_from_samples(
    samples: torch.Tensor, gt: torch.Tensor
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
        }

        for name, func in ci_methods.items():
            kwargs = {"bdim": 0}
            if name in ["percentile", "basic"]:
                kwargs["alpha"] = 0.05
            else:
                kwargs["delta"] = 0.05

            lower, upper = func(item_samples, **kwargs)
            lower = lower.clamp(0, 1)
            upper = upper.clamp(0, 1)

            # Coverage
            in_bounds = (item_target >= lower) & (item_target <= upper)
            sim_cov = in_bounds.all().float().item()
            ind_cov = in_bounds.float().mean().item()
            width = (upper - lower).mean().item()

            metrics[f"{name}_sim_cov"].append(sim_cov)
            metrics[f"{name}_ind_cov"].append(ind_cov)
            metrics[f"{name}_width"].append(width)

            # Use Gaussian Width for Correlation/R2 as standard
            if name == "gaussian":
                width_map = upper - lower
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
        # Search in boundary_sampling dir
        # Pattern: boundary_diffusion:dataset:intensity:*.h5
        search_dir = "results/boundary_sampling"
        # The files are like: boundary_diffusion:composite:10000.0:True:10-20:0:timestamp.h5
        # We match dataset and intensity.
        # Note: intensity string format might vary (10000.0 vs 1e4).
        # existing files have .0 suffix for floats.
        pattern = f"{search_dir}/boundary_diffusion:{dataset}:{intensity}*:*.h5"
        files = glob(pattern)

        # Filter out _metrics.csv files if glob matched them (it shouldn't with .h5)
        valid_files = [f for f in files if "metrics" not in f]

        return sorted(valid_files)

    # Standard / Bootstrapping
    # Construct glob pattern
    # Filename format: model:dataset:intensity:sparse:range:timestamp.parquet
    # We assume sparse=True for bootstrapping as per code

    search_model = f"bootstrapping_{model}" if is_bootstrapping else model
    # Relax pattern if needed, but this matches user provided example
    pattern = f"results/runs/{search_model}:{dataset}:{intensity}:True:*.parquet"
    files = glob(pattern)

    # If is_bootstrapping and we find nothing, maybe we need to look for chunks.
    # Actually, the pattern above matches ALL ranges.
    # The issue is usually we pick *one* file.
    # But if we have chunks, we will have multiple files (different ranges) with different timestamps potentially.
    # We need to group them or select a consistent set.
    # For now, let's just return all valid files and let calculate_metrics handle them.

    valid_files = []
    for f in files:
        try:
            # We relax the check to just valid parquet + H5 existence
            # The strict SLURM ID check is removed as user might run locally or differently
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
        except Exception as e:
            print(f"DEBUG: Error processing {f}: {e}")
            continue

    # Sort by timestamp (newest first)
    def get_ts(x):
        try:
            val = pd.read_parquet(x)["timestamp"].iloc[0]
            return pd.to_datetime(val)
        except Exception:
            return pd.Timestamp.min

    valid_files.sort(key=get_ts, reverse=True)
    return valid_files


def calculate_metrics(
    dataset: str, intensity: float, method: str, n_bootstraps: int | None = None
) -> dict[str, float]:
    """Computes all UQ statistics for a given setting."""

    # helper to aggregate results
    all_metrics = defaultdict(list)

    if method == "boundary":
        boundary_files = find_files(dataset, intensity, method, is_bootstrapping=False)
        if not boundary_files:
            return {}

        for b_file in boundary_files:
            try:
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

                # Load GT
                gt = get_ground_truth(dataset, (start, end))

                # To CUDA
                samples = torch.tensor(samples).cuda()

                # Resize if needed
                if gt.shape[-2:] != samples.shape[-2:]:
                    gt = F.interpolate(
                        gt.unsqueeze(1), size=samples.shape[-2:], mode="bilinear"
                    ).squeeze(1)

                n = min(len(gt), len(samples))
                gt = gt[:n]
                samples = samples[:n]

                # Compute Stats
                chunk_metrics = compute_stats_from_samples(samples, gt)

                for k, v in chunk_metrics.items():
                    all_metrics[k].extend(v)

            except Exception as e:
                print(f"Error processing {b_file}: {e}")

    else:
        # Bootstrapping / Ensemble
        if method == "unet_ensemble":
            boot_files = find_files(dataset, intensity, method, is_bootstrapping=False)
            is_bootstrapping = False
        else:
            boot_files = find_files(
                dataset,
                intensity,
                method,
                is_bootstrapping=True,
                n_bootstraps=n_bootstraps,
            )
            is_bootstrapping = True

        if not boot_files:
            return {}

        # New Logic: If bootstrapping, we might have multiple chunks (10-20, 20-30, etc.)
        # If we have a full file (10-110), use that.
        # Otherwise, try to aggregate chunks.

        files_to_process = []

        # Check for full range first (simplistic check: assuming 10-110 is the target full range, but it might vary)
        # Actually, let's just process all found files, but filter out duplicates (same range, older timestamp).
        # find_files sorts by timestamp (newest first).

        processed_ranges = set()

        for f in boot_files:
            try:
                # Parse Range from filename
                # Model:Dataset:Intensity:Sparse:Range:Seed:Timestamp.parquet
                # bootstrapping_fbp:lung:1000000.0:True:10-20:0:timestamp.parquet

                parts = Path(f).name.split(":")
                # We need to be careful with splitting if timestamp has colons, but usually it doesn't in the filename part before extension?
                # Actually timestamp format in `run.py` uses `datetime.now()` which str() has colons.
                # But let's assume the order is fixed.
                # Range is at index 4 (0-indexed) if split by ':' works somewhat cleanly.
                # Let's search for range pattern.
                range_part = None
                for p in parts:
                    if "-" in p and p.replace("-", "").isdigit():
                        range_part = p
                        break

                if not range_part:
                    continue

                start, end = map(int, range_part.split("-"))
                r_tuple = (start, end)

                # If we already processed this range (or a superset/subset logic?), skip?
                # Simplistic: If exact range already seen, skip (since files are sorted by time, we see newest first).
                if r_tuple in processed_ranges:
                    continue

                # Also, if we have "10-110" and "10-20", do we process both?
                # Ideally we want a non-overlapping set covering the max area.
                # Implementing a greedy coverage?
                # For now: Just process valid non-overlapping chunks if possible, or just all unique newest ranges and let user/stats handle it?
                # Actually, `compute_stats_from_samples` returns lists. We extend `all_metrics`.
                # If we process "10-110" AND "10-20", we double count 10-20 metrics.
                # Simple heuristic: If we find a large range (e.g. > 50 images), prefer that and ignore smaller chunks?
                # OR: If we are in chunked mode (bootstrapping), we prefer consistent chunks?

                # Let's just collect all unique ranges, then perform a check to avoid overlap?
                # Overlap check:
                is_overlap = False
                for pr in processed_ranges:
                    # Check overlap: max(start1, start2) < min(end1, end2)
                    if max(start, pr[0]) < min(end, pr[1]):
                        is_overlap = True
                        break

                if is_overlap and not is_bootstrapping:
                    # For ensembles, we usually just want one file per seed? Or allow duplicates?
                    # Existing logic took `boot_files[0]`.
                    # Let's keep `continue`.
                    continue
                elif is_overlap and is_bootstrapping:
                    # For bootstrapping, if we have 10-110 and 10-20, they overlap.
                    # We should prefer one. Since we sort by timestamp, the newest one wins.
                    # If I ran 10-110 yesterday, and 10-20 today, I probably want 10-20.
                    # So we process 10-20 and then skip 10-110 later?
                    # Yes, `continue` works if we encounter the overlapping one later.
                    continue

                processed_ranges.add(r_tuple)
                files_to_process.append(f)

            except Exception:
                continue

        if not files_to_process:
            # Fallback to old behavior if something fails
            files_to_process = [boot_files[0]]

        for boot_file in files_to_process:
            try:
                df_boot = pd.read_parquet(boot_file)
                # Ensure metadata matches (sanity check)

                start = int(df_boot["image_start_index"].iloc[0])
                end = int(df_boot["image_end_index"].iloc[0])

                preds_boot_np = load_h5_data(boot_file)
                preds_boot = torch.tensor(preds_boot_np).cuda()

                # Format to (N, S, H, W)
                # Ensemble: (N, Steps, Members, H, W) -> (N, Members, H, W)
                # Boot: (N, 1, Samples, H, W) -> (N, Samples, H, W)

                if method == "unet_ensemble":
                    if preds_boot.ndim == 5:
                        preds_boot = preds_boot[:, -1, ...]
                else:
                    if preds_boot.ndim == 5:
                        preds_boot = preds_boot[:, 0, ...]

                # Load GT
                gt = get_ground_truth(dataset, (start, end))

                if gt.shape[-2:] != preds_boot.shape[-2:]:
                    gt = F.interpolate(
                        gt.unsqueeze(1), size=preds_boot.shape[-2:], mode="bilinear"
                    ).squeeze(1)

                n = min(len(gt), len(preds_boot))
                gt = gt[:n]
                preds_boot = preds_boot[:n]

                chunk_metrics = compute_stats_from_samples(preds_boot, gt)

                for k, v in chunk_metrics.items():
                    all_metrics[k].extend(v)

            except Exception as e:
                print(f"Error processing {dataset} {method} file {boot_file}: {e}")
                continue

    # Average
    if not all_metrics:
        # print("No metrics computed.")
        return {}

    return {k: float(np.nanmean(v)) for k, v in all_metrics.items()}


import click


@click.command()
@click.option(
    "--n-bootstraps", type=int, default=None, help="Filter by number of bootstraps"
)
def main(n_bootstraps):
    # Gather Data
    results = {d: {m: defaultdict(list) for m in METHODS} for d in DATASETS}

    files_found = False

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        for method in METHODS:
            for intensity in TOTAL_INTENSITIES:
                metrics = calculate_metrics(
                    dataset, intensity, method, n_bootstraps=n_bootstraps
                )
                if metrics:
                    for k, v in metrics.items():
                        results[dataset][method][k].append(v)
                    results[dataset][method]["intensity"].append(intensity)
                    files_found = True
                else:
                    # print(f"  Missing data for {dataset} {method} {intensity}")
                    pass

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
    ]
    ci_metrics = ["sim_cov", "ind_cov", "width"]

    metrics_to_plot = []
    titles = {}

    for method_name in ci_methods:
        method_title = method_name.replace("_", " ").title()
        if method_name == "gaussian_cons":
            method_title = "Gaussian (Cons.)"

        for metric in ci_metrics:
            key = f"{method_name}_{metric}"
            metrics_to_plot.append(key)

            metric_title = metric.replace("_", " ").title()
            if metric == "sim_cov":
                metric_title = "Sim. Coverage"
            if metric == "ind_cov":
                metric_title = "Ind. Coverage"

            titles[key] = f"{method_title} {metric_title}"

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

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
        # axes is (3,) array

        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[col_idx]

            # Loop methods
            for method in METHODS:
                data = results[dataset][method]
                if not data or not data["intensity"]:
                    continue

                if metric_key not in data:
                    continue

                # Sort by intensity
                ints = np.array(data["intensity"])
                vals = np.array(data[metric_key])

                if len(ints) != len(vals):
                    print(
                        f"Warning: Length mismatch for {dataset} {method} {metric_key}"
                    )
                    continue

                sort_idx = np.argsort(ints)

                ax.plot(
                    ints[sort_idx],
                    vals[sort_idx],
                    label=MODEL_NAMES.get(method, method),
                    color=get_model_colors().get(method, "black"),
                    marker="x",
                )

            ax.set_xscale("log")
            ax.set_xlabel("Total Intensity")
            ax.set_title(dataset.title())
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Y-label only on first plot
            if col_idx == 0:
                ax.set_ylabel(titles.get(metric_key, metric_key))
                ax.legend()

        # Save
        out_path = out_dir / f"{metric_key}.pdf"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
