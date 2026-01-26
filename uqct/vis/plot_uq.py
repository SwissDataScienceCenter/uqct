from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Callable
from uqct.debugging import plot_img

import json
import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

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
    student_t_ci,
    student_t_bonferroni_ci,
)
from uqct.vis.style import MODEL_NAMES, get_model_colors

# Configuration matching the user's setup
TOTAL_INTENSITIES = [1e6, 1e7, 1e8, 1e9]
DATASETS = ["lung", "composite", "lamino"]
METHODS = ["fbp", "unet", "unet_ensemble", "distance_maximization", "boundary"]


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
        search_dir = "results/boundary_sampling"
        pattern = f"{search_dir}/boundary_diffusion:{dataset}:{intensity}*:*.h5"
        files = glob(pattern)
        return sorted(files)
    elif model == "distance_maximization":
        search_dir = "results/uncertainty_distance"
        pattern = f"{search_dir}/diffusion:{dataset}:{intensity}:*.h5"
        files = glob(pattern)
        return sorted(files)

    search_model = f"bootstrapping_{model}" if is_bootstrapping else model
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
            sim_cov = in_ci.all((-2, -1)).float().mean()
            ind_cov = in_ci.float().mean()
            width = (ub - lb).abs().mean()

            all_metrics["distance_maximization_sim_cov"].append(sim_cov.cpu().item())
            all_metrics["distance_maximization_ind_cov"].append(ind_cov.cpu().item())
            all_metrics["distance_maximization_width"].append(width.cpu().item())
            all_metrics["chosen_sim_cov"].append(sim_cov.cpu().item())
            all_metrics["chosen_ind_cov"].append(ind_cov.cpu().item())
            all_metrics["chosen_width"].append(width.cpu().item())
    else:
        # Bootstrapping / Ensemble
        if method == "unet_ensemble":
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

        processed_ranges = set()

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

    return {k: float(np.nanmean(v)) for k, v in all_metrics.items()}


@click.command()
@click.option(
    "--n-bootstraps", type=int, default=None, help="Filter by number of bootstraps"
)
def main(n_bootstraps):
    # Gather Data
    results = {d: {m: defaultdict(list) for m in METHODS} for d in DATASETS}

    files_found = False

    if Path("results/uq_comparison.json").exists():
        results = json.load(open("results/uq_comparison.json", "r"))
        files_found = True
    else:
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
        json.dump(results, open("results/uq_comparison.json", "w"))

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
        method_title = method_name.replace("_", " ").title()
        if method_name == "gaussian_cons":
            method_title = "Gaussian (Cons.)"

        for metric in ci_metrics:
            key = f"{method_name}_{metric}"
            metrics_to_plot.append(key)

            metric_title = metric.replace("_", " ").title()
            if metric == "sim_cov":
                metric_title = r"Sim. Coverage (\%)"
            if metric == "ind_cov":
                metric_title = r"Ind. Coverage (\%)"

            if method_name == "chosen":
                titles[key] = f"{metric_title}"
            else:
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

        fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), constrained_layout=True)
        # axes is (3,) array

        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[col_idx]

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
                    elif "ind_cov" in metric_key:
                        vals = np.array(data["distance_maximization_ind_cov"])
                    elif "width" in metric_key:
                        vals = np.array(data["distance_maximization_width"])
                else:
                    vals = np.array(data[metric_key])

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
                else:
                    vals_plot = vals[sort_idx]

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

            ax.set_xscale("log")
            ax.set_xlabel("Total Intensity")
            ax.set_title(f"{dataset.title()} Dataset")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Y-label only on first plot
            if col_idx == 0:
                ax.set_ylabel(titles.get(metric_key, metric_key))
                ax.legend(loc="best")

        # Save
        out_path = out_dir / f"sparse_{metric_key}.pdf"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
