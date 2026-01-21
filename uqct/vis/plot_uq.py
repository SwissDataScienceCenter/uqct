from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.uq import simultaneous_ci, error_correlation
from uqct.vis.style import MODEL_NAMES, get_model_colors

# Configuration matching the user's setup
TOTAL_INTENSITIES = [1e6, 1e7, 1e8, 1e9]
DATASETS = ["lung", "composite", "lamino"]
METHODS = ["fbp", "unet"]

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


def load_h5_data(parquet_path: str) -> np.ndarray:
    """Loads prediction data from the corresponding H5 file."""
    h5_path = parquet_path.replace(".parquet", ".h5")
    with h5py.File(h5_path, "r") as f:
        # shape: (N, T, R, H, W)
        return f["preds"][:]


def get_ground_truth(dataset: str, image_range: tuple[int, int]) -> torch.Tensor:
    """Loads ground truth images for a given dataset and range."""
    # We can use the existing setup_experiment to get GT, but that might be heavy.
    # Alternatively, use get_dataset directly.
    from uqct.datasets.utils import get_dataset

    _, test_set = get_dataset(dataset, True)

    # Check if range is valid
    start, end = image_range
    indices = range(start, end)

    gt_list = []
    for i in indices:
        # test_set[i] is (C, H, W) -> (1, H, W) usually
        gt_list.append(test_set[i])

    # Stack -> (N, C, H, W)
    gt = torch.stack(gt_list, dim=0)
    return gt.squeeze(1).cuda()  # (N, H, W)


def find_files(
    dataset: str,
    intensity: float,
    model: str,
    is_bootstrapping: bool,
) -> list[str]:
    """Finds run files matching the criteria."""

    # Construct glob pattern
    # Filename format: model:dataset:intensity:sparse:range:timestamp.parquet
    # We assume sparse=True for bootstrapping as per code

    search_model = f"bootstrapping_{model}" if is_bootstrapping else model
    # Relax pattern if needed, but this matches user provided example
    pattern = f"results/runs/{search_model}:{dataset}:{intensity}:True:*.parquet"
    files = glob(pattern)
    # print(f"DEBUG: Pattern: {pattern}, Found: {len(files)}")

    valid_files = []
    for f in files:
        try:
            # We relax the check to just valid parquet + H5 existence
            # The strict SLURM ID check is removed as user might run locally or differently
            h5_path = f.replace(".parquet", ".h5")
            if Path(h5_path).exists():
                valid_files.append(f)
        except Exception as e:
            print(f"DEBUG: Error processing {f}: {e}")
            continue

    # Sort by timestamp (newest first)
    valid_files.sort(
        key=lambda x: pd.read_parquet(x)["timestamp"].iloc[0], reverse=True
    )
    return valid_files


def calculate_metrics(dataset: str, intensity: float, method: str) -> dict[str, float]:
    """Computes the 3 metrics for a given setting."""

    # 1. Load Bootstrapping Run
    boot_files = find_files(dataset, intensity, method, is_bootstrapping=True)
    if not boot_files:
        return {}

    # 2. Load Standard Run (for true error)
    std_files = find_files(dataset, intensity, method, is_bootstrapping=False)
    # Allow missing standard run (handled below)

    # Take the latest
    boot_file = boot_files[0]
    std_file = std_files[0] if std_files else None

    # Parse image range to load GT
    df_boot = pd.read_parquet(boot_file)
    start = int(df_boot["image_start_index"].iloc[0])
    end = int(df_boot["image_end_index"].iloc[0])

    # Load Data
    preds_boot_np = load_h5_data(boot_file)  # (N, 1, B, H, W)

    # Handle Standard Predictions
    preds_std_np = None
    if std_file:
        h5_std = std_file.replace(".parquet", ".h5")
        if Path(h5_std).exists():
            preds_std_np = load_h5_data(std_file)

    if preds_std_np is None:
        print(
            f"  Warning: Standard run missing/incomplete for {dataset} {method} {intensity}. Using Bootstrap Mean."
        )

    # Create Tensors
    preds_boot = torch.tensor(preds_boot_np).cuda()

    if preds_std_np is None:
        # Compute mean from bootstraps
        if preds_boot.ndim == 5:
            preds_std = preds_boot.mean(dim=2).squeeze(1)
        else:
            preds_std = preds_boot.mean(dim=1)
    else:
        preds_std = torch.tensor(preds_std_np).cuda()

    # Squeeze dimensions
    if preds_boot_np.ndim == 5:
        preds_boot = preds_boot[:, 0, ...]

    if preds_std_np is not None:
        if preds_std.ndim == 5:
            preds_std = preds_std[:, -1, 0, ...]
        elif preds_std.ndim == 4:
            preds_std = preds_std[:, -1, ...]

    # Load GT
    gt = get_ground_truth(dataset, (start, end))

    # Resize GT if needed
    if gt.shape[-2:] != preds_boot.shape[-2:]:
        # GT is (N, H, W), interpolate expects (N, C, H, W)
        gt = F.interpolate(
            gt.unsqueeze(1),
            size=preds_boot.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    # Ensure shapes match
    if len(gt) != len(preds_boot):
        n = min(len(gt), len(preds_boot))
        gt = gt[:n]
        preds_boot = preds_boot[:n]
        preds_std = preds_std[:n]

    sim_covs = []
    ind_covs = []
    corrs = []

    for i in range(len(gt)):
        samples = preds_boot[i]  # (B, H, W)

        # Simultaneous CI
        # bdim=0 because samples is (B, H, W)
        lower, upper = simultaneous_ci(samples, delta=0.05, bdim=0)

        lower = lower.clamp(0, 1)
        upper = upper.clamp(0, 1)

        target = gt[i]

        # 1. Simultaneous Coverage
        in_bounds = (target >= lower) & (target <= upper)
        sim_cov = in_bounds.all().float().item()
        sim_covs.append(sim_cov)

        # 2. Independent Coverage
        ind_cov = in_bounds.float().mean().item()
        ind_covs.append(ind_cov)

        # 3. Correlation
        width = upper - lower
        error = (preds_std[i] - target).abs()

        # error_correlation expects (..., H, W)
        # We can pass single image tensors (H, W) -> output scalar
        corr = error_correlation(width, error, circle_mask=False).item()
        corrs.append(corr)

    return {
        "sim_cov": np.mean(sim_covs),
        "ind_cov": np.mean(ind_covs),
        "corr": np.mean(corrs),
    }


def main():
    # Gather Data
    results = {d: {m: defaultdict(list) for m in METHODS} for d in DATASETS}

    # Iterate
    # This might be slow if standard sequential.
    # But script is one-off.

    files_found = False

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        for method in METHODS:
            for intensity in TOTAL_INTENSITIES:
                metrics = calculate_metrics(dataset, intensity, method)
                if metrics:
                    results[dataset][method]["sim_cov"].append(metrics["sim_cov"])
                    results[dataset][method]["ind_cov"].append(metrics["ind_cov"])
                    results[dataset][method]["corr"].append(metrics["corr"])
                    results[dataset][method]["intensity"].append(intensity)
                    files_found = True
                else:
                    print(f"  Missing data for {dataset} {method} {intensity}")

    if not files_found:
        print("No valid data found to plot.")
        return

    # Plotting
    metrics_cols = ["sim_cov", "ind_cov", "corr"]
    metric_titles = {
        "sim_cov": "Simultaneous Coverage",
        "ind_cov": "Independent Coverage",
        "corr": "Correlation w/ Error",
    }
    metric_filenames = {
        "sim_cov": "bootstrapping_sim_cov.pdf",
        "ind_cov": "bootstrapping_ind_cov.pdf",
        "corr": "bootstrapping_corr.pdf",
    }

    # Loop over metrics -> Create one figure per metric
    for metric_key in metrics_cols:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

        # If only 1 dataset, axes is not array, but we have 3 datasets hardcoded
        # DATASETS = ["lung", "composite", "lamino"]
        # So axes should be (3,)

        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[col_idx]

            # Loop methods
            for method in METHODS:
                data = results[dataset][method]
                if not data or not data["intensity"]:
                    continue

                # Sort by intensity
                ints = np.array(data["intensity"])
                vals = np.array(data[metric_key])
                sort_idx = np.argsort(ints)

                ax.plot(
                    ints[sort_idx],
                    vals[sort_idx],
                    label=MODEL_NAMES.get(method, method),
                    color=get_model_colors().get(method, "black"),
                    marker="o",
                    markersize=4,
                )

            ax.set_xscale("log")
            ax.set_xlabel("Total Intensity")
            ax.set_title(dataset.title())
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Y-label only on first plot
            if col_idx == 0:
                ax.set_ylabel(metric_titles[metric_key])
                ax.legend()

        # Save
        out_path = metric_filenames[metric_key]
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
