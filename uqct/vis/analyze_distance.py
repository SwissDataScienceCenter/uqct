import uqct
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import torch
import torch.nn.functional as F

from uqct.vis.style import (
    MODEL_NAMES,
    ICML_TEXT_WIDTH,
    ICML_COLUMN_WIDTH,
    ICML_COLUMN_HEIGHT,
)
from uqct.eval.run import setup_experiment
from uqct.ct import circular_mask

# Setup
RESULTS_DIR = Path("results/uncertainty_distance")
PLOTS_DIR = Path("plots/distance_analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["lung", "composite", "lamino"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
MODEL = "diffusion"
TARGET_CHUNK_START = 10
TARGET_CHUNK_END = 20
BATCH_INDEX = 0  # Index 10 is 0th in 10-20 chunk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask = circular_mask(128, dtype=torch.float32).to(device)
max_uncertainty = mask.sum().item() / mask.numel()


def parse_filename(filename):
    parts = filename.stem.split(":")
    if len(parts) < 8:
        return None
    return {
        "model": parts[0],
        "dataset": parts[1],
        "intensity": float(parts[2]),
        "sparse": parts[3] == "True",
        "start": int(parts[4].split("-")[0]),
        "end": int(parts[4].split("-")[1]),
        "seed": int(parts[5]),
        "timestamp": parts[6],
        "path": filename,
    }


def find_runs():
    files = list(RESULTS_DIR.glob("*.h5"))
    runs = []
    for f in files:
        meta = parse_filename(f)
        if meta and meta["model"] == MODEL:
            runs.append(meta)
    return runs


all_runs = find_runs()
print(f"Found {len(all_runs)} runs for {MODEL}")


def get_latest_runs(runs):
    latest = {}
    for r in runs:
        k = (r["dataset"], r["intensity"], r["start"], r["end"])
        if k not in latest:
            latest[k] = r
        else:
            if r["timestamp"] > latest[k]["timestamp"]:
                latest[k] = r
    return list(latest.values())


cleaned_runs = get_latest_runs(all_runs)

gt_cache = {}


def get_gt_chunk(dataset, start, end):
    key = (dataset, start, end)
    if key in gt_cache:
        return gt_cache[key]

    print(f"Loading GT for {dataset} {start}-{end}...")
    gt_hr, _, _ = setup_experiment(dataset, (start, end), 1e6, True, 0, 32)
    if gt_hr.shape[-1] > 128:
        gt_lr = F.interpolate(gt_hr.unsqueeze(1), size=(128, 128), mode="area").squeeze(
            1
        )
    else:
        gt_lr = gt_hr

    gt_cache[key] = gt_lr.to(device)
    return gt_cache[key]


# Store list of image means per (dataset, intensity) for SEM calculation
dataset_intensity_stats = defaultdict(
    lambda: {"uncertainty_means": [], "coverage_means": []}
)

for r in cleaned_runs:
    ds = r["dataset"]
    inten = r["intensity"]
    start = r["start"]
    end = r["end"]

    try:
        gt_batch = get_gt_chunk(ds, start, end)
    except Exception as e:
        print(f"Skipping GT load for {ds}: {e}")
        gt_batch = None

    with h5py.File(r["path"], "r") as f:
        # Calculate mean per image in the chunk
        # uncertainty shape: (N, H, W)
        u = f["uncertainty"][:]
        # Mean over pixels (axis 1, 2) -> (N,)
        image_means = u.mean(axis=(1, 2))
        dataset_intensity_stats[(ds, inten)]["uncertainty_means"].extend(
            image_means.tolist()
        )

        if gt_batch is not None and "maximizers" in f:
            mins = torch.from_numpy(f["maximizers"][:, 0]).to(device)
            maxs = torch.from_numpy(f["maximizers"][:, 1]).to(device)
            n_imgs = min(len(gt_batch), len(mins))
            # Coverage per image
            covered = (gt_batch[:n_imgs] >= mins[:n_imgs]) & (
                gt_batch[:n_imgs] <= maxs[:n_imgs]
            )
            # Mean over pixels -> (N,)
            cov_per_img = covered.float().mean(dim=(1, 2)).cpu().numpy()
            dataset_intensity_stats[(ds, inten)]["coverage_means"].extend(
                cov_per_img.tolist()
            )

final_stats = {}
for k, v in dataset_intensity_stats.items():
    u_vals = np.array(v["uncertainty_means"])
    c_vals = np.array(v["coverage_means"])

    final_stats[k] = {
        "uncertainty_mean": np.mean(u_vals),
        "uncertainty_sem": (
            np.std(u_vals, ddof=1) / np.sqrt(len(u_vals)) if len(u_vals) > 1 else 0.0
        ),
        "coverage_mean": np.mean(c_vals) if len(c_vals) > 0 else None,
        "coverage_sem": (
            np.std(c_vals, ddof=1) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0.0
        ),
    }

print("Data processing complete.")

# --- Plot 1: Uncertainty (CI Width) vs Intensity ---
# 1 col, 3 rows
fig, axes = plt.subplots(3, 1, figsize=(ICML_COLUMN_WIDTH * 1.0, 5.0), sharex=True)

for i, ds in enumerate(DATASETS):
    ax = axes[i]
    intensities = []
    means = []
    sems = []

    for inten in INTENSITIES:
        if (ds, inten) in final_stats:
            intensities.append(inten)
            means.append(final_stats[(ds, inten)]["uncertainty_mean"])
            sems.append(final_stats[(ds, inten)]["uncertainty_sem"])

    xy = sorted(zip(intensities, means, sems))
    if xy:
        x, y, e = zip(*xy)
        x = np.array(x)
        y = np.array(y)
        e = np.array(e)

        ax.plot(x, y, marker="x", label=f"{ds.capitalize()}")
        ax.fill_between(x, y - e, y + e, alpha=0.3)

    ax.set_xscale("log")
    ax.set_title(ds.capitalize())
    ax.set_ylabel("CI Width")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    if i == 2:
        ax.set_xlabel("Total Intensity")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "uncertainty_vs_intensity.pdf")
plt.savefig(PLOTS_DIR / "uncertainty_vs_intensity.png")

# --- Plot 4: Coverage Rate vs Intensity ---
fig, axes = plt.subplots(3, 1, figsize=(ICML_COLUMN_WIDTH * 1.0, 5.0), sharex=True)

for i, ds in enumerate(DATASETS):
    ax = axes[i]
    intensities = []
    means = []
    sems = []

    for inten in INTENSITIES:
        if (ds, inten) in final_stats:
            if final_stats[(ds, inten)]["coverage_mean"] is not None:
                intensities.append(inten)
                means.append(final_stats[(ds, inten)]["coverage_mean"])
                sems.append(final_stats[(ds, inten)]["coverage_sem"])

    xy = sorted(zip(intensities, means, sems))
    if xy:
        x, y, e = zip(*xy)
        x = np.array(x)
        y = np.array(y)
        e = np.array(e)

        ax.plot(x, y, marker="x", label=f"{ds.capitalize()}")
        ax.fill_between(x, y - e, y + e, alpha=0.3)

    ax.set_xscale("log")
    ax.set_title(ds.capitalize())
    ax.set_ylabel("Coverage Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    if i == 2:
        ax.set_xlabel("Total Intensity")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_vs_intensity.pdf")
plt.savefig(PLOTS_DIR / "coverage_vs_intensity.png")

print("Saved stats plots")


# --- Plot 2: Uncertainty Evolution (Visual) ---
# Tighter spacing, vmin=0, vmax=1
fig, axes = plt.subplots(
    3, 7, figsize=(ICML_TEXT_WIDTH, 4.5 * 3 / 5), constrained_layout=True
)

for i, ds in enumerate(DATASETS):
    gt_img = None
    if (ds, TARGET_CHUNK_START, TARGET_CHUNK_END) in gt_cache:
        gt_img = (
            gt_cache[(ds, TARGET_CHUNK_START, TARGET_CHUNK_END)][BATCH_INDEX]
            .cpu()
            .numpy()
        )
    ax_gt = axes[i, 0]
    if gt_img is not None:
        ax_gt.imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    else:
        ax_gt.text(0.5, 0.5, "GT N/A", ha="center", va="center")
    ax_gt.axis("off")
    if i == 0:
        ax_gt.set_title("GT", fontsize=8)

    ax_gt.text(
        -0.2,
        0.5,
        ds.capitalize(),
        transform=ax_gt.transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )

    for j, inten in enumerate(INTENSITIES):
        ax = axes[i, j + 1]
        run = None
        for r in cleaned_runs:
            if (
                r["dataset"] == ds
                and np.isclose(r["intensity"], inten)
                and r["start"] == TARGET_CHUNK_START
            ):
                run = r
                break
        if run:
            with h5py.File(run["path"], "r") as f:
                unc = f["uncertainty"][BATCH_INDEX]
                # Force vmin=0, vmax=1
                im = ax.imshow(unc, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            # Use 10^k formatting
            exponent = int(np.log10(inten))
            ax.set_title(f"$10^{{{exponent}}}$", fontsize=8)

# Reduce margins manually if needed, constrained_layout handles mostly
plt.savefig(PLOTS_DIR / "uncertainty_evolution.pdf")
print("Saved uncertainty_evolution plots")

# --- Plot 3: Lung/Composite Bounds Evolution ---
DS_TARGET = "composite"
fig, axes = plt.subplots(
    2,
    6,
    figsize=(ICML_TEXT_WIDTH, ICML_TEXT_WIDTH / 3.0),
    gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    constrained_layout=True,
)

for j, inten in enumerate(INTENSITIES):
    upper_ax = axes[0, j]
    lower_ax = axes[1, j]

    run = None
    for r in cleaned_runs:
        if (
            r["dataset"] == DS_TARGET
            and np.isclose(r["intensity"], inten)
            and r["start"] == TARGET_CHUNK_START
        ):
            run = r
            break

    if run:
        with h5py.File(run["path"], "r") as f:
            mms = f["maximizers"][BATCH_INDEX]
            min_img = mms[0]
            max_img = mms[1]
            upper_ax.imshow(max_img, cmap="gray", vmin=0, vmax=1)
            lower_ax.imshow(min_img, cmap="gray", vmin=0, vmax=1)

    upper_ax.axis("off")
    lower_ax.axis("off")

    if j == 0:
        upper_ax.text(
            -0.1,
            0.5,
            "Upper",
            transform=upper_ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )
        lower_ax.text(
            -0.1,
            0.5,
            "Lower",
            transform=lower_ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )

    # Label logic: 10^4, 10^5...
    exponent = int(np.log10(inten))
    upper_ax.set_title(f"$10^{{{exponent}}}$", fontsize=8)

plt.savefig(PLOTS_DIR / f"{DS_TARGET}_bounds_evolution.pdf")
print(f"Saved {DS_TARGET}_bounds_evolution plots")

# --- Plot 5: Individual Replicates ---
DS_REP = "composite"
INTEN_REP = 1e7
fig = plt.figure(figsize=(ICML_TEXT_WIDTH, 1), constrained_layout=True)
# Reduce height

run = None
for r in cleaned_runs:
    if (
        r["dataset"] == DS_REP
        and np.isclose(r["intensity"], INTEN_REP)
        and r["start"] == TARGET_CHUNK_START
    ):
        run = r
        break

if run:
    with h5py.File(run["path"], "r") as f:
        if "replicates" in f:
            reps = f["replicates"][BATCH_INDEX]
            k = reps.shape[0]
            breakpoint()

            if k > 1:
                axes = fig.subplots(1, k)
            else:
                axes = np.array([fig.subplots(1, 1)])
            axes = axes.flatten()

            for i in range(k):
                ax = axes[i]
                ax.imshow(reps[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

        else:
            print("No replicates dataset found.")
else:
    print("Run for replicates not found.")

plt.savefig(PLOTS_DIR / "replicates_example.pdf")
print("Saved replicates_example plots")
