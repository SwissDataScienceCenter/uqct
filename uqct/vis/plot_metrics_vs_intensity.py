from collections import defaultdict
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.vis.style import MODEL_NAMES, get_model_colors

plt.rcParams.update(
    {
        "text.usetex": True,  # Keep LaTeX for proper math rendering
        "font.family": "sans-serif",  # Switch to sans-serif
        "font.sans-serif": ["Helvetica"],  # Helvetica is the standard print sans
        "font.size": 10,  # Match ICML 10pt body
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,  # 8pt or 9pt for legend
        "xtick.labelsize": 8,  # 8pt for ticks
        "ytick.labelsize": 8,
        "figure.figsize": (6.75, 2.5),  # Full width
    }
)

total_intensities = list(map(float, [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]))
seeds = list(range(10))
gt_ranges_10_batch = [(x * 10, x * 10 + 10) for x in range(1, 11)]
gt_ranges_20_batch = [(10 + x * 20, x * 20 + 30) for x in range(0, 5)]
datasets = ["lamino", "composite", "lung"]
valid_jobids = [54717753, 54757527]


def get_data(
    predictor: str, dataset: str, gt_ranges: list[tuple[int, int]]
) -> np.ndarray:
    files = sorted(glob(f"results/runs/{predictor}:{dataset}*.parquet"))
    toti2psnr = defaultdict(list)
    not_found = []

    for total_intensity in total_intensities:
        for gt_range in gt_ranges:
            for seed in seeds:
                prefix = f"results/runs/{predictor}:{dataset}:{total_intensity}:True:{gt_range[0]}-{gt_range[1]}:{seed}"
                matches = [x for x in files if x.startswith(prefix)]
                if not matches:
                    not_found.append((total_intensity, gt_range, seed, prefix))
                    continue

                # assert (
                #     len(matches) == 1
                # ), f"Not exactly one match for {total_intensity=}, {gt_range=}, {seed=}, {prefix=}: {matches=}"
                psnr = np.array([x for x in pd.read_parquet(matches[0])["psnr"]])
                toti2psnr[total_intensity].append(psnr)
    if len(not_found) > 0:
        print("Not found:")
        for x in not_found:
            print(x)
    out = np.array(list(toti2psnr.values()))[..., -1]

    if len(gt_ranges) > 1:
        out = np.permute_dims(out, (0, 2, 1))
    out = out.reshape(out.shape[0], -1)
    return out


def get_stats(dataset: str) -> dict[str, dict[str, float]]:
    data_diffusion = get_data("diffusion", dataset, gt_ranges_10_batch)
    data_fbp = get_data("fbp", dataset, [(10, 110)])
    data_mle = get_data("mle", dataset, gt_ranges_20_batch)
    data_unet = get_data("unet", dataset, [(10, 110)])
    data_unet_ensemble = get_data("unet_ensemble", dataset, [(10, 110)])
    assert (
        data_diffusion.shape
        == data_fbp.shape
        == data_mle.shape
        == data_unet.shape
        == data_unet_ensemble.shape
    ), (
        f"Shape mismatch for {dataset=}: {data_diffusion.shape=}, {data_fbp.shape=}, {data_mle.shape=}, {data_unet.shape=}, {data_unet_ensemble.shape=}"
    )

    stats = {
        predictor: {
            "mean": np.mean(data, axis=-1),
            "std": np.std(data, axis=-1),
        }
        for predictor, data in zip(
            ["fbp", "mle", "unet", "unet_ensemble", "diffusion"],
            [data_fbp, data_mle, data_unet, data_unet_ensemble, data_diffusion],
        )
    }
    return stats


def get_all_stats():
    return {dataset: get_stats(dataset) for dataset in datasets}


def plot_dataset(dataset: str, stats: dict[str, dict[str, float]], ax: Any) -> None:
    ax.set_title(dataset.title())
    model2color = get_model_colors()
    for predictor, mean_std in stats.items():
        ax.plot(
            total_intensities,
            mean_std["mean"],
            marker="x",
            color=model2color[predictor],
            label=MODEL_NAMES[predictor],
        )
        ax.fill_between(
            total_intensities,
            mean_std["mean"] - mean_std["std"],
            mean_std["mean"] + mean_std["std"],
            color=model2color[predictor],
            alpha=0.2,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Total Intensity")
        ax.set_ylabel("PSNR (dB)")
        ax.legend()


all_stats = get_all_stats()

fig, axes = plt.subplots(1, 3, constrained_layout=True)
for dataset, ax in zip(datasets, axes):
    plot_dataset(dataset, all_stats[dataset], ax)
    ax.tick_params(direction="in", length=4)
    ax.grid(True, linestyle="--", alpha=0.3)  # Adds readability
fig.savefig("psnr.pdf", bbox_inches="tight")  # 'tight' ensures no clipping
