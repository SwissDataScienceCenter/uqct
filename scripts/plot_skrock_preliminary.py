"""Quick-look visualisation of the *preliminary* SK-ROCK sparse sweep.

The preliminary sweep is the ``seed == 0`` slice of the full SK-ROCK grid
(3 datasets x 6 total-intensity levels x images [10, 110)); the remaining
9 seeds (the full 1800-cell SLURM array) are not run yet.

This reads every ``results/runs/skrock:*.parquet`` (and, for context, the same
seed-0 cells for a few reference models) and plots PSNR / SSIM / RMSE / mixture
NLL of the posterior mean against total intensity, one row per dataset.

Usage::

    uv run python scripts/plot_skrock_preliminary.py
    uv run python scripts/plot_skrock_preliminary.py --no-baselines --seed 0
"""

from __future__ import annotations

import glob
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.utils import get_results_dir

REF_MODELS = ["fbp", "mle", "unet", "unet_ensemble"]
PRETTY = {
    "skrock": "SK-ROCK",
    "fbp": "FBP",
    "mle": "MLE",
    "unet": "U-Net",
    "unet_ensemble": "U-Net Ens.",
}
METRICS = [
    ("psnr", "PSNR (dB)", "max"),
    ("ssim", "SSIM", "max"),
    ("rmse", "RMSE", "min"),
    ("nll_pred_mix", "mixture NLL (pred)", "min"),
]


def _last(x) -> float:
    return float(np.asarray(x).reshape(-1)[-1])


def _collect(model: str, seed: int | None) -> pd.DataFrame:
    runs = get_results_dir() / "runs"
    rows = []
    for f in sorted(glob.glob(str(runs / f"{model}:*.parquet"))):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            if seed is not None and int(r.get("seed", 0)) != seed:
                continue
            rows.append(
                {
                    "model": model,
                    "dataset": r["dataset"],
                    "intensity": float(r["total_intensity"]),
                    "image": int(r["image_start_index"]),
                    "seed": int(r.get("seed", 0)),
                    **{m: _last(r[m]) for m, _, _ in METRICS if m in r},
                }
            )
    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Restrict to a single seed (the preliminary sweep is seed 0).",
)
@click.option(
    "--baselines/--no-baselines",
    default=True,
    show_default=True,
    help="Overlay FBP / MLE / U-Net / U-Net-Ens. on the same cells.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path (default: <results>/plots/skrock_preliminary.png).",
)
def main(seed: int, baselines: bool, out: Path | None):
    models = ["skrock"] + (REF_MODELS if baselines else [])
    data = pd.concat([_collect(m, seed) for m in models], ignore_index=True)
    if data.empty:
        raise SystemExit("No matching runs found.")

    datasets = ["lung", "composite", "lamino"]
    datasets = [d for d in datasets if d in set(data.dataset)]
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(models)}

    n_rows, n_cols = len(datasets), len(METRICS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False, sharex=True
    )

    for ri, ds in enumerate(datasets):
        for ci, (metric, ylabel, _) in enumerate(METRICS):
            ax = axes[ri][ci]
            for m in models:
                sub = data[(data.model == m) & (data.dataset == ds)]
                if sub.empty or metric not in sub:
                    continue
                g = sub.groupby("intensity")[metric]
                x = np.array(sorted(g.groups))
                mean = g.mean().reindex(x).to_numpy()
                sem = (g.std() / np.sqrt(g.count())).reindex(x).to_numpy()
                ax.errorbar(
                    x,
                    mean,
                    yerr=sem,
                    marker="o",
                    ms=3,
                    lw=1.4,
                    capsize=2,
                    color=colors[m],
                    label=PRETTY.get(m, m),
                    zorder=3 if m == "skrock" else 2,
                )
            ax.set_xscale("log")
            if metric in ("rmse", "nll_pred_mix"):
                ax.set_yscale("log")
            if ri == 0:
                ax.set_title(ylabel)
            if ci == 0:
                ax.set_ylabel(f"{ds}", fontweight="bold")
            if ri == n_rows - 1:
                ax.set_xlabel("total intensity $N_0$")
            ax.grid(alpha=0.3, which="both")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    n_imgs = int(
        data[data.model == "skrock"].groupby(["dataset", "intensity"]).size().min()
    )
    fig.suptitle(
        f"SK-ROCK preliminary sweep (seed {seed}, {n_imgs} images/cell) "
        f"— posterior-mean metrics vs. dose",
        y=1.06,
    )
    fig.tight_layout()

    out = out or (get_results_dir() / "plots" / "skrock_preliminary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
