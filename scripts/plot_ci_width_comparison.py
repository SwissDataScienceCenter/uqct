"""Compare per-pixel 95% credible-interval width across UQ methods at the last
schedule step (full angle set), per (dataset, total_intensity) cell.

Loads only the last time-step slice from each ``results/runs/<method>:*.h5`` for
seed 0, computes the mean 95% CI width within the circular FOV, and saves a
parquet + a per-dataset plot under ``results/plots/``.

Methods compared (those with a replicate axis R > 1):
    skrock, bootstrapping_{fbp,unet}, unet_ensemble, diffusion.

Equivariant bootstrap is skipped: only 12 stale h5s exist, all on lung, and they
predate the new equivariant_bootstrapping (no estimator suffix) layout.
"""

from __future__ import annotations

import glob
import re
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from uqct.ct import circular_mask
from uqct.utils import get_results_dir

sys.stdout.reconfigure(line_buffering=True)

METHODS = [
    ("skrock", "tab:blue"),
    ("bootstrapping_fbp", "tab:orange"),
    ("bootstrapping_unet", "tab:green"),
    ("unet_ensemble", "tab:red"),
    ("diffusion", "tab:purple"),
]
RE = re.compile(
    r"^(?P<m>[\w_]+):(?P<ds>[a-z]+):(?P<i>[\d.e+-]+):True:(\d+)-(\d+):(?P<s>\d+):"
)
DELTA = 0.05  # 95% central CI -> width = q(0.975) - q(0.025)


def main() -> None:
    runs = get_results_dir() / "runs"
    plots = get_results_dir() / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    mask = circular_mask(128).to(dev).bool()
    mask_sum = mask.sum()
    qs = torch.tensor([DELTA / 2, 1 - DELTA / 2], device=dev)

    out_pq = plots / "ci_width_comparison.parquet"
    # Resume from a partial parquet so a kill mid-method doesn't lose finished methods.
    if out_pq.exists():
        prev = pd.read_parquet(out_pq)
        done = set(prev.method.unique())
        print(f"resuming: {sorted(done)} already in {out_pq} ({len(prev)} rows)")
        rows = prev.to_dict("records")
    else:
        done = set()
        rows = []
    for m, _color in METHODS:
        if m in done:
            print(f"{m}: cached, skipping")
            continue
        files = sorted(glob.glob(str(runs / f"{m}:*.h5")))
        print(f"{m}: {len(files)} files")
        t0 = time.time()
        n_used = 0
        for k, f in enumerate(files):
            match = RE.match(f.split("/")[-1])
            if not match:
                continue
            ds, intensity, seed = match["ds"], float(match["i"]), int(match["s"])
            if seed != 0:
                continue
            try:
                with h5py.File(f, "r") as h:
                    preds = h["preds"]
                    if preds.ndim != 5:
                        continue
                    sl = preds[:, -1, :, :, :]  # (N, R, H, W) at full-angle-set step
            except Exception as e:
                print(f"  skip {f}: {e}")
                continue
            x = torch.from_numpy(sl).to(dev).float()
            q = torch.quantile(x, qs, dim=1)  # (2, N, H, W)
            width = q[1] - q[0]
            fov = mask.expand_as(width)
            ciw95 = ((width * fov).sum((-1, -2)) / mask_sum).cpu()
            for i, w in enumerate(ciw95.tolist()):
                rows.append(
                    {
                        "method": m,
                        "dataset": ds,
                        "intensity": intensity,
                        "image": i,
                        "ciw95": w,
                    }
                )
            n_used += sl.shape[0]
            del x, q, width, fov
            if k % 30 == 0:
                print(f"  {k + 1}/{len(files)}  {time.time() - t0:.1f}s")
        print(f"  {m}: {n_used} images in {time.time() - t0:.1f}s")
        # Checkpoint after each method so an interruption doesn't redo finished work.
        pd.DataFrame(rows).to_parquet(out_pq)
        print(f"  checkpointed {out_pq} ({len(rows)} rows total)")

    df = pd.DataFrame(rows)
    df.to_parquet(out_pq)
    print(f"wrote {out_pq}  ({len(df)} rows)")

    print("\n=== mean 95% CI width by (method, dataset, intensity) ===")
    agg = (
        df.groupby(["method", "dataset", "intensity"])["ciw95"]
        .mean()
        .unstack("intensity")
    )
    print(agg.round(4))

    datasets = ["lung", "composite", "lamino"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, ds in zip(axes, datasets):
        for m, color in METHODS:
            sub = df[(df.method == m) & (df.dataset == ds)]
            if sub.empty:
                continue
            g = sub.groupby("intensity")["ciw95"]
            x = np.array(sorted(g.groups))
            mean = g.mean().reindex(x).to_numpy()
            sem = (g.std() / np.sqrt(g.count())).reindex(x).to_numpy()
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                marker="o",
                ms=3.5,
                lw=1.4,
                capsize=2,
                color=color,
                label=m,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ds)
        ax.set_xlabel("total intensity $N_0$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean 95% CI width (FOV)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Per-pixel 95% credible-interval width vs. dose, seed 0", y=1.02)
    fig.tight_layout()
    out_png = plots / "ci_width_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
