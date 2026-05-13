"""UQ-calibration quick-look for the preliminary SK-ROCK sparse sweep.

For every ``results/runs/skrock:*.h5`` (the seed-0 sweep over the
3 datasets x 6 total-intensity levels x images [10, 110)), this loads the saved
posterior samples ``(N, T=1, R=500, H, W)``, reconstructs the matching
area-downsampled GT, and reports, per (dataset, intensity):

* posterior-mean PSNR (sanity vs. the parquet metrics);
* mean per-pixel 90 % credible-interval width over the FOV (and over object pixels);
* empirical coverage = fraction of GT pixels inside their per-pixel 90 % CI;
* mean per-pixel posterior std.

It writes a parquet of per-image rows and a 3xK figure (rows = datasets).

Usage::

    uv run python scripts/plot_skrock_uq.py
    uv run python scripts/plot_skrock_uq.py --q 0.05   # 90% CI (default)
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.ct import circular_mask
from uqct.datasets.utils import get_dataset
from uqct.utils import get_results_dir

_RE = re.compile(r"skrock:([^:]+):([^:]+):True:(\d+)-(\d+):(\d+):")


def _gt_lr(dataset: str, lo: int, hi: int, device) -> torch.Tensor:
    """Area-downsampled (128) GT for test indices [lo, hi), exactly as
    ``evaluate_and_save`` builds it (no FOV mask applied here)."""
    _, test = get_dataset(dataset, True)
    gt = torch.stack([test[i] for i in range(lo, hi)], 0).to(device).squeeze(1)
    gt = F.interpolate(gt[:, None], (128, 128), mode="area")[:, 0]
    return gt


@click.command()
@click.option(
    "--q",
    type=float,
    default=0.05,
    show_default=True,
    help="Lower tail prob; CI is [q, 1-q] (0.05 -> 90% CI).",
)
@click.option(
    "--obj-thresh",
    type=float,
    default=0.05,
    show_default=True,
    help="GT > thresh defines 'object' pixels for the object-only stats.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(q: float, obj_thresh: float, out: Path | None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = get_results_dir() / "runs"
    files = sorted(glob.glob(str(runs / "skrock:*.h5")))
    if not files:
        raise SystemExit("No skrock h5 files found.")
    mask = circular_mask(128).to(dev).bool()

    gt_cache: dict[tuple, torch.Tensor] = {}
    rows = []
    cov_level = 1.0 - 2.0 * q
    for f in files:
        m = _RE.search(f)
        ds, ti, lo, hi = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
        key = (ds, lo, hi)
        if key not in gt_cache:
            gt_cache[key] = _gt_lr(ds, lo, hi, dev)
        gt = gt_cache[key]  # (n, H, W)
        with h5py.File(f, "r") as h:
            x = torch.from_numpy(h["preds"][:]).to(dev).float()
        x = x.reshape(x.shape[0], -1, 128, 128)  # (n, R, H, W), T folded into R (T=1)
        ql = torch.quantile(x, torch.tensor([q, 1 - q], device=dev), dim=1)  # (2,n,H,W)
        loq, hiq = ql[0], ql[1]
        mean, std, width = x.mean(1), x.std(1), (ql[1] - ql[0])
        inside = (gt >= loq) & (gt <= hiq)
        fov = mask.expand_as(mean)
        obj = fov & (gt > obj_thresh)

        def red(t, msk):
            msk = msk.expand_as(t)
            return ((t * msk).sum((-1, -2)) / msk.sum((-1, -2)).clamp_min(1)).cpu()

        err = red((mean - gt) ** 2, fov)
        psnr = 10 * torch.log10(1.0 / err.clamp_min(1e-12))
        for i in range(mean.shape[0]):
            rows.append(
                {
                    "dataset": ds,
                    "intensity": ti,
                    "image": lo + i,
                    "psnr": float(psnr[i]),
                    "ciw_fov": float(red(width, fov)[i]),
                    "ciw_obj": float(red(width, obj)[i]),
                    "cov_fov": float(red(inside.float(), fov)[i]),
                    "cov_obj": float(red(inside.float(), obj)[i]),
                    "pstd_fov": float(red(std, fov)[i]),
                }
            )
    d = pd.DataFrame(rows)
    pq = get_results_dir() / "plots" / "skrock_uq_per_image.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(pq)

    agg = (
        d.groupby(["dataset", "intensity"])
        .mean(numeric_only=True)
        .drop(columns="image")
    )
    print(f"target coverage = {cov_level:.2f}")
    print(agg.round(4).to_string())

    # ---- figure ----
    datasets = [x for x in ("lung", "composite", "lamino") if x in set(d.dataset)]
    panels = [
        ("psnr", "posterior-mean PSNR (dB)", None),
        ("cov_fov", f"coverage @ {int(cov_level*100)}%  (FOV)", cov_level),
        ("cov_obj", f"coverage @ {int(cov_level*100)}%  (object px)", cov_level),
        ("ciw_fov", "mean 90% CI width (FOV)", None),
        ("pstd_fov", "mean posterior std (FOV)", None),
    ]
    fig, axes = plt.subplots(
        len(datasets),
        len(panels),
        figsize=(3.1 * len(panels), 2.7 * len(datasets)),
        squeeze=False,
        sharex=True,
    )
    for ri, ds in enumerate(datasets):
        for ci, (col, ylabel, hline) in enumerate(panels):
            ax = axes[ri][ci]
            sub = d[d.dataset == ds]
            g = sub.groupby("intensity")[col]
            xs = np.array(sorted(g.groups))
            mean = g.mean().reindex(xs).to_numpy()
            sem = (g.std() / np.sqrt(g.count())).reindex(xs).to_numpy()
            ax.errorbar(
                xs,
                mean,
                yerr=sem,
                marker="o",
                ms=3.5,
                lw=1.5,
                capsize=2,
                color="tab:blue",
            )
            ax.set_xscale("log")
            if col.startswith("ciw") or col.startswith("pstd"):
                ax.set_yscale("log")
            if hline is not None:
                ax.axhline(hline, ls="--", lw=1, color="k", alpha=0.6)
                ax.set_ylim(min(0.4, np.nanmin(mean) - 0.05), 1.02)
            if ri == 0:
                ax.set_title(ylabel)
            if ci == 0:
                ax.set_ylabel(ds, fontweight="bold")
            if ri == len(datasets) - 1:
                ax.set_xlabel("total intensity $N_0$")
            ax.grid(alpha=0.3, which="both")
    n_imgs = int(d.groupby(["dataset", "intensity"]).size().min())
    fig.suptitle(
        f"SK-ROCK preliminary sweep — per-pixel UQ calibration "
        f"(seed 0, {n_imgs} images/cell; dashed = nominal {int(cov_level*100)}%)",
        y=1.04,
    )
    fig.tight_layout()
    out = out or (get_results_dir() / "plots" / "skrock_uq.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}\nwrote {pq}")


if __name__ == "__main__":
    main()
