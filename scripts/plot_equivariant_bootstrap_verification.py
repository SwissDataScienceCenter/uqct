"""Quick verification plots for the equivariant bootstrap (FBPUNet estimator).

For each image in a single (dataset, intensity) result file we display ground
truth, mean reconstruction, |error|, and uncertainty (= CI width), followed by
a binned reliability curve of mean uncertainty vs. mean |error| across pixels.

Note on the /2 convention: the codebase stores uncertainty as the *full* CI
width ``hi - lo`` (see ``uqct/vis/plot_uq.py``). We compare it against the raw
absolute error |x_hat - gt| (no /2). For a calibrated symmetric CI, the
half-width ``width/2`` should track |error| at the chosen quantile.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from uqct.ct import circular_mask
from uqct.uq import percentile_ci
from uqct.vis.plot_uq import get_ground_truth

# Project convention: alpha is the error level (1 - target coverage).
ALPHA = 0.05


def _parse_run_filename(path: str) -> dict:
    """Parses ``equivariant_bootstrapping:<dataset>:<intensity>:<sparse>:<range>:<seed>:<ts>.h5``."""
    name = Path(path).name.replace(".h5", "")
    _model, dataset, intensity, _sparse, image_range, seed, _ts = name.split(":")[:7]
    start, end = (int(x) for x in image_range.split("-"))
    return {
        "dataset": dataset,
        "intensity": float(intensity),
        "image_range": (start, end),
        "seed": int(seed),
    }


def _load_preds(h5_path: str) -> torch.Tensor:
    """Returns bootstrap samples shaped ``(N, R, H, W)`` from a result file."""
    with h5py.File(h5_path, "r") as f:
        preds = f["preds"][:]  # (N, T=1, R, H, W)
    preds = torch.from_numpy(preds).float().squeeze(1)  # (N, R, H, W)
    return preds


def _binned_curve(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin pixels by uncertainty, return (mean_unc, mean_err, count) per bin."""
    valid = mask.bool().flatten()
    u = uncertainty.flatten()[valid].cpu().numpy()
    e = error.flatten()[valid].cpu().numpy()

    edges = np.quantile(u, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-12  # ensure rightmost bin is closed
    idx = np.clip(np.digitize(u, edges) - 1, 0, n_bins - 1)

    mean_u = np.zeros(n_bins)
    mean_e = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = sel.sum()
        if counts[b] > 0:
            mean_u[b] = u[sel].mean()
            mean_e[b] = e[sel].mean()
    return mean_u, mean_e, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a specific equivariant_bootstrapping:*.h5 file. "
        "Defaults to the most recent file in results/runs/.",
    )
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default="results/figures/equivariant_bootstrap_verification.png",
    )
    args = parser.parse_args()

    if args.file is None:
        candidates = sorted(glob("results/runs/equivariant_bootstrapping:*.h5"))
        if not candidates:
            raise FileNotFoundError("No equivariant_bootstrapping:*.h5 files found.")
        args.file = candidates[-1]
    print(f"Loading: {args.file}")

    meta = _parse_run_filename(args.file)
    print(f"Meta: {meta}")

    samples = _load_preds(args.file)  # (N, R, H, W)
    n_total, n_boot, h, w = samples.shape

    gt = get_ground_truth(meta["dataset"], meta["image_range"]).cpu()  # (N, H_gt, W_gt)
    if gt.shape[-1] != h:
        gt = F.interpolate(gt.unsqueeze(1), size=(h, w), mode="area").squeeze(1)
    gt = gt[:n_total]

    mask = circular_mask(h, device=samples.device, dtype=samples.dtype)

    mean = samples.mean(dim=1) * mask
    err = (mean - gt).abs() * mask
    lo, hi = percentile_ci(samples, delta=ALPHA, bdim=1)
    width = (hi - lo).clamp(0, 1) * mask  # full CI width

    n_show = min(args.n_images, n_total)

    fig = plt.figure(figsize=(12, 3 * n_show + 4))
    gs = fig.add_gridspec(n_show + 1, 4, height_ratios=[1] * n_show + [1.2])

    img_kwargs = dict(cmap="gray", vmin=0, vmax=1)
    err_vmax = float(err[:n_show].max().item())
    unc_vmax = float(width[:n_show].max().item())
    print(f"Per-fig vmax: error={err_vmax:.4f}, uncertainty={unc_vmax:.4f}")

    titles = [
        "Ground truth",
        "Mean reconstruction",
        r"$|\hat{x} - x|$",
        rf"CI width ($\alpha={ALPHA}$)",
    ]

    def _bare(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_show):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(gt[i].cpu(), **img_kwargs)
        ax.set_ylabel(f"img {meta['image_range'][0] + i}")
        _bare(ax)
        if i == 0:
            ax.set_title(titles[0])

        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(mean[i].cpu(), **img_kwargs)
        _bare(ax)
        if i == 0:
            ax.set_title(titles[1])

        ax = fig.add_subplot(gs[i, 2])
        im = ax.imshow(err[i].cpu(), cmap="magma", vmin=0, vmax=err_vmax)
        _bare(ax)
        if i == 0:
            ax.set_title(titles[2])
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = fig.add_subplot(gs[i, 3])
        im = ax.imshow(width[i].cpu(), cmap="magma", vmin=0, vmax=unc_vmax)
        _bare(ax)
        if i == 0:
            ax.set_title(titles[3])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Bottom row: pooled uncertainty-vs-error curve.
    ax_curve = fig.add_subplot(gs[n_show, :2])
    mean_u, mean_e, counts = _binned_curve(width, err, mask.expand_as(err), n_bins=20)
    ax_curve.plot(mean_u, mean_e, "o-", label="binned mean |error|")
    lim = max(mean_u.max(), mean_e.max())
    diag = np.linspace(0, lim, 100)
    ax_curve.plot(
        diag, diag / 2, "--", color="gray", label=r"$y = x/2$ (calibrated half-width)"
    )
    ax_curve.plot(diag, diag, ":", color="lightgray", label=r"$y = x$")
    ax_curve.set_xlabel("uncertainty (CI width)")
    ax_curve.set_ylabel(r"$|\hat{x} - x|$")
    ax_curve.set_title(
        f"binned reliability — {meta['dataset']} / "
        f"intensity {meta['intensity']:.0e}"
    )
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)

    # Bottom-right: pixel-level scatter sample for a sanity check.
    ax_sc = fig.add_subplot(gs[n_show, 2:])
    expanded_mask = mask.expand_as(err).bool().flatten()
    u_all = width.flatten()[expanded_mask].cpu().numpy()
    e_all = err.flatten()[expanded_mask].cpu().numpy()
    if u_all.size > 50_000:
        rng = np.random.default_rng(0)
        keep = rng.choice(u_all.size, size=50_000, replace=False)
        u_all, e_all = u_all[keep], e_all[keep]
    ax_sc.scatter(u_all, e_all, s=2, alpha=0.1)
    lim = max(u_all.max(), e_all.max())
    diag = np.linspace(0, lim, 100)
    ax_sc.plot(diag, diag / 2, "--", color="gray")
    ax_sc.plot(diag, diag, ":", color="lightgray")
    ax_sc.set_xlabel("uncertainty (CI width)")
    ax_sc.set_ylabel(r"$|\hat{x} - x|$")
    ax_sc.set_title("pixel scatter (subsample)")
    ax_sc.grid(alpha=0.3)

    fig.suptitle(
        f"Equivariant bootstrap (FBPUNet) — {meta['dataset']} / "
        f"intensity {meta['intensity']:.0e}, R={n_boot}",
        y=0.995,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
