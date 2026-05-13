"""Side-by-side: ordinary bootstrap vs equivariant bootstrap (FBPUNet estimator).

Single example image (configurable). Plots GT, mean, half-CI-width, and
|true error| per method, plus a reliability curve where the x-axis is the
binned true error and the y-axis is the binned half-CI-width. The diagonal
y = x marks the locus where the predicted half-width equals the expected
|error|. The CI uses the project-wide error level alpha = 0.05.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from uqct.ct import circular_mask
from uqct.eval.run import setup_experiment
from uqct.models.unet import FBPUNet
from uqct.other_methods.equivariant_bootstrapping import _reconstruct
from uqct.uq import error_correlation, percentile_ci, sparsification_error
from uqct.vis.plot_uq import get_ground_truth

# Project convention: alpha is the error level (1 - target coverage).
ALPHA = 0.05


def _load(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        preds = f["preds"][:]  # (N, T=1, R, H, W)
    return torch.from_numpy(preds).float().squeeze(1)  # (N, R, H, W)


def _binned_reliability(
    err: torch.Tensor,
    half_width: torch.Tensor,
    mask: torch.Tensor,
    n_bins: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin pixels by true |error|, return (mean_err, mean_half_width, count)."""
    valid = mask.expand_as(err).bool().flatten()
    e = err.flatten()[valid].cpu().numpy()
    h = half_width.flatten()[valid].cpu().numpy()
    edges = np.quantile(e, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(e, edges) - 1, 0, n_bins - 1)
    mean_e = np.zeros(n_bins)
    mean_h = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = sel.sum()
        if counts[b] > 0:
            mean_e[b] = e[sel].mean()
            mean_h[b] = h[sel].mean()
    return mean_e, mean_h, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--equivariant", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-range", nargs=2, type=int, required=True)
    parser.add_argument("--total-intensity", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image-idx",
        type=int,
        default=0,
        help="Which image (within the loaded slice) to display.",
    )
    parser.add_argument("--out", default="results/figures/bootstrap_comparison.png")
    parser.add_argument("--label-equiv", default="equivariant")
    args = parser.parse_args()

    pred_o = _load(args.ordinary)  # (N, R_o, H, W)
    pred_e = _load(args.equivariant)
    n = min(pred_o.shape[0], pred_e.shape[0])
    pred_o, pred_e = pred_o[:n], pred_e[:n]
    h = pred_o.shape[-1]

    gt = get_ground_truth(args.dataset, tuple(args.image_range)).cpu()[:n]
    if gt.shape[-1] != h:
        gt = F.interpolate(gt.unsqueeze(1), size=(h, h), mode="area").squeeze(1)

    mask = circular_mask(h, device=torch.device("cpu"), dtype=pred_o.dtype)

    # x_hat = FBPUNet(y) on the *original* measurement (no bootstrapping).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, experiment, _ = setup_experiment(
        dataset=args.dataset,
        image_range=tuple(args.image_range),
        total_intensity=args.total_intensity,
        sparse=True,
        seed=args.seed,
        schedule_length=1,
        schedule_start=199,
        schedule_type="linear",
        n_angles=200,
        max_angle=180,
    )
    model = FBPUNet(
        dataset=args.dataset,
        member=0,
        sparse=True,
        batch_size=32,
        model_device=device,
    )
    x_hat = _reconstruct(experiment, model).cpu()[:n] * mask

    lo_o, hi_o = percentile_ci(pred_o, delta=ALPHA, bdim=1)
    lo_e, hi_e = percentile_ci(pred_e, delta=ALPHA, bdim=1)
    half_o = ((hi_o - lo_o) * 0.5) * mask
    half_e = ((hi_e - lo_e) * 0.5) * mask
    err = (x_hat - gt).abs() * mask  # |x_hat - x|, shared between methods

    valid = mask.expand_as(err).bool()
    # Per-image then averaged: Pearson corr(half-width, |error|) and AUSE.
    corr_o = error_correlation(half_o, err).mean().item()
    corr_e = error_correlation(half_e, err).mean().item()
    ause_o = sparsification_error(half_o, err).mean().item()
    ause_e = sparsification_error(half_e, err).mean().item()
    print(f"x_hat     |err|={err[valid].mean():.4f}")
    print(
        f"ordinary  R={pred_o.shape[1]}, half_w={half_o[valid].mean():.4f}, "
        f"corr={corr_o:+.3f}, AUSE={ause_o:.4f}"
    )
    print(
        f"equivar.  R={pred_e.shape[1]}, half_w={half_e[valid].mean():.4f}, "
        f"corr={corr_e:+.3f}, AUSE={ause_e:.4f}"
    )

    # === One example image ===
    i = args.image_idx
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.1])

    img_kw = dict(cmap="gray", vmin=0, vmax=1)
    err_vmax = float(err[i].max())
    hw_vmax = float(max(half_o[i].max(), half_e[i].max()))
    err_kw = dict(cmap="magma", vmin=0, vmax=err_vmax)
    hw_kw = dict(cmap="magma", vmin=0, vmax=hw_vmax)

    panels = [
        ("GT", gt[i], img_kw),
        (r"$\hat{x}$ (FBPUNet)", x_hat[i], img_kw),
        (r"$|\hat{x} - x|$", err[i], err_kw),
        (rf"ordinary half-CI ($\alpha={ALPHA}$)", half_o[i], hw_kw),
        (rf"{args.label_equiv} half-CI ($\alpha={ALPHA}$)", half_e[i], hw_kw),
        (r"equiv − ordinary half-CI", half_e[i] - half_o[i], None),
    ]
    for c, (title, data, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        if kw is None:
            d_lim = float(data.abs().max().clamp_min(1e-6))
            im = ax.imshow(data, cmap="bwr", vmin=-d_lim, vmax=d_lim)
        else:
            im = ax.imshow(data, **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        if c >= 2:
            plt.colorbar(im, ax=ax, fraction=0.046)

    # === Reliability: x = true error, y = half CI width ===
    ax_rel = fig.add_subplot(gs[1, :3])
    e_o, h_o, _ = _binned_reliability(err, half_o, mask, n_bins=25)
    e_e, h_e, _ = _binned_reliability(err, half_e, mask, n_bins=25)
    ax_rel.plot(
        e_o,
        h_o,
        "o-",
        label=f"ordinary (r={corr_o:+.2f}, AUSE={ause_o:.3f})",
        color="tab:blue",
    )
    ax_rel.plot(
        e_e,
        h_e,
        "s-",
        label=f"{args.label_equiv} (r={corr_e:+.2f}, AUSE={ause_e:.3f})",
        color="tab:red",
    )
    lim = float(max(e_o.max(), e_e.max(), h_o.max(), h_e.max()))
    diag = np.linspace(0, lim, 100)
    ax_rel.plot(diag, diag, "--", color="gray", label=r"$y = x$")
    ax_rel.set_xlabel(r"true error $|\hat{x} - x|$ (binned)")
    ax_rel.set_ylabel(rf"half-CI width ($\alpha={ALPHA}$)")
    ax_rel.set_title("Reliability — half-CI width vs. true error")
    ax_rel.legend(fontsize=8)
    ax_rel.grid(alpha=0.3)
    ax_rel.set_aspect("equal")

    # === Per-pixel scatter ===
    ax_sc = fig.add_subplot(gs[1, 3:])
    e_all = err.flatten()[valid.flatten()].cpu().numpy()
    ho = half_o.flatten()[valid.flatten()].cpu().numpy()
    he = half_e.flatten()[valid.flatten()].cpu().numpy()
    rng = np.random.default_rng(0)
    if e_all.size > 30_000:
        keep = rng.choice(e_all.size, size=30_000, replace=False)
        e_all, ho, he = e_all[keep], ho[keep], he[keep]
    ax_sc.scatter(e_all, ho, s=2, alpha=0.05, color="tab:blue", label="ordinary")
    ax_sc.scatter(e_all, he, s=2, alpha=0.05, color="tab:red", label=args.label_equiv)
    lim = float(max(e_all.max(), ho.max(), he.max()))
    diag = np.linspace(0, lim, 100)
    ax_sc.plot(diag, diag, "--", color="gray", label=r"$y = x$")
    ax_sc.set_xlabel(r"true error $|\hat{x} - x|$")
    ax_sc.set_ylabel(rf"half-CI width ($\alpha={ALPHA}$)")
    ax_sc.set_title("pixel scatter (subsample)")
    leg = ax_sc.legend(fontsize=8, markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax_sc.grid(alpha=0.3)

    fig.suptitle(
        f"Bootstrap comparison (FBPUNet) — {args.dataset}, "
        f"intensity {args.total_intensity:.0e}, image {args.image_range[0] + i}",
        y=1.0,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
