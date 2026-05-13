"""Quality check for the SK-ROCK sampler (``uqct.other_methods.skrock``).

Runs the sampler on a few test slices at high total intensity and reports, for the
returned posterior samples:

* PSNR of the posterior mean vs. (area-downsampled) GT, with the FBP init PSNR as a baseline;
* mean width of the per-pixel 90% credible interval over the FOV, and over object pixels only;
* empirical coverage: fraction of GT pixels inside their per-pixel 90% CI (FOV / object pixels);
* mean per-pixel posterior std.

Usage::

    uv run python scripts/check_skrock.py            # defaults below
    uv run python scripts/check_skrock.py --burnin 1500 --samples 800 --prior none
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from uqct.ct import circular_mask, fbp, sinogram_from_counts
from uqct.eval.run import setup_experiment
from uqct.other_methods.skrock import get_skrock_predictor


def _summary(
    samples: torch.Tensor, gt_lr: torch.Tensor, mask: torch.Tensor, q: float = 0.05
):
    """``samples`` (N, R, H, W) -> dict of per-image metric tensors."""
    fov = mask.bool()
    obj = fov & (gt_lr > 0.05)
    mean = samples.mean(1)
    lo = torch.quantile(samples, q, dim=1)
    hi = torch.quantile(samples, 1.0 - q, dim=1)
    inside = (gt_lr >= lo) & (gt_lr <= hi)
    width = hi - lo
    std = samples.std(1)

    def reduce_(t, m):
        m = m.expand_as(t)
        return (t * m).sum((-1, -2)) / m.sum((-1, -2)).clamp_min(1)

    err = reduce_((mean - gt_lr) ** 2, fov)
    return {
        "psnr": 10 * torch.log10(1.0 / err),
        "ciw_fov": reduce_(width, fov),
        "ciw_obj": reduce_(width, obj),
        "cov_fov": reduce_(inside.float(), fov),
        "cov_obj": reduce_(inside.float(), obj),
        "std_fov": reduce_(std, fov),
    }


def _fmt(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lung")
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--burnin", type=int, default=700)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--prior", default="tv", choices=["tv", "none"])
    p.add_argument("--intensities", default="1e8,1e9")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    intensities = [float(x) for x in args.intensities.split(",")]
    print(
        f"dataset={args.dataset} n_images={args.n_images} burnin={args.burnin} "
        f"samples={args.samples} prior={args.prior} intensities={intensities}"
    )

    for total_intensity in intensities:
        gt, experiment, schedule = setup_experiment(
            args.dataset,
            (0, args.n_images),
            total_intensity,
            sparse=True,
            seed=args.seed,
            schedule_length=1,
            schedule_start=199,
            schedule_type="linear",
            n_angles=200,
            max_angle=180,
        )
        gt_lr = F.interpolate(gt.unsqueeze(1), (128, 128), mode="area")[:, 0]
        mask = circular_mask(128, device=gt.device, dtype=gt.dtype)
        fbp_init = fbp(
            sinogram_from_counts(experiment.counts, experiment.intensities).clip(0),
            experiment.angles,
        ).clip(0, 1)
        fbp_err = (((fbp_init - gt_lr) ** 2) * mask).sum((-1, -2)) / mask.sum()
        fbp_psnr = 10 * torch.log10(1.0 / fbp_err)

        t0 = time.time()
        predictor_fn = get_skrock_predictor(
            n_burnin=args.burnin,
            n_samples=args.samples,
            prior=args.prior,
            verbose=True,
            seed=args.seed,
            total_intensity_for_calibration=total_intensity,
        )
        out = predictor_fn(experiment, schedule)  # (N, 1, R, H, W)
        samples = out[:, 0]  # (N, R, H, W)
        m = _summary(samples, gt_lr, mask)
        dt = time.time() - t0
        print(
            f"\n--- I={total_intensity:.0e}  ({dt:.0f}s)\n"
            f"  FBP-init PSNR : {fbp_psnr.mean():.2f}  {_fmt(fbp_psnr)}\n"
            f"  PSNR(mean)    : {m['psnr'].mean():.2f}  {_fmt(m['psnr'])}\n"
            f"  CI width 90%  : FOV {m['ciw_fov'].mean():.4f} {_fmt(m['ciw_fov'])} | obj {m['ciw_obj'].mean():.4f} {_fmt(m['ciw_obj'])}\n"
            f"  coverage 90%  : FOV {m['cov_fov'].mean():.3f} {_fmt(m['cov_fov'])} | obj {m['cov_obj'].mean():.3f} {_fmt(m['cov_obj'])}\n"
            f"  mean post.std : {m['std_fov'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
