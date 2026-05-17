"""Calibrate the equivariant bootstrap ``(rotation_std_deg, flip)`` per
(dataset, total_intensity); the base estimator is always ``FBPUNet``.

For each cell, sweeps ``SIGMAS x FLIPS`` and keeps the configuration with the
smallest mean ``|emp_cov - (1 - ALPHA)|`` on the first 10 test images
(``image_range = (0, 10)``, disjoint from the reported sweep ``[10, 110)``) at
``ALPHA = 0.05``, using ``N_BOOTSTRAPS = 100`` replicates per configuration.
Overrides land in ``uqct/settings.toml`` under
``[[eval-sparse.equivariant_bootstrapping.calibrated]]``.
"""

from __future__ import annotations

import math
from itertools import product

import einops
import torch
import torch.nn.functional as F

from uqct.ct import circular_mask
from uqct.eval.run import setup_experiment
from uqct.models.unet import FBPUNet
from uqct.other_methods.equivariant_bootstrapping import (
    get_equivariant_bootstrap_predictor,
)
from uqct.uq import percentile_ci
from uqct.utils import get_root_dir

DATASETS = ("lung", "composite", "lamino")
ALPHA = 0.05
INTENSITIES = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
SIGMAS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
FLIPS = (False, True)
N_BOOTSTRAPS = 100
SEED = 0
IMAGE_RANGE = (0, 10)


def _downsample_gt(gt: torch.Tensor) -> torch.Tensor:
    if gt.shape[-1] == 128:
        return gt
    gt = einops.rearrange(gt, "n h w -> n 1 h w")
    gt = F.interpolate(gt, size=(128, 128), mode="area")
    return einops.rearrange(gt, "n 1 h w -> n h w")


def _calibration_error(samples: torch.Tensor, gt_lr: torch.Tensor) -> float:
    """Mean |emp_cov - (1 - ALPHA)|, averaged over images.

    ``ALPHA`` is the error level (project convention), so the target coverage is
    ``1 - ALPHA`` and ``percentile_ci`` consumes ``ALPHA`` directly as ``delta``.

    ``samples`` has shape (N, R, H, W); ``gt_lr`` has shape (N, H, W).
    """
    mask = circular_mask(samples.shape[-1], device=samples.device)
    mask_sum = mask.sum()
    target_cov = 1.0 - ALPHA
    lo, hi = percentile_ci(samples, delta=ALPHA, bdim=1)
    covered = ((gt_lr >= lo) & (gt_lr <= hi)).float() * mask
    emp_cov = covered.sum(dim=(-1, -2)) / mask_sum
    return float((emp_cov - target_cov).abs().mean().item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load FBPUNet once per dataset; reused across intensities & configs.
    unet_per_dataset: dict[str, FBPUNet] = {}
    for dataset in DATASETS:
        print(f"Loading FBPUNet for {dataset}...")
        unet_per_dataset[dataset] = FBPUNet(
            dataset=dataset,
            member=0,
            sparse=True,
            batch_size=32,
            model_device=device,
        )

    calibrated: list[dict] = []

    for dataset in DATASETS:
        print(f"\n===== Dataset: {dataset} =====")
        model = unet_per_dataset[dataset]

        for intensity in INTENSITIES:
            gt, experiment, schedule = setup_experiment(
                dataset=dataset,
                image_range=IMAGE_RANGE,
                total_intensity=intensity,
                sparse=True,
                seed=SEED,
                schedule_length=1,
                schedule_start=199,
                schedule_type="linear",
                n_angles=200,
                max_angle=180,
            )
            gt_lr = _downsample_gt(gt).to(device)

            scores: dict[tuple[float, bool], float] = {}
            for sigma, flip in product(SIGMAS, FLIPS):
                torch.manual_seed(SEED)
                predictor = get_equivariant_bootstrap_predictor(
                    model,
                    n_bootstraps=N_BOOTSTRAPS,
                    rotation_std_deg=sigma,
                    flip=flip,
                )
                preds = predictor(experiment, schedule)  # (N, 1, R, H, W)
                samples = preds[:, 0]
                scores[(sigma, flip)] = _calibration_error(samples, gt_lr)

            best = min(scores, key=scores.get)
            best_sigma, best_flip = best
            print(
                f"  intensity=1e{int(round(math.log10(intensity)))}: "
                f"best sigma={best_sigma}, flip={best_flip} "
                f"(cal_err={scores[best]:.4f})"
            )
            for (sigma, flip), err in sorted(scores.items()):
                marker = " *" if (sigma, flip) == best else ""
                print(f"      sigma={sigma:>5}, flip={str(flip):>5}: {err:.4f}{marker}")

            calibrated.append(
                {
                    "dataset": dataset,
                    "intensity": intensity,
                    "rotation_std_deg": best_sigma,
                    "flip": best_flip,
                    "calibration_error": scores[best],
                }
            )

    _write_calibrated_to_settings(calibrated)


def _write_calibrated_to_settings(calibrated: list[dict]) -> None:
    settings_path = get_root_dir() / "uqct" / "settings.toml"
    text = settings_path.read_text()
    marker = (
        "# === Calibrated per-(dataset, intensity) overrides (FBPUNet estimator) ==="
    )
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"

    lines = [marker]
    for entry in calibrated:
        lines.append("[[eval-sparse.equivariant_bootstrapping.calibrated]]")
        lines.append(f'dataset = "{entry["dataset"]}"')
        lines.append(f"intensity = {entry['intensity']:.1e}")
        lines.append(f"rotation_std_deg = {entry['rotation_std_deg']}")
        lines.append(f"flip = {str(entry['flip']).lower()}")
        lines.append(f"# calibration_error = {entry['calibration_error']:.4f}")
        lines.append("")

    settings_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines).rstrip() + "\n")
    print(f"\nWrote {len(calibrated)} calibrated overrides to {settings_path}")


if __name__ == "__main__":
    main()
