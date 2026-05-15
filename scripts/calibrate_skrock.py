"""Calibrate the SK-ROCK TV-prior weight (``beta``) per (dataset, total_intensity).

For each cell, sweeps multiples of the MAP-heuristic ``beta`` (``MULTIPLIERS``)
and keeps the one with the smallest Expected Calibration Error over the nominal
grid ``Q``; ties within ``ECE_TOL`` are broken toward the larger ``beta``.
Calibration uses the first 10 test images (``image_range = (0, 10)``), disjoint
from the reported sweep ``[10, 110)``, with a reduced SK-ROCK budget; per-cell
results are cached on disk so re-runs only fill in the missing combinations.

Overrides land in ``uqct/settings.toml`` under
``[[eval-sparse.skrock.calibrated]]``; cells not listed fall back to
``tv_weight_calibration`` (the global default in ``uqct.other_methods.skrock``).
"""

from __future__ import annotations

import json
import math
import tomllib

import einops
import torch
import torch.nn.functional as F

from uqct.ct import circular_mask
from uqct.datasets.utils import DatasetName
from uqct.eval.run import setup_experiment
from uqct.other_methods.skrock import _calibrate_tv_weight, get_skrock_predictor
from uqct.uq import percentile_ci
from uqct.utils import get_results_dir, get_root_dir

# The full grid is 3 datasets x 6 intensities x len(MULTIPLIERS) SK-ROCK runs --
# ~hours. Trim DATASETS / INTENSITIES for partial runs; results are cached (see
# CACHE_PATH) and the settings.toml overrides merge, so partial runs accumulate.
DATASETS: tuple[DatasetName, ...] = ("lung", "composite", "lamino")
INTENSITIES = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
# Multiples of the MAP-heuristic beta. Spans 1/32 x heuristic up to 4 x.
MULTIPLIERS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_MULT = (
    2.0  # the production default (uqct.other_methods.skrock tv_weight_calibration)
)
IMAGE_RANGE = (0, 10)
SEED = 0
ALPHA = 0.05  # project convention: error level (target coverage 1 - ALPHA)
# Nominal coverage grid for ECE; includes 1 - ALPHA = 0.95.
Q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# When several multipliers tie on ECE within this tolerance, prefer the larger
# beta (tighter intervals / less posterior-mean bias).
ECE_TOL = 0.01

# Reduced SK-ROCK budget for calibration (production defaults: 1000 / 1000; the
# production sweep uses 500 / 500). Keep these stable -- the on-disk cache is keyed
# by them, so changing them invalidates every cached cell.
N_BURNIN = 400
N_SAMPLES = 350
N_STAGES = 10
ETA = 0.05
DT_PERC = 0.95

CACHE_PATH = get_results_dir() / "caches" / "skrock_calib_cache.json"


def _budget_tag() -> str:
    return f"b{N_BURNIN}_s{N_SAMPLES}_st{N_STAGES}_eta{ETA}_dt{DT_PERC}"


def _cache_key(dataset: str, intensity: float, mult: float) -> str:
    return f"{dataset}:{intensity:.0e}:x{mult}:{_budget_tag()}"


def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))


def _downsample_gt(gt: torch.Tensor) -> torch.Tensor:
    if gt.shape[-1] == 128:
        return gt
    gt = einops.rearrange(gt, "n h w -> n 1 h w")
    gt = F.interpolate(gt, size=(128, 128), mode="area")
    return einops.rearrange(gt, "n 1 h w -> n h w")


def _cell_metrics(samples: torch.Tensor, gt_lr: torch.Tensor) -> dict[str, float]:
    """``samples`` (N, R, H, W), ``gt_lr`` (N, H, W). Returns image-averaged
    ``ece`` (mean_q mean_image |emp_cov(q) - q|), ``cov95`` / ``ciw95`` (the
    1 - ALPHA interval), and ``psnr`` (posterior-mean PSNR over the mask)."""
    mask = circular_mask(samples.shape[-1], device=samples.device)
    mask_sum = mask.sum()
    abs_gaps = []
    cov95 = ciw95 = float("nan")
    for q in Q:
        lo, hi = percentile_ci(
            samples, delta=1.0 - q, bdim=1
        )  # central-mass-q interval
        covered = ((gt_lr >= lo) & (gt_lr <= hi)).float() * mask
        emp_cov = covered.sum(dim=(-1, -2)) / mask_sum  # (N,)
        abs_gaps.append((emp_cov - q).abs().mean().item())
        if abs(q - (1.0 - ALPHA)) < 1e-9:
            cov95 = emp_cov.mean().item()
            ciw95 = (((hi - lo) * mask).sum(dim=(-1, -2)) / mask_sum).mean().item()
    mean = samples.mean(dim=1)
    mse = (((mean - gt_lr) ** 2) * mask).sum(dim=(-1, -2)) / mask_sum
    psnr = (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean().item()
    return {
        "ece": float(sum(abs_gaps) / len(abs_gaps)),
        "cov95": cov95,
        "ciw95": ciw95,
        "psnr": psnr,
    }


def _pick_multiplier(results: dict[float, dict]) -> float:
    """Smallest-ECE multiplier; ties within ECE_TOL broken toward the largest beta."""
    min_ece = min(results[m]["ece"] for m in MULTIPLIERS)
    return max(m for m in MULTIPLIERS if results[m]["ece"] <= min_ece + ECE_TOL)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = _load_cache()
    print(
        f"device={device}  datasets={DATASETS}  intensities={INTENSITIES}  "
        f"multipliers={MULTIPLIERS}  budget={_budget_tag()}  "
        f"objective: min ECE (no coverage floor)  cache={len(cache)} cells"
    )
    calibrated: list[dict] = []

    for dataset in DATASETS:
        print(f"\n===== dataset: {dataset} =====")
        for intensity in INTENSITIES:
            heuristic_beta = _calibrate_tv_weight(intensity)
            experiment = (
                schedule
            ) = gt_lr = None  # built lazily, only if a cell misses cache

            results: dict[float, dict] = {}  # mult -> metrics dict (+ "beta")
            for mult in MULTIPLIERS:
                beta = mult * heuristic_beta
                key = _cache_key(dataset, intensity, mult)
                m = cache.get(key)
                if m is None:
                    if experiment is None or gt_lr is None or schedule is None:
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
                    preds = get_skrock_predictor(
                        n_burnin=N_BURNIN,
                        n_samples=N_SAMPLES,
                        n_stages=N_STAGES,
                        eta=ETA,
                        dt_perc=DT_PERC,
                        prior="tv",
                        tv_weight=beta,
                        verbose=False,
                        seed=SEED,
                    )(experiment, schedule)  # (N, 1, R, H, W)
                    m = _cell_metrics(preds[:, 0], gt_lr)
                    cache[key] = m
                    _save_cache(cache)
                m = {**m, "beta": beta}
                results[mult] = m

            best_mult = _pick_multiplier(results)
            best = results[best_mult]
            i_lbl = f"1e{int(round(math.log10(intensity)))}"
            print(f"  intensity={i_lbl}  heuristic_beta={heuristic_beta:.3e}")
            for mult in MULTIPLIERS:
                r = results[mult]
                tag = " <- prod default" if mult == DEFAULT_MULT else ""
                tag += "  *PICK*" if mult == best_mult else ""
                print(
                    f"    x{mult:<7}  beta={r['beta']:.3e}  cov@95={r['cov95']:.3f}  "
                    f"CIw@95={r['ciw95']:.4f}  ECE={r['ece']:.4f}  PSNR={r['psnr']:.2f}{tag}"
                )
            if best_mult != DEFAULT_MULT:
                print(
                    f"    => override x{best_mult} (ECE {best['ece']:.4f}, cov@95 {best['cov95']:.3f}, "
                    f"CIw@95 {best['ciw95']:.4f}, PSNR {best['psnr']:.2f}; "
                    f"x{DEFAULT_MULT} default ECE {results[DEFAULT_MULT]['ece']:.4f} "
                    f"cov@95 {results[DEFAULT_MULT]['cov95']:.3f})."
                )
                calibrated.append(
                    {
                        "dataset": dataset,
                        "intensity": intensity,
                        "tv_weight": best["beta"],
                        "tv_weight_multiplier": best_mult,
                        "cov95": best["cov95"],
                        "ciw95": best["ciw95"],
                        "ece": best["ece"],
                        "ece_default": results[DEFAULT_MULT]["ece"],
                    }
                )
            else:
                print(
                    f"    => x{DEFAULT_MULT} default is the ECE-min (ECE {best['ece']:.4f})."
                )

    covered: set[tuple[str, float]] = {
        (str(d), float(i)) for d in DATASETS for i in INTENSITIES
    }
    _write_calibrated_to_settings(calibrated, covered)


def _write_calibrated_to_settings(
    calibrated: list[dict], covered: set[tuple[str, float]]
) -> None:
    """Merge ``calibrated`` (keyed by (dataset, intensity)) into the
    ``[[eval-sparse.skrock.calibrated]]`` block of settings.toml. Cells in
    ``covered`` are first dropped from whatever was previously recorded (so a full
    sweep replaces the block; a partial sweep keeps the cells it did not touch)."""
    settings_path = get_root_dir() / "uqct" / "settings.toml"
    text = settings_path.read_text()
    marker = "# === Calibrated SK-ROCK TV weight per (dataset, intensity) ==="

    with open(settings_path, "rb") as f:
        existing = (
            tomllib.load(f)
            .get("eval-sparse", {})
            .get("skrock", {})
            .get("calibrated", [])
        )
    merged: dict[tuple[str, float], dict] = {
        (e["dataset"], float(e["intensity"])): {
            "dataset": e["dataset"],
            "intensity": float(e["intensity"]),
            "tv_weight": float(e["tv_weight"]),
            "tv_weight_multiplier": e.get("tv_weight_multiplier"),
            "cov95": e.get("cov95"),
            "ciw95": e.get("ciw95"),
            "ece": e.get("ece"),
            "ece_default": e.get("ece_default"),
        }
        for e in existing
        if (e["dataset"], float(e["intensity"])) not in covered
    }
    for entry in calibrated:
        merged[(entry["dataset"], float(entry["intensity"]))] = entry

    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"

    lines = [marker]
    lines.append(
        "# Smallest-ECE multiple of the MAP-heuristic beta on the first 10 test images"
    )
    lines.append(
        "# (pure ECE objective, no coverage floor); written by scripts/calibrate_skrock.py."
    )
    lines.append("# Cells not listed use tv_weight_calibration.")
    for _, entry in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append("[[eval-sparse.skrock.calibrated]]")
        lines.append(f'dataset = "{entry["dataset"]}"')
        lines.append(f"intensity = {entry['intensity']:.1e}")
        lines.append(f"tv_weight = {entry['tv_weight']:.6e}")
        if entry.get("tv_weight_multiplier") is not None:
            lines.append(f"tv_weight_multiplier = {entry['tv_weight_multiplier']}")
        comment_bits = []
        if entry.get("ece") is not None:
            comment_bits.append(f"ECE {entry['ece']:.4f}")
        if entry.get("cov95") is not None:
            comment_bits.append(f"cov@95 {entry['cov95']:.3f}")
        if entry.get("ciw95") is not None:
            comment_bits.append(f"CIw@95 {entry['ciw95']:.4f}")
        if entry.get("ece_default") is not None:
            comment_bits.append(f"(x{DEFAULT_MULT} ECE {entry['ece_default']:.4f})")
        if comment_bits:
            lines.append("# " + "  ".join(comment_bits))
        lines.append("")

    settings_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines).rstrip() + "\n")
    print(f"\nWrote {len(merged)} calibrated SK-ROCK overrides to {settings_path}")


if __name__ == "__main__":
    main()
