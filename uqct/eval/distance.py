import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import math
import click
import h5py
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from uqct.ct import Experiment, circular_mask, nll
from uqct.eval.nll_constraints import (
    compute_nll_trajectory,
    compute_constraint_violation,
)
from uqct.eval.run import setup_experiment, CTSettings
from uqct.logging import get_logger
from uqct.utils import get_results_dir, load_runs

logger = get_logger(__name__)

DatasetName = Literal["lung", "composite", "lamino"]


@dataclass
class DistanceRun:
    ct_settings: CTSettings
    model: str
    seed: int
    # Results
    uncertainty_mean: float
    # Metadata
    initial_lr: float

    # Artifacts
    uncertainty_images: np.ndarray  # The computed uncertainty images
    distance_maximizers: np.ndarray  # The maximized images

    # Optional detailed stats
    mean_pixel_uncertainty: np.ndarray | None = None

    run_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    slurm_job_id: str | int | None = None
    extra: dict[str, Any] | None = None

    def dump_parquet(self) -> None:
        # Create DataFrame
        data = asdict(self.ct_settings)
        data.update(
            {
                "model": self.model,
                "seed": self.seed,
                "uncertainty_mean": self.uncertainty_mean,
                "initial_lr": self.initial_lr,
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "slurm_job_id": self.slurm_job_id,
            }
        )

        if self.mean_pixel_uncertainty is not None:
            data["mean_pixel_uncertainty"] = self.mean_pixel_uncertainty.tobytes()
            data["mean_pixel_uncertainty_shape"] = list(
                self.mean_pixel_uncertainty.shape
            )

        if self.extra:
            data.update(self.extra)

        df = pd.DataFrame([data])

        # Save
        output_dir = get_results_dir() / "uncertainty_distance"
        output_dir.mkdir(exist_ok=True, parents=True)

        file_name = (
            f"{self.model}:{self.ct_settings.dataset}:{self.ct_settings.total_intensity}:"
            f"{self.ct_settings.sparse}:{self.ct_settings.image_start_index}-"
            f"{self.ct_settings.image_end_index}:{self.seed}:{self.timestamp}"
        )

        fp_parquet = output_dir / (file_name + ".parquet")
        fp_h5 = output_dir / (file_name + ".h5")

        df.to_parquet(fp_parquet, index=False)

        with h5py.File(fp_h5, "w") as f:
            f.create_dataset(
                "uncertainty",
                data=self.uncertainty_images,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "maximizers",
                data=self.distance_maximizers,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

        logger.info(f"Saved distance uncertainty data at \n- {fp_parquet}\n- {fp_h5}")


def find_prediction_files(
    model: str,
    dataset: str,
    total_intensity: float,
    sparse: bool,
    seed: int,
    image_range: tuple[int, int],
) -> list[tuple[Path, tuple[int, int]]]:
    """
    Finds all H5 files containing predictions that overlap with the requested image range.
    Returns a list of (path, (file_start, file_end)) tuples, sorted by start index.
    Handles split files (chunks) by selecting the most recent file for each chunk.
    """
    runs_dir = get_results_dir() / "runs"

    # format: {model}:{dataset}:{total_intensity}:{sparse}:{start}-{end}:{seed}:{timestamp}.h5
    pattern = f"{model}:{dataset}:{total_intensity}:{sparse}:*:{seed}:*.h5"
    candidates = list(runs_dir.glob(pattern))

    req_start, req_end = image_range

    # Group by (start, end) -> list of (path, timestamp)
    chunks: dict[tuple[int, int], list[tuple[Path, float]]] = {}

    for p in candidates:
        try:
            parts = p.name.split(":")
            # Check intensity
            intensity_str = parts[2]
            if not np.isclose(float(intensity_str), total_intensity, rtol=1e-5):
                continue

            # Parse range
            range_str = parts[4]
            file_start, file_end = map(int, range_str.split("-"))

            # Check overlap with request
            # Two ranges [a, b) and [c, d) overlap if a < d and c < b
            if file_start < req_end and req_start < file_end:
                timestamp = p.stat().st_mtime
                if (file_start, file_end) not in chunks:
                    chunks[(file_start, file_end)] = []
                chunks[(file_start, file_end)].append((p, timestamp))

        except Exception:
            continue

    if not chunks:
        raise FileNotFoundError(
            f"No prediction files found covering {image_range} for {model}, {dataset}, {total_intensity}, {sparse}, {seed}"
        )

    # For each chunk, pick the latest
    selected_files = []
    for (start, end), files in chunks.items():
        # Sort by timestamp desc
        files.sort(key=lambda x: x[1], reverse=True)
        selected_files.append((files[0][0], (start, end)))

    # Sort by file start index
    selected_files.sort(key=lambda x: x[1][0])

    return selected_files


def project_to_confidence_set(
    theta: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor,
    confcoefs: torch.Tensor,
    mask: torch.Tensor,
    optimizer: optim.Optimizer,
    max_steps: int = 100,
) -> tuple[torch.Tensor, int]:
    """
    Project theta such that cumulative NLL <= nll_threshold_cum.

    Args:
        theta (torch.Tensor): Initial image batch. Shape (N, H, W).
        experiment (Experiment): Experiment object.
        schedule (torch.Tensor): Schedule.
        confcoefs (torch.Tensor): Confidence coefficients (N).
        mask (torch.Tensor): Valid pixel mask.
        max_steps (int): Max steps.
        optimizer (optim.Optimizer): Optimizer.

    Returns:
        tuple[torch.Tensor, int]: Projected theta and number of steps taken.
    """
    theta_proj = optimizer.param_groups[0]["params"][0]
    theta_proj.data.copy_((theta * mask).clip(0, 1))

    steps = 0
    with torch.enable_grad():
        for _ in range(max_steps):
            optimizer.zero_grad()

            theta_curr = theta_proj * mask

            nlls = nll(
                theta_curr,
                experiment.counts,
                experiment.intensities,
                experiment.angles,
            )[:, schedule[0] :].sum((-2, -1))

            violations = nlls > confcoefs

            if violations.any() == False:
                break
            loss = nlls[violations].sum()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                theta_proj.data.clamp_(0, 1)
                if mask is not None:
                    theta_proj.data[..., ~mask] = 0.0

            steps += 1

    return theta_proj.detach().clip(0, 1) * mask, steps


def distance_maximization(
    pred: torch.Tensor,
    confcoef: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor,
    lr: float = 1,
    lr_reduce_threshold: int = 10,
    patience: int = 5,
    max_steps: int = 10000,
    projection_lr: float = 1e-2,
    projection_steps: int = 10000,
    verbose: bool = True,
    use_l2_grad: bool = True,
    theta_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Maximizes ||theta - pred||_2 s.t. CumSum(NLL(theta)) <= nll_threshold_cum.

    Args:
        pred (torch.Tensor): Prediction batch (N, H, W).
        confcoef (torch.Tensor): Confidence coefficient (N).
    """

    device = pred.device

    # 1. Initialization
    mask = circular_mask(pred.shape[-1], device=device, dtype=torch.bool)

    theta_opt = torch.full_like(pred, 0.5)
    if theta_init is not None:
        theta_opt = theta_init.detach().clone()

    if mask is not None:
        theta_opt[..., ~mask] = 0.0

    best_dist = torch.zeros(pred.shape[0], device=device)
    patience_counter = 0

    # Persistent projection state
    proj_param = theta_opt.detach().clone()  # Shape to match
    proj_param.requires_grad_()
    proj_optimizer = optim.Adam([proj_param], lr=projection_lr)

    best_theta = theta_opt.detach().clone()

    total_proj_steps = 0
    step = 0
    best_dist_mean = 0
    prev_best_dist_mean = 0

    for step in (pbar := tqdm(range(1, max_steps + 1), disable=not verbose)):
        # Check if finished
        if step > 1 and best_dist_mean - prev_best_dist_mean < 1e-5:
            patience_counter += 1
        else:
            patience_counter = 0
        if patience_counter > patience:
            break

        prev_best_dist_mean = best_dist_mean

        # Explicit gradient ascent
        if use_l2_grad:
            diff = theta_opt - pred
            # Normalize gradient to unit length per sample (similar magnitude to sign)
            # diff shape: (N, H, W) -> flatten to (N, -1) for norm
            flat_diff = diff.view(diff.shape[0], -1)
            norm = flat_diff.norm(p=2, dim=-1)
            # Avoid division by zero
            grad = diff / (norm[:, None, None] + 1e-8)
        else:
            grad = torch.sign(theta_opt - pred)
        theta_proposed = theta_opt + lr * grad

        # Apply bounds/mask
        theta_proposed.data.clamp_(0, 1)
        if mask is not None:
            theta_proposed.data[..., ~mask] = 0.0

        # Project ALL
        theta_projected, p_steps = project_to_confidence_set(
            theta_proposed,
            experiment,
            schedule,
            confcoef,
            mask,
            max_steps=projection_steps,
            optimizer=proj_optimizer,
        )

        if p_steps > lr_reduce_threshold:
            lr *= 0.9

        theta_opt = theta_projected

        # Calculate Distance and Update Best
        with torch.no_grad():
            d = torch.norm((theta_opt - pred).view(pred.shape[0], -1), p=2, dim=-1)

        improved = d > best_dist

        best_dist[improved] = d[improved]
        best_theta[improved] = theta_opt[improved]

        total_proj_steps += p_steps

        best_dist_mean = best_dist.mean().item()
        pbar.set_postfix(
            {
                "best_dist": best_dist_mean,
                "proj_steps": p_steps,
                "lr": lr,
                "patience": f"{patience_counter}/{patience}",
            }
        )
    pbar.close()
    avg_proj_steps = total_proj_steps / step
    return best_theta, {"proj_steps": avg_proj_steps, "opt_steps": step}


def pairwise_distance_maximization(
    pred: torch.Tensor,
    confcoef: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor,
    lr: float = 1e-3,
    lr_reduce_threshold: int = 10,
    rotations: int = 2,
    patience: int = 5,
    max_steps: int = 10000,
    use_l2_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds two images (theta1, theta2) on the boundary of the confidence set that are maximally distant.
    Recycles distance_maximization by alternating the reference 'pred'.
    """
    device = pred.device
    N, H, _ = pred.shape

    # Initialize
    print(f"{pred.shape=}")
    mask = circular_mask(H, device=device, dtype=torch.bool)
    theta1 = pred.clone()
    theta2 = torch.full_like(pred, 0.5)

    theta1.clamp_(0, 1)
    theta2.clamp_(0, 1)
    if mask is not None:
        theta1[..., ~mask] = 0.0
        theta2[..., ~mask] = 0.0

    best_dist = torch.zeros(N, device=device)
    best_theta1 = theta1.clone()
    best_theta2 = theta2.clone()

    for rotation in range(rotations):
        # 1. Optimize theta1 to be far from theta2 (fixed)
        theta2, _ = distance_maximization(
            pred=theta1,  # Maximize ||theta1 - theta2||
            confcoef=confcoef,
            experiment=experiment,
            schedule=schedule,
            lr=lr,
            patience=patience if rotation > 0 else 1,
            max_steps=max_steps,
            lr_reduce_threshold=lr_reduce_threshold,
            use_l2_grad=use_l2_grad,
            theta_init=theta2,
        )

        # 2. Optimize theta2 to be far from theta1 (fixed)
        theta1, _ = distance_maximization(
            pred=theta2,  # Maximize ||theta2 - theta1||
            confcoef=confcoef,
            experiment=experiment,
            schedule=schedule,
            lr=lr,
            max_steps=max_steps,
            lr_reduce_threshold=lr_reduce_threshold,
            patience=patience if rotation > 0 else 1,
            use_l2_grad=use_l2_grad,
            theta_init=theta1,
        )

        with torch.no_grad():
            dist = torch.norm((theta1 - theta2).view(N, -1), p=2, dim=-1)
            improved = dist > best_dist

            if improved.any():
                best_dist[improved] = dist[improved]
                best_theta1[improved] = theta1[improved]
                best_theta2[improved] = theta2[improved]

    return best_theta1, best_theta2


def check_confidence_set_violation(
    maximizers: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor,
    confcoefs: torch.Tensor,
) -> None:
    """
    Verifies that the maximized images satisfy the cumulative NLL constraints.
    Raises RuntimeError if a violation is detected.
    """
    with torch.no_grad():
        nlls = nll(
            maximizers, experiment.counts, experiment.intensities, experiment.angles
        )[:, schedule[0] :].sum((-2, -1))
        violations = nlls > confcoefs
        if violations.any():
            raise RuntimeError("Constraint violation detected after optimization!")


@click.command()
@click.option(
    "--dataset", type=click.Choice(["lung", "composite", "lamino"]), required=True
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model name used for prediction (e.g. diffusion, unet)",
)
@click.option("--total-intensity", type=float, required=True)
@click.option("--sparse/--dense", default=True)
@click.option("--seed", type=int, default=0)
@click.option("--image-range", nargs=2, type=int, default=(10, 170))
@click.option(
    "--lr",
    type=float,
    default=2.0,
    help="Initial learning rate/step size for distance maximization (pixel values)",
)
@click.option("--patience", type=int, default=5)
@click.option("--max-steps", type=int, default=1000)
@click.option("--last-only", default=False, is_flag=True)
def main(
    dataset,
    model,
    total_intensity,
    sparse,
    seed,
    image_range,
    lr,
    # projection_lr,
    patience,
    max_steps,
    last_only,
):
    schedule_length = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_dir = get_results_dir() / "runs"
    runs = load_runs(runs_dir, dataset, total_intensity, True, tuple())
    df = pd.concat(runs.values(), ignore_index=True)
    df = df[
        (df["model"] == model)
        & (df["seed"] == seed)
        & (df["image_start_index"] == image_range[0])
        & (df["image_end_index"] == image_range[1])
    ].reset_index(drop=True)
    gt_hr, experiment, schedule = setup_experiment(
        dataset,
        image_range,
        total_intensity,
        sparse,
        seed,
        schedule_length,
    )
    gt_lr = F.interpolate(
        gt_hr.unsqueeze(0), gt_hr.shape[-1] // 2, mode="area"
    ).squeeze(0)
    assert schedule is not None

    batch_uncertainties = []
    batch_maximizers = []

    batch_nll_preds = df["nll_pred"].tolist()
    batch_nll_preds = np.array(batch_nll_preds)

    nll_pred_full = torch.tensor(
        batch_nll_preds, device=device, dtype=torch.float32
    )  # (B, S_full)
    log_inv_delta = math.log(1.0 / 0.05)
    confcoefs = nll_pred_full.sum(-1) + log_inv_delta
    B, r = experiment.counts.shape[0], experiment.counts.shape[-1]

    p_batch = torch.full((B, r, r), 0.5, device=device)

    best_theta_0, best_theta_1 = pairwise_distance_maximization(
        p_batch,
        confcoefs,
        experiment,
        schedule,
        lr=lr,
        patience=patience,
        max_steps=max_steps,
    )
    check_confidence_set_violation(best_theta_0, experiment, schedule, confcoefs)
    check_confidence_set_violation(best_theta_1, experiment, schedule, confcoefs)

    # maximizers_batch: (B, S, H, W)
    # p_batch: (B, S, H, W)
    u_p_batch = (best_theta_0 - best_theta_1).abs()
    batch_uncertainties.append(u_p_batch.cpu().numpy())
    batch_maximizers.append(
        np.stack([best_theta_0.cpu().numpy(), best_theta_1.cpu().numpy()], axis=1)
    )

    batch_uncertainties = np.concatenate(batch_uncertainties, axis=0)  # (N, S, H, W)
    batch_uncertainties[
        ..., ~circular_mask(batch_uncertainties.shape[-1], dtype=torch.bool).numpy()
    ] = 0
    batch_maximizers = np.concatenate(batch_maximizers, axis=0)

    mean_u = np.mean(batch_uncertainties)

    ct_settings = CTSettings(
        dataset=dataset,
        total_intensity=total_intensity,
        sparse=sparse,
        image_start_index=image_range[0],
        image_end_index=image_range[1],
        intensity_schedule=None,
        angle_schedule=schedule.tolist(),
    )

    # Compute mean pixel uncertainty
    # batch_uncertainties shape: (N, S, H, W)
    mean_pixel_uncertainty = np.mean(batch_uncertainties, axis=0)  # (S, H, W)

    # Store binary search count instead of projection steps
    run = DistanceRun(
        ct_settings=ct_settings,
        model=model,
        seed=seed,
        uncertainty_mean=float(mean_u),
        initial_lr=lr,
        uncertainty_images=batch_uncertainties,
        distance_maximizers=batch_maximizers,
        mean_pixel_uncertainty=mean_pixel_uncertainty,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    run.dump_parquet()


if __name__ == "__main__":
    main()
