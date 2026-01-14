import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal
from uuid import uuid4

import einops
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.ct import Experiment, nll_mixture_angle_schedule, sample_observations
from uqct.datasets.utils import get_dataset
from uqct.logging import get_logger
from uqct.metrics import get_metrics
from uqct.training.unet import N_ANGLES
from uqct.utils import get_results_dir

logger = get_logger(__name__)


@dataclass
class CTSettings:
    dataset: str
    total_intensity: float
    sparse: bool
    image_start_index: int
    image_end_index: int
    pred_angles: list[int] | None = None
    intensity_schedule: list[float] | None = None
    num_rounds: int | None = None


@dataclass
class Metrics:
    psnr: list[list[float]]
    ssim: list[list[float]]
    rmse: list[list[float]]
    zeroone: list[list[float]]
    l1: list[list[float]]
    nll_pred: list[list[float]]
    nll_gt: list[list[float]]


@dataclass
class Run:
    ct_settings: CTSettings
    model: str
    metrics: Metrics
    seed: int
    preds: np.ndarray

    run_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    slurm_job_id: str | int | None = None
    extra: dict[str, Any] | None = None

    def __str__(self) -> str:
        s = []
        s.append(f"Run Summary (ID: {self.run_id})")
        if self.slurm_job_id:
            s.append(f"  SLURM Job ID: {self.slurm_job_id}")
        s.append(f"  Timestamp: {self.timestamp}")
        s.append(f"  Model: {self.model}")
        s.append(f"  Dataset: {self.ct_settings.dataset}")
        s.append(f"  Total Intensity: {self.ct_settings.total_intensity:.2e}")
        s.append(f"  Sparse: {self.ct_settings.sparse}")
        s.append(f"  Seed: {self.seed}")

        if self.extra:
            s.append("  Extra Metadata:")
            for k, v in self.extra.items():
                s.append(f"    {k}: {v}")

        s.append("  Metrics (Mean over all images and steps):")
        metrics_dict = asdict(self.metrics)
        for k, v in metrics_dict.items():
            # v is list[list[float]]
            # Flatten and calculate mean
            try:
                values = [item for sublist in v for item in sublist]
                if values:
                    mean_val = sum(values) / len(values)
                    s.append(f"    {k}: {mean_val:.4f}")
                else:
                    s.append(f"    {k}: N/A")
            except Exception:
                s.append(f"    {k}: Error calculating mean")

        return "\n".join(s)

    def __repr__(self) -> str:
        return self.__str__()

    def dump_parquet(self) -> None:
        # Load metrics into dataframe
        metrics_dict = asdict(self.metrics)
        df = pd.DataFrame(metrics_dict)
        for k, v in asdict(self.ct_settings).items():
            df[k] = v

        # Load extra data into dataframe
        if self.extra:
            for k, v in self.extra.items():
                df[k] = v

        # Inject metadata
        df["model"] = self.model
        df["run_id"] = self.run_id
        df["timestamp"] = self.timestamp
        df["slurm_job_id"] = self.slurm_job_id
        df["seed"] = self.seed

        # Locate place to save the data
        output_dir = get_results_dir() / "runs"
        output_dir.mkdir(exist_ok=True, parents=True)
        file_name = (
            f"{self.model}:{self.ct_settings.dataset}:{self.ct_settings.total_intensity}:{self.ct_settings.sparse}:"
            f"{self.ct_settings.image_start_index}-{self.ct_settings.image_end_index}:{self.seed}:{self.timestamp}"
        )
        fp_parquet = output_dir / (file_name + ".parquet")
        fp_preds = output_dir / (file_name + ".h5")
        df.to_parquet(fp_parquet, index=False)

        with h5py.File(fp_preds, "w") as f:
            f.create_dataset("preds", data=self.preds, dtype="float32")

        logger.info(f"Saved run data at \n- {fp_parquet}\n- {fp_preds}")


def setup_experiment(
    dataset: Literal["lung", "lamino", "composite"],
    image_range: tuple[int, int],
    total_intensity: float,
    sparse: bool,
    seed: int,
    schedule_length: int,
    schedule_start: int = 10,
    schedule_type: Literal["exp", "linear"] = "exp",
    n_angles: int = N_ANGLES,
    max_angle: int = 180,
) -> tuple[torch.Tensor, Experiment, torch.Tensor | None]:
    """Deterministically computes experiment (and angle schedule if sparse setting).

    Arguments:
        dataset (`str`): One of "lung", "composite" or "lamino"
        image_range (`tuple[int, int]`): Range of test set indices; `image_range[0]` is the first test set index, `image_range[1] - 1` is the last test set index, i.e. the upper bound is exclusive.
        total_intensity (`float`): Total intensity over all angles/rounds
        seed (`int`): Random seed
        sparse (`bool`): Whether we are in a sparse setting
        schedule_start (`int`): Start of the schedule
        schedule_type (`Literal['exp', 'linear']`): Whether to use an exponential or linear schedule
        schedule_length (`int`): Number of angles/rounds to use for the schedule
        n_angles (`int`): Number of angles
        max_angle (`int`): Maximum angle

    Returns:
        `tuple[torch.Tensor, Experiment, torch.Tensor | None]`: Ground truth images (high resolution), experiment object, and schedule if sparse == True, otherwise None.
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _, test_set = get_dataset(dataset, True)

    # Seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    n_gt = image_range[1] - image_range[0]
    gt = (
        torch.stack([test_set[i] for i in range(image_range[0], image_range[1])], dim=0)
        .to(device)
        .squeeze(1)
    )

    angles = torch.from_numpy(np.linspace(0, max_angle, n_angles, endpoint=False)).to(
        device
    )
    n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)
    if sparse:
        intensities = intensities.view(1, 1, 1).expand(n_gt, n_angles, -1) / (
            n_angles * n_detectors_hr
        )
        if schedule_type == "linear":
            schedule = (
                torch.linspace(
                    schedule_start, n_angles - 1, steps=schedule_length, device=device
                )
                .round()
                .int()
            )
        else:
            schedule = (
                torch.logspace(
                    math.log10(schedule_start),
                    math.log10(n_angles - 1),
                    steps=schedule_length,
                    device=device,
                )
                .round()
                .int()
            )
        if (schedule[:-1] == schedule[1:]).any():
            raise ValueError("Schedule must be strictly increasing")
    else:
        n_rounds = schedule_length
        intensities = intensities.view(1, 1, 1, 1).expand(
            n_gt, n_rounds, n_angles, -1
        ) / (n_angles * n_detectors_hr * n_rounds)
        schedule = None
    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts, intensities_lr, angles, sparse)
    print(f"Experiment: {experiment}")
    return gt, experiment, schedule


def evaluate_and_save(
    preds: torch.Tensor,
    gt: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor | None,
    ct_settings: CTSettings,
    model_name: str,
    seed: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Evaluates predictions against ground truth and saves the results.

    Args:
        preds: (N, T, R, H, W) -> Always 5D
        gt: (N, H, W)
        experiment: Experiment object
        schedule: Schedule tensor or None
        ct_settings: CTSettings object
        model_name: Name of the model
        seed: Random seed
        extra_metadata: Additional metadata to save
    """
    n_gt = len(gt)

    # Ensure preds has shape (N, T, R, H, W) for metrics calculation
    # If it happens to be 4D (N, T, H, W), we unsqueeze R=1
    if preds.ndim == 4:
        preds_metrics = preds.unsqueeze(2)
    else:
        preds_metrics = preds

    # Calculate metrics on the average prediction across replicates
    preds_mean = preds_metrics.mean(dim=2)  # (N, T, H, W)

    # Downsample GT for metrics
    gt_lr = F.interpolate(
        einops.rearrange(gt, "n w h -> n 1 w h"), (128, 128), mode="area"
    )
    gt_lr = einops.rearrange(gt_lr, "n 1 w h -> n w h")

    metric2lists = defaultdict(list)
    for image_index in range(n_gt):
        # Iterate over "timesteps" or "samples" dimension
        for t in range(preds_mean.shape[1]):
            image_gt = gt_lr[image_index]
            image_pred = preds_mean[image_index, t]
            for k, v in get_metrics(image_gt, image_pred).items():
                if image_index + 1 > len(metric2lists[k]):
                    metric2lists[k].append(list())
                metric2lists[k][image_index].append(v)
    metric2lists = dict(metric2lists)

    # NLL calculation
    if experiment.sparse:
        assert schedule is not None, "Expecting schedule to not be None."

        # Ensure preds has shape (N, T, R, H, W) for NLL calculation
        # If it came in as 4D, we already handled it above for metrics, but check again for safety/consistency
        if preds.ndim == 4:
            preds_nll = einops.rearrange(preds, "n t w h -> n t 1 w h")
        else:
            preds_nll = preds

        # Ensure prediction time dimension matches schedule length strict
        # User requested to not repeat across dimension to make it work.
        assert preds_nll.shape[1] == len(schedule), (
            f"Prediction time dimension {preds_nll.shape[1]} must match "
            f"schedule length {len(schedule)}."
        )

        preds_nll = preds_nll.contiguous()

        # nll_mixture_angle_schedule expects (..., s, n_preds, H, W)
        # Our preds_nll is (N, s, R, H, W).
        # We pass it directly. R corresponds to n_preds.

        nlls_pred = nll_mixture_angle_schedule(
            preds_nll,
            experiment.counts,
            experiment.intensities,
            experiment.angles,
            schedule,
            reduce=False,
        )

        # For GT, we treat it as single replicate (R=1).
        # We need (N, s, 1, H, W).
        gt_expanded = einops.repeat(gt_lr, "n w h -> n t 1 w h", t=len(schedule))
        gt_expanded = gt_expanded.contiguous()

        nlls_gt = nll_mixture_angle_schedule(
            gt_expanded,
            experiment.counts,
            experiment.intensities,
            experiment.angles,
            schedule,
            reduce=False,
        )
    else:
        # Placeholder for dense setting or raise NotImplementedError as before
        raise NotImplementedError(
            "Dense setting not fully supported for NLL calculation yet."
        )

    metrics = Metrics(
        psnr=metric2lists["PSNR"],
        ssim=metric2lists["SS"],
        rmse=metric2lists["RMSE"],
        zeroone=metric2lists["ZeroOne"],
        l1=metric2lists["L1"],
        nll_pred=nlls_pred.tolist(),
        nll_gt=nlls_gt.tolist(),
    )

    # Try to capture SLURM ID
    # Priority: SLURM_ARRAY_JOB_ID (for arrays) -> SLURM_JOB_ID (for normal)
    slurm_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID"))

    run = Run(
        ct_settings=ct_settings,
        model=model_name,
        metrics=metrics,
        seed=seed,
        preds=preds.cpu().numpy(),
        slurm_job_id=slurm_id,
        extra=extra_metadata,
    )
    logger.info(run)
    run.dump_parquet()


def run_evaluation(
    dataset: Literal["lung", "lamino", "composite"],
    sparse: bool,
    total_intensity: float,
    image_range: tuple[int, int],
    seed: int,
    model_name: str,
    predictor_fn: Callable[[Experiment, torch.Tensor | None], torch.Tensor],
    n_angles: int,
    schedule_start: int,
    schedule_type: Literal["exp", "linear"],
    schedule_length: int,
    max_angle: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Unified evaluation execution logic.
    """
    gt, experiment, schedule = setup_experiment(
        dataset,
        image_range,
        total_intensity,
        sparse,
        seed,
        schedule_length,
        schedule_start,
        schedule_type,
        n_angles,
        max_angle,
    )

    preds = predictor_fn(experiment, schedule)

    ct_settings = CTSettings(
        dataset=dataset,
        total_intensity=total_intensity,
        sparse=sparse,
        image_start_index=image_range[0],
        image_end_index=image_range[1],
    )

    evaluate_and_save(
        preds=preds,
        gt=gt,
        experiment=experiment,
        schedule=schedule,
        ct_settings=ct_settings,
        model_name=model_name,
        seed=seed,
        extra_metadata=extra_metadata,
    )
