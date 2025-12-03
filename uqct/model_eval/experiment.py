import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd
import torch

from uqct.ct import Experiment, sample_observations
from uqct.datasets.utils import get_dataset
from uqct.training.unet import N_ANGLES
from uqct.utils import get_root_dir


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
    extra: dict[str, Any] | None = None

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
        df["seed"] = self.seed

        # Locate place to save the data
        output_dir = get_root_dir() / "results" / "model_eval"
        output_dir.mkdir(exist_ok=True, parents=True)
        file_name = (
            f"{self.model}:{self.ct_settings.dataset}:{self.ct_settings.total_intensity}:{self.ct_settings.sparse}:"
            f"{self.ct_settings.image_start_index}-{self.ct_settings.image_end_index}:{self.timestamp}"
        )
        fp_parquet = output_dir / (file_name + ".parquet")
        fp_preds = output_dir / (file_name + ".h5")
        df.to_parquet(fp_parquet, index=False)

        with h5py.File(fp_preds, "w") as f:
            f.create_dataset("preds", data=self.preds, dtype="float32")

        print(f"Saved run data at \n- {fp_parquet}\n- {fp_preds}")


def setup_experiment(
    dataset: Literal["lung", "lamino", "composite"],
    image_range: tuple[int, int],
    total_intensity: float,
    sparse: bool,
    seed: int,
) -> tuple[torch.Tensor, Experiment, torch.Tensor | None]:
    """Deterministically computes experiment (and angle schedule if sparse setting).

    Arguments:
        dataset (`str`): One of "lung", "composite" or "lamino"
        image_range (`tuple[int, int]`): Range of test set indices; `image_range[0]` is the first test set index, `image_range[1] - 1` is the last test set index, i.e. the upper bound is exclusive.
        total_intensity (`float`): Total intensity over all angles/rounds
        sparse (`bool`): Whether we are in a sparse setting

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

    angles = torch.from_numpy(np.linspace(0, 180, N_ANGLES, endpoint=False)).to(device)
    n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)
    if sparse:
        intensities = intensities.view(1, 1, 1).expand(n_gt, N_ANGLES, -1) / (
            N_ANGLES * n_detectors_hr
        )
        schedule = (
            torch.logspace(math.log10(10), math.log10(199), steps=10, device=device)
            .round()
            .int()
        )
    else:
        n_rounds = 1
        intensities = intensities.view(1, 1, 1, 1).expand(
            n_gt, n_rounds, N_ANGLES, -1
        ) / (N_ANGLES * n_detectors_hr * n_rounds)
        schedule = None
    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts, intensities_lr, angles, sparse)
    return gt, experiment, schedule
