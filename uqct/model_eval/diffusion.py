import json
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional

import click
import lovely_tensors as lt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uqct.ct import (
    Experiment,
    sample_observations,
)
from uqct.datasets.utils import get_dataset
from uqct.debugging import plot_img
from uqct.metrics import get_metrics
from uqct.models.diffusion import Diffusion, get_guidance_loss_fn
from uqct.utils import get_root_dir

DatasetName = Literal["lung", "composite", "lamino"]


@click.command()
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lung", "composite", "lamino"]),
    help="Which dataset to generate samples for",
)
@click.option(
    "--sparse",
    default=False,
    type=bool,
    help="Whether to generate samples for the sparse setting",
)
@click.option(
    "--cond",
    default=False,
    type=bool,
    help="Whether to use a conditional diffusion model",
)
@click.option("--total-intensity", default=1e7, type=float, help="Total intensity")
@click.option(
    "--guidance-lr", default=None, type=Optional[float], help="Guidance learning rate"
)
@click.option(
    "--gradient-steps",
    default=20,
    type=int,
    help="Number of guidance steps per denoising step",
)
@click.option("--n-images", default=2, type=int, help="Number of test set images")
def main(
    dataset: DatasetName,
    sparse: bool,
    cond: bool,
    total_intensity: float,
    gradient_steps: int,
    guidance_lr: float | None,
    n_images: int,
):
    lt.monkey_patch()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _, test_set = get_dataset(dataset, True)
    n_gt = min(n_images, len(test_set))
    gt = torch.stack([test_set[i] for i in range(n_gt)], dim=0).to(device)

    n_angles = 200

    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)
    n_detectors_hr = gt.shape[-1]
    intensities = torch.tensor(total_intensity, device=device)
    if sparse:
        intensities = intensities.view(1, 1, 1, 1).expand(n_gt, -1, n_angles, -1) / (
            n_angles * n_detectors_hr
        )
        schedule = torch.tensor([1, 25, 50, 75, 100, 125, 150, 175, 200])
    else:
        n_rounds = 1
        intensities = intensities.view(1, 1, 1, 1).expand(
            n_gt, n_rounds, n_angles, -1
        ) / (n_angles * n_detectors_hr * n_rounds)
        schedule = None
    counts = sample_observations(gt, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts, intensities_lr, angles, sparse)
    diffusion = Diffusion(
        dataset,
        batch_size=32,
        lr=guidance_lr,
        gradient_steps=gradient_steps,
        cond=cond,
        onnx=True,
        verbose=True,
    )

    guidance_loss_fn = get_guidance_loss_fn(experiment, schedule)
    sample = diffusion.sample(experiment, 1, schedule, guidance_loss_fn)
    gt_lr = F.interpolate(gt, (128, 128), mode="area")
    sample = sample.reshape(-1, 128, 128)
    plot_img(*gt_lr, *sample.reshape(-1, 128, 128), name="diffusion", share_range=True)
    gt_lr_rep = gt_lr.repeat_interleave(
        len(schedule) if schedule is not None else n_rounds,  # type: ignore
        0,
    )
    plot_img(*gt_lr, *gt_lr_rep.reshape(-1, 128, 128), name="gt_lr", share_range=True)
    metrics_list = [get_metrics(gt, x) for gt, x in zip(gt_lr_rep, sample)]
    summary = defaultdict(lambda: 0.0)

    for metrics in metrics_list:
        for k, v in metrics.items():
            summary[k] += v
    for k in summary.keys():
        summary[k] = float(summary[k]) / len(metrics_list)
    summary = dict(summary)
    # 1. Convert the list of metrics (one dict per image) to a DataFrame
    df = pd.DataFrame(metrics_list)

    # 2. Inject the configuration parameters as columns
    # This allows you to "group by" these columns later
    df["dataset"] = dataset
    df["sparse"] = sparse
    df["cond"] = cond
    df["total_intensity"] = total_intensity
    df["gradient_steps"] = gradient_steps
    df["guidance_lr"] = guidance_lr
    df["n_images"] = n_images

    # 3. Add metadata (Run ID and Timestamp)
    # This distinguishes this specific run from an identical run done later
    run_id = str(uuid.uuid4())[:8]
    df["run_id"] = run_id
    df["timestamp"] = datetime.now()

    # 4. Save as Parquet
    # We create a descriptive filename, but the UUID ensures no collisions
    output_dir = get_root_dir() / "results" / "eval_diffusion"
    output_dir.mkdir(exist_ok=True, parents=True)

    filename = f"{dataset}_{total_intensity:.0e}_steps{gradient_steps}_{run_id}.parquet"
    save_path = output_dir / filename

    df.to_parquet(save_path, index=False)

    print(f"Run saved to: {save_path}")
    print(json.dumps(summary))  # Keep this for quick console checking


if __name__ == "__main__":
    main()
