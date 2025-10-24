import inspect
import json
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any, Literal

import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from uqct.ct import Experiment, fbp, sample_observations, sinogram_from_counts
from uqct.datasets.utils import DatasetName, get_dataset
from uqct.models.diffusion import Diffusion, get_guidance_loss_fn
from uqct.models.unet import FBPUNetEnsemble
from uqct.training.unet import N_BINS_HR
from uqct.utils import get_cache_dir

N_ROUNDS = 200
N_ANGLES = 200
ANGULAR_RANGE = 180
L = 5


def _linspace_exclusive(
    start: float,
    end: float,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if steps <= 0:
        return torch.empty(0, device=device)
    values = np.linspace(start, end, steps, endpoint=False, dtype=np.float32)
    return torch.from_numpy(values).to(device=device)


def get_base_dir():
    if Path("/cluster").exists():
        return Path("/cluster/scratch/mgaetzner/uqct/")
    else:
        return Path("./")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_avg_image(train_set: Subset[torch.Tensor], dataset_name: str) -> torch.Tensor:
    if not dataset_name:
        raise ValueError("Dataset name must be provided for average image caching.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cache_dir = get_cache_dir()
    cache_path = cache_dir / "avg_image_cache.h5"
    key = str(dataset_name)

    if cache_path.exists():
        with h5py.File(cache_path, "r") as cache_file:
            if key in cache_file:
                cached = torch.from_numpy(cache_file[key][...])
                return cached.to(device=device)
    print("Computing average image...")

    if (num_cpus := cpu_count()) is None:
        num_cpus = 1
    data_loader = DataLoader(
        train_set, batch_size=64, prefetch_factor=1, num_workers=min(1, num_cpus)
    )
    sample = train_set[0]
    out = torch.zeros_like(sample, device=device)
    for batch in tqdm(data_loader, desc="avg image"):
        out += batch.to(device).sum(0)
    avg_img = out / len(train_set)

    avg_cpu = avg_img.detach().cpu().to(torch.float32).numpy()
    with h5py.File(cache_path, "a") as cache_file:
        if key in cache_file:
            del cache_file[key]
        cache_file.create_dataset(key, data=avg_cpu)
    return avg_img


def get_fbps(experiment: Experiment, **_) -> torch.Tensor:
    if experiment.sparse:
        sino = sinogram_from_counts(experiment.counts, experiment.intensities)
        fbps = list()
        for i in range(1, len(experiment.angles)):
            fbps.append(fbp(sino[..., :i, :], experiment.angles[:i]))
        fbps = torch.stack(fbps)
    else:
        sino = sinogram_from_counts(
            experiment.counts.cumsum(-3), experiment.intensities.cumsum(-2)
        )
        fbps = fbp(
            sino,
            experiment.angles,
        )
    return fbps


def get_unet_preds(
    experiment: Experiment, dataset: DatasetName, mixture: bool
) -> torch.Tensor:
    model = FBPUNetEnsemble(dataset, experiment.sparse)
    if mixture:
        preds = model.predict(experiment)
    else:
        preds = model.predict(experiment, aggregate="mean")
    return preds


def get_diffusion_preds(
    experiment: Experiment, dataset: DatasetName, **_
) -> torch.Tensor:
    model = Diffusion(dataset)
    preds = model.sample(
        experiment, replicates=10, guidance_loss_fn=get_guidance_loss_fn(experiment)
    )
    return preds


def get_mle(experiment: Experiment, **_) -> torch.Tensor: ...
def get_map(experiment: Experiment, **_) -> torch.Tensor: ...


predictors = dict(
    fbp=get_fbps,
    mle=get_mle,
    map=get_map,
    unet=get_unet_preds,
    diffusion=get_diffusion_preds,
)


def get_predictions(
    experiment: Experiment,
    avg_image: torch.Tensor,
    t_start: int,
    predictor_name: Literal["fbp", "mle", "map", "unet", "diffusion"],
    mixture: bool,
    dataset: DatasetName,
    device: torch.device,
) -> torch.Tensor:
    r"""
    Arguments:
        `experiment` (`Experiment`): Self-explanatory.
        `avg_image` (`torch.Tensor`): Average training set image with shape `(1, 128, 128)`
        `t_start` (`int`): Confidence sequence start time. Implies that we want $S = (C_{t_start}, \dots, C_{200})$
        `predictor_name` (`str`): Defines which predictor for $\theta$ we use, choose one of `'fbp'`, `'mle'`, `'map'`, `'unet'`, `'diffusion'`.
        `mixture` (`bool`): Whether 10 or 1 prediction should be returned. If the predictor yields only one prediction (FBPs, MLEs and MAPs), the prediction gets repeated.
        `dataset` (`DatasetName`): Which dataset we trained the predictor on.
        `device` (`torch.device`): Self-explanatory.
    Returns:
        `preds` (`torch.Tensor`): If `mixture == True` the shape is `(B, n_angles, 1, 128, 128)` (sparse), or `(B, n_rounds, 1, 128, 128)` (dense). Otherwise, the shape is `(B, n_angles, 10, 1, 128, 128)` (sparse) or `B, n_rounds, 10, 1, 128, 128)` (dense).
    """
    assert t_start >= 1
    params = {
        "experiment": experiment,
        "avg_image": avg_image,
        "t_start": t_start,
        "predictor_name": predictor_name,
        "mixture": mixture,
        "dataset": dataset,
        "device": device,
    }
    pred_fn = predictors[predictor_name]
    preds = pred_fn(**params)
    if t_start == 1:
        avg_image.view((preds.ndim - avg_image.ndim) * (1,), *avg_image.shape)
        if preds.ndim == 5:
            avg_image.expand(preds.shape[0], -1, -1, -1, -1)
        else:
            avg_image.expand(preds.shape[0], -1, preds.shape[2], -1, -1, -1)
        torch.cat([avg_image, preds])
    return preds


def run_cseq(kwargs: dict[str, Any]) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args_str = "\n".join(f"\t{k}: {v}" for k, v in kwargs.items())
    print(f"Arguments:\n{args_str}")

    base_dir = get_base_dir()
    cseq_dir = base_dir / "cseqs" / "slm" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cseq_dir.mkdir(exist_ok=True, parents=True)

    # Saving metadata about confidence sequence
    json.dump(kwargs, (cseq_dir / "metadata.json").open("w"))
    print(f"Saved arguments to '{cseq_dir}'")

    # Retrieve image
    train_set, test_set = get_dataset(kwargs["dataset"], True)

    # Retrieve average image if necessary
    avg_image = torch.zeros_like(train_set[0])
    if kwargs["t_start"] == 1:
        avg_image = get_avg_image(train_set, kwargs["dataset"])

    start_idx, end_idx = kwargs["test_set_range"]
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid test set range: {(start_idx, end_idx)}")
    indices = list(range(start_idx, min(end_idx, len(test_set))))
    if not indices:
        raise ValueError(
            f"Requested empty slice from test set with indices {(start_idx, end_idx)}."
        )
    images = torch.stack([test_set[i] for i in indices], dim=0).to(device)

    # Simulate measurements
    set_seeds(kwargs["seed"])
    images = images.to(dtype=torch.float32)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    elif images.ndim != 3:
        raise ValueError(
            f"Expected batched images with shape (N, 1, H, W) or (N, H, W), got {tuple(images.shape)}."
        )
    if images.shape[-1] % 2 != 0:
        raise ValueError(
            "Images must have even side length to downsample detector bins."
        )

    total_intensity = float(kwargs["total_intensity"])
    sparse = bool(kwargs["sparse"])

    angles = _linspace_exclusive(0, ANGULAR_RANGE, N_ANGLES, device=device)
    if kwargs["sparse"]:
        intensities = torch.tensor(
            total_intensity / N_ANGLES / N_BINS_HR, device=device
        )
        intensities = intensities.view(3 * (1,)).expand(-1, len(angles), -1)
    else:
        intensities = torch.tensor(
            total_intensity / N_ROUNDS / N_ANGLES / N_BINS_HR, device=device
        )
        intensities = intensities.view(4 * (1,)).expand(-1, -1, len(angles), -1)
    counts_lr = sample_observations(images, intensities, angles)
    intensities_lr = intensities * 2
    experiment = Experiment(counts_lr, intensities_lr, angles, sparse)
    print(experiment)

    # Obtain predictions
    print("Obtaining predictions...")
    preds = get_predictions(
        experiment,
        avg_image,
        kwargs["t_start"],
        kwargs["predictor"],
        kwargs["mixture"],
        dataset=kwargs["dataset"],
        device=device,
    )
    breakpoint()

    # Save predictions with metrics

    # Compute confidence sequence

    # Save confidence coefficients, NLL of true image


@click.command()
@click.option(
    "--sparse",
    default=False,
    is_flag=True,
    help="sequence of linearly increasing angles or number number of (dense) scanning rounds",
)
@click.option(
    "--predictor",
    default="mle",
    type=click.Choice(["fbp", "mle", "map", "unet", "diffusion"]),
    help="which predictor to use",
)
@click.option(
    "--mixture",
    default=False,
    type=bool,
    help="Whether to mix over 10 dirac deltas.",
)
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lamino", "lung", "composite"]),
    help="which dataset to use",
)
@click.option(
    "--test-set-range",
    default=(0, 1),
    type=(int, int),
    help="test set range (zero-indexed, exclusive), i.e. which images to run the sequence for",
)
@click.option(
    "--total-intensity",
    default=1e9,
    type=click.FloatRange(min=1e4),
    help="TOTAL exposure time after T angles or rounds",
)
@click.option(
    "--t_start",
    default=1,
    type=click.IntRange(min=1),
    help="after how many time steps to start the sequence",
)
@click.option(
    "--seed",
    default=0,
    type=click.IntRange(min=0),
    help="random seed",
)
def main(**kwargs):
    run_cseq(kwargs)


if __name__ == "__main__":
    try:
        import lovely_tensors as lt

        lt.monkey_patch()
    except Exception as _:
        pass
    main()
