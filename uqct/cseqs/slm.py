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
from uqct.models.diffusion import Diffusion
from uqct.models.unet import FBPUNetEnsemble
from uqct.utils import get_cache_dir

N_ROUNDS = 200
N_ANGLES = 200
ANGULAR_RANGE = 180
L = 5


def _linspace_exclusive(
    start: float,
    end: float,
    steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    values = np.linspace(start, end, steps, endpoint=False, dtype=np.float32)
    return torch.from_numpy(values).to(device=device, dtype=dtype)


def _expand_to_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    orig_shape = tensor.shape
    out = tensor
    while out.ndim < len(shape):
        out = out.unsqueeze(0)
    if out.ndim > len(shape):
        extra_dims = out.ndim - len(shape)
        if not all(dim == 1 for dim in out.shape[:extra_dims]):
            raise ValueError(
                f"Cannot broadcast tensor of shape {orig_shape} to target shape {shape}."
            )
        out = out.view(*out.shape[extra_dims:])
    expand_sizes = []
    for current, target in zip(out.shape, shape):
        if current == target:
            expand_sizes.append(current)
        elif current == 1:
            expand_sizes.append(target)
        else:
            raise ValueError(
                f"Cannot broadcast tensor of shape {orig_shape} to target shape {shape}."
            )
    return out.expand(*expand_sizes)


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


# class DenseCTScan:
#     def __init__(
#         self,
#         image: torch.Tensor,
#         exposure: float,
#         T: int,
#         device: torch.device = torch.device("cpu"),
#     ):
#         self.image = image.to(device)
#         self.exposure = exposure
#         self.T = T
#         self.device = device
#         self.angles = torch.from_numpy(
#             np.linspace(0, ANGULAR_RANGE, N_ANGLES, endpoint=False)
#         ).to(device)
#         self.proj_geom_lr, self.vol_geom_lr = get_astra_geometry_3d(self.angles, 128, 1)
#         proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(self.angles, 256, T)
#         self.op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)
#         I_0_hr = self.exposure / N_ANGLES / self.T
#         scale = L / self.image.shape[-1]
#         radon = self.op.forward(self.image.expand(T, -1, -1).contiguous())
#         counts_hr = poisson(I_0_hr * torch.exp(-scale * radon))  # (N_ROUNDS, 200, 256)

#         # (N_ROUNDS, 200, 128)
#         self.counts = counts_hr.view(
#             counts_hr.shape[0], counts_hr.shape[1], 128, 2
#         ).sum(-1)
#         self.I_0 = I_0_hr * 2

#         # Lazy
#         self.sinogram = None
#         self.fbp = None

#     def get_counts(self, t: int) -> torch.Tensor:
#         """
#         Returns a (T, N_ANGLES, 128) tensor of Poisson counts.
#         """
#         return self.counts[:t]

#     def get_sinogram(self, t_start: int, t: int) -> torch.Tensor:
#         return sinogram(
#             self.counts[t_start - 1 : t].sum(0, keepdim=True),
#             self.I_0 * (t - t_start + 1),
#             L,
#         ).clip(0)

#     def get_fbp(self, t: int) -> torch.Tensor:
#         if t <= 0:
#             raise ValueError(f"Expected t >= 1, got t = {t}")
#         sinogram = self.get_sinogram(1, t)
#         return iradon_astra(
#             sinogram.transpose(1, 2), self.vol_geom_lr, self.proj_geom_lr
#         ).clip(0, 1)

#     def get_fbps(self, t_start: int) -> torch.Tensor:
#         fbps = list()
#         for t in range(t_start - 1, N_ROUNDS - 1):
#             fbps.append(self.get_fbp(t))
#         return torch.stack(fbps)


# class SparseCTScan:
#     def __init__(
#         self,
#         image: torch.Tensor,
#         exposure: float,
#         t_end: int,
#         device: torch.device = torch.device("cpu"),
#     ):
#         self.image = image
#         self.exposure = exposure
#         self.t_end = t_end
#         self.device = device
#         self.angles = linspace(0, ANGULAR_RANGE, N_ANGLES)

#         r = image.shape[-1]
#         self.angles_np = self.angles.detach().cpu().numpy()
#         proj_geom_hr, vol_geom_hr = get_astra_geometry_2d(self.angles_np, r)

#         n_angles = int(proj_geom_hr["ProjectionAngles"].shape[0])
#         n_det = int(proj_geom_hr["DetectorCount"])

#         # --- link CPU arrays to ASTRA ---
#         img_np = image[0].detach().contiguous().cpu().to(torch.float32).numpy()
#         sino_np = np.empty((n_angles, n_det), dtype=np.float32)

#         vol_id = astra.data2d.link("-vol", vol_geom_hr, img_np)
#         sino_id = astra.data2d.link("-sino", proj_geom_hr, sino_np)

#         try:
#             try:
#                 cfg = astra.astra_dict("FP_CUDA")
#             except Exception:
#                 cfg = astra.astra_dict("FP")
#             cfg["VolumeDataId"] = vol_id
#             cfg["ProjectionDataId"] = sino_id
#             alg_id = astra.algorithm.create(cfg)
#             astra.algorithm.run(alg_id, 1)
#             astra.algorithm.delete(alg_id)
#         finally:
#             astra.data2d.delete([sino_id, vol_id])

#         sino_t = torch.from_numpy(sino_np).to(device)
#         self.I_0 = exposure / N_ANGLES
#         scale = L / r
#         counts = poisson(self.I_0 * torch.exp(-scale * sino_t))  # (N_ANGLES, 256)
#         self.counts = counts.view(N_ANGLES, r // 2, 2).sum(-1)  # (N_ANGLES, 128)
#         self.sinogram = None
#         self.fbp = None

#     def get_counts(self, t: int) -> torch.Tensor:
#         return self.counts[:t]

#     def get_sinogram(self, t_start: int, t: int) -> torch.Tensor:
#         if isinstance(self.sinogram, torch.Tensor):
#             return self.sinogram[t_start - 1 : t]
#         I_0_lr = self.I_0 * 2

#         # (n_angles, 128)
#         self.sinogram = sinogram(self.counts, I_0_lr, L).clamp_min_(0)
#         return self.sinogram[t_start - 1 : t]

#     def get_fbp(self, t: int) -> torch.Tensor:
#         sinogram = self.get_sinogram(1, t)
#         proj_geom_lr, vol_geom_lr = get_astra_geometry_2d(
#             self.angles_np[:t], sinogram.shape[-1]
#         )
#         self.fbp = fbp_single_from_forward(
#             vol_geom=vol_geom_lr,
#             proj_geom=proj_geom_lr,
#             sino_t=sinogram,
#             filter_name="ramp",
#             circle=True,
#         ).clip(0, 1)
#         return self.fbp

#     def get_fbps(self, t_start) -> torch.Tensor:
#         fbps = list()
#         for t in range(t_start, N_ANGLES + 1):
#             fbps.append(self.get_fbp(t))
#         return torch.stack(fbps)


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


def get_predictions(
    experiment: Experiment,
    avg_img: torch.Tensor,
    t_start: int,
    predictor_name: Literal["fbp", "mle", "map", "unet", "diffusion"],
    *,
    dataset: DatasetName | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    assert t_start >= 1

    def _experiment_to_device(
        exp: Experiment, target: torch.device | None
    ) -> Experiment:
        if target is None:
            return exp
        if (
            exp.counts.device == target
            and exp.intensities.device == target
            and exp.angles.device == target
        ):
            return exp
        return Experiment(
            exp.counts.to(target),
            exp.intensities.to(target),
            exp.angles.to(target),
            exp.sparse,
        )

    if predictor_name == "fbp":
        start_step = max(t_start, 1)
        fbps: list[torch.Tensor] = []

        if experiment.sparse:
            total_steps = experiment.counts.shape[-2]
            if total_steps < start_step:
                raise ValueError(
                    f"Requested t_start={t_start} but only {total_steps} angles available."
                )
            for step in range(start_step, total_steps + 1):
                counts_prefix = experiment.counts[..., :step, :]
                intensities_prefix = experiment.intensities[..., :step, :]
                sino = sinogram_from_counts(counts_prefix, intensities_prefix, L)
                sino.clamp_min_(0.0)
                recon = fbp(sino, experiment.angles[:step]).clamp_(0.0, 1.0)
                fbps.append(recon)
        else:
            total_steps = experiment.counts.shape[-3]
            if total_steps < start_step:
                raise ValueError(
                    f"Requested t_start={t_start} but only {total_steps} rounds available."
                )
            counts = experiment.counts
            intensities = experiment.intensities
            angles = experiment.angles
            for step in range(start_step, total_steps + 1):
                counts_cum = counts[..., :step, :, :].sum(dim=-3)
                intensities_cum = intensities[..., :step, :, :].sum(dim=-3)
                sino = sinogram_from_counts(counts_cum, intensities_cum, L)
                sino.clamp_min_(0.0)
                recon = fbp(sino, angles).clamp_(0.0, 1.0)
                fbps.append(recon)

        fbp_stack = torch.stack(fbps, dim=0)
        if t_start <= 1:
            target_shape = fbp_stack.shape[1:]
            avg = avg_img.to(device=fbp_stack.device, dtype=fbp_stack.dtype)
            avg = _expand_to_shape(avg, target_shape)
            fbp_stack = torch.cat([avg.unsqueeze(0), fbp_stack], dim=0)
        return fbp_stack
    if predictor_name == "unet":
        if dataset is None:
            raise ValueError("Dataset must be provided to run the UNet predictor.")
        ensemble = FBPUNetEnsemble(dataset, experiment.sparse, num_workers=0)
        if device is not None:
            ensemble.to(device)
        with torch.inference_mode():
            preds = ensemble.predict(experiment, aggregate="mean")
        return preds
    if predictor_name == "diffusion":
        if dataset is None:
            raise ValueError("Dataset must be provided to run the diffusion predictor.")
        diff_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        experiment_diff = _experiment_to_device(experiment, diff_device)
        diffusion = Diffusion(dataset, experiment_diff)
        with torch.inference_mode():
            samples = diffusion.sample(1)
        return samples.squeeze(0)
    else:
        raise NotImplementedError(
            f"Getting predictions for predictor '{predictor_name}' is not implemented yet."
        )


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
    avg_img = torch.zeros_like(train_set[0])
    if kwargs["t_start"] == 1:
        print("Computing average image...")
        avg_img = get_avg_image(train_set, kwargs["dataset"])

    start_idx, end_idx = kwargs["test_set_range"]
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid test set range: {(start_idx, end_idx)}")
    indices = list(range(start_idx, min(end_idx, len(test_set))))
    if not indices:
        raise ValueError(
            f"Requested empty slice from test set with indices {(start_idx, end_idx)}."
        )
    image = torch.stack([test_set[i] for i in indices], dim=0)

    # Simulate measurements
    set_seeds(kwargs["seed"])
    image = image.to(dtype=torch.float32)
    if image.ndim == 4 and image.shape[1] == 1:
        image = image[:, 0]
    elif image.ndim != 3:
        raise ValueError(
            f"Expected batched images with shape (N, 1, H, W) or (N, H, W), got {tuple(image.shape)}."
        )
    if image.shape[-1] % 2 != 0:
        raise ValueError(
            "Images must have even side length to downsample detector bins."
        )

    measurement_device = image.device
    batch_size = image.shape[0]
    total_exposure = float(kwargs["exposure"])
    t_end = int(kwargs["t_end"])
    sparse = bool(kwargs["sparse"])
    n_detectors_lr = image.shape[-1] // 2

    if sparse:
        n_angles = max(t_end, 0)
        angles = _linspace_exclusive(
            0.0,
            float(ANGULAR_RANGE),
            n_angles,
            device=measurement_device,
            dtype=image.dtype,
        )
        if n_angles > 0:
            intensities_hr = torch.full(
                (batch_size, n_angles, 1),
                total_exposure / n_angles,
                device=measurement_device,
                dtype=image.dtype,
            )
            counts_lr = sample_observations(image, intensities_hr, angles)
            intensities_lr = intensities_hr * 2.0
        else:
            intensities_lr = torch.zeros(
                batch_size, 0, 1, device=measurement_device, dtype=image.dtype
            )
            counts_lr = torch.zeros(
                batch_size,
                0,
                n_detectors_lr,
                device=measurement_device,
                dtype=image.dtype,
            )
    else:
        n_angles = N_ANGLES
        angles = _linspace_exclusive(
            0.0,
            float(ANGULAR_RANGE),
            n_angles,
            device=measurement_device,
            dtype=image.dtype,
        )
        T = max(t_end, 0)
        if T > 0:
            intensities_hr = torch.full(
                (batch_size, T, n_angles, 1),
                total_exposure / (n_angles * T),
                device=measurement_device,
                dtype=image.dtype,
            )
            images_rep = image.unsqueeze(1).expand(-1, T, -1, -1).contiguous()
            counts_lr = sample_observations(images_rep, intensities_hr, angles)
            intensities_lr = intensities_hr * 2.0
        else:
            intensities_lr = torch.zeros(
                batch_size,
                0,
                n_angles,
                1,
                device=measurement_device,
                dtype=image.dtype,
            )
            counts_lr = torch.zeros(
                batch_size,
                0,
                n_angles,
                n_detectors_lr,
                device=measurement_device,
                dtype=image.dtype,
            )
    experiment = Experiment(counts_lr, intensities_lr, angles, sparse)

    # Obtain predictions
    print("Obtaining predictions...")
    preds = get_predictions(
        experiment,
        avg_img,
        kwargs["t_start"],
        kwargs["predictor"],
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
    "--exposure",
    default=1e9,
    type=click.FloatRange(min=1e4),
    help="TOTAL exposure time after T angles or rounds",
)
@click.option(
    "--t_end",
    default=200,
    type=click.IntRange(min=0),
    help="number of angles (sparse) or rounds (dense)",
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
    main()
