import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import astra
import click
import numpy as np
import torch

from uqct.ct import (AstraParallelOp3D, fbp_single_from_forward,
                     forward_and_fbp_2d, get_astra_geometry_2d,
                     get_astra_geometry_3d, iradon_astra, linspace, poisson,
                     sinogram_ct)
from uqct.datasets.utils import get_dataset
from uqct.debugging import plot_img

N_ANGLES = 360
L = 5


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


class DenseCTScan:
    def __init__(
        self,
        image: torch.Tensor,
        exposure: float,
        T: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.image = image.to(device)
        self.exposure = exposure
        self.T = T
        self.device = device
        self.angles = linspace(0, 360, N_ANGLES).to(device)
        self.proj_geom_lr, self.vol_geom_lr = get_astra_geometry_3d(self.angles, 128, T)
        proj_geom_hr, vol_geom_hr = get_astra_geometry_3d(self.angles, 256, T)
        self.op = AstraParallelOp3D(proj_geom_hr, vol_geom_hr)
        I_0_hr = self.exposure / N_ANGLES / self.T
        scale = L / self.image.shape[-1]
        radon = self.op.forward(self.image.expand(T, -1, -1).contiguous())
        counts_hr = poisson(I_0_hr * torch.exp(-scale * radon))  # (B, 200, 256)
        self.counts = counts_hr.view(
            counts_hr.shape[0], counts_hr.shape[1], 128, 2
        ).sum(
            -1
        )  # (B, 200, 128)
        self.I_0 = I_0_hr * 2

        # Lazy
        self.sinogram = None
        self.fbp = None

    def get_counts(self, t: int) -> torch.Tensor:
        """
        Returns a (T, N_ANGLES, 128) tensor of Poisson counts.
        """
        return self.counts[:t]

    def get_sinogram(self, t_start: int, t: int) -> torch.Tensor:
        if isinstance(self.sinogram, torch.Tensor):
            return self.sinogram[t_start - 1 : t]
        self.sinogram = sinogram_ct(self.counts, self.I_0, L).clip(0)
        return self.sinogram

    def get_fbp(self, t: int) -> torch.Tensor:
        sinogram = self.get_sinogram(1, t)
        return iradon_astra(
            sinogram.transpose(1, 2), self.vol_geom_lr, self.proj_geom_lr
        ).clip(0, 1)


class SparseCTScan:
    def __init__(
        self,
        image: torch.Tensor,
        exposure: float,
        t_end: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.image = image
        self.exposure = exposure
        self.t_end = t_end
        self.device = device
        self.angles = linspace(0, 360, N_ANGLES)

        r = image.shape[-1]
        angles_np = self.angles.detach().cpu().numpy()
        proj_geom_hr, vol_geom_hr = get_astra_geometry_2d(angles_np, r)
        self.proj_geom_lr, self.vol_geom_lr = get_astra_geometry_2d(angles_np, r // 2)

        n_angles = int(proj_geom_hr["ProjectionAngles"].shape[0])
        n_det = int(proj_geom_hr["DetectorCount"])

        # --- link CPU arrays to ASTRA ---
        img_np = image[0].detach().contiguous().cpu().to(torch.float32).numpy()
        sino_np = np.empty((n_angles, n_det), dtype=np.float32)

        vol_id = astra.data2d.link("-vol", vol_geom_hr, img_np)
        sino_id = astra.data2d.link("-sino", proj_geom_hr, sino_np)

        try:
            try:
                cfg = astra.astra_dict("FP_CUDA")
            except Exception:
                cfg = astra.astra_dict("FP")
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, 1)
            astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([sino_id, vol_id])

        sino_t = torch.from_numpy(sino_np).to(device)
        self.I_0 = exposure / N_ANGLES
        scale = L / r
        counts = poisson(self.I_0 * torch.exp(-scale * sino_t))  # (N_ANGLES, 256)
        self.counts = counts.view(N_ANGLES, r // 2, 2).sum(-1)  # (N_ANGLES, 128)
        self.sinogram = None
        self.fbp = None

    def get_counts(self, t: int) -> torch.Tensor:
        return self.counts[:t]

    def get_sinogram(self, t_start: int, t: int) -> torch.Tensor:
        if isinstance(self.sinogram, torch.Tensor):
            return self.sinogram[t_start - 1 : t]
        I_0_lr = self.I_0 * 2

        # (n_angles, 128)
        self.sinogram = sinogram_ct(self.counts, I_0_lr, L).clamp_min_(0)
        return self.sinogram[t_start:t]

    def get_fbp(self, t: int) -> torch.Tensor:
        sinogram = self.get_sinogram(1, t)
        self.fbp = fbp_single_from_forward(
            vol_geom=self.vol_geom_lr,
            proj_geom=self.proj_geom_lr,
            sino_t=sinogram,
            filter_name="ramp",
            circle=True,
        ).clip(0, 1)
        return self.fbp


def get_predictions(
    scan: SparseCTScan | DenseCTScan,
    t_start: int,
    predictor_name: Literal["fbp", "mle", "map", "unet", "diffusion"],
) -> torch.Tensor:
    if predictor_name == "fbp":
        # TODO: Implement mu_0 prediction: do it
        return scan.get_fbps(t_start)
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
    set_seeds(0)  # fix order of test set images
    _, test_set = get_dataset(kwargs["dataset"], True)
    image = test_set[kwargs["test_set_idx"]]

    # Simulate measurements
    set_seeds(kwargs["seed"])
    if kwargs["sparse"]:
        scan = SparseCTScan(image, kwargs["exposure"], kwargs["t_end"], device)
    else:
        scan = DenseCTScan(image, kwargs["exposure"], kwargs["t_end"], device)

    # Obtain predictions
    get_predictions(scan, kwargs["t_start"], kwargs["t_end"])

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
    type=click.Choice(["mle", "map", "unet", "diffusion"]),
    help="which predictor to use",
)
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lamino", "lung", "composite"]),
    help="which dataset to use",
)
@click.option(
    "--test-set-idx",
    default=0,
    type=click.IntRange(min=0),
    help="test set index, i.e. which image to run the sequence for",
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
