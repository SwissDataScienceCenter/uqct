from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Self
import pickle
import tempfile


import numpy as np
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from tqdm.auto import tqdm

from uqct.ct import (
    Experiment,
    circular_mask,
    sample_observations,
    prepare_inputs_from_experiment,
)
from uqct.datasets.utils import DatasetName, get_dataset
from uqct.training.unet import (
    MAX_TOTAL_INTENSITY,
    MIN_TOTAL_INTENSITY,
    build_unet,
    sample_intensities,
)
from uqct.utils import get_checkpoint_dir


class FBPUNetEnsemble:
    def __init__(
        self,
        dataset: DatasetName,
        sparse: bool,
        *,
        batch_size: int = 64,
        num_workers: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.dataset = dataset
        self.sparse = sparse
        self.batch_size = batch_size
        self.num_workers = num_workers

        def _load_unet(member: int) -> FBPUNet:
            return FBPUNet(
                dataset,
                member,
                sparse,
                batch_size=batch_size,
                num_workers=num_workers,
                model_device=device,
            )

        with ThreadPoolExecutor() as executor:
            self.unets = tuple(executor.map(_load_unet, list(range(10))))

        if self.unets:
            first_param = next(self.unets[0].unet.parameters(), None)
            self._device = (
                first_param.device if first_param is not None else torch.device("cpu")
            )
        else:
            self._device = torch.device("cpu")

    def reinit(self, dataset: DatasetName, sparse: bool) -> Self:
        self.__init__(
            dataset, sparse, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return self

    def to_disk(self) -> None:
        self.pickle_path = tempfile.NamedTemporaryFile(delete=False).name
        pickle.dump(self.unets, open(self.pickle_path, "wb"))
        del self.unets

    def to(self, device: torch.device | str) -> FBPUNetEnsemble:
        target = torch.device(device)
        for unet in self.unets:
            unet.unet.to(target)
        self._device = target
        return self

    def predict(
        self,
        experiment: Experiment,
        schedule: torch.Tensor | None = None,
        *,
        out_device: torch.device | None = None,
        aggregate: Literal["none", "mean", "median"] = "mean",
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate ensemble predictions for an `Experiment`.

        Parameters
        ----------
        `experiment` : `Experiment`
            Experiment providing the measurement tensors required for inference.
        `out_device` : `torch.device | None`
            Target device for the output tensor. Defaults to each model's configured output device.
        `aggregate` : `Literal["none", "mean", "median"]`
            Aggregation strategy applied across ensemble members.
        `verbose` : `bool`
            Whether to show a progress bar.

        Returns
        -------
        `torch.Tensor`
            Tensor of shape `(..., rounds, M, 1, H, W)` (dense) or `(..., M, 1, H, W)` (sparse) when `aggregate="none"` with `M` ensemble members, otherwise `(..., rounds, 1, H, W)` (dense) or `(..., 1, H, W)` (sparse) after aggregation, all values clipped to `[0, 1]`.
        """
        fbp_lr, intensity_lr, class_labels = prepare_inputs_from_experiment(
            experiment, schedule
        )
        preds = []
        pbar = tqdm(self.unets) if verbose else self.unets
        maybe_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for unet in pbar:
            prev_device = unet.unet.device
            unet.to(maybe_cuda)
            preds.append(
                unet._predict_from_tensors(
                    fbp_lr, intensity_lr, class_labels, out_device=out_device
                )
            )
            unet.to(prev_device)

        stacked = torch.stack(preds, dim=-4)
        if stacked.dtype == torch.float16 and stacked.device.type == "cpu":
            stacked = stacked.to(torch.float32)
        if aggregate == "none":
            return stacked
        if aggregate == "median":
            return stacked.median(dim=-4).values
        if aggregate == "mean":
            return stacked.mean(dim=-4)
        raise ValueError(f"Unknown aggregate mode: {aggregate}")


class FBPUNet:
    def __init__(
        self,
        dataset: DatasetName,
        member: int,
        sparse: bool,
        *,
        batch_size: int = 64,
        num_workers: int = 4,
        model_device: torch.device = torch.device("cpu"),
        out_device: torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        label = "sparse" if sparse else "dense"
        ckpt_dir = get_checkpoint_dir() / f"unet_{label}"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory '{ckpt_dir}' does not exist."
            )
        prefix = f"unet_{label}_128_{dataset}_{member}"
        ckpt_path = ckpt_dir / f"{prefix}.pt"

        unet = build_unet(sparse).to(model_device)  # type: ignore
        load_unet_ckpt(unet, ckpt_path, verbose)

        self.unet = unet.eval()  # inference only
        self.sparse = sparse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_device = out_device

    def to(self, device: torch.device, dtype: torch.dtype = torch.float16) -> "FBPUNet":
        self.unet.to(device, dtype=dtype)  # type: ignore
        return self

    def _predict_from_tensors(
        self,
        fbp_lr: torch.Tensor,
        total_intensity: torch.Tensor,
        class_labels: torch.Tensor | None,
        *,
        out_device: torch.device | None,
    ) -> torch.Tensor:
        """
        Run inference on batched tensors produced by `prepare_inputs_from_experiment`.

        Parameters
        ----------
        fbp_lr : torch.Tensor
            Tensor of shape `(..., 1, H, W)` or `(..., H, W)` containing filtered backprojection inputs in `[0, 1]`.
        total_intensity : torch.Tensor
            Tensor of shape `(..., 1)` containing total intensity per sample.
        class_labels : torch.Tensor | None
            Tensor of shape `(...,)` with integer class labels when the model is sparse, otherwise `None`.
        out_device : torch.device | None
            Target device for the output tensor. Defaults to the device specified at init when `None`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(N, 1, H, W)` containing predictions in `[0, 1]`.
        """
        device = next(self.unet.parameters()).device

        out_shape = fbp_lr.shape
        batch_dims = fbp_lr.shape[:-2]
        batch_prod = math.prod(batch_dims)

        fbp_lr = fbp_lr.view(batch_prod, 1, *fbp_lr.shape[-2:])
        total_intensity = total_intensity.flatten()
        if class_labels is not None:
            class_labels = class_labels.flatten()

        out_device = out_device if out_device is not None else self.out_device
        preds = []
        for i in range(0, len(fbp_lr), self.batch_size):
            fbp_b = fbp_lr[i : i + self.batch_size]
            intensity_b = total_intensity[i : i + self.batch_size]
            cls_b = None
            if class_labels is not None:
                cls_b = class_labels[i : i + self.batch_size]
            x = fbp_b.to(device, non_blocking=True) * 2.0 - 1.0
            intensity_norm = (
                (intensity_b.to(device, non_blocking=True) - MIN_TOTAL_INTENSITY)
                / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY)
                * 999
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    y = self.unet(
                        x,
                        timestep=intensity_norm.flatten(),
                        class_labels=(cls_b.to(device) if cls_b is not None else None),
                        return_dict=False,
                    )[0]

            pred = ((y + 1.0) / 2.0).clamp(0.0, 1.0)
            if out_device is not None:
                pred = pred.to(out_device)
            preds.append(pred)

        out = torch.cat(preds, dim=0).view(out_shape)
        out.mul_(circular_mask(out_shape[-1]).to(out.device))
        return out

    def predict(
        self,
        experiment: Experiment,
        schedule: torch.Tensor | None,
        *,
        out_device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Run inference on an `Experiment`, returning normalized reconstructions.

        Parameters
        ----------
        experiment : Experiment
            Experiment containing counts, intensities, and angles used to build model inputs.
        schedule :  torch.Tensor | None
            1D tensor containing the number of angles or rounds for which to generate predictions.
        out_device : torch.device | None
            Target device for the output tensor. Defaults to the device specified at init when `None`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(..., 1, H, W)` containing predictions in `[0, 1]`.
        """
        fbp_lr, intensity_lr, class_labels = prepare_inputs_from_experiment(
            experiment, schedule
        )
        return self._predict_from_tensors(
            fbp_lr,
            intensity_lr,
            class_labels,
            out_device=out_device,
        ).float()


def load_unet_ckpt(unet: UNet2DModel, ckpt_path: Path, verbose: bool = False) -> None:
    with ckpt_path.open("rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
        sd = ckpt["unet"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        unet.load_state_dict(sd, strict=True)
        if verbose:
            print(
                f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}"
            )


if __name__ == "__main__":
    from uqct.debugging import plot_img

    try:
        import lovely_tensors as lt

        lt.monkey_patch()
    except Exception as _:
        pass

    def build_dense_experiment(
        images: torch.Tensor, angles: torch.Tensor, n_rounds: int = 10
    ) -> Experiment:
        samples_cpu = images.to("cpu")
        angles_cpu = angles.to("cpu")
        batch = samples_cpu.shape[0]
        intensities = sample_intensities(
            batch,
            MIN_TOTAL_INTENSITY * n_rounds,
            MAX_TOTAL_INTENSITY,
            device=samples_cpu.device,
        )
        intensities /= n_rounds * len(angles) * images.shape[-1]
        intensities = intensities.view(-1, 1, 1, 1).expand(
            -1, n_rounds, len(angles), -1
        )
        counts = sample_observations(samples_cpu, intensities, angles_cpu)
        intensities_lr = intensities * 2
        return Experiment(counts, intensities_lr, angles_cpu, sparse=False)

    def build_sparse_experiment(
        images: torch.Tensor,
        angles: torch.Tensor,
        *,
        seed: int | None = 0,
    ) -> Experiment:
        samples_cpu = images.to("cpu")
        angles_cpu = angles.to("cpu")
        batch = samples_cpu.shape[0]

        if seed is not None:
            prev_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        else:
            prev_state = None

        try:
            intensities = sample_intensities(
                batch,
                MIN_TOTAL_INTENSITY * len(angles),
                MAX_TOTAL_INTENSITY,
                device=samples_cpu.device,
            ) / (len(angles) * images.shape[-1])
            intensities = intensities.reshape(-1, 1, 1, 1).expand(
                -1, 1, len(angles_cpu), 1
            )
            counts = sample_observations(samples_cpu, intensities, angles_cpu)
            intensities_lr = intensities * 2
        finally:
            if prev_state is not None:
                torch.random.set_rng_state(prev_state)

        return Experiment(counts, intensities_lr, angles_cpu, sparse=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_label = "dense"
    dataset = "lamino"

    _, test_set = get_dataset(dataset, True)
    num_examples = min(2, len(test_set))

    gt = torch.stack([test_set[i] for i in range(num_examples)], dim=0).to(device)
    n_angles = 200
    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False))

    ckpt_dir = get_checkpoint_dir()
    ckpt_path = (
        ckpt_dir / f"unet_{model_label}" / f"unet_{model_label}_128_{dataset}_0.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for dataset='{dataset}' (model='{model_label}'): {ckpt_path}"
        )

    if model_label == "sparse":
        exp = build_sparse_experiment(gt, angles)
    else:
        exp = build_dense_experiment(gt, angles)
    exp.to(device)
    gt_lr = torch.nn.functional.interpolate(gt, (128, 128), mode="area")

    model = FBPUNetEnsemble(dataset, model_label == "sparse", batch_size=16)
    fbps, _, _ = prepare_inputs_from_experiment(exp)
    preds = model.predict(exp, aggregate="mean", verbose=True)
    print(preds)

    if model_label == "sparse":
        time_points = (50, 100, 150, 199)
        preds = preds.view(num_examples, n_angles, 128, 128)
        fbps = fbps.view(num_examples, n_angles, 128, 128)
        preds = preds[:, time_points]
        fbps = fbps[:, time_points]
    else:
        rounds = (0, 3, 6, 9)
        preds = preds[:, rounds]
        fbps = fbps[:, rounds]
    plot_img(
        *gt,
        *preds.view(-1, 128, 128),
        *fbps.view(-1, 128, 128),
        name=f"unet_{model_label}",
    )

    # # TODO: Remove
    model = FBPUNet(dataset, 0, model_label == "sparse")
    assert ckpt_path.exists()
    # ckpt_path = Path(
    #     f"runs/unet_dense/2025-10-23_20-24_lamino_48_500_3e-05_0.37_0.0043_{i}/ckpts/best.pt"
    # )
    # ckpt_path = Path(
    #     f""
    # )
    # load_unet_ckpt(model.unet, ckpt_path)
    #
    # fbps, _, _ = prepare_inputs_from_experiment(exp)
    # fbps = fbps.view(-1, 128, 128).to(device)
    # preds = model.predict(exp)
    # preds = preds.view(-1, 128, 128).to(device)
    #
    # l1_pred = torch.nn.functional.l1_loss(preds, gt_lr)
    # l1_fbp = torch.nn.functional.l1_loss(fbps, gt_lr)
    # print(f"Performance: {l1_pred=}, {l1_fbp=}")
    # plot_img(*fbps, *preds, *gt)
