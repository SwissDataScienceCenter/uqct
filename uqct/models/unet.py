from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import torch
from uqct.ct import (
    Experiment,
    fbp,
    sample_observations,
    sinogram_from_counts,
    circular_mask,
)
from torch.utils.data import TensorDataset, DataLoader
from uqct.datasets.utils import DatasetName, get_dataset
from uqct.utils import get_checkpoint_dir
from uqct.training.unet import (
    MAX_TOTAL_INTENSITY,
    MIN_TOTAL_INTENSITY,
    N_ANGLES,
    N_BINS_HR,
    build_unet,
    sample_intensities,
)


class FBPUNetEnsemble:
    def __init__(
        self,
        dataset: DatasetName,
        sparse: bool,
        *,
        batch_size: int = 64,
        num_workers: int = 4,
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

    def to(self, device: torch.device | str) -> FBPUNetEnsemble:
        target = torch.device(device)
        for unet in self.unets:
            unet.unet.to(target)
        self._device = target
        return self

    def predict(
        self,
        experiment: Experiment,
        *,
        out_device: torch.device | None = None,
        aggregate: Literal["none", "mean", "median"] = "mean",
    ) -> torch.Tensor:
        """
        Generate ensemble predictions for an `Experiment`.

        Parameters
        ----------
        experiment : Experiment
            Experiment providing the measurement tensors required for inference.
        out_device : torch.device | None
            Target device for the output tensor. Defaults to each model's configured output device.
        aggregate : Literal["none", "mean", "median"]
            Aggregation strategy applied across ensemble members.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(..., rounds, M, 1, H, W)` (dense) or `(..., M, 1, H, W)` (sparse) when `aggregate="none"` with `M`
            ensemble members, otherwise `(..., rounds, 1, H, W)` (dense) or `(..., 1, H, W)` (sparse) after aggregation, all values clipped to `[0, 1]`.
        """
        fbp_lr, intensity_lr, class_labels = FBPUNet._prepare_inputs_from_experiment(
            experiment
        )
        preds = []
        for unet in self.unets:
            preds.append(
                unet._predict_from_tensors(
                    fbp_lr, intensity_lr, class_labels, out_device=out_device
                )
            )

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = build_unet(sparse).to(device)  # type: ignore
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["unet"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        unet.load_state_dict(sd, strict=True)
        if verbose:
            print(
                f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}"
            )
        self.unet = unet.eval()  # inference only
        self.sparse = sparse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_device = out_device

    @staticmethod
    def _prepare_inputs_from_experiment(
        experiment: Experiment,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        experiment : Experiment
            Experiment containing counts, intensities, and angles used to build model inputs.

        Returns
        -------
        torch.Tensor:
            Tensor of shape `(..., 1, H, W)` or `(..., H, W)` containing filtered backprojection inputs in `[0, 1]`.
        torch.Tensor:
            Tensor of shape `(..., 1)` containing intensity per sample.
        torch.Tensor | None:
            Tensor of shape `(...,)` with integer class labels when the model is sparse, otherwise `None`.
        """
        if not experiment.sparse:
            counts = experiment.counts.cumsum(dim=-3)
            intensities = experiment.intensities.cumsum(dim=-3)
            sino = sinogram_from_counts(counts, intensities).clamp_min(0.0)
            fbp_lr = fbp(sino, experiment.angles).clamp(0.0, 1.0)
            n_angles = None
            return (
                fbp_lr.unsqueeze(-3),
                intensities.sum(-2) * counts.shape[-1],
                n_angles,
            )

        num_angles = len(experiment.angles)
        fbps = list()
        intensities = list()
        for i in range(1, num_angles + 1):
            angles_i = experiment.angles[:i]
            counts_i = experiment.counts[..., :i, :]
            intensities_i = experiment.intensities[..., :i, :]
            sino_i = sinogram_from_counts(counts_i, intensities_i)
            fbp_i = fbp(sino_i, angles_i)
            fbps.append(fbp_i)
            intensities.append(intensities_i.sum((-2, -1)))
        fbps = torch.stack(fbps, dim=-4)
        intensities = torch.stack(intensities, dim=-2) * experiment.counts.shape[-1]
        class_labels = (
            torch.arange(1, num_angles + 1)
            .view(*((intensities.ndim - 2) * (1,)), num_angles)
            .expand(*intensities.shape[:-2], -1)
        )
        return fbps, intensities, class_labels

    def _predict_from_tensors(
        self,
        fbp_lr: torch.Tensor,
        total_intensity: torch.Tensor,
        class_labels: torch.Tensor | None,
        *,
        out_device: torch.device | None,
    ) -> torch.Tensor:
        """
        Run inference on batched tensors produced by `_prepare_inputs_from_experiment`.

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
        batch_dims = fbp_lr.shape[:-3]
        batch_prod = math.prod(batch_dims)

        fbp_lr = fbp_lr.view(batch_prod, *fbp_lr.shape[-3:])
        total_intensity = total_intensity.view(batch_prod, 1)
        if class_labels is not None:
            class_labels = class_labels.flatten()

        fbp_cpu = fbp_lr.to("cpu")
        intensity_cpu = total_intensity.to("cpu")
        class_cpu = class_labels.to("cpu") if class_labels is not None else None

        if class_cpu is None:
            dataset = TensorDataset(fbp_cpu, intensity_cpu)
        else:
            dataset = TensorDataset(fbp_cpu, intensity_cpu, class_cpu)

        out_device = out_device if out_device is not None else self.out_device
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        preds = []
        torch.set_grad_enabled(False)
        for batch in loader:
            if class_cpu is None:
                fbp_b, intensity_b = batch
                cls_b = None
            else:
                fbp_b, intensity_b, cls_b = batch

            x = fbp_b.to(device, non_blocking=True) * 2.0 - 1.0
            intensity_norm = (
                (intensity_b.to(device, non_blocking=True) - MIN_TOTAL_INTENSITY)
                / (MAX_TOTAL_INTENSITY - MIN_TOTAL_INTENSITY)
                * 999
            )

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
        *,
        out_device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Run inference on an `Experiment`, returning normalized reconstructions.

        Parameters
        ----------
        experiment : Experiment
            Experiment containing counts, intensities, and angles used to build model inputs.
        out_device : torch.device | None
            Target device for the output tensor. Defaults to the device specified at init when `None`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(..., 1, H, W)` containing predictions in `[0, 1]`.
        """
        fbp_lr, intensity_lr, class_labels = self._prepare_inputs_from_experiment(
            experiment
        )
        return self._predict_from_tensors(
            fbp_lr,
            intensity_lr,
            class_labels,
            out_device=out_device,
        )


if __name__ == "__main__":
    try:
        import lovely_tensors as lt

        lt.monkey_patch()
    except Exception as _:
        pass

    def build_dense_experiment(
        samples: torch.Tensor, angles: torch.Tensor
    ) -> Experiment:
        samples_cpu = samples.to("cpu")
        angles_cpu = angles.to("cpu")
        batch = samples_cpu.shape[0]
        intensities = sample_intensities(batch, device=samples_cpu.device) / (
            N_ANGLES * N_BINS_HR
        )
        intensities = intensities.reshape(-1, 1, 1, 1).expand(-1, 1, len(angles_cpu), 1)
        counts = sample_observations(samples_cpu, intensities, angles_cpu)
        intensities_lr = (intensities * 2).squeeze(1)
        return Experiment(counts, intensities_lr.unsqueeze(1), angles_cpu, sparse=False)

    def build_sparse_experiment(
        samples: torch.Tensor,
        angles: torch.Tensor,
        *,
        seed: int | None = 0,
    ) -> Experiment:
        samples_cpu = samples.to("cpu")
        angles_cpu = angles.to("cpu")
        batch = samples_cpu.shape[0]

        if seed is not None:
            prev_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        else:
            prev_state = None

        try:
            intensities = sample_intensities(batch, device=samples_cpu.device) / (
                N_ANGLES * N_BINS_HR
            )
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
    model_label = "sparse"
    dataset = "lung"

    _, test_set = get_dataset(dataset, True)
    num_examples = min(10, len(test_set))

    gt = torch.stack([test_set[i] for i in range(num_examples)], dim=0).to(device)
    angles = torch.from_numpy(np.linspace(0, 180, 50, endpoint=False))

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
    model = FBPUNetEnsemble("lung", model_label == "sparse")
    preds = model.predict(exp, aggregate="none")
    print(preds)
