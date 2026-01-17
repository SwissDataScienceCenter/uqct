import click
import torch
import h5py
import json
from uqct.ct import Experiment, sinogram_from_counts, fbp, circular_mask
from uqct.eval.run import run_evaluation, setup_experiment
from uqct.utils import get_results_dir


def get_bootstrap_predictor(
    n_bootstraps: int, comparison: bool = False, minmax: bool = False
):
    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        if not experiment.sparse or schedule is None:
            raise NotImplementedError(
                "Bootstrapping only implemented for sparse sequential setting."
            )

        preds_all = []

        for t, n_avail in enumerate(schedule):
            n_avail_val = int(n_avail.item())

            indices = torch.randint(
                0,
                n_avail_val,
                (n_bootstraps, n_avail_val),
                device=experiment.counts.device,
            )

            preds_t = []

            for b in range(n_bootstraps):
                idxs = indices[b]
                angles_b = experiment.angles[idxs]

                counts_slice = experiment.counts[..., :n_avail_val, :]
                counts_b = counts_slice[..., idxs, :]

                intensities_slice = experiment.intensities[..., :n_avail_val, :]
                intensities_b = intensities_slice[..., idxs, :]

                sino_b = sinogram_from_counts(counts_b, intensities_b)
                fbp_recon = fbp(sino_b, angles_b)

                preds_t.append(fbp_recon)

            preds_t_stacked = torch.stack(preds_t, dim=1)  # (N, B, H, W)

            if comparison:
                # Compute min and max across bootstraps -> (N, 2, H, W)
                if minmax:
                    lb = preds_t_stacked.min(dim=1).values
                    ub = preds_t_stacked.max(dim=1).values
                else:
                    mean = preds_t_stacked.mean(dim=1)
                    std = preds_t_stacked.std(dim=1)
                    lb = mean - std / 2
                    ub = mean + std / 2
                preds_t_stacked = torch.stack([lb, ub], dim=1)

            preds_all.append(preds_t_stacked)

        # Stack time -> (N, T, R, H, W) where R = n_bootstraps or 2
        preds = torch.stack(preds_all, dim=1)

        preds = preds.clamp(0, 1)
        preds.mul_(circular_mask(preds.shape[-1], device=preds.device))

        return preds

    return predictor_fn


def run_bootstrapping(
    dataset,
    sparse,
    total_intensity,
    image_range,
    seed,
    n_angles,
    schedule_start,
    schedule_type,
    schedule_length,
    max_angle,
    n_bootstraps,
    comparison: bool = False,
):
    predictor_fn = get_bootstrap_predictor(n_bootstraps, comparison)

    if comparison:
        # manual setup and save
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

        output_dir = get_results_dir() / "other_uq_methods" / "bootstrapping"
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = (
            f"{dataset}:{total_intensity}:{sparse}:"
            f"{image_range[0]}-{image_range[1]}:{seed}"
        )
        fp_h5 = output_dir / (file_name + ".h5")

        metadata = {
            "dataset": dataset,
            "sparse": sparse,
            "total_intensity": total_intensity,
            "image_range": image_range,
            "seed": seed,
            "n_angles": n_angles,
            "schedule_start": schedule_start,
            "schedule_type": schedule_type,
            "schedule_length": schedule_length,
            "max_angle": max_angle,
            "n_bootstraps": n_bootstraps,
            "method": "bootstrapping_minmax",
        }

        with h5py.File(fp_h5, "w") as f:
            f.create_dataset("bounds", data=preds.cpu().numpy(), dtype="float32")
            f.attrs["metadata"] = json.dumps(metadata)

        print(f"Saved bootstrapping comparison bounds to {fp_h5}")

    else:
        run_evaluation(
            dataset=dataset,
            sparse=sparse,
            total_intensity=total_intensity,
            image_range=image_range,
            seed=seed,
            model_name="bootstrapping",
            predictor_fn=predictor_fn,
            n_angles=n_angles,
            schedule_start=schedule_start,
            schedule_type=schedule_type,
            schedule_length=schedule_length,
            max_angle=max_angle,
            extra_metadata={"n_bootstraps": n_bootstraps},
        )
