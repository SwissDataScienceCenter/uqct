import click
import torch
import h5py
import json
from typing import Literal
from uqct.ct import Experiment, sinogram_from_counts, fbp, circular_mask
from uqct.eval.run import run_evaluation, setup_experiment
from uqct.utils import get_results_dir, get_checkpoint_dir
from uqct.models.unet import FBPUNet, build_unet, load_unet_ckpt


def get_bootstrap_predictor(
    n_bootstraps: int,
    method: str = "fbp",  # "fbp" or "unet"
    comparison: bool = False,
    minmax: bool = False,
):

    # We return a closure-creator so we can inject the model instance later if needed
    def get_predictor_with_model(model=None):
        def predictor_fn(
            experiment: Experiment, schedule: torch.Tensor | None
        ) -> torch.Tensor:
            if not experiment.sparse:
                raise NotImplementedError(
                    "Bootstrapping only implemented for sparse sequential setting."
                )

            # Ignore passed schedule. Use full available angles (final time step implied).
            nonlocal model
            if method == "unet" and model is not None:
                # FBPUNet handles device internal to some extent but let's ensure
                pass

            preds_all = []

            # Use full available angles
            n_avail_val = experiment.angles.shape[0]

            # Single time step logic: resample angles
            indices = torch.randint(
                0,
                n_avail_val,
                (n_bootstraps, n_avail_val),
                device=experiment.counts.device,
            )

            # Batching Logic for Images to prevent OOM
            # Experiment has shape (N, ...)
            N_images = experiment.counts.shape[0]
            batch_size = 10

            final_preds = []

            for i in range(0, N_images, batch_size):
                end_i = min(i + batch_size, N_images)

                # Slice experiment data for this batch
                # counts: (N, n_angles, n_det)
                counts_batch = experiment.counts[i:end_i]
                intensities_batch = experiment.intensities[i:end_i]
                # angles are shared

                batch_preds_t = []
                for b in range(n_bootstraps):
                    idxs = indices[b]
                    angles_b = experiment.angles[idxs]

                    # Select along angle dimension (dim -2 usually for counts)
                    counts_b = counts_batch[..., idxs, :]
                    intensities_b = intensities_batch[..., idxs, :]

                    sino_b = sinogram_from_counts(counts_b, intensities_b)
                    fbp_recon = fbp(sino_b, angles_b)

                    if method == "unet" and model is not None:
                        with torch.no_grad():
                            # Unet input expectation: (N, 1, H, W)
                            if fbp_recon.ndim == 3:
                                inp = fbp_recon.unsqueeze(1)
                            else:
                                inp = fbp_recon

                            # Total intensity slice
                            tot = experiment.total_intensity[i:end_i]
                            if tot.ndim == 1:
                                tot = tot.unsqueeze(1)

                            # Create class labels (Model likely expects them if sparse)
                            # Assuming label 0 is correct default for this task where dataset is fixed
                            class_labels = torch.zeros(
                                len(inp), dtype=torch.long, device=inp.device
                            )

                            # FBPUNet._predict_from_tensors(fbp_lr, total_intensity, class_labels, ...)
                            pred = model._predict_from_tensors(
                                inp, tot, class_labels, out_device=inp.device
                            )
                            batch_preds_t.append(pred.squeeze(1))  # Back to (N, H, W)
                    else:
                        batch_preds_t.append(fbp_recon)

                # Stack bootstrap samples for this batch -> (Batch, B, H, W)
                batch_preds_stacked = torch.stack(batch_preds_t, dim=1)

                # Apply comparison/stats immediately if needed to save memory?
                # Or keep full samples if n_bootstraps is small (10).
                # (10, 10, 128, 128) per batch is small.
                # But we join them later.

                if comparison:
                    if minmax:
                        lb = batch_preds_stacked.min(dim=1).values
                        ub = batch_preds_stacked.max(dim=1).values
                    else:
                        mean = batch_preds_stacked.mean(dim=1)
                        std = batch_preds_stacked.std(dim=1)
                        lb = mean - std / 2
                        ub = mean + std / 2
                    batch_preds_stacked = torch.stack([lb, ub], dim=1)

                final_preds.append(batch_preds_stacked)

            # Cat batches -> (N, ...)
            preds_t_stacked = torch.cat(final_preds, dim=0)

            # Append to list (representing SINGLE time step)
            preds_all.append(preds_t_stacked)

            # Stack time -> (N, T, R, H, W). T=1.
            preds = torch.stack(preds_all, dim=1)
            preds = preds.clamp(0, 1)
            preds.mul_(circular_mask(preds.shape[-1], device=preds.device))
            return preds

        return predictor_fn

    return get_predictor_with_model


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
    method: str = "fbp",
    comparison: bool = False,
):

    # 1. Setup Model if needed
    model_instance = None
    if method == "unet":
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Initializing U-Net (dataset={dataset}, member=0)...")
            model_instance = FBPUNet(
                dataset=dataset,
                member=0,
                sparse=sparse,
                batch_size=32,  # Batch size for U-Net internal ops if used directly, but we use _predict_from_tensors
                model_device=device,
            )
            print("U-Net initialized successfully.")
        except Exception as e:
            print(f"Error initializing FBPUNet: {e}")
            model_instance = None

    # 2. Create Predictor
    wrapper = get_bootstrap_predictor(n_bootstraps, method, comparison)
    predictor_fn = wrapper(model_instance)

    # 3. Schedule Override: Start at n_angles (max index effectively)
    # This signals to valid schedule generators to produce a single point at the end.
    schedule_start_override = 200  # n_angles
    schedule_length_override = 1
    schedule_type_override = "linear"

    # 4. Run Evaluation
    if comparison:
        gt, experiment, schedule = setup_experiment(
            dataset,
            image_range,
            total_intensity,
            sparse,
            seed,
            schedule_length_override,
            schedule_start_override,
            schedule_type_override,
            n_angles,
            max_angle,
        )

        preds = predictor_fn(experiment, schedule)

        output_dir = get_results_dir() / "other_uq_methods" / "bootstrapping"
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = (
            f"{method}:{dataset}:{total_intensity}:{sparse}:"
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
            "method": f"bootstrapping_{method}",
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
            model_name=f"bootstrapping_{method}",
            predictor_fn=predictor_fn,
            n_angles=n_angles,
            schedule_start=schedule_start_override,
            schedule_type=schedule_type_override,
            schedule_length=schedule_length_override,
            max_angle=max_angle,
            extra_metadata={"n_bootstraps": n_bootstraps, "method": method},
        )
