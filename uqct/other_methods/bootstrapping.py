import torch

from uqct.ct import Experiment, sinogram_from_counts, fbp, circular_mask
from uqct.eval.run import run_evaluation
from uqct.models.unet import FBPUNet


def get_bootstrap_predictor(
    n_bootstraps: int,
    method: str = "fbp",  # "fbp" or "unet"
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
            batch_size = 32

            final_preds = []

            for i in range(0, N_images, batch_size):
                end_i = min(i + batch_size, N_images)

                # Slice experiment data for this batch
                # counts: (N, n_angles, n_det)
                counts_batch = experiment.counts[i:end_i]
                intensities_batch = experiment.intensities[i:end_i]

                batch_preds_t = []
                for b in range(n_bootstraps):
                    idxs = indices[b]
                    angles_b = experiment.angles[idxs]

                    # Select along angle dimension (dim -2 usually for counts)
                    counts_b = counts_batch[..., idxs, :]
                    intensities_b = intensities_batch[..., idxs, :]

                    if method == "unet" and model is not None:
                        # Construct temporary experiment for this bootstrap sample
                        exp_b = Experiment(
                            counts=counts_b,
                            intensities=intensities_b,
                            angles=angles_b,
                            sparse=True,
                        )
                        # Predict using all angles (last step)
                        # We need to pass a schedule to get a prediction at the specific number of angles
                        schedule = torch.tensor([len(angles_b)], device=counts_b.device)
                        # pred shape: (Batch, 1, 1, H, W)
                        pred = model.predict(
                            exp_b, schedule=schedule, out_device=counts_b.device
                        )
                        batch_preds_t.append(pred.squeeze(1).squeeze(1))
                    else:
                        sino_b = sinogram_from_counts(counts_b, intensities_b).clip(0)
                        fbp_recon = fbp(sino_b, angles_b).clip(0, 1)
                        batch_preds_t.append(fbp_recon)

                # Stack bootstrap samples for this batch -> (Batch, B, H, W)
                batch_preds_stacked = torch.stack(batch_preds_t, dim=1)
                final_preds.append(batch_preds_stacked)
            preds_t_stacked = torch.cat(final_preds, dim=0)
            preds_all.append(preds_t_stacked)
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
    max_angle,
    n_bootstraps,
    method: str = "fbp",
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
    wrapper = get_bootstrap_predictor(n_bootstraps, method)
    predictor_fn = wrapper(model_instance)

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name=f"bootstrapping_{method}",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=199,
        schedule_type="linear",
        schedule_length=1,
        max_angle=max_angle,
        extra_metadata={"n_bootstraps": n_bootstraps, "method": method},
    )
