"""Equivariant bootstrap for uncertainty quantification.

Implements the parametric bootstrap of Pereyra & Tachella (AISTATS 2024,
"Equivariant bootstrapping for uncertainty quantification in imaging inverse
problems") for our CT setup, with the ``FBPUNet`` estimator as ``x_hat``.

Algorithm (per bootstrap sample i = 1 ... N, given the estimator
``x_hat = FBPUNet(y)``):
    1.  Draw a random group element g_i (here: an image-plane rotation by
        theta_i, optionally composed with axis flips).
    2.  Simulate y_tilde_i ~ P(A T_{g_i} x_hat).
    3.  Reconstruct x_hat_i = FBPUNet(y_tilde_i).
    4.  Invert: x_tilde_i = T_{g_i}^{-1} x_hat_i.

The empirical {x_tilde_i} approximates the sampling distribution of FBPUNet(Y).
"""

import torch
import torchvision.transforms.functional as TF

from uqct.ct import Experiment, circular_mask, poisson, radon
from uqct.eval.run import run_evaluation
from uqct.models.unet import FBPUNet

_LENGTH_SCALE = 5.0


def _rotate(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate every image in ``images`` by ``angle_deg`` (CCW, bilinear).

    ``images`` has shape ``(..., H, W)``; output keeps the same shape.
    """
    orig_shape = images.shape
    flat = images.reshape(-1, 1, *orig_shape[-2:])
    rot = TF.rotate(
        flat,
        angle=float(angle_deg),
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0.0],
    )
    return rot.view(orig_shape)


def _reconstruct(experiment: Experiment, model: FBPUNet) -> torch.Tensor:
    """Run ``FBPUNet`` on a (bootstrap) measurement.

    Returns a tensor of shape ``(N, H, W)`` clamped to ``[0, 1]``.
    """
    exp = Experiment(
        counts=experiment.counts,
        intensities=experiment.intensities,
        angles=experiment.angles,
        sparse=True,
    )
    # Single time step using all angles. prepare_inputs_from_experiment slices
    # `experiment.angles[:i]`, so passing i = len(angles) keeps every angle.
    schedule = torch.tensor([len(experiment.angles)], device=experiment.counts.device)
    # FBPUNet.predict returns (N, T, 1, H, W) for a length-T schedule.
    pred = model.predict(exp, schedule=schedule, out_device=experiment.counts.device)
    return pred.squeeze(-3).squeeze(-3).clip(0, 1)


def get_equivariant_bootstrap_predictor(
    model: FBPUNet,
    n_bootstraps: int,
    rotation_std_deg: float = 8.0,
    flip: bool = False,
):
    """Build the equivariant-bootstrap predictor closure for ``run_evaluation``.

    Matches the parameterization of the reference implementation
    (https://github.com/tachella/equivariant_bootstrap, ``bootstrap`` in
    ``helper_fcns.py``): the per-iteration rotation angle is drawn as
    ``N(0, rotation_std_deg^2)`` rather than uniformly. The paper's tomography
    experiments use ``rotation_std_deg = 8``. ``flip`` enables independent
    Bernoulli(0.5) horizontal and vertical flips of the augmented signal.
    """

    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        del schedule  # we always use the full angle set, like other bootstraps.

        if not experiment.sparse:
            raise NotImplementedError(
                "Equivariant bootstrap only implemented for the sparse setting."
            )

        device = experiment.counts.device

        # Initial reconstruction x_hat = FBPUNet(y) at LR (128x128).
        x_hat = _reconstruct(experiment, model)
        mask = circular_mask(x_hat.shape[-1], device=device, dtype=x_hat.dtype)
        x_hat = x_hat * mask

        n_det_lr = experiment.counts.shape[-1]
        scale_lr = _LENGTH_SCALE / n_det_lr

        generator = torch.Generator(device=device).manual_seed(
            int(torch.randint(0, 2**31 - 1, (1,)).item())
        )

        samples: list[torch.Tensor] = []
        for _ in range(n_bootstraps):
            # Gaussian angle, matching `rotate(xg, np.random.randn() * max_angle)`
            # in the reference implementation.
            theta = float(
                torch.randn(1, device=device, generator=generator).item()
                * rotation_std_deg
            )

            flip_lr = (
                flip and torch.rand(1, device=device, generator=generator).item() > 0.5
            )
            flip_ud = (
                flip and torch.rand(1, device=device, generator=generator).item() > 0.5
            )

            # 1. Apply forward transform T_g to x_hat.
            x_hat_rot = _rotate(x_hat, theta)
            if flip_lr:
                x_hat_rot = torch.flip(x_hat_rot, dims=(-1,))
            if flip_ud:
                x_hat_rot = torch.flip(x_hat_rot, dims=(-2,))
            x_hat_rot = x_hat_rot * mask

            # 2. Simulate bootstrap measurement y_tilde ~ P(A T_g x_hat).
            sino_rot = radon(x_hat_rot, experiment.angles)
            clean = experiment.intensities * torch.exp(-scale_lr * sino_rot)
            counts_b = poisson(clean, generator=generator)
            experiment_b = Experiment(
                counts_b,
                experiment.intensities,
                experiment.angles,
                experiment.sparse,
            )

            # 3. Reconstruct x_hat(y_tilde).
            x_b = _reconstruct(experiment_b, model)

            # 4. Invert the transform: T_g^{-1} = flip_ud, flip_lr, rotate(-theta).
            if flip_ud:
                x_b = torch.flip(x_b, dims=(-2,))
            if flip_lr:
                x_b = torch.flip(x_b, dims=(-1,))
            x_b_inv = _rotate(x_b, -theta) * mask
            samples.append(x_b_inv.clamp(0, 1))

        # Stack to (N, R, H, W) then add the singleton time dimension expected
        # by run.evaluate_and_save: (N, T=1, R, H, W).
        preds = torch.stack(samples, dim=1).unsqueeze(1)
        preds.mul_(mask)
        return preds

    return predictor_fn


def run_equivariant_bootstrapping(
    dataset,
    sparse,
    total_intensity,
    image_range,
    seed,
    n_angles,
    max_angle,
    n_bootstraps,
    rotation_std_deg: float = 8.0,
    flip: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing FBPUNet (dataset={dataset}, member=0)...")
    model = FBPUNet(
        dataset=dataset,
        member=0,
        sparse=sparse,
        batch_size=32,
        model_device=device,
    )
    print("FBPUNet initialized successfully.")

    predictor_fn = get_equivariant_bootstrap_predictor(
        model, n_bootstraps, rotation_std_deg, flip
    )

    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="equivariant_bootstrapping",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=199,
        schedule_type="linear",
        schedule_length=1,
        max_angle=max_angle,
        extra_metadata={
            "n_bootstraps": n_bootstraps,
            "rotation_std_deg": rotation_std_deg,
            "flip": flip,
        },
    )
