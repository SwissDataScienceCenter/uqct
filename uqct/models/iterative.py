import math
from typing import Literal

import einops
import torch
from torch import optim
from tqdm.auto import tqdm

from uqct.ct import Experiment, circular_mask, fbp, sinogram_from_counts
from uqct.models.diffusion import get_guidance_loss_fn

ReconstructionMethod = Literal["mle", "map"]


def tv_prior(image: torch.Tensor) -> torch.Tensor:
    """Compute Total Variation (TV) prior."""
    diff_h = image[..., 1:, :] - image[..., :-1, :]
    diff_w = image[..., :, 1:] - image[..., :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def initialize_fbp(experiment: Experiment, schedule) -> torch.Tensor:
    """
    Initialize reconstruction using FBP.
    Returns: (N, S, H, W)
    """
    device = experiment.counts.device
    n_gt = experiment.counts.shape[0]

    if experiment.sparse:
        assert schedule is not None
        n_angles_total = experiment.counts.shape[-2]

        # Create mask for angles based on schedule
        # shape: (S, n_angles)
        schedule_indices = schedule.to(device)
        angle_indices = torch.arange(n_angles_total, device=device).unsqueeze(
            0
        )  # (1, n_angles)
        mask = angle_indices < schedule_indices.unsqueeze(1)  # (S, n_angles)

        # Prepare sinogram
        # counts: (N, n_angles, n_det)
        sino_full = sinogram_from_counts(
            experiment.counts, experiment.intensities
        )  # (N, n_angles, n_det)

        # Expand to (N, S, n_angles, n_det)
        sino_expanded = sino_full.unsqueeze(1).expand(-1, len(schedule), -1, -1).clone()

        # Apply mask (zero out future angles)
        # mask: (S, n_angles) -> (1, S, n_angles, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        sino_expanded = sino_expanded * mask_expanded

        # Flatten for FBP: (N*S, n_angles, n_det)
        sino_flat = einops.rearrange(sino_expanded, "n s a d -> (n s) a d")

        # Run FBP
        # fbp scales by pi / (2 * n_angles_total)
        x_flat = fbp(sino_flat, experiment.angles)  # (N*S, H, W)

        # Correct scaling
        # We want scale pi / (2 * schedule[s])
        # Factor = (pi / (2 * schedule)) / (pi / (2 * total)) = total / schedule
        scale_factor = n_angles_total / schedule_indices.float()  # (S,)
        scale_factor = einops.repeat(scale_factor, "s -> (n s) 1 1", n=n_gt)
        x_flat = x_flat * scale_factor

        x = einops.rearrange(x_flat, "(n s) h w -> n s h w", n=n_gt)

    else:
        # Dense: loop over time/rounds
        # counts: (N, T, n_angles, n_det)
        # For dense, we want cumulative counts/intensities
        counts_csum = experiment.counts.cumsum(1)
        intensities_csum = experiment.intensities.cumsum(1)

        # Flatten: (N*T, n_angles, n_det)
        sino_flat = sinogram_from_counts(
            einops.rearrange(counts_csum, "n t a d -> (n t) a d"),
            einops.rearrange(intensities_csum, "n t a d -> (n t) a d"),
        )

        x_flat = fbp(sino_flat, experiment.angles)  # (N*T, H, W)
        x = einops.rearrange(x_flat, "(n t) h w -> n t h w", n=n_gt)

    return x


def reconstruct(
    experiment: Experiment,
    schedule,
    method: ReconstructionMethod,
    lr: float,
    patience: int,
    tv_weight: float,
    max_steps: int,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Perform iterative reconstruction (MLE or MAP).

    Returns:
        torch.Tensor: Reconstructed images (N, S, H, W)
    """
    device = experiment.counts.device
    side_length = experiment.counts.shape[-1]

    if tv_weight < 0.0:
        # 5e5 is good for total intensity 1e9
        # 5e4 is good for total intensity 1e4
        # Log linearly interpolate
        tv_weight = 5 * 10 ** (
            4 + (math.log10(experiment.total_intensity.mean().item()) - 4) / 5.0
        )
        print(
            f"Interpolated TV Weight: {tv_weight:.2e} (Exposure: {experiment.total_intensity.mean().item():.2e})"
        )

    # Initialize using FBP
    # Shape: (N, S, H, W)
    x_init = initialize_fbp(experiment, schedule)

    # Shape: (1, N, S, H, W)
    x = x_init.clone()
    x = torch.nn.Parameter(x).requires_grad_()

    optimizer = optim.Adam([x], lr=lr)

    # Get NLL loss function
    nll_loss_fn = get_guidance_loss_fn(experiment, schedule)

    mask = circular_mask(side_length, device=device, dtype=torch.bool)

    best_loss = float("inf")
    patience_counter = 0
    best_x = x.data.clone()

    it = tqdm(
        range(max_steps), disable=not verbose, desc=f"{method.upper()} Optimization"
    )

    for _ in it:
        optimizer.zero_grad()

        # Apply constraints
        xp = (x * mask).clip(0)

        loss = nll_loss_fn(xp)

        # Add TV prior for MAP
        if method == "map":
            loss += tv_weight * tv_prior(xp)

        loss.backward()
        optimizer.step()

        # Enforce mask on parameters
        with torch.no_grad():
            x.data[..., ~mask] = 0.0
            x.data[x.data < 0] = 0.0
            x.data[x.data > 1] = 1.0

        current_loss = loss.item()

        # Convergence check
        if current_loss < best_loss:
            best_loss = current_loss
            best_x = xp.data.clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Converged at step {_} with loss {best_loss:.4e}")
            break

        it.set_postfix({"loss": f"{current_loss:.4e}", "best": f"{best_loss:.4e}"})
    return best_x
