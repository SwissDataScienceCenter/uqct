import torch
from uqct.ct import Experiment, nll
import einops


def compute_nll_trajectory(
    images: torch.Tensor,
    experiment: Experiment,
    schedule: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the NLL trajectory for a batch of images over the schedule.

    Args:
        images: (N, S, H, W).
        experiment: Experiment matching N.
        schedule: (S,) tensor.

    Returns:
        nlls_cum: (N, S) containing cumulative NLLs for each step in schedule.
        nlls_loss: (N,) containing NLLs for each sample.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be 4D (N, S, H, W), got {images.shape}")

    S = images.shape[-3]
    S_sched = len(schedule)

    if S != S_sched:
        raise ValueError(
            f"images schedule dim {S} does not match schedule length {S_sched}"
        )

    device = images.device
    n_angles = experiment.counts.shape[-2]
    counts_exp = einops.repeat(experiment.counts, "... T A -> ... S T A", S=S)
    intensities_exp = einops.repeat(experiment.intensities, "... T 1 -> ... S T 1", S=S)
    nlls = nll(images, counts_exp, intensities_exp, experiment.angles)
    split_points = torch.cat(
        [schedule.to(device), torch.tensor([len(experiment.angles)], device=device)]
    )
    start = schedule.min()
    end = split_points[1:].unsqueeze(1)  # (S, 1)
    angles_idx = torch.arange(n_angles, device=device).unsqueeze(0)  # (1, A)
    mask = (angles_idx >= start) & (angles_idx < end)  # (S, A)
    nlls[..., ~mask, :] = 0
    nlls_loss = einops.reduce(nlls, "... A -> ...", "mean")
    nlls_loss = einops.reduce(nlls_loss, "... -> 1", "sum")
    nlls = einops.reduce(nlls, "... T A -> ... ", "sum")
    return nlls, nlls_loss


def compute_constraint_violation(
    nll_cum: torch.Tensor, threshold_trajectory_cum: torch.Tensor
) -> torch.Tensor:
    """
    Computes constraint violation: sum(relu(cumsum(nll) - threshold_cum)).

    Args:
        nll_cum: (N, S)
        threshold_trajectory_cum: (N, S) - Already cumulative!

    Returns:
        violation: (N,) scalar violation per sample (sum over schedule)
    """
    diff = nll_cum - threshold_trajectory_cum
    visual_violation = torch.clamp(diff, min=0)
    return visual_violation.sum(dim=1)
