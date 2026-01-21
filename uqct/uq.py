from uqct.ct import circular_mask

import torch
from typing import Dict


def mean_std(samples, bdim=0):
    mean = samples.mean(dim=bdim)
    std = samples.std(dim=bdim, unbiased=True)
    return mean, std


def gaussian_ci(
    samples: torch.Tensor,
    delta: float = 0.05,
    bdim = 0
) -> Dict[str, torch.Tensor]:
    """
    Gaussian (mean ± z·std) confidence interval.
    """
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - delta / 2))
    mean, std = mean_std(samples, bdim=bdim)

    lo = mean - z * std
    hi = mean + z * std

    return lo, hi


def percentile_ci(
    samples: torch.Tensor,
    alpha: float = 0.05,
    bdim = 0
) -> Dict[str, torch.Tensor]:
    """
    Percentile bootstrap confidence interval.
    """
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    lo = torch.quantile(samples, q_lo, dim=bdim)
    hi = torch.quantile(samples, q_hi, dim=bdim)

    return lo, hi


def basic_ci(
    samples: torch.Tensor,
    alpha: float = 0.05,
    bdim = 0
) -> Dict[str, torch.Tensor]:
    """
    Basic (bias-corrected) bootstrap interval:
    [2θ̂ − q_hi, 2θ̂ − q_lo]
    """
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    mean = samples.mean(dim=bdim)
    ql = torch.quantile(samples, q_lo, dim=bdim)
    qh = torch.quantile(samples, q_hi, dim=bdim)

    lo = 2 * mean - qh
    hi = 2 * mean - ql

    return lo, hi


def studentized_ci(
    samples: torch.Tensor,
    delta: float = 0.05,
    eps: float = 1e-8,
    bdim = 0
) -> Dict[str, torch.Tensor]:
    """
    Studentized (t-type) bootstrap confidence interval.
    """
    mean, std = mean_std(samples, bdim=bdim)

    t = (samples - mean.unsqueeze(bdim)) / (std.unsqueeze(bdim) + eps)

    q_lo = delta / 2
    q_hi = 1 - delta / 2

    t_hi = torch.quantile(t, q_lo, dim=bdim)
    t_lo = torch.quantile(t, q_hi, dim=bdim)

    lo = mean - t_lo * std
    hi = mean - t_hi * std

    return lo, hi


def simultaneous_ci(
    samples: torch.Tensor,
    delta: float = 0.05,
    eps: float = 1e-8,
    bdim = 0
) -> Dict[str, torch.Tensor]:
    """
    Simultaneous (global) confidence band using max-T bootstrap.
    Provides ~95% coverage over all pixels jointly.
    """
    mean, std = mean_std(samples, bdim=bdim)

    t = (samples - mean.unsqueeze(bdim)) / (std.unsqueeze(bdim) + eps)

    # flatten spatial dims only
    t_flat = t.flatten(start_dim=bdim + 1)
    max_t = torch.max(torch.abs(t_flat), dim=-1).values

    t_star = torch.quantile(max_t, 1 - delta)

    lo = mean - t_star * std
    hi = mean + t_star * std

    return lo, hi


def coverage(
    ci_lo: torch.Tensor,
    ci_hi: torch.Tensor,
    target: torch.Tensor,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Empirical pointwise coverage fraction, optionally within a circular mask.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        covered = ((target >= ci_lo) & (target <= ci_hi)).float() * mask
        return covered.sum(dim=(-1, -2, -3)) / mask.sum()
    else:
        return ((target >= ci_lo) & (target <= ci_hi)).float().mean(dim=(-1, -2, -3))

def error_correlation(
    ci_width: torch.Tensor,
    error: torch.Tensor,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Computes the Pearson correlation coefficient between absolute error and confidence interval width.
    Args:
        ci_width (torch.Tensor): (..., H, W) Width of confidence interval.
        error (torch.Tensor): (..., H, W) Error tensor.
        circle_mask (bool): If True, applies a circular mask before computing correlation.
    Returns:
        torch.Tensor: Correlation coefficient(s), shape: (...,)
    """
    if circle_mask:
        mask = circular_mask(error.shape[-1], device=error.device)
        ci_width = ci_width * mask
        error = error * mask

    # Flatten spatial dims
    batch_dims = ci_width.shape[:-2]
    width_flat = ci_width.reshape(*batch_dims, -1)
    error_flat = error.reshape(*batch_dims, -1)

    # Subtract mean
    width_flat = width_flat - width_flat.mean(dim=-1, keepdim=True)
    error_flat = error_flat - error_flat.mean(dim=-1, keepdim=True)
    # Compute correlation
    numerator = (width_flat * error_flat).sum(dim=-1)
    denominator = (
        torch.sqrt((width_flat ** 2).sum(dim=-1)) *
        torch.sqrt((error_flat ** 2).sum(dim=-1))
    )
    corr = numerator / (denominator + 1e-8)
    return corr


def error_r2(
    ci_width: torch.Tensor,
    error: torch.Tensor,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Computes the coefficient of determination (R^2) between confidence interval width and error.
    Args:
        ci_width (torch.Tensor): (..., H, W) Width of confidence interval.
        error (torch.Tensor): (..., H, W) Error tensor.
        circle_mask (bool): If True, applies a circular mask before computing R^2.
    Returns:
        torch.Tensor: R^2 values for each image in the batch. Output shape: (...,)
    """
    if circle_mask:
        mask = circular_mask(error.shape[-1], device=error.device)
        ci_width = ci_width * mask
        error = error * mask

    # Flatten spatial dims
    batch_dims = ci_width.shape[:-2]
    width_flat = ci_width.reshape(*batch_dims, -1)
    error_flat = error.reshape(*batch_dims, -1)

    # Compute R^2
    ss_res = ((error_flat - width_flat) ** 2).sum(dim=-1)
    ss_tot = ((error_flat - error_flat.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2