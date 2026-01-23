import torch
from typing import Dict

from uqct.ct import circular_mask


def mean_std(samples, bdim=0):
    mean = samples.mean(dim=bdim)
    std = samples.std(dim=bdim, unbiased=True)
    return mean, std


def gaussian_ci(
    samples: torch.Tensor, delta: float = 0.05, bdim=0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian (mean ± z·std) confidence interval.
    """
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - delta / 2))
    mean, std = mean_std(samples, bdim=bdim)

    lo = mean - z * std
    hi = mean + z * std

    return lo, hi


def gaussian_conservative_ci(
    samples: torch.Tensor, delta: float = 0.05, bdim=0, n_pixels: int = 128 * 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian (mean ± z·std) confidence interval.
    """
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - delta / 2 / n_pixels))
    mean, std = mean_std(samples, bdim=bdim)

    lo = mean - z * std
    hi = mean + z * std

    return lo, hi


def percentile_ci(
    samples: torch.Tensor, alpha: float = 0.05, bdim=0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Percentile bootstrap confidence interval.
    """
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    lo = torch.quantile(samples, q_lo, dim=bdim)
    hi = torch.quantile(samples, q_hi, dim=bdim)

    return lo, hi


def basic_ci(
    samples: torch.Tensor, alpha: float = 0.05, bdim=0
) -> tuple[torch.Tensor, torch.Tensor]:
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
    samples: torch.Tensor, delta: float = 0.05, eps: float = 1e-8, bdim=0
) -> tuple[torch.Tensor, torch.Tensor]:
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
    samples: torch.Tensor, delta: float = 0.05, eps: float = 1e-8, bdim=0
) -> tuple[torch.Tensor, torch.Tensor]:
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


def twod_to_threed(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.unsqueeze(0) if t.ndim == 2 else t for t in tensors)


def coverage(
    ci_lo: torch.Tensor,
    ci_hi: torch.Tensor,
    target: torch.Tensor,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Empirical pointwise coverage fraction, optionally within a circular mask.
    """
    ci_lo, ci_hi, target = twod_to_threed(ci_lo, ci_hi, target)
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        covered = ((target >= ci_lo) & (target <= ci_hi)).float() * mask
        return covered.sum(dim=(-1, -2, -3)) / mask.sum()
    else:
        return ((target >= ci_lo) & (target <= ci_hi)).float().mean(dim=(-1, -2, -3))


def simultaneous_coverage(
    ci_lo: torch.Tensor,
    ci_hi: torch.Tensor,
    target: torch.Tensor,
    circle_mask: bool = True,
) -> bool:
    """
    Returns True if all (masked) pixels are covered by the interval.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        covered = ((target >= ci_lo) & (target <= ci_hi)) | (~mask.bool())
        return covered.all().item()
    else:
        covered = (target >= ci_lo) & (target <= ci_hi)
        return covered.all().item()


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
    denominator = torch.sqrt((width_flat**2).sum(dim=-1)) * torch.sqrt(
        (error_flat**2).sum(dim=-1)
    )
    corr = numerator / (denominator + 1e-8)
    return corr


def error_r2(
    ci_width: torch.Tensor,
    error: torch.Tensor,
    circle_mask: bool = True,
    linear_fit: bool = False,
) -> torch.Tensor:
    """
    Computes the coefficient of determination (R^2) between confidence interval width and error.
    Args:
        ci_width (torch.Tensor): (..., H, W) Width of confidence interval.
        error (torch.Tensor): (..., H, W) Error tensor.
        circle_mask (bool): If True, applies a circular mask before computing R^2.
        linear_fit (bool): If True, computes R^2 of an optimal linear fit (correlation^2).
    Returns:
        torch.Tensor: R^2 values for each image in the batch. Output shape: (...,)
    """
    if linear_fit:
        # Arguments are passed directly to `error_correlation`.
        return error_correlation(ci_width, error, circle_mask) ** 2

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


def sparsification_error(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    circle_mask: bool = True,
    n_bins: int = 100,
) -> torch.Tensor:
    """
    Computes the Area Under the Sparsification Error curve (AUSE).
    Measures how well uncertainty predicts high-error regions.
    The AUSE is the difference between the area under the sparsification curve
    sorted by uncertainty (Model) and the area under the curve sorted by error (Oracle).
    Lower is better. AUSE=0 means perfect ranking.

    Args:
        uncertainty (torch.Tensor): (..., H, W) Uncertainty map (e.g., ci_width or std).
        error (torch.Tensor): (..., H, W) Absolute error map.
        circle_mask (bool): If True, applies circular mask.
        n_bins (int): Number of bins for the sparsification curve.

    Returns:
        torch.Tensor: AUSE values for each image in the batch. Output shape: (...,)
    """
    if circle_mask:
        mask = circular_mask(error.shape[-1], device=error.device)
        uncertainty = uncertainty * mask
        error = error * mask

        # Flatten and keep only valid pixels
        batch_dims = error.shape[:-2]
        error_flat = error.reshape(*batch_dims, -1)
        unc_flat = uncertainty.reshape(*batch_dims, -1)

        # Note: If we just mask with * mask, the outside become 0.
        # 0 uncertainty might be interpreted as high confidence.
        # But 0 error is good.
        # We need to filter indices where mask is 1.
        # Assuming simple batch processing where mask is same for all is tricky if vectorizing.
        # However, usually we compute mean over batch.

        # To strictly compute sparsification on valid pixels only:
        valid_indices = mask.flatten().nonzero().squeeze()
        if valid_indices.numel() > 0:
            error_flat = error_flat.index_select(-1, valid_indices)
            unc_flat = unc_flat.index_select(-1, valid_indices)
        else:
            # Should not happen in CT
            return torch.zeros(batch_dims, device=error.device)

    else:
        # Flatten spatial dims
        batch_dims = error.shape[:-2]
        error_flat = error.reshape(*batch_dims, -1)
        unc_flat = uncertainty.reshape(*batch_dims, -1)

    # Number of valid pixels
    n_pixels = error_flat.shape[-1]

    # Vectorized sorting

    # 1. Sort by Uncertainty (Model) Ascending
    #    (Removing high uncertainty first = keeping low uncertainty pixels)
    #    So curve x-axis is "fraction removed".
    #    At x=0 (0 removed), we have all pixels. Mean error is global mean.
    #    At x=0.99 (99% removed), we have top 1% uncertain pixels removed (or keep 1% most certain).
    #    Wait, usually sparsification plots Error vs Fraction Removed.
    #    Order: Remove highest uncertainty first.
    #    So we define "remaining" set as those with lowest uncertainty.
    #    So we sort by uncertainty ascending.
    #    Remaining k% = first k% of sorted array.

    _, unc_indices = torch.sort(unc_flat, dim=-1, descending=False)

    # 2. Sort by Error (Oracle) Ascending
    #    Oracle removes highest error first.
    #    Remaining k% = first k% of sorted array (lowest error).

    _, err_indices = torch.sort(error_flat, dim=-1, descending=False)

    # Reorder error
    error_by_unc = torch.gather(error_flat, -1, unc_indices)
    error_by_err = torch.gather(error_flat, -1, err_indices)

    # Compute cumulative sums -> cumulative means of remaining pixels
    # cumsum[i] is sum of elements 0..i
    cum_error_model = torch.cumsum(error_by_unc, dim=-1)
    cum_error_oracle = torch.cumsum(error_by_err, dim=-1)

    counts = torch.arange(1, n_pixels + 1, device=error.device, dtype=torch.float32)

    mean_error_model = cum_error_model / counts
    mean_error_oracle = cum_error_oracle / counts

    # AUSE = Mean absolute difference between curves
    diff = (mean_error_model - mean_error_oracle).abs()
    ause = diff.mean(dim=-1)

    return ause
