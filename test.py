import uqct

_ = uqct.getLogger()

import torch
from scipy import stats


def run_full_comparison_simulation(
    trials_per_scale=10000, k=10, d=16384, target_conf=0.95
):
    """
    Compares Marginal, Šidák, and Bonferroni bounds across different noise scales.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Full Comparison on {device} ---")
    print(f"Configuration: k={k}, d={d}, Target={target_conf}")

    # 1. Setup Critical Values (CPU/Scipy)
    df = k - 1
    alpha = 1 - target_conf

    # --- A. Marginal (Uncorrected) ---
    # alpha_marg = 0.05
    t_marg_val = stats.t.ppf(1 - alpha / 2, df)

    # --- B. Šidák (Exact for Independent) ---
    # coverage_per_dim = 0.95^(1/d)
    coverage_per_dim = target_conf ** (1 / d)
    alpha_sidak = 1 - coverage_per_dim
    t_sidak_val = stats.t.ppf(1 - alpha_sidak / 2, df)

    # --- C. Bonferroni (Conservative Approx) ---
    # alpha_bonf = 0.05 / d
    alpha_bonf = alpha / d
    t_bonf_val = stats.t.ppf(1 - alpha_bonf / 2, df)

    # Move to GPU
    t_marg = torch.tensor(t_marg_val, device=device, dtype=torch.float32)
    t_sidak = torch.tensor(t_sidak_val, device=device, dtype=torch.float32)
    t_bonf = torch.tensor(t_bonf_val, device=device, dtype=torch.float32)

    print(f"\nCritical Values (t-score):")
    print(f"  Marginal:   {t_marg.item():.4f}")
    print(f"  Šidák:      {t_sidak.item():.4f}")
    print(f"  Bonferroni: {t_bonf.item():.4f}")
    print("-" * 105)

    # Table Header
    # We compare Width (Efficiency) and Joint Coverage (Safety)
    header = f"{'Sigma (σ)':<10} | {'Marg Width':<12} {'J.Cov%':<7} | {'Šidák Width':<12} {'J.Cov%':<7} | {'Bonf Width':<12} {'J.Cov%':<7}"
    print(header)
    print("-" * 105)

    sigmas = [0.001, 0.1, 1.0, 10.0, 1000.0]

    for sigma in sigmas:
        # Counters for Joint Coverage (Did ALL d dimensions fall inside?)
        cov_marg = 0
        cov_sidak = 0
        cov_bonf = 0

        # Accumulators for Widths
        width_marg_sum = 0.0
        width_sidak_sum = 0.0
        width_bonf_sum = 0.0

        for _ in range(trials_per_scale):
            # X ~ N(0, sigma^2)
            X = torch.randn(k, d, device=device) * sigma

            # Statistics
            means = X.mean(dim=0)
            se = X.std(dim=0) / (k**0.5)

            # --- Check Joint Coverage ---
            # Joint coverage fails if ANY dimension is outside.
            # Max t-stat determines the worst-case dimension.
            max_t = (means.abs() / se).max()

            if max_t <= t_marg:
                cov_marg += 1
            if max_t <= t_sidak:
                cov_sidak += 1
            if max_t <= t_bonf:
                cov_bonf += 1

            # --- Calculate Average Widths ---
            # Width = 2 * t * se. We take the mean across d dimensions for this trial.
            mean_se = se.mean()
            width_marg_sum += (2 * t_marg * mean_se).item()
            width_sidak_sum += (2 * t_sidak * mean_se).item()
            width_bonf_sum += (2 * t_bonf * mean_se).item()

        # Averages
        print(
            f"{sigma:<10} | "
            f"{width_marg_sum / trials_per_scale:<12.5f} {cov_marg / trials_per_scale:<7.2%} | "
            f"{width_sidak_sum / trials_per_scale:<12.5f} {cov_sidak / trials_per_scale:<7.2%} | "
            f"{width_bonf_sum / trials_per_scale:<12.5f} {cov_bonf / trials_per_scale:<7.2%}"
        )


if __name__ == "__main__":
    run_full_comparison_simulation()
