import numpy as np
from scipy import stats


def compute_confidence_intervals_numpy(
    samples, confidence=0.95, method="sidak", use_sqrt_k=False
):
    """
    Replication of the logic in uqct.other_methods.bootstrapping using numpy.
    samples: (k, d)
    """
    k, d = samples.shape
    df = k - 1

    means = np.mean(samples, axis=0)

    # The crucial part we are testing
    if use_sqrt_k:
        se = np.std(samples, axis=0, ddof=1) / np.sqrt(
            k
        )  # Original code (approx, using ddof=1 for sample std)
    else:
        se = np.std(samples, axis=0, ddof=1)  # My proposed fix

    alpha = 1.0 - confidence

    if method == "sidak":
        coverage_per_dim = confidence ** (1.0 / d)
        alpha_adj = 1.0 - coverage_per_dim
    else:
        alpha_adj = alpha

    t_val = stats.t.ppf(1 - alpha_adj / 2, df)

    margin = t_val * se
    lower = means - margin
    upper = means + margin

    return lower, upper


def verify_bootstrap_coverage():
    """
    Simulates a simple mean estimation problem to check coverage.
    """
    np.random.seed(42)

    n_experiments = 1000
    n_samples = 50  # Size of original data
    n_bootstraps = 100  # Number of bootstrap samples

    mu_true = 0.0
    sigma_true = 1.0

    print(f"Running {n_experiments} experiments...")
    print(f"Data - N(0,1), Size: {n_samples}, Bootstraps: {n_bootstraps}")
    print("-" * 30)

    for use_sqrt_k in [False, True]:
        label = "SE = STD / SQRT(K) [Original]" if use_sqrt_k else "SE = STD [Proposed]"
        covered_count = 0

        for _ in range(n_experiments):
            # 1. Generate Data
            data = np.random.randn(n_samples) * sigma_true + mu_true

            # 2. Bootstrap
            indices = np.random.randint(0, n_samples, (n_bootstraps, n_samples))
            boot_samples = data[indices]  # (B, N)

            # Compute Estimator (Mean) for each bootstrap sample
            # Each row is a bootstrap dataset. We calculate the mean of that dataset.
            # This gives us the bootstrap distribution of the mean.
            boot_means = np.mean(boot_samples, axis=1).reshape(-1, 1)  # (B, 1)

            # 3. Compute CI
            lower, upper = compute_confidence_intervals_numpy(
                boot_means, confidence=0.95, method="sidak", use_sqrt_k=use_sqrt_k
            )

            # Check coverage
            if lower[0] <= mu_true <= upper[0]:
                covered_count += 1

        coverage = covered_count / n_experiments
        print(f"Method: {label}")
        print(f"  Empirical Coverage: {coverage:.4f}")

    print("-" * 30)
    print(
        "Conclusion: The method with coverage closest to 0.95 is correct for estimating the parameter CI."
    )


if __name__ == "__main__":
    verify_bootstrap_coverage()
