import torch
import numpy as np
from uqct.uq import sparsification_error


def reference_ause(uncertainty, error):
    """
    Reference implementation of AUSE using numpy and explicit steps.
    Assumes 1D arrays.
    """
    u = uncertainty.flatten().cpu().numpy()
    e = error.flatten().cpu().numpy()
    n = len(e)

    # 1. Oracle Curve (Sort by Error Descending - remove high error first)
    # Wait, my implementation sorted Ascending (remove high uncertainty first).
    # Let's clarify the definition.
    # Sparsification curve:
    # X-axis: fraction of pixels REMOVED. (0 to 1)
    # Y-axis: Mean Error of REMAINING pixels.
    # Policy: Remove pixels with HIGHEST uncertainty.
    # So we sort pixels by uncertainty DESCENDING.
    # Then we iteratively remove them.
    # Remaining set at step k (removing k pixels) is the bottom N-k pixels of the sorted array.
    # So if we sort by uncertainty ASCENDING, the remaining set corresponds to indices [0 : N-k].
    # So my code: `unc_indices = torch.sort(unc_flat, descending=False)`
    # `cum_error` computes sum of [0], [0..1], [0..2]...
    # `cum_error[i]` is sum of `i+1` pixels.
    # This corresponds to keeping `i+1` lowest uncertainty pixels.
    # This matches "Removing Top (N-(i+1))".
    # This seems correct for "Sparsification Error Curve".

    # Let's replicate this logic in numpy.

    # Model: Sort by uncertainty
    # We want to keep low uncertainty.
    # Sort U ascending.
    idx_u = np.argsort(u)  # Ascending
    e_sorted_by_u = e[idx_u]

    # Oracle: Sort by error
    # We want to keep low error.
    # Sort E ascending.
    idx_e = np.argsort(e)
    e_sorted_by_e = e[idx_e]

    # Compute running means
    # fractions remaining = [1/N, 2/N, ..., 1.0]
    # removing [ (N-1)/N, ..., 0 ]

    mse_model = np.cumsum(e_sorted_by_u) / np.arange(1, n + 1)
    mse_oracle = np.cumsum(e_sorted_by_e) / np.arange(1, n + 1)

    diff = np.abs(mse_model - mse_oracle)
    ause = np.mean(diff)
    return ause


def test_ause():
    print("Testing AUSE Implementation...")

    # Case 1: Perfect Ranking
    # Uncertainty is exactly Error.
    print("\nCase 1: Perfect Ranking (Uncertainty ~ Error)")
    error = torch.rand(1000)
    uncertainty = error.clone()  # Perfect
    ause_my = sparsification_error(
        uncertainty.unsqueeze(0), error.unsqueeze(0), circle_mask=False
    ).item()
    ause_ref = reference_ause(uncertainty, error)
    print(f"  My AUSE: {ause_my:.6f}")
    print(f"  Ref AUSE: {ause_ref:.6f}")
    assert np.isclose(ause_my, 0.0), "Perfect ranking should be 0"
    assert np.isclose(ause_my, ause_ref), " implementations differ"

    # Case 2: Inverted Ranking
    # Uncertainty is opposite of Error.
    # Pixels with High Error have Low Uncertainty.
    # This is the worst case.
    # Sorting by uncertainty ASC gives High errors first.
    # Curve starts high and goes down.
    # Oracle starts low and goes up.
    print("\nCase 2: Inverted Ranking (Uncertainty ~ 1/Error)")
    error = torch.linspace(0.1, 1.0, 1000)
    uncertainty = 1.0 / error  # High error -> Low uncertainty
    ause_my = sparsification_error(
        uncertainty.unsqueeze(0), error.unsqueeze(0), circle_mask=False
    ).item()
    ause_ref = reference_ause(uncertainty, error)
    print(f"  My AUSE: {ause_my:.6f}")
    print(f"  Ref AUSE: {ause_ref:.6f}")
    assert np.isclose(ause_my, ause_ref)

    # Case 3: Random Ranking
    print("\nCase 3: Random Ranking")
    error = torch.rand(1000)
    uncertainty = torch.rand(1000)
    ause_my = sparsification_error(
        uncertainty.unsqueeze(0), error.unsqueeze(0), circle_mask=False
    ).item()
    ause_ref = reference_ause(uncertainty, error)
    print(f"  My AUSE: {ause_my:.6f}")
    print(f"  Ref AUSE: {ause_ref:.6f}")
    assert np.isclose(ause_my, ause_ref)

    # Case 4: With Masking
    print("\nCase 4: With Masking")
    # 4x4 image
    error = torch.randn(1, 1, 4, 4).abs()
    uncertainty = torch.randn(1, 1, 4, 4).abs()
    # Mask: center 2x2 is 1 (circle logic approx)
    # Actually uqct.ct.circular_mask creates specific mask
    # We will let the function create it and compare against manual masking

    # Let's trust the function's internal masking logic, just check if it runs
    # and if perfect ranking still gives 0 inside mask.

    # Mock "perfect inside mask"
    from uqct.ct import circular_mask

    mask = circular_mask(4, device="cpu")
    error = torch.rand(1, 1, 4, 4) * mask  # 0 outside
    uncertainty = error.clone()  # 0 outside, perfect inside

    # Add noise outside mask to uncertainty to ensure it doesn't accidentally sort 0s perfectly?
    # Actually, if both are 0 outside, they are "perfectly correlated" there too.
    # Let's invert uncertainty outside mask.
    uncertainty = uncertainty * mask + torch.rand(1, 1, 4, 4) * (1 - mask)

    # Inside mask: Perfect.
    # Ideally AUSE should be 0 because we only care about inside mask.

    ause_my = sparsification_error(uncertainty, error, circle_mask=True).item()
    print(f"  My AUSE (Perfect inside, Random outside): {ause_my:.6f}")
    assert np.isclose(ause_my, 0.0), "Should correspond to perfect ranking inside mask"


if __name__ == "__main__":
    test_ause()
