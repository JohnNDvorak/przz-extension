#!/usr/bin/env python3
"""
Test the Integral Ratio Approach for g derivation.

The log factor split at the coefficient level FAILED because the Beta moment
is an "emergent property" - it comes from integration with (1-u)^{2K-1} weight,
not from pointwise coefficient ratios.

NEW APPROACH:
Compute the ratio of I1 with vs without log factor prefactor.

If I1_with_log = I1 (what we normally compute, includes (1/θ + x + y) factor)
And I1_without_log = M1 (just the (1/θ) × F_xy part, no cross-terms)

Then: internal_correction = I1_with_log / I1_without_log

And if this equals g_baseline, we confirm:
- I1 already has the correction internally
- g_I1 = 1.0 is correct

Created: 2025-12-27 (Phase 46)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials, Polynomial
from src.evaluator.g_functional import compute_I1_I2_totals


def compute_I1_without_log_factor(
    R: float,
    theta: float,
    polynomials: dict,
    n_quad: int = 60,
) -> float:
    """
    Compute I1 as if there was NO log factor prefactor.

    The normal I1 integrand is:
        d²/dxdy [(1/θ + x + y) × F(x,y)] = (1/θ) × F_xy + F_x + F_y

    Without the log factor, we'd just have:
        d²/dxdy [(1/θ) × F(x,y)] = (1/θ) × F_xy

    This function computes the (1/θ) × F_xy contribution only.

    Actually, this is hard to compute because the current I1 implementation
    includes the log factor intrinsically.

    ALTERNATIVE: Compute M1 from the log factor split directly.
    """
    from src.unified_s12.logfactor_split import split_logfactor_for_pair

    K = 3
    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    M1_total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        split = split_logfactor_for_pair(pair_key, theta, R, K, polynomials, n_quad)
        weight = factorial_norm[pair_key] * symmetry_factor[pair_key]
        M1_total += weight * split.main_coeff

    return M1_total


def main():
    print("=" * 70)
    print("INTEGRAL RATIO APPROACH FOR g DERIVATION")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3
    n_quad = 60

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    print(f"g_baseline = 1 + θ/(2K(2K+1)) = {g_baseline:.8f}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Test with real Q
    print("=" * 70)
    print("TEST WITH REAL Q POLYNOMIAL")
    print("=" * 70)

    for R in [1.3036, 1.1167]:
        print(f"\nR = {R}:")

        # Compute I1 with log factor (normal computation)
        I1_with_log, I2 = compute_I1_I2_totals(R, theta, polys, n_quad)

        # Compute I1 without log factor (M1 only)
        M1 = compute_I1_without_log_factor(R, theta, polys, n_quad)

        # The ratio should give the internal correction
        if abs(M1) > 1e-15:
            internal_correction = I1_with_log / M1
        else:
            internal_correction = float('nan')

        gap_pct = (internal_correction / g_baseline - 1) * 100

        print(f"  I1 (with log factor) = {I1_with_log:.8f}")
        print(f"  M1 (without log factor) = {M1:.8f}")
        print(f"  Ratio I1/M1 = {internal_correction:.8f}")
        print(f"  g_baseline = {g_baseline:.8f}")
        print(f"  Gap = {gap_pct:+.4f}%")

    # Test with Q=1
    print("\n" + "=" * 70)
    print("TEST WITH Q=1")
    print("=" * 70)

    Q_unity = Polynomial(np.array([1.0]))
    polys_q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}

    for R in [1.3036, 1.1167]:
        print(f"\nR = {R}:")

        I1_with_log, I2 = compute_I1_I2_totals(R, theta, polys_q1, n_quad)
        M1 = compute_I1_without_log_factor(R, theta, polys_q1, n_quad)

        if abs(M1) > 1e-15:
            internal_correction = I1_with_log / M1
        else:
            internal_correction = float('nan')

        gap_pct = (internal_correction / g_baseline - 1) * 100

        print(f"  I1 (with log factor) = {I1_with_log:.8f}")
        print(f"  M1 (without log factor) = {M1:.8f}")
        print(f"  Ratio I1/M1 = {internal_correction:.8f}")
        print(f"  g_baseline = {g_baseline:.8f}")
        print(f"  Gap = {gap_pct:+.4f}%")

    # Key insight check
    print("\n" + "=" * 70)
    print("KEY INSIGHT CHECK")
    print("=" * 70)
    print()
    print("If I1/M1 ≈ g_baseline, then:")
    print("  - I1 already includes the Beta moment correction internally")
    print("  - External g_I1 = 1.0 is correct")
    print()
    print("If I1/M1 ≠ g_baseline, then:")
    print("  - The log factor split is happening at the wrong level")
    print("  - Need a different approach")


if __name__ == "__main__":
    main()
