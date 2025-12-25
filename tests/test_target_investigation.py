"""
tests/test_target_investigation.py
Phase 15 Follow-up: Is the target exactly 5?

The asymmetry (κ overshoots +0.84%, κ* undershoots -1.33%) suggests
either:
1. The target isn't exactly 5
2. There's an R-dependent correction term

Let's investigate what target would give zero average error.
"""

import numpy as np
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.g_product_full import compute_zeta_factors


def find_optimal_target():
    """Find what target value would give zero average error."""
    print("\n" + "=" * 70)
    print("INVESTIGATION: WHAT TARGET GIVES ZERO ERROR?")
    print("=" * 70)

    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        results[benchmark] = {
            'R': R,
            'B_over_A': result['B_over_A'],
            'A': result['exp_coefficient'],
            'B': result['constant_offset'],
        }

    ba_k = results['kappa']['B_over_A']
    ba_ks = results['kappa_star']['B_over_A']

    print(f"\nActual B/A values:")
    print(f"  κ:  {ba_k:.6f}")
    print(f"  κ*: {ba_ks:.6f}")
    print(f"  Average: {(ba_k + ba_ks) / 2:.6f}")

    # If we used the average as target:
    avg_target = (ba_k + ba_ks) / 2
    print(f"\nIf target = {avg_target:.6f}:")
    print(f"  κ error: {(ba_k - avg_target) / avg_target * 100:+.4f}%")
    print(f"  κ* error: {(ba_ks - avg_target) / avg_target * 100:+.4f}%")

    # What does the current error imply about the "true" target?
    # If κ is +0.84% high and κ* is -1.33% low, and we assume
    # both should have similar absolute errors, the target is between them
    print(f"\nError-weighted optimal target:")
    # Weight inversely by error magnitude
    weighted_target = (ba_k * abs(ba_ks - 5) + ba_ks * abs(ba_k - 5)) / (abs(ba_ks - 5) + abs(ba_k - 5))
    print(f"  Weighted target: {weighted_target:.6f}")


def investigate_mirror_constant():
    """
    The mirror formula uses m = exp(R) + (2K-1).
    What if (2K-1) should be (2K-1) + f(R) for some function f?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: R-DEPENDENT CORRECTION TO MIRROR CONSTANT")
    print("=" * 70)

    # For each benchmark, find what constant gives B/A = 5 exactly
    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        # Current: B = D + 5*A, so B/A = D/A + 5
        # We want: B/A = 5, so D/A should be 0
        # But D/A = delta = (B/A - 5) ≠ 0

        # If we replaced 5 with (5 + c), we'd get:
        # B' = D + (5+c)*A
        # B'/A = D/A + 5 + c
        # For B'/A = 5, need c = -D/A = -delta

        A = result['exp_coefficient']
        D = result['D']
        delta = D / A
        correction = -delta

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  Current B/A: {result['B_over_A']:.6f}")
        print(f"  D/A (delta): {delta:.6f}")
        print(f"  To get B/A=5, need (2K-1)+c where c = {correction:.6f}")
        print(f"  That means constant = {5 + correction:.6f}")

    # Is there a pattern? Let's compute c as a function of R
    print("\n\nPattern search: Is correction related to R?")
    print("-" * 50)

    R_values = [1.0, 1.1, 1.1167, 1.2, 1.3, 1.3036, 1.4, 1.5]
    polys_k = load_przz_k3_polynomials("kappa")

    for R in R_values:
        # Use kappa polynomials with different R values
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys_k, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        delta = result['D'] / result['exp_coefficient']
        correction = -delta

        # Try various R-dependent forms
        form1 = -0.1 * (R - 1.2)  # Linear in R
        form2 = -0.2 / R  # 1/R term
        form3 = -0.1 * (R - 1)**2  # Quadratic

        print(f"R={R:.4f}: c={correction:+.4f}, "
              f"linear={form1:+.4f}, 1/R={form2:+.4f}, quad={form3:+.4f}")


def investigate_higher_order_terms():
    """
    Check if there are higher-order terms in the Laurent expansion we're dropping.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: HIGHER-ORDER LAURENT TERMS")
    print("=" * 70)

    # The Laurent expansion of ζ'/ζ around s=1 is:
    # (ζ'/ζ)(1+ε) = -1/ε + γ + γ₁ε + γ₂ε² + ...
    # where γ₁ = γ₁ (Stieltjes constant), etc.

    # We use only the first two terms: -1/ε + γ
    # At ε = -R: -1/(-R) + γ = 1/R + γ

    # The first Stieltjes constant γ₁ ≈ -0.0728158...
    gamma1 = -0.0728158454836767248605863758749997319

    print("Laurent expansion: (ζ'/ζ)(1+ε) = -1/ε + γ + γ₁ε + O(ε²)")
    print(f"  γ (Euler-Mascheroni) = 0.5772156649...")
    print(f"  γ₁ (first Stieltjes) = {gamma1:.10f}")

    for R in [1.3036, 1.1167]:
        epsilon = -R
        order0 = -1/epsilon  # = 1/R
        order1 = gamma1 * epsilon  # = -γ₁ * R

        current = 1/R + 0.5772156649015329

        zf = compute_zeta_factors(R, precision=100)
        actual = zf.logderiv_actual

        print(f"\nR = {R}:")
        print(f"  Current (1/R + γ): {current:.10f}")
        print(f"  With γ₁ term: {current + order1:.10f}")
        print(f"  Actual (ζ'/ζ)(1-R): {actual:.10f}")
        print(f"  Difference from actual:")
        print(f"    Current: {(current - actual) / actual * 100:+.4f}%")
        print(f"    With γ₁: {((current + order1) - actual) / actual * 100:+.4f}%")


def investigate_d_component():
    """
    D = I₁₂(+R) + I₃₄(+R) is the "contamination" term.
    What makes D different between κ and κ*?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: D COMPONENT BREAKDOWN")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  D = {result['D']:.6f}")
        print(f"  A = {result['exp_coefficient']:.6f}")
        print(f"  D/A = {result['D']/result['exp_coefficient']:.6f}")
        print(f"\n  I₁₂(+R) components:")
        for k, v in result['i12_plus_pieces'].items():
            print(f"    {k}: {v:+.6f}")
        print(f"\n  I₃₄(+R) components:")
        for k, v in result['i34_plus_pieces'].items():
            print(f"    {k}: {v:+.6f}")

        # The issue is that D has opposite signs for κ and κ*
        # κ: D > 0, so δ > 0, B/A > 5
        # κ*: D < 0, so δ < 0, B/A < 5


def test_different_j12_j34_balance():
    """
    What if the error is in the relative weight of j12 vs j34?

    Currently: B = D + 5*A = (I12_plus + I34_plus) + 5*I12_minus

    What if there should be a factor on I34?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: I₃₄ WEIGHT FACTOR")
    print("=" * 70)

    # For each benchmark, find what I34 weight would give B/A = 5 exactly
    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        i12_plus = result['i12_plus_total']
        i12_minus = result['i12_minus_total']
        i34_plus = result['i34_plus_total']

        # Current: B = i12_plus + i34_plus + 5*i12_minus
        # B/A = (i12_plus + i34_plus + 5*i12_minus) / i12_minus
        #     = i12_plus/i12_minus + i34_plus/i12_minus + 5

        # If we weight i34 by factor w:
        # B' = i12_plus + w*i34_plus + 5*i12_minus
        # B'/A = i12_plus/i12_minus + w*i34_plus/i12_minus + 5
        # For B'/A = 5: w*i34_plus/i12_minus = -i12_plus/i12_minus
        # w = -i12_plus / i34_plus

        w_optimal = -i12_plus / i34_plus if abs(i34_plus) > 1e-14 else float('inf')

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  i12_plus: {i12_plus:.6f}")
        print(f"  i34_plus: {i34_plus:.6f}")
        print(f"  i12_minus: {i12_minus:.6f}")
        print(f"  Current weight on i34: 1.0")
        print(f"  Optimal weight to get B/A=5: {w_optimal:.4f}")


if __name__ == "__main__":
    find_optimal_target()
    investigate_mirror_constant()
    investigate_higher_order_terms()
    investigate_d_component()
    test_different_j12_j34_balance()
