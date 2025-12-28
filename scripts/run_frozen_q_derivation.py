#!/usr/bin/env python3
"""
Phase 45.4, Approach 1: Frozen-Q Decomposition Derivation

This script attempts to derive g_I1 and g_I2 from first principles using
the frozen-Q decomposition.

HYPOTHESIS:
===========
Q appears differently in I1 vs I2:
- I1: Q(Arg_α) × Q(Arg_β) where Arg depends on (x,y) → derivatives hit Q
- I2: Q(t)² frozen weight → Q is never differentiated

The difference between "normal" and "frozen" Q isolates the Q-derivative
contribution. If g_I1 ≈ 1.0 because Q-derivatives provide self-correction,
we should be able to derive this from the frozen-Q ratio.

DERIVATION APPROACH:
====================
1. Compute I1 at +R and -R with three Q modes: normal, frozen, none
2. The frozen-Q I1 should behave like I2 (no Q derivatives)
3. The ratio I1_normal/I1_frozen at -R modifies the effective g

If frozen-Q needs g_baseline correction, but normal-Q needs g_I1:
  g_I1 = g_baseline × (I1_frozen(-R) / I1_normal(-R))

For I2 (always uses frozen Q):
  g_I2 = g_baseline × (adjustment from Q reweighting)

Created: 2025-12-27 (Phase 45.4)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode
from src.evaluator.correction_policy import (
    compute_g_baseline,
    compute_base,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


@dataclass
class FrozenQDerivationResult:
    """Result of frozen-Q derivation for one benchmark."""
    benchmark: str
    R: float

    # I1 totals at +R with different Q modes
    I1_plus_normal: float
    I1_plus_frozen: float
    I1_plus_no_Q: float

    # I1 totals at -R with different Q modes
    I1_minus_normal: float
    I1_minus_frozen: float
    I1_minus_no_Q: float

    # Derived g values
    g_I1_derived: float
    g_I2_derived: float

    # Calibrated values for comparison
    g_I1_calibrated: float = G_I1_CALIBRATED
    g_I2_calibrated: float = G_I2_CALIBRATED


def compute_I1_total_with_Q_mode(
    R: float,
    theta: float,
    polynomials: Dict,
    q_mode: str,
    n_quad: int = 60,
) -> float:
    """
    Compute total I1 across all pairs with specified Q mode.

    Returns the weighted sum across all 6 pairs.
    """
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1_val = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode=q_mode, n_quad_u=n_quad, n_quad_t=n_quad
        )

        norm = f_norm[pair_key] * symmetry[pair_key]
        total += I1_val * norm

    return total


def derive_g_from_frozen_q(
    benchmark_name: str,
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> FrozenQDerivationResult:
    """
    Derive g_I1 and g_I2 from frozen-Q decomposition.

    The key insight is that I1 with frozen Q behaves like I2 (no Q derivatives).
    The ratio of I1_normal to I1_frozen at -R tells us how g_I1 differs from
    what frozen-Q would need.

    DERIVATION FORMULA:
    ===================

    Hypothesis 1: Frozen-Q I1 needs g_baseline (like I2)
    Hypothesis 2: Normal-Q I1 needs g_I1 (which is ≈ 1.0)

    The mirror formula at -R:
        contribution = g × base × I1(-R)

    If frozen-Q and normal-Q give the same c contribution, then:
        g_frozen × I1_frozen(-R) = g_normal × I1_normal(-R)
        g_normal = g_frozen × (I1_frozen(-R) / I1_normal(-R))

    If g_frozen = g_baseline, then:
        g_I1 = g_baseline × (I1_frozen(-R) / I1_normal(-R))
    """
    g_baseline = compute_g_baseline(theta, K)
    base = compute_base(R, K)

    print(f"\n{'='*70}")
    print(f"  FROZEN-Q DERIVATION: {benchmark_name}")
    print(f"{'='*70}")
    print(f"\nParameters: R={R}, θ={theta:.6f}, K={K}")
    print(f"g_baseline = 1 + θ/(2K(2K+1)) = {g_baseline:.8f}")
    print()

    # Compute I1 totals at +R
    print("Computing I1 at +R with different Q modes...")
    I1_plus_normal = compute_I1_total_with_Q_mode(R, theta, polynomials, "normal", n_quad)
    I1_plus_frozen = compute_I1_total_with_Q_mode(R, theta, polynomials, "frozen", n_quad)
    I1_plus_no_Q = compute_I1_total_with_Q_mode(R, theta, polynomials, "none", n_quad)

    print(f"  I1(+R, normal): {I1_plus_normal:+.10f}")
    print(f"  I1(+R, frozen): {I1_plus_frozen:+.10f}")
    print(f"  I1(+R, no_Q):   {I1_plus_no_Q:+.10f}")
    print()

    # Compute I1 totals at -R
    print("Computing I1 at -R with different Q modes...")
    I1_minus_normal = compute_I1_total_with_Q_mode(-R, theta, polynomials, "normal", n_quad)
    I1_minus_frozen = compute_I1_total_with_Q_mode(-R, theta, polynomials, "frozen", n_quad)
    I1_minus_no_Q = compute_I1_total_with_Q_mode(-R, theta, polynomials, "none", n_quad)

    print(f"  I1(-R, normal): {I1_minus_normal:+.10f}")
    print(f"  I1(-R, frozen): {I1_minus_frozen:+.10f}")
    print(f"  I1(-R, no_Q):   {I1_minus_no_Q:+.10f}")
    print()

    # Compute Q-mode ratios
    ratio_frozen_normal_minus = I1_minus_frozen / I1_minus_normal if abs(I1_minus_normal) > 1e-15 else 1.0
    ratio_frozen_normal_plus = I1_plus_frozen / I1_plus_normal if abs(I1_plus_normal) > 1e-15 else 1.0

    print("Q-mode ratios:")
    print(f"  I1_frozen(-R) / I1_normal(-R) = {ratio_frozen_normal_minus:.8f}")
    print(f"  I1_frozen(+R) / I1_normal(+R) = {ratio_frozen_normal_plus:.8f}")
    print()

    # =========================================================================
    # DERIVATION APPROACH 1: Simple ratio
    # =========================================================================
    # Hypothesis: frozen-Q needs g_baseline, normal-Q needs g_I1
    # g_I1 = g_baseline × (I1_frozen(-R) / I1_normal(-R))

    g_I1_approach1 = g_baseline * ratio_frozen_normal_minus

    print("APPROACH 1: Simple ratio")
    print("-" * 50)
    print(f"  g_I1 = g_baseline × (I1_frozen / I1_normal)")
    print(f"  g_I1 = {g_baseline:.8f} × {ratio_frozen_normal_minus:.8f}")
    print(f"  g_I1 = {g_I1_approach1:.8f}")
    print(f"  Calibrated g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"  Difference: {g_I1_approach1 - G_I1_CALIBRATED:+.8f} ({(g_I1_approach1/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # DERIVATION APPROACH 2: Q-derivative effect inversion
    # =========================================================================
    # The Q derivative effect = I1_normal - I1_frozen
    # This effect creates the difference between g_I1 and g_baseline

    q_deriv_effect_minus = I1_minus_normal - I1_minus_frozen
    q_deriv_effect_plus = I1_plus_normal - I1_plus_frozen

    print("APPROACH 2: Q-derivative effect")
    print("-" * 50)
    print(f"  Q-deriv effect at -R: {q_deriv_effect_minus:+.10f}")
    print(f"  Q-deriv effect at +R: {q_deriv_effect_plus:+.10f}")

    # The Q-derivative effect fraction at -R
    q_deriv_fraction = q_deriv_effect_minus / I1_minus_frozen if abs(I1_minus_frozen) > 1e-15 else 0
    print(f"  Q-deriv fraction at -R: {q_deriv_fraction:+.8f}")

    # If g_I1 = 1.0 + delta, and frozen needs g_baseline...
    # delta_g = g_baseline - g_I1 ≈ 0.013 (if g_I1 = 1.0)
    # This should relate to q_deriv_fraction

    # Approach 2a: g_I1 = 1.0 (if Q derivatives provide full self-correction)
    g_I1_approach2a = 1.0
    print(f"\n  Approach 2a: g_I1 = 1.0 (pure self-correction hypothesis)")
    print(f"    Calibrated g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"    Difference: {1.0 - G_I1_CALIBRATED:+.8f}")

    # Approach 2b: g_I1 = 1 / (1 + q_deriv_fraction)
    g_I1_approach2b = g_baseline / (1 + q_deriv_fraction) if (1 + q_deriv_fraction) != 0 else g_baseline
    print(f"\n  Approach 2b: g_I1 = g_baseline / (1 + q_deriv_fraction)")
    print(f"    g_I1 = {g_baseline:.8f} / {1 + q_deriv_fraction:.8f}")
    print(f"    g_I1 = {g_I1_approach2b:.8f}")
    print(f"    Calibrated g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"    Difference: {g_I1_approach2b - G_I1_CALIBRATED:+.8f} ({(g_I1_approach2b/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # DERIVATION FOR g_I2
    # =========================================================================
    # I2 always uses frozen Q (Q(t)²), so g_I2 relates to g_baseline via
    # the Q-reweighting effect (frozen vs no_Q)

    print("APPROACH FOR g_I2:")
    print("-" * 50)

    q_reweight_ratio_minus = I1_minus_frozen / I1_minus_no_Q if abs(I1_minus_no_Q) > 1e-15 else 1.0
    q_reweight_ratio_plus = I1_plus_frozen / I1_plus_no_Q if abs(I1_plus_no_Q) > 1e-15 else 1.0

    print(f"  Q-reweight ratio at -R: {q_reweight_ratio_minus:.8f}")
    print(f"  Q-reweight ratio at +R: {q_reweight_ratio_plus:.8f}")

    # Hypothesis: I2 needs g_baseline modified by Q-reweighting asymmetry
    # If Q reweighting changes I(-R) relative to I(+R), it affects the mirror ratio
    g_I2_approach = g_baseline * (q_reweight_ratio_plus / q_reweight_ratio_minus)

    print(f"\n  g_I2 = g_baseline × (Q_reweight_plus / Q_reweight_minus)")
    print(f"  g_I2 = {g_baseline:.8f} × ({q_reweight_ratio_plus:.8f} / {q_reweight_ratio_minus:.8f})")
    print(f"  g_I2 = {g_I2_approach:.8f}")
    print(f"  Calibrated g_I2 = {G_I2_CALIBRATED:.8f}")
    print(f"  Difference: {g_I2_approach - G_I2_CALIBRATED:+.8f} ({(g_I2_approach/G_I2_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # Use approach 1 as the derived values
    g_I1_derived = g_I1_approach1
    g_I2_derived = g_I2_approach

    return FrozenQDerivationResult(
        benchmark=benchmark_name,
        R=R,
        I1_plus_normal=I1_plus_normal,
        I1_plus_frozen=I1_plus_frozen,
        I1_plus_no_Q=I1_plus_no_Q,
        I1_minus_normal=I1_minus_normal,
        I1_minus_frozen=I1_minus_frozen,
        I1_minus_no_Q=I1_minus_no_Q,
        g_I1_derived=g_I1_derived,
        g_I2_derived=g_I2_derived,
    )


def main():
    print()
    print("=" * 70)
    print("  PHASE 45.4, APPROACH 1: FROZEN-Q DERIVATION")
    print("=" * 70)
    print()
    print("Testing hypothesis: Q-derivative effects explain g_I1 ≈ 1.0")
    print()
    print(f"Calibrated targets:")
    print(f"  g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"  g_I2 = {G_I2_CALIBRATED:.8f}")

    # Load polynomial sets
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4 / 7
    K = 3

    # Run derivation for both benchmarks
    result_kappa = derive_g_from_frozen_q(
        "κ BENCHMARK", R=1.3036, theta=theta, polynomials=polys_kappa, K=K, n_quad=60
    )

    result_kappa_star = derive_g_from_frozen_q(
        "κ* BENCHMARK", R=1.1167, theta=theta, polynomials=polys_kappa_star, K=K, n_quad=60
    )

    # Summary
    print()
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print()

    print("Derived g_I1 values:")
    print(f"  κ benchmark:  {result_kappa.g_I1_derived:.8f}")
    print(f"  κ* benchmark: {result_kappa_star.g_I1_derived:.8f}")
    print(f"  Calibrated:   {G_I1_CALIBRATED:.8f}")
    print()

    print("Derived g_I2 values:")
    print(f"  κ benchmark:  {result_kappa.g_I2_derived:.8f}")
    print(f"  κ* benchmark: {result_kappa_star.g_I2_derived:.8f}")
    print(f"  Calibrated:   {G_I2_CALIBRATED:.8f}")
    print()

    # Check success
    g_I1_avg = (result_kappa.g_I1_derived + result_kappa_star.g_I1_derived) / 2
    g_I2_avg = (result_kappa.g_I2_derived + result_kappa_star.g_I2_derived) / 2

    g_I1_match = abs(g_I1_avg - G_I1_CALIBRATED) / G_I1_CALIBRATED < 0.01  # Within 1%
    g_I2_match = abs(g_I2_avg - G_I2_CALIBRATED) / G_I2_CALIBRATED < 0.01  # Within 1%

    print("Average derived values:")
    print(f"  g_I1_avg = {g_I1_avg:.8f} (error: {(g_I1_avg/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print(f"  g_I2_avg = {g_I2_avg:.8f} (error: {(g_I2_avg/G_I2_CALIBRATED - 1)*100:+.4f}%)")
    print()

    if g_I1_match and g_I2_match:
        print("STATUS: DERIVATION SUCCESSFUL")
        print("  The frozen-Q decomposition derives g_I1 and g_I2 within 1%!")
        return True
    elif g_I1_match or g_I2_match:
        print("STATUS: PARTIAL SUCCESS")
        if g_I1_match:
            print("  g_I1 derivation works!")
        if g_I2_match:
            print("  g_I2 derivation works!")
        print("  Proceed to Approach 2 (Beta Moment) for the other component.")
        return False
    else:
        print("STATUS: DERIVATION NEEDS REFINEMENT")
        print("  The frozen-Q approach does not directly derive g_I1 and g_I2.")
        print("  Proceed to Approach 2 (Beta Moment) or Approach 3 (Log Factor).")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
