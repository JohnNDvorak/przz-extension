#!/usr/bin/env python3
"""
Phase 60.1: Test True PRZZ Unified Bracket Method

This script tests whether the frozen Q(t)² mode correctly implements
PRZZ's actual method and produces results matching their κ = 0.417 target.

From PRZZ TeX Line 1544:
    Q(D_α)Q(D_β)[T^{-tα-tβ}] |_{α=β=-R/L} = Q(t)² e^{2Rt}

The Q operators act on T^{-tα-tβ} which has NO x,y dependence.
Q(t)² is just a scalar multiplier at each quadrature point t.

Created: 2025-12-29
Phase: 60.1
"""

import sys
import math
sys.path.insert(0, "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension")

from src.unified_s12_evaluator_v3 import (
    run_dual_benchmark_v3,
    compute_t_integral_factor,
    compute_scalar_baseline_factor,
)


def test_frozen_vs_legacy_q():
    """
    Compare frozen Q(t)² mode against legacy affine-dependent mode.
    """
    print("=" * 70)
    print("PHASE 60.1: FROZEN Q(t)² vs LEGACY AFFINE Q TEST")
    print("=" * 70)

    theta = 4.0 / 7.0

    # PRZZ targets
    targets = {
        "kappa": {"R": 1.3036, "c": 2.137, "kappa": 0.417293962},
        "kappa_star": {"R": 1.1167, "c": 1.938, "kappa": 0.327833316},
    }

    # Run with LEGACY mode (affine-dependent Q)
    print("\n*** LEGACY MODE (affine-dependent Q) ***")
    legacy_kappa, legacy_kappa_star = run_dual_benchmark_v3(
        include_Q=True,
        frozen_q=False,  # Legacy: Q(A_α) × Q(A_β) with x,y dependence
        normalize_scalar_baseline=False,
    )

    # Run with FROZEN Q(t)² mode (PRZZ correct)
    print("\n*** FROZEN Q(t)² MODE (PRZZ correct) ***")
    frozen_kappa, frozen_kappa_star = run_dual_benchmark_v3(
        include_Q=True,
        frozen_q=True,  # PRZZ: Q(t)² as scalar
        normalize_scalar_baseline=False,
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: FROZEN Q(t)² vs LEGACY")
    print("=" * 70)

    for name, legacy, frozen, target in [
        ("kappa", legacy_kappa, frozen_kappa, targets["kappa"]),
        ("kappa_star", legacy_kappa_star, frozen_kappa_star, targets["kappa_star"]),
    ]:
        R = target["R"]
        c_target = target["c"]
        kappa_target = target["kappa"]

        # Normalization factor
        F_R = compute_t_integral_factor(R)
        scalar_baseline = compute_scalar_baseline_factor(R)

        print(f"\n{'='*60}")
        print(f"BENCHMARK: {name.upper()} (R={R})")
        print(f"{'='*60}")

        print(f"\n  PRZZ Targets:")
        print(f"    c_target     = {c_target:.6f}")
        print(f"    κ_target     = {kappa_target:.9f}")

        print(f"\n  Normalization factors:")
        print(f"    F(R) = (e^{{2R}}-1)/(2R) = {F_R:.6f}")
        print(f"    F(R)/2                   = {scalar_baseline:.6f}")

        print(f"\n  S12 values (unnormalized):")
        print(f"    Legacy (affine Q)   = {legacy.S12_unnormalized:.10f}")
        print(f"    Frozen Q(t)²        = {frozen.S12_unnormalized:.10f}")

        # Check if the ratio matches the expected normalization
        if frozen.S12_unnormalized != 0:
            ratio = legacy.S12_unnormalized / frozen.S12_unnormalized
            print(f"    Ratio (legacy/frozen) = {ratio:.6f}")

        # What κ would we get from each?
        # Using normalized S12 as a proxy for c (ignoring I₃₄ for now)
        for mode_name, result in [("Legacy", legacy), ("Frozen", frozen)]:
            S12_norm = result.S12_unnormalized / scalar_baseline
            # κ = 1 - log(c)/R, and if S12_norm ≈ c (rough approximation)
            if S12_norm > 0:
                kappa_est = 1 - math.log(S12_norm) / R
                gap_pct = (kappa_est - kappa_target) / kappa_target * 100
                print(f"\n  {mode_name} mode:")
                print(f"    S12 (normalized by F(R)/2) = {S12_norm:.6f}")
                print(f"    κ estimate (if c ≈ S12)    = {kappa_est:.6f}")
                print(f"    Gap from κ_target          = {gap_pct:+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The frozen Q(t)² mode implements PRZZ's actual method where:
  - Q operators act on T^{-tα-tβ} which has NO x,y dependence
  - Q(t)² is just a scalar multiplier at each t

  The legacy affine mode uses:
  - Q(A_α) × Q(A_β) where A_α = t + θ(t-1)x + θt·y
  - This adds x,y dependence through Q'(t)

  If frozen mode significantly changes results, it means the Q eigenvalue
  structure was contributing incorrectly to the xy coefficient.
""")


def test_t_integral_factor():
    """
    Test that the t-integral of e^{2Rt} gives the expected factor.
    """
    print("\n" + "=" * 70)
    print("TEST: t-INTEGRAL FACTOR")
    print("=" * 70)

    for R in [1.3036, 1.1167, 1.0, 0.5]:
        from scipy.integrate import quad

        # Numerical integration
        numerical, _ = quad(lambda t: math.exp(2 * R * t), 0, 1)

        # Analytical formula
        analytical = (math.exp(2 * R) - 1) / (2 * R)

        # Our function
        our_F = compute_t_integral_factor(R)

        print(f"\n  R = {R}:")
        print(f"    Numerical ∫e^{{2Rt}}dt = {numerical:.10f}")
        print(f"    Analytical (e^2R-1)/2R = {analytical:.10f}")
        print(f"    Our compute_t_integral_factor = {our_F:.10f}")
        print(f"    Match: {abs(numerical - our_F) < 1e-10}")


def test_theoretical_link_formula():
    """
    Test the theoretical link formula that connects unified bracket to c.

    From the DQ identity:
        Direct - exp(2R)×Mirror = (-2Rθ) × bracket

    Rearranging and substituting into assembly formula:
        c = Direct + m×Mirror + S34
          = (exp(2R) + m)×Mirror + (-2Rθ)×bracket + S34

    Where:
    - bracket = unified S12 computation (unnormalized)
    - Mirror = S12(-R) from split-channel
    - S34 = S34(+R) from split-channel
    - m = exp(R) + 5 (our empirical formula)
    """
    print("\n" + "=" * 70)
    print("TEST: THEORETICAL LINK FORMULA")
    print("=" * 70)
    print("""
    Formula: c = (exp(2R) + m)×S12(-R) + (-2Rθ)×bracket + S34(+R)

    This connects the unified bracket to the empirical c value.
    """)

    from src.kappa_engine import KappaEngine

    theta = 4.0 / 7.0

    # Get split-channel values from KappaEngine
    for name, engine_fn, R, c_target, kappa_target in [
        ("kappa", KappaEngine.from_przz_kappa, 1.3036, 2.137, 0.417293962),
        ("kappa_star", KappaEngine.from_przz_kappa_star, 1.1167, 1.938, 0.327833316),
    ]:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {name.upper()} (R={R})")
        print(f"{'='*60}")

        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()

        # Split-channel values
        S12_plus = integrals.S12_plus
        S12_minus = integrals.S12_minus
        S34_plus = integrals.S34_plus

        # Empirical m
        m = math.exp(R) + 5

        # Empirical c from split-channel
        c_empirical = S12_plus + m * S12_minus + S34_plus
        kappa_empirical = 1 - math.log(c_empirical) / R

        print(f"\n  Split-channel values:")
        print(f"    S12(+R) = {S12_plus:.10f}")
        print(f"    S12(-R) = {S12_minus:.10f}")
        print(f"    S34(+R) = {S34_plus:.10f}")
        print(f"    m = exp(R) + 5 = {m:.6f}")
        print(f"    c_empirical = S12+ + m×S12- + S34 = {c_empirical:.6f}")
        print(f"    κ_empirical = 1 - log(c)/R = {kappa_empirical:.6f}")

        # Now compute unified bracket
        from src.unified_s12_evaluator_v3 import run_dual_benchmark_v3

        # Get unified bracket for this benchmark
        if name == "kappa":
            frozen_result, _ = run_dual_benchmark_v3(
                include_Q=True, frozen_q=True, normalize_scalar_baseline=False
            )
            legacy_result, _ = run_dual_benchmark_v3(
                include_Q=True, frozen_q=False, normalize_scalar_baseline=False
            )
            bracket_frozen = frozen_result.S12_unnormalized
            bracket_legacy = legacy_result.S12_unnormalized
        else:
            _, frozen_result = run_dual_benchmark_v3(
                include_Q=True, frozen_q=True, normalize_scalar_baseline=False
            )
            _, legacy_result = run_dual_benchmark_v3(
                include_Q=True, frozen_q=False, normalize_scalar_baseline=False
            )
            bracket_frozen = frozen_result.S12_unnormalized
            bracket_legacy = legacy_result.S12_unnormalized

        # Theoretical link formula
        # c = (exp(2R) + m)×S12(-R) + (-2Rθ)×bracket + S34(+R)
        exp_2R = math.exp(2 * R)
        coeff_mirror = exp_2R + m
        coeff_bracket = -2 * R * theta

        for bracket_name, bracket in [("Legacy", bracket_legacy), ("Frozen", bracket_frozen)]:
            c_derived = coeff_mirror * S12_minus + coeff_bracket * bracket + S34_plus

            if c_derived > 0:
                kappa_derived = 1 - math.log(c_derived) / R
            else:
                kappa_derived = float('nan')

            rel_error_c = (c_derived - c_target) / c_target * 100
            rel_error_kappa = (kappa_derived - kappa_target) / kappa_target * 100 if not math.isnan(kappa_derived) else float('nan')

            print(f"\n  Theoretical link ({bracket_name} bracket):")
            print(f"    bracket = {bracket:.10f}")
            print(f"    (exp(2R) + m)×S12(-R) = {coeff_mirror:.4f} × {S12_minus:.6f} = {coeff_mirror * S12_minus:.6f}")
            print(f"    (-2Rθ)×bracket        = {coeff_bracket:.4f} × {bracket:.6f} = {coeff_bracket * bracket:.6f}")
            print(f"    S34(+R)               = {S34_plus:.6f}")
            print(f"    c_derived = {c_derived:.6f}")
            print(f"    c_target  = {c_target:.6f}")
            print(f"    c error   = {rel_error_c:+.2f}%")
            print(f"    κ_derived = {kappa_derived:.6f}")
            print(f"    κ_target  = {kappa_target:.6f}")
            print(f"    κ error   = {rel_error_kappa:+.2f}%")

        # Reverse check: what bracket would be needed for c_target?
        bracket_needed = (c_target - coeff_mirror * S12_minus - S34_plus) / coeff_bracket
        print(f"\n  Reverse engineering:")
        print(f"    bracket_needed for c_target = {bracket_needed:.10f}")
        print(f"    bracket_legacy             = {bracket_legacy:.10f} (ratio: {bracket_legacy/bracket_needed:.4f})")
        print(f"    bracket_frozen             = {bracket_frozen:.10f} (ratio: {bracket_frozen/bracket_needed:.4f})")


if __name__ == "__main__":
    test_t_integral_factor()
    test_frozen_vs_legacy_q()
    test_theoretical_link_formula()
