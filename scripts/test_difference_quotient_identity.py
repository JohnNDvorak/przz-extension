#!/usr/bin/env python3
"""
Phase 59B: Verify Difference Quotient Identity
===============================================

The PRZZ difference quotient identity (lines 1502-1511) is:
    [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

At α = β = -Rθ, this becomes:
    LHS = [S12(+R) - exp(2R) × S12(-R)] / (-2Rθ)
    RHS = unified_bracket (integrated over t)

This test verifies that the unified bracket correctly implements the
DIFFERENCE QUOTIENT identity (subtraction), NOT the mirror assembly
formula (addition).

NOTE: The mirror assembly c = S12(+R) + m × S12(-R) is a DIFFERENT formula
that is empirically validated. It is NOT derived from the DQ identity.

Created: 2025-12-29
"""

import sys
import math
sys.path.insert(0, "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension")

from src.unified_s12_evaluator_v3 import compute_S12_unified_v3
from src.kappa_engine import KappaEngine
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def test_difference_quotient_identity():
    """
    Verify: [S12(+R) - exp(2R) × S12(-R)] / (-2Rθ) ≈ unified_bracket

    The unified bracket is the RHS of the DQ identity.
    The LHS computes the difference quotient from split-channel values.
    """
    theta = 4/7

    print("=" * 70)
    print("PHASE 59B: DIFFERENCE QUOTIENT IDENTITY TEST")
    print("Verify: [S12(+R) - exp(2R)×S12(-R)] / (-2Rθ) ≈ unified_bracket")
    print("=" * 70)

    for benchmark, R, load_fn, engine_fn in [
        ("kappa", 1.3036, load_przz_polynomials, KappaEngine.from_przz_kappa),
        ("kappa_star", 1.1167, load_przz_polynomials_kappa_star, KappaEngine.from_przz_kappa_star),
    ]:
        print(f"\n--- {benchmark.upper()} (R={R}) ---")

        # Load polynomials
        P1, P2, P3, Q = load_fn(enforce_Q0=False)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q.to_monomial()}

        # Compute split-channel values
        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()

        S12_plus = integrals.S12_plus
        S12_minus = integrals.S12_minus
        exp_2R = math.exp(2 * R)

        # LHS of DQ identity: [S12(+R) - exp(2R) × S12(-R)] / (-2Rθ)
        numerator = S12_plus - exp_2R * S12_minus
        denominator = -2 * R * theta
        LHS = numerator / denominator

        # RHS: Unified bracket (unnormalized)
        unified_result = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polys,
            n_quad_u=80, n_quad_t=80,
            normalization_mode="none"  # Get raw bracket value
        )
        RHS_raw = unified_result.S12_total

        # The unified bracket integrates over t, which gives factor F(R) = (exp(2R)-1)/(2R)
        # So the bracket output should be compared to LHS × some factor
        F_R = (math.exp(2*R) - 1) / (2 * R)

        print(f"\n  Split-channel values:")
        print(f"    S12(+R)     = {S12_plus:.10f}")
        print(f"    S12(-R)     = {S12_minus:.10f}")
        print(f"    exp(2R)     = {exp_2R:.10f}")

        print(f"\n  DQ identity LHS:")
        print(f"    S12(+R) - exp(2R)×S12(-R) = {numerator:.10f}")
        print(f"    / (-2Rθ) = / {denominator:.10f}")
        print(f"    LHS = {LHS:.10f}")

        print(f"\n  Unified bracket (RHS):")
        print(f"    RHS_raw = {RHS_raw:.10f}")
        print(f"    F(R) = (exp(2R)-1)/(2R) = {F_R:.10f}")

        # Try different normalization comparisons
        print(f"\n  Comparison attempts:")
        print(f"    LHS / RHS_raw = {LHS / RHS_raw:.6f}")
        print(f"    LHS × F(R) / RHS_raw = {LHS * F_R / RHS_raw:.6f}")
        print(f"    RHS_raw / LHS = {RHS_raw / LHS:.6f}")
        print(f"    RHS_raw / (LHS × F(R)) = {RHS_raw / (LHS * F_R):.6f}")

        # The unified bracket multiplies by log factor L(1+θ(x+y))
        # At xy extraction, this gives a factor related to 1/θ
        print(f"\n  With θ factor:")
        print(f"    LHS × θ / RHS_raw = {LHS * theta / RHS_raw:.6f}")
        print(f"    LHS / (θ × RHS_raw) = {LHS / (theta * RHS_raw):.6f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("""
  The unified bracket computes the RHS of the difference quotient identity.
  The LHS is [S12(+R) - exp(2R)×S12(-R)] / (-2Rθ).

  These should match (up to normalization factors).

  IMPORTANT: This is DIFFERENT from the mirror assembly formula:
    c = S12(+R) + m × S12(-R)

  The DQ identity uses SUBTRACTION (Direct - exp(2R)×Mirror).
  The assembly formula uses ADDITION (Direct + m×Mirror).

  The m = exp(R) + 5 formula is an EMPIRICAL assembly rule, not derived
  from the DQ identity directly.
""")


if __name__ == "__main__":
    test_difference_quotient_identity()
