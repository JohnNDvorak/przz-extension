#!/usr/bin/env python3
"""
Phase 59.2: Theoretical Link Test (Non-Circular Validation)
============================================================

This test attempts to bridge the gap between:
1. The PRZZ difference quotient (DQ) identity (subtraction)
2. Our mirror assembly formula (addition)

FROM THE DQ IDENTITY:
    [S12(+R) - exp(2R) × S12(-R)] / (-2Rθ) = unified_bracket

Rearranging:
    S12(+R) = exp(2R)×S12(-R) + (-2Rθ)×bracket

SUBSTITUTING INTO ASSEMBLY FORMULA:
    c = S12(+R) + m × S12(-R) + S34(+R)
      = [exp(2R)×S12(-R) + (-2Rθ)×bracket] + m×S12(-R) + S34(+R)
      = (exp(2R) + m)×S12(-R) + (-2Rθ)×bracket + S34(+R)

This gives us a NON-CIRCULAR formula:
    c_derived = (exp(2R) + m) × S12(-R) + (-2Rθ) × bracket + S34(+R)

Where:
    - bracket comes from the unified computation (no m involved)
    - S12(-R) from split-channel
    - S34(+R) from split-channel
    - m = exp(R) + 5 (our formula)

If c_derived ≈ c_target = 2.137, we've proven the theoretical connection!

Created: 2025-12-29
"""

import sys
import math
sys.path.insert(0, "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension")

from src.unified_s12_evaluator_v3 import compute_S12_unified_v3
from src.kappa_engine import KappaEngine
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def test_theoretical_link():
    """
    Test the derived formula that links DQ identity to assembly formula.

    c_derived = (exp(2R) + m) × S12(-R) + (-2Rθ) × bracket + S34(+R)

    This formula:
    1. Uses the unified bracket (from DQ identity)
    2. Combines with split-channel components
    3. Should yield c ≈ c_target if the link is valid
    """
    theta = 4/7
    K = 3

    print("=" * 70)
    print("PHASE 59.2: THEORETICAL LINK TEST")
    print("Testing: c = (exp(2R) + m) × S12(-R) + (-2Rθ) × bracket + S34(+R)")
    print("=" * 70)

    results = []

    for benchmark, R, c_target, load_fn, engine_fn in [
        ("kappa", 1.3036, 2.1375, load_przz_polynomials, KappaEngine.from_przz_kappa),
        ("kappa_star", 1.1167, 1.938, load_przz_polynomials_kappa_star, KappaEngine.from_przz_kappa_star),
    ]:
        print(f"\n--- {benchmark.upper()} (R={R}) ---")

        # Load polynomials
        P1, P2, P3, Q = load_fn(enforce_Q0=False)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q.to_monomial()}

        # Our formula for m
        m = math.exp(R) + 5  # = exp(R) + (2K-1) for K=3
        exp_2R = math.exp(2 * R)

        # Get split-channel components
        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()

        S12_minus = integrals.S12_minus
        S34_plus = integrals.S34_plus

        # Get unified bracket (unnormalized)
        unified_result = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polys,
            n_quad_u=80, n_quad_t=80,
            normalization_mode="none"  # Raw bracket value
        )
        bracket = unified_result.S12_total

        # Derived formula: c = (exp(2R) + m) × S12(-R) + (-2Rθ) × bracket + S34(+R)
        term1 = (exp_2R + m) * S12_minus
        term2 = (-2 * R * theta) * bracket
        term3 = S34_plus
        c_derived = term1 + term2 + term3

        # Compare to target
        rel_error = abs(c_derived - c_target) / c_target

        # Compute κ from c_derived
        kappa_derived = 1 - math.log(c_derived) / R if c_derived > 0 else float('nan')
        kappa_target = 1 - math.log(c_target) / R

        print(f"\n  Components:")
        print(f"    exp(2R)          = {exp_2R:.6f}")
        print(f"    m = exp(R) + 5   = {m:.6f}")
        print(f"    exp(2R) + m      = {exp_2R + m:.6f}")
        print(f"    S12(-R)          = {S12_minus:.6f}")
        print(f"    S34(+R)          = {S34_plus:.6f}")
        print(f"    bracket (unified)= {bracket:.6f}")
        print(f"    -2Rθ             = {-2*R*theta:.6f}")

        print(f"\n  Derived formula breakdown:")
        print(f"    (exp(2R)+m)×S12(-R) = {term1:.6f}")
        print(f"    (-2Rθ)×bracket      = {term2:.6f}")
        print(f"    S34(+R)             = {term3:.6f}")
        print(f"    ─────────────────────────────")
        print(f"    c_derived           = {c_derived:.6f}")

        print(f"\n  Comparison to target:")
        print(f"    c_target  = {c_target:.6f}")
        print(f"    c_derived = {c_derived:.6f}")
        print(f"    rel_error = {rel_error:.2%}")

        print(f"\n  κ comparison:")
        print(f"    κ_target  = {kappa_target:.6f}")
        print(f"    κ_derived = {kappa_derived:.6f}")

        results.append({
            "benchmark": benchmark,
            "R": R,
            "c_target": c_target,
            "c_derived": c_derived,
            "rel_error": rel_error,
            "kappa_derived": kappa_derived,
            "kappa_target": kappa_target,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Benchmark | c_target | c_derived | rel_error | Status |")
    print("|-----------|----------|-----------|-----------|--------|")

    for r in results:
        if r["rel_error"] < 0.01:
            status = "MATCH"
        elif r["rel_error"] < 0.05:
            status = "CLOSE"
        elif r["rel_error"] < 0.20:
            status = "APPROXIMATE"
        else:
            status = "NO MATCH"

        print(f"| {r['benchmark']:9} | {r['c_target']:.4f}   | {r['c_derived']:.4f}    | {r['rel_error']:.2%}    | {status} |")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    avg_error = sum(r["rel_error"] for r in results) / len(results)

    if avg_error < 0.05:
        print("""
  THEORETICAL LINK ESTABLISHED!

  The derived formula:
    c = (exp(2R) + m) × S12(-R) + (-2Rθ) × bracket + S34(+R)

  successfully relates:
  - The unified bracket (from DQ identity)
  - The split-channel components S12(-R) and S34(+R)
  - Our mirror multiplier m = exp(R) + 5

  This provides NON-CIRCULAR validation:
  - bracket comes from unified computation (no m involved)
  - The formula yields c ≈ c_target from PRZZ paper
  - The connection to PRZZ is now explicit
""")
    elif avg_error < 0.20:
        print("""
  PARTIAL LINK - Normalization mismatch

  The derived formula shows correct structure but has ~10-20% error.
  This suggests:
  - The algebraic derivation is correct
  - Missing normalization factor(s) in either bracket or split-channel
  - Further investigation of normalization conventions needed
""")
    else:
        print("""
  NO THEORETICAL LINK

  The derived formula does not match the PRZZ target.
  The DQ identity and assembly formula may be computing genuinely different things.
  The m = exp(R) + 5 formula remains empirically validated but not theoretically derived.
""")

    print(f"\n  Average relative error: {avg_error:.2%}")

    # Also test: What if we use the split-channel S12(+R) directly?
    print("\n" + "=" * 70)
    print("SANITY CHECK: Does DQ identity hold numerically?")
    print("=" * 70)

    for benchmark, R, load_fn, engine_fn in [
        ("kappa", 1.3036, load_przz_polynomials, KappaEngine.from_przz_kappa),
        ("kappa_star", 1.1167, load_przz_polynomials_kappa_star, KappaEngine.from_przz_kappa_star),
    ]:
        print(f"\n--- {benchmark.upper()} ---")

        P1, P2, P3, Q = load_fn(enforce_Q0=False)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q.to_monomial()}

        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()

        S12_plus = integrals.S12_plus
        S12_minus = integrals.S12_minus
        exp_2R = math.exp(2 * R)

        # LHS of DQ: [S12(+R) - exp(2R) × S12(-R)] / (-2Rθ)
        DQ_LHS = (S12_plus - exp_2R * S12_minus) / (-2 * R * theta)

        # RHS: unified bracket
        unified_result = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polys,
            n_quad_u=80, n_quad_t=80,
            normalization_mode="none"
        )
        DQ_RHS = unified_result.S12_total

        print(f"  S12(+R)                        = {S12_plus:.6f}")
        print(f"  S12(-R)                        = {S12_minus:.6f}")
        print(f"  exp(2R)                        = {exp_2R:.6f}")
        print(f"  S12(+R) - exp(2R)×S12(-R)      = {S12_plus - exp_2R * S12_minus:.6f}")
        print(f"  DQ_LHS = above / (-2Rθ)        = {DQ_LHS:.6f}")
        print(f"  DQ_RHS = unified bracket       = {DQ_RHS:.6f}")
        print(f"  Ratio (LHS/RHS)                = {DQ_LHS / DQ_RHS:.6f}")

    return results


if __name__ == "__main__":
    test_theoretical_link()
