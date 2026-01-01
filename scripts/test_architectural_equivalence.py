#!/usr/bin/env python3
"""
Phase 59: Architectural Equivalence Test
=========================================

Compare unified S12 vs split-channel S12(+R) + m × S12(-R)

If these match to machine precision, the "derivation gap" becomes a
notational translation, not a physical discrepancy.

Created: 2025-12-29
"""

import sys
import math
sys.path.insert(0, "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension")

from src.unified_s12_evaluator_v3 import compute_S12_unified_v3
from src.kappa_engine import KappaEngine, compute_mirror_multiplier
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def test_s12_equivalence():
    """
    Compare S12 from both architectures.

    Method 1 (Unified): Uses difference quotient identity that combines
    direct+mirror via t-integral, then normalizes by F(R)/2

    Method 2 (Split-channel): Computes S12(+R), S12(-R) separately,
    combines with m = g_total × (exp(R) + 5)
    """
    theta = 4/7
    K = 3

    print("=" * 70)
    print("PHASE 59: ARCHITECTURAL EQUIVALENCE TEST")
    print("Comparing unified S12 vs split-channel S12(+R) + m × S12(-R)")
    print("=" * 70)

    results = []

    for benchmark, R, load_fn, engine_fn in [
        ("kappa", 1.3036, load_przz_polynomials, KappaEngine.from_przz_kappa),
        ("kappa_star", 1.1167, load_przz_polynomials_kappa_star, KappaEngine.from_przz_kappa_star),
    ]:
        print(f"\n--- {benchmark.upper()} (R={R}) ---")

        # Load polynomials
        P1, P2, P3, Q = load_fn(enforce_Q0=False)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q.to_monomial()}

        # Method 1: Unified (with scalar normalization)
        print("  Computing unified S12...")
        unified_result = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polys,
            n_quad_u=80, n_quad_t=80,
            normalization_mode="scalar"
        )
        S12_unified = unified_result.S12_total

        # Method 2: Split-channel
        print("  Computing split-channel S12...")
        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()
        corrections = compute_mirror_multiplier(theta, K, R, integrals.f_I1)
        S12_split = integrals.S12_plus + corrections.m * integrals.S12_minus

        # Compare
        rel_error = abs(S12_split - S12_unified) / abs(S12_split)

        print(f"\n  Results:")
        print(f"    S12_unified = {S12_unified:.10f}")
        print(f"    S12_split   = {S12_split:.10f}")
        print(f"    Difference  = {abs(S12_split - S12_unified):.2e}")
        print(f"    Rel error   = {rel_error:.2e} ({rel_error*100:.4f}%)")

        # Breakdown for diagnostics
        print(f"\n  Split-channel breakdown:")
        print(f"    S12(+R) = {integrals.S12_plus:.10f}")
        print(f"    S12(-R) = {integrals.S12_minus:.10f}")
        print(f"    m       = {corrections.m:.10f}")
        print(f"    m×S12(-R) = {corrections.m * integrals.S12_minus:.10f}")

        print(f"\n  Unified normalization:")
        print(f"    S12_unnormalized = {unified_result.S12_unnormalized:.10f}")
        print(f"    F(R)/2 factor    = {unified_result.scalar_baseline_factor:.10f}")

        results.append({
            "benchmark": benchmark,
            "R": R,
            "S12_unified": S12_unified,
            "S12_split": S12_split,
            "rel_error": rel_error,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Benchmark | R | S12_unified | S12_split | rel_error | Status |")
    print("|-----------|------|-------------|-----------|-----------|--------|")

    for r in results:
        if r["rel_error"] < 1e-6:
            status = "EQUIVALENT"
        elif r["rel_error"] < 1e-3:
            status = "APPROX"
        elif r["rel_error"] < 0.05:
            status = "SIMILAR"
        else:
            status = "DIFFERENT"

        print(f"| {r['benchmark']:9} | {r['R']:.4f} | {r['S12_unified']:.6f} | {r['S12_split']:.6f} | {r['rel_error']:.2e} | {status} |")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    avg_error = sum(r["rel_error"] for r in results) / len(results)

    if avg_error < 1e-6:
        print("\n  EQUIVALENT to machine precision")
        print("  → Derivation gap is closed: split-channel ≡ unified")
        print("  → m = exp(R) + (2K-1) is the exact PRZZ t-integral result")
    elif avg_error < 1e-3:
        print("\n  APPROXIMATELY equivalent")
        print("  → Minor numerical differences (quadrature, normalization)")
        print("  → Derivation gap is small; architectures are consistent")
    elif avg_error < 0.05:
        print("\n  SIMILAR but not equivalent")
        print("  → m = exp(R) + (2K-1) is a good approximation")
        print("  → Derivation gap exists but is controlled")
    else:
        print("\n  DIFFERENT architectures")
        print("  → Significant discrepancy requires investigation")
        print("  → The unified and split-channel compute different things")

    print(f"\n  Average relative error: {avg_error:.2e}")

    return results


if __name__ == "__main__":
    test_s12_equivalence()
