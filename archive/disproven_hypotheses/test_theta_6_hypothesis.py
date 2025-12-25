"""
src/test_theta_6_hypothesis.py
Test the θ/6 additive term hypothesis at both R benchmarks.

KEY FINDING from raw_vs_przz_diagnostic:
- Addition needed ≈ (θ/6) × c_raw
- c_target ≈ c_raw × (1 + θ/6)

But we know from two-benchmark test that the gap is R-DEPENDENT.
If (1 + θ/6) works at R=1.3036, does it also work at R=1.1167?

This test will determine if θ/6 is the answer or if we need something more complex.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_c_full, compute_kappa


THETA = 4/7
R1 = 1.3036
R2 = 1.1167

# Target c values
C1_TARGET = 2.13745440613217263636  # For R=1.3036
# For R=1.1167, derive from kappa* ≈ 0.408
KAPPA_STAR = 0.407511457  # From earlier analysis
C2_TARGET = math.exp(R2 * (1 - KAPPA_STAR))


def test_theta_6_at_both_benchmarks(n_quad: int = 60, verbose: bool = True) -> Dict[str, Any]:
    """
    Test if (1 + θ/6) factor works at both R values.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Factor
    factor = 1 + THETA / 6

    results = {}

    for R, c_target, name in [(R1, C1_TARGET, "Benchmark 1"), (R2, C2_TARGET, "Benchmark 2")]:
        result = evaluate_c_full(THETA, R, n_quad, polys, mode="main", return_breakdown=False)
        c_raw = result.total

        # Apply (1 + θ/6) factor
        c_corrected = c_raw * factor

        # Compare with target
        gap_raw = (c_raw - c_target) / c_target * 100
        gap_corrected = (c_corrected - c_target) / c_target * 100

        results[name] = {
            "R": R,
            "c_raw": c_raw,
            "c_corrected": c_corrected,
            "c_target": c_target,
            "gap_raw_pct": gap_raw,
            "gap_corrected_pct": gap_corrected,
        }

    if verbose:
        print("\n" + "=" * 70)
        print("TEST: (1 + θ/6) FACTOR AT BOTH BENCHMARKS")
        print("=" * 70)

        print(f"\nFactor = 1 + θ/6 = 1 + {THETA:.6f}/6 = {factor:.10f}")

        for name in ["Benchmark 1", "Benchmark 2"]:
            r = results[name]
            print(f"\n--- {name}: R = {r['R']} ---")
            print(f"  c_raw:        {r['c_raw']:.10f}")
            print(f"  c_corrected:  {r['c_corrected']:.10f}")
            print(f"  c_target:     {r['c_target']:.10f}")
            print(f"  Gap (raw):    {r['gap_raw_pct']:+.4f}%")
            print(f"  Gap (corrected): {r['gap_corrected_pct']:+.4f}%")

        print(f"\n--- ANALYSIS ---")
        gap1 = results["Benchmark 1"]["gap_corrected_pct"]
        gap2 = results["Benchmark 2"]["gap_corrected_pct"]

        print(f"  Benchmark 1 corrected gap: {gap1:+.4f}%")
        print(f"  Benchmark 2 corrected gap: {gap2:+.4f}%")
        print(f"  Difference: {abs(gap1 - gap2):.4f}%")

        if abs(gap1) < 1.0 and abs(gap2) < 1.0:
            print(f"\n  *** (1 + θ/6) WORKS AT BOTH BENCHMARKS! ***")
            print(f"  This suggests a missing global factor, not R-dependent structure.")
        elif abs(gap1) < 1.0 and abs(gap2) > 1.0:
            print(f"\n  (1 + θ/6) only works at Benchmark 1")
            print(f"  The gap is R-DEPENDENT - need different approach.")
        else:
            print(f"\n  (1 + θ/6) doesn't match at either benchmark well.")

        print("=" * 70)

    return results


def scan_factor_both_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """
    Scan different factors to find what matches both benchmarks.

    If no single factor works for both, the gap is R-dependent.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    n_quad = 60

    # Get raw c at both R values
    c1_raw = evaluate_c_full(THETA, R1, n_quad, polys, mode="main", return_breakdown=False).total
    c2_raw = evaluate_c_full(THETA, R2, n_quad, polys, mode="main", return_breakdown=False).total

    # Factor needed at each benchmark
    factor1_needed = C1_TARGET / c1_raw
    factor2_needed = C2_TARGET / c2_raw

    # Scan factors
    factors_to_test = [
        ("1", 1.0),
        ("1 + θ/6", 1 + THETA/6),
        ("1 + θ/5", 1 + THETA/5),
        ("1 + θ/7", 1 + THETA/7),
        ("1 + θ²/6", 1 + THETA**2/6),
        ("1 + 0.1", 1.1),
        ("factor1_needed", factor1_needed),
        ("factor2_needed", factor2_needed),
    ]

    results = []
    for name, factor in factors_to_test:
        c1_corr = c1_raw * factor
        c2_corr = c2_raw * factor

        gap1 = (c1_corr - C1_TARGET) / C1_TARGET * 100
        gap2 = (c2_corr - C2_TARGET) / C2_TARGET * 100

        results.append({
            "name": name,
            "factor": factor,
            "gap1_pct": gap1,
            "gap2_pct": gap2,
            "max_gap": max(abs(gap1), abs(gap2)),
        })

    # Sort by max gap
    results.sort(key=lambda x: x["max_gap"])

    if verbose:
        print("\n" + "=" * 70)
        print("FACTOR SCAN: WHAT WORKS FOR BOTH BENCHMARKS?")
        print("=" * 70)

        print(f"\n  Factor needed at R=1.3036: {factor1_needed:.10f}")
        print(f"  Factor needed at R=1.1167: {factor2_needed:.10f}")
        print(f"  Factor difference: {abs(factor1_needed - factor2_needed):.10f}")
        print(f"  Relative difference: {abs(factor1_needed - factor2_needed)/factor1_needed*100:.2f}%")

        print(f"\n--- Factor Scan ---")
        print(f"  {'Factor':25} {'Value':>12} {'Gap1':>10} {'Gap2':>10} {'Max':>10}")
        print("  " + "-" * 70)

        for r in results:
            print(f"  {r['name']:25} {r['factor']:>12.6f} {r['gap1_pct']:>+9.4f}% {r['gap2_pct']:>+9.4f}% {r['max_gap']:>9.4f}%")

        print(f"\n--- CONCLUSION ---")
        best = results[0]
        if best["max_gap"] < 1.0:
            print(f"  Best factor '{best['name']}' gets both gaps < 1%")
        else:
            print(f"  No single factor gets both gaps < 1%")
            print(f"  The gap is inherently R-DEPENDENT")
            print(f"  → Cannot be fixed with a global multiplicative factor")

        print("=" * 70)

    return {
        "factor1_needed": factor1_needed,
        "factor2_needed": factor2_needed,
        "scan_results": results,
    }


if __name__ == "__main__":
    # Test θ/6 hypothesis
    test_theta_6_at_both_benchmarks(verbose=True)

    # Scan factors
    scan_factor_both_benchmarks(verbose=True)
