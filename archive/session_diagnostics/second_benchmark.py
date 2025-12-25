"""
src/second_benchmark.py
Second PRZZ Benchmark: κ* Simple Zeros

The first benchmark (κ with R=1.3036) is insufficient to distinguish
"global normalization factor" from "missing additive term".

PRZZ provides a SECOND benchmark: κ* (simple zeros) with different R.

PRZZ TeX References:
- Lines 2596-2600: κ* simple zeros result
- R* = 1.1167
- κ* = 0.407511457

Key Test:
If the ~1.096 factor appears for BOTH benchmarks, it's likely a
"global normalization" issue.

If the factor differs, it's more likely a "missing term family".
"""

from __future__ import annotations
import math
from typing import Dict

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_c_full


# Second PRZZ benchmark values (κ* simple zeros)
# From PRZZ TeX lines 2596-2600
KAPPA_STAR_TARGET = 0.407511457
R_STAR = 1.1167


def compute_c_from_kappa(kappa: float, R: float) -> float:
    """Compute c from κ using: κ = 1 - log(c)/R → c = exp(R(1-κ))"""
    return math.exp(R * (1 - kappa))


def compute_kappa_from_c(c: float, R: float) -> float:
    """Compute κ from c using: κ = 1 - log(c)/R"""
    return 1 - math.log(c) / R


def test_second_benchmark(
    theta: float = 4/7,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Test our computation against BOTH PRZZ benchmarks.

    Benchmark 1: κ = 0.417293962, R = 1.3036
    Benchmark 2: κ* = 0.407511457, R* = 1.1167

    If the same factor appears for both, it's likely global.
    If factors differ, it's likely a missing term.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Benchmark 1: Original (κ, R)
    R1 = 1.3036
    kappa1_target = 0.417293962
    c1_target = compute_c_from_kappa(kappa1_target, R1)

    result1 = evaluate_c_full(theta, R1, n=n_quad, polynomials=polys, mode="main")
    c1_computed = result1.total
    kappa1_computed = compute_kappa_from_c(c1_computed, R1)
    factor1 = c1_target / c1_computed

    # Benchmark 2: Simple zeros (κ*, R*)
    R2 = R_STAR
    kappa2_target = KAPPA_STAR_TARGET
    c2_target = compute_c_from_kappa(kappa2_target, R2)

    result2 = evaluate_c_full(theta, R2, n=n_quad, polynomials=polys, mode="main")
    c2_computed = result2.total
    kappa2_computed = compute_kappa_from_c(c2_computed, R2)
    factor2 = c2_target / c2_computed

    # Compare factors
    factor_diff = abs(factor1 - factor2) / factor1

    # Reference factor (1 + θ/6)
    ref_factor = 1 + theta/6

    results = {
        "benchmark1": {
            "R": R1,
            "kappa_target": kappa1_target,
            "c_target": c1_target,
            "c_computed": c1_computed,
            "kappa_computed": kappa1_computed,
            "factor": factor1,
        },
        "benchmark2": {
            "R": R2,
            "kappa_target": kappa2_target,
            "c_target": c2_target,
            "c_computed": c2_computed,
            "kappa_computed": kappa2_computed,
            "factor": factor2,
        },
        "factor_diff_percent": factor_diff * 100,
        "ref_factor_1_plus_theta_6": ref_factor,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("TWO-BENCHMARK TEST: Global Factor vs Missing Term")
        print("=" * 80)

        print(f"\n{'─'*40}")
        print("BENCHMARK 1: κ (PRZZ TeX 2586)")
        print(f"{'─'*40}")
        print(f"  R = {R1}")
        print(f"  κ_target = {kappa1_target}")
        print(f"  c_target = {c1_target:.10f}")
        print(f"  c_computed = {c1_computed:.10f}")
        print(f"  κ_computed = {kappa1_computed:.10f}")
        print(f"  Factor needed: {factor1:.10f}")

        print(f"\n{'─'*40}")
        print("BENCHMARK 2: κ* Simple Zeros (PRZZ TeX 2596-2600)")
        print(f"{'─'*40}")
        print(f"  R* = {R2}")
        print(f"  κ*_target = {kappa2_target}")
        print(f"  c*_target = {c2_target:.10f}")
        print(f"  c*_computed = {c2_computed:.10f}")
        print(f"  κ*_computed = {kappa2_computed:.10f}")
        print(f"  Factor needed: {factor2:.10f}")

        print(f"\n{'─'*40}")
        print("COMPARISON")
        print(f"{'─'*40}")
        print(f"  Factor 1: {factor1:.10f}")
        print(f"  Factor 2: {factor2:.10f}")
        print(f"  Difference: {factor_diff*100:.4f}%")
        print(f"  (1 + θ/6): {ref_factor:.10f}")

        print(f"\n{'─'*40}")
        print("INTERPRETATION")
        print(f"{'─'*40}")
        if factor_diff < 0.01:  # Less than 1% difference
            print("  ✓ Factors are CONSISTENT across benchmarks")
            print("  → Supports 'global normalization' hypothesis")
            print(f"  → Missing factor is approximately {(factor1+factor2)/2:.6f}")
        else:
            print("  ✗ Factors DIFFER between benchmarks")
            print("  → Suggests 'missing term family' rather than global factor")
            print("  → Need to investigate R-dependent terms")

        # Check against (1 + θ/6)
        err1 = abs(factor1 - ref_factor) / ref_factor
        err2 = abs(factor2 - ref_factor) / ref_factor
        print(f"\n  Match to (1+θ/6):")
        print(f"    Benchmark 1: {err1*100:.4f}% off")
        print(f"    Benchmark 2: {err2*100:.4f}% off")

        print("=" * 80)

    return results


def investigate_r_dependence(
    theta: float = 4/7,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Check how the "missing factor" depends on R.

    If the factor is truly global (like a normalization), it should
    be independent of R.

    If it depends on R, we're likely missing an R-dependent term.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Test several R values
    R_values = [1.0, 1.1167, 1.2, 1.3036, 1.4]
    results = []

    for R in R_values:
        result = evaluate_c_full(theta, R, n=n_quad, polynomials=polys, mode="main")
        c_computed = result.total
        results.append({
            "R": R,
            "c_computed": c_computed,
        })

    if verbose:
        print("\n" + "=" * 70)
        print("R-DEPENDENCE OF COMPUTED c")
        print("=" * 70)
        print(f"\n  {'R':>8} | {'c_computed':>15}")
        print(f"  {'-'*8} | {'-'*15}")
        for r in results:
            print(f"  {r['R']:>8.4f} | {r['c_computed']:>15.10f}")

        print(f"\n  Note: c should decrease as R increases (more restrictive bound)")
        print("=" * 70)

    return {"r_scan": results}


if __name__ == "__main__":
    test_second_benchmark(verbose=True)
    investigate_r_dependence(verbose=True)
