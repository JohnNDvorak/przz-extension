#!/usr/bin/env python3
"""
Production script for computing κ using the derived m formula.

The m formula is fully derived from PRZZ first principles:

    m(K, R) = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]

Components:
-----------
1. exp(R): From difference quotient T^{-(α+β)} at α=β=-R/L (PRZZ line 1502-1511)
2. (2K-1): From unified bracket B/A ratio (Phase 32)
3. 1+θ/(2K(2K+1)): From product rule cross-terms on log factor (Phase 34C)
   - This equals 1 + θ × Beta(2, 2K) where Beta(2, 2K) = 1/(2K(2K+1))
   - Arises from: d²/dxdy[(1/θ + x + y)×F] = (1/θ)×F_xy + F_x + F_y
   - Integration with (1-u)^{2K-1} weights yields the Beta moment

Accuracy: ±0.15% on both κ and κ* benchmarks
Residual source: Q polynomial interaction (Phase 35 analysis)

Assembly:
---------
    c = S12(+R) + m × S12(-R) + S34(+R)
    κ = 1 - log(c)/R

Created: 2025-12-26 (Phase 36)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.decomposition import compute_decomposition, compute_mirror_multiplier


# Benchmark targets from PRZZ
BENCHMARKS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.137454406132173,
        "kappa_target": 0.417293962,
        "loader": load_przz_polynomials,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.9381,  # exp(1.1167 * (1 - 0.407511457))
        "kappa_target": 0.407511457,  # From PRZZ paper
        "loader": load_przz_polynomials_kappa_star,
    },
}


def print_derivation_summary():
    """Print the derivation summary for the m formula."""
    theta = 4 / 7
    K = 3

    print("=" * 80)
    print("PRODUCTION KAPPA COMPUTATION (Derived Formula)")
    print("=" * 80)
    print()
    print("Formula: m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]")
    print()
    print("DERIVATION SUMMARY")
    print("-" * 80)
    print(f"  Component 1: exp(R)")
    print(f"    Source:    Difference quotient T^{{-(α+β)}} at α=β=-R/L")
    print(f"    PRZZ Ref:  Lines 1502-1511")
    print()
    print(f"  Component 2: (2K-1) = {2*K-1}")
    print(f"    Source:    Unified bracket B/A ratio")
    print(f"    Verified:  Phase 32 polynomial ladder (17 tests)")
    print()
    print(f"  Component 3: 1 + θ/(2K(2K+1)) = 1 + {theta:.6f}/{2*K*(2*K+1)} = {1 + theta/(2*K*(2*K+1)):.8f}")
    print(f"    Source:    Product rule cross-terms on log factor")
    print(f"    PRZZ Ref:  Lines 1530, 2391-2409, 2472")
    print(f"    Math:      Equals 1 + θ × Beta(2, 2K)")
    print()


def run_benchmark(benchmark_key: str, n_quad: int = 60) -> dict:
    """
    Run a single benchmark and return results.

    Args:
        benchmark_key: "kappa" or "kappa_star"
        n_quad: Quadrature points

    Returns:
        Dictionary with computed values and gaps
    """
    bm = BENCHMARKS[benchmark_key]
    theta = 4 / 7
    K = 3

    # Load polynomials
    P1, P2, P3, Q = bm["loader"]()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute decomposition with derived formula
    decomp = compute_decomposition(
        theta=theta,
        R=bm["R"],
        K=K,
        polynomials=polynomials,
        kernel_regime="paper",
        n_quad=n_quad,
        mirror_formula="derived",
    )

    # Extract values
    c = decomp.total
    kappa = 1 - math.log(c) / bm["R"]

    # Calculate gaps
    c_gap_pct = (c / bm["c_target"] - 1) * 100
    kappa_gap_pp = (kappa - bm["kappa_target"]) * 100  # percentage points

    return {
        "name": bm["name"],
        "R": bm["R"],
        "c_computed": c,
        "c_target": bm["c_target"],
        "c_gap_pct": c_gap_pct,
        "kappa_computed": kappa,
        "kappa_target": bm["kappa_target"],
        "kappa_gap_pp": kappa_gap_pp,
        "decomp": decomp,
    }


def print_benchmark_result(result: dict):
    """Print results for a single benchmark."""
    print("=" * 80)
    print(f"BENCHMARK: {result['name']} (R={result['R']})")
    print("=" * 80)
    print()
    print(f"  c_computed:  {result['c_computed']:.10f}")
    print(f"  c_target:    {result['c_target']:.10f}")
    print(f"  c_gap:       {result['c_gap_pct']:+.4f}%")
    print()
    print(f"  κ_computed:  {result['kappa_computed']:.10f}")
    print(f"  κ_target:    {result['kappa_target']:.10f}")
    print(f"  κ_gap:       {result['kappa_gap_pp']:+.4f} pp")
    print()

    # Show decomposition components
    d = result["decomp"]
    print(f"  Decomposition:")
    print(f"    S12(+R):   {d.S12_plus:+.10f}")
    print(f"    S12(-R):   {d.S12_minus:+.10f}")
    print(f"    S34:       {d.S34:+.10f}")
    print(f"    m:         {d.mirror_mult:.8f}")
    print(f"    Formula:   {d.mirror_formula}")
    print()


def print_summary(results: list):
    """Print summary of all benchmark results."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    all_pass = all(abs(r["c_gap_pct"]) < 0.20 for r in results)

    if all_pass:
        print("  Both benchmarks pass within ±0.20%")
    else:
        print("  WARNING: Some benchmarks exceed ±0.20% gap")

    print()
    print("  Residual source: Q polynomial interaction (Phase 35 analysis)")
    print("    - When P=real, Q=1: formula matches prediction to +0.05%")
    print("    - Adding Q polynomial creates ~0.4% deviation")
    print("    - Net effect: ±0.15% residual after partial cancellation")
    print()

    if all_pass:
        print("  Production formula: VALIDATED")
    else:
        print("  Production formula: NEEDS INVESTIGATION")
    print()


def main():
    """Main entry point."""
    # Print derivation summary
    print_derivation_summary()

    # Run both benchmarks
    results = []
    for benchmark_key in ["kappa", "kappa_star"]:
        result = run_benchmark(benchmark_key)
        results.append(result)
        print_benchmark_result(result)

    # Print summary
    print_summary(results)

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
