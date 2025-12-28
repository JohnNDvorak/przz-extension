#!/usr/bin/env python3
"""
Phase 43: Component-Separated Mirror Test

Tests whether applying different mirror multipliers to I1 and I2 can
eliminate the ±0.15% residual.

Hypothesis: The Beta moment correction θ/(2K(2K+1)) was derived from the
log factor L = 1/θ + x + y, which only appears in I1 (derivative term).
I2 has no log factor, so it shouldn't get this correction.

Formula tested:
  c = I1(+R) + m_I1 × I1(-R) + I2(+R) + m_I2 × I2(-R) + S34

where:
  m_I1 = g_I1 × base  (with correction)
  m_I2 = g_I2 × base  (different correction or no correction)

Created: 2025-12-27 (Phase 43)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


@dataclass
class SeparatedMirrorResult:
    """Result of separated mirror test."""
    benchmark: str
    R: float
    c_target: float

    # Components
    I1_plus: float
    I1_minus: float
    I2_plus: float
    I2_minus: float
    S34: float

    # Multipliers used
    g_I1: float
    g_I2: float
    m_I1: float
    m_I2: float

    # Results
    c_computed: float
    c_gap_pct: float

    # Comparison
    c_baseline: float
    c_baseline_gap_pct: float


def compute_S34(theta: float, R: float, polynomials: Dict, n_quad: int = 60) -> float:
    """Compute S34 = I3 + I4 (no mirror)."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += full_norm * result.value

    return S34


def test_separated_mirror(
    benchmark: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    g_I1: float,
    g_I2: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> SeparatedMirrorResult:
    """Test separated mirror formula."""
    # Compute I1 and I2 at +R and -R
    I1_plus, I2_plus = compute_I1_I2_totals(R, theta, polynomials, n_quad)
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)

    # Compute S34
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # Compute separated multipliers
    base = math.exp(R) + (2 * K - 1)
    m_I1 = g_I1 * base
    m_I2 = g_I2 * base

    # Assemble c with separated multipliers
    c_computed = I1_plus + m_I1 * I1_minus + I2_plus + m_I2 * I2_minus + S34
    c_gap_pct = (c_computed / c_target - 1) * 100

    # Also compute baseline for comparison
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    m_baseline = g_baseline * base
    S12_plus = I1_plus + I2_plus
    S12_minus = I1_minus + I2_minus
    c_baseline = S12_plus + m_baseline * S12_minus + S34
    c_baseline_gap_pct = (c_baseline / c_target - 1) * 100

    return SeparatedMirrorResult(
        benchmark=benchmark,
        R=R,
        c_target=c_target,
        I1_plus=I1_plus,
        I1_minus=I1_minus,
        I2_plus=I2_plus,
        I2_minus=I2_minus,
        S34=S34,
        g_I1=g_I1,
        g_I2=g_I2,
        m_I1=m_I1,
        m_I2=m_I2,
        c_computed=c_computed,
        c_gap_pct=c_gap_pct,
        c_baseline=c_baseline,
        c_baseline_gap_pct=c_baseline_gap_pct,
    )


def print_comparison(results: List[SeparatedMirrorResult], config_name: str):
    """Print comparison table."""
    print()
    print(f"Configuration: {config_name}")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'g_I1':<10} | {'g_I2':<10} | {'c_gap':<12} | {'baseline_gap':<12} | {'improvement':<12}")
    print("-" * 90)
    for r in results:
        improvement = abs(r.c_baseline_gap_pct) - abs(r.c_gap_pct)
        print(f"{r.benchmark:<12} | {r.g_I1:<10.6f} | {r.g_I2:<10.6f} | {r.c_gap_pct:+12.4f}% | {r.c_baseline_gap_pct:+12.4f}% | {improvement:+12.4f}%")


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print()
    print("=" * 90)
    print("PHASE 43: COMPONENT-SEPARATED MIRROR TEST")
    print("=" * 90)
    print()
    print(f"Baseline g = 1 + θ/(2K(2K+1)) = {g_baseline:.6f}")
    print()
    print("Testing different (g_I1, g_I2) configurations...")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Configuration 1: Current baseline (both get g_baseline)
    print("=" * 90)
    print("CONFIG 1: Baseline (g_I1 = g_I2 = g_baseline)")
    print("=" * 90)
    results_1 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              g_baseline, g_baseline, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              g_baseline, g_baseline, theta, K, n_quad),
    ]
    print_comparison(results_1, "g_I1 = g_I2 = g_baseline")

    # Configuration 2: I1 gets baseline, I2 gets 1.0
    print()
    print("=" * 90)
    print("CONFIG 2: Only I1 gets Beta correction (g_I1 = g_baseline, g_I2 = 1.0)")
    print("=" * 90)
    results_2 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              g_baseline, 1.0, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              g_baseline, 1.0, theta, K, n_quad),
    ]
    print_comparison(results_2, "g_I1 = g_baseline, g_I2 = 1.0")

    # Configuration 3: I1 gets 1.0, I2 gets baseline (inverted)
    print()
    print("=" * 90)
    print("CONFIG 3: Inverted (g_I1 = 1.0, g_I2 = g_baseline)")
    print("=" * 90)
    results_3 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              1.0, g_baseline, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              1.0, g_baseline, theta, K, n_quad),
    ]
    print_comparison(results_3, "g_I1 = 1.0, g_I2 = g_baseline")

    # Configuration 4: Both get 1.0 (no Beta correction)
    print()
    print("=" * 90)
    print("CONFIG 4: No correction (g_I1 = g_I2 = 1.0)")
    print("=" * 90)
    results_4 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              1.0, 1.0, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              1.0, 1.0, theta, K, n_quad),
    ]
    print_comparison(results_4, "g_I1 = g_I2 = 1.0")

    # Configuration 5: Higher g_I1, lower g_I2
    g_I1_high = g_baseline * 1.02
    g_I2_low = g_baseline * 0.98
    print()
    print("=" * 90)
    print(f"CONFIG 5: Adjusted (g_I1 = {g_I1_high:.6f}, g_I2 = {g_I2_low:.6f})")
    print("=" * 90)
    results_5 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              g_I1_high, g_I2_low, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              g_I1_high, g_I2_low, theta, K, n_quad),
    ]
    print_comparison(results_5, f"g_I1 = {g_I1_high:.4f}, g_I2 = {g_I2_low:.4f}")

    # Configuration 6: Inverse of Config 5
    g_I1_low = g_baseline * 0.98
    g_I2_high = g_baseline * 1.02
    print()
    print("=" * 90)
    print(f"CONFIG 6: Inverse adjusted (g_I1 = {g_I1_low:.6f}, g_I2 = {g_I2_high:.6f})")
    print("=" * 90)
    results_6 = [
        test_separated_mirror("kappa", 1.3036, c_target_kappa, polynomials_kappa,
                              g_I1_low, g_I2_high, theta, K, n_quad),
        test_separated_mirror("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star,
                              g_I1_low, g_I2_high, theta, K, n_quad),
    ]
    print_comparison(results_6, f"g_I1 = {g_I1_low:.4f}, g_I2 = {g_I2_high:.4f}")

    # Compute what g_I1 and g_I2 would need to be to hit target for each benchmark
    print()
    print("=" * 90)
    print("COMPUTING OPTIMAL g VALUES FOR EACH BENCHMARK")
    print("=" * 90)

    for name, R, c_target, polys in [
        ("kappa", 1.3036, c_target_kappa, polynomials_kappa),
        ("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star),
    ]:
        I1_plus, I2_plus = compute_I1_I2_totals(R, theta, polys, n_quad)
        I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polys, n_quad)
        S34 = compute_S34(theta, R, polys, n_quad)
        base = math.exp(R) + (2 * K - 1)

        # Current baseline
        S12_plus = I1_plus + I2_plus
        S12_minus = I1_minus + I2_minus

        # If g_I1 = g_I2 = g, solve for g to hit target
        # c = S12_plus + g*base*S12_minus + S34 = c_target
        # g = (c_target - S12_plus - S34) / (base * S12_minus)
        g_needed_uniform = (c_target - S12_plus - S34) / (base * S12_minus)

        # If g_I2 = 1.0, solve for g_I1 to hit target
        # c = I1_plus + g_I1*base*I1_minus + I2_plus + 1.0*base*I2_minus + S34 = c_target
        # g_I1 = (c_target - I1_plus - I2_plus - base*I2_minus - S34) / (base * I1_minus)
        g_I1_needed = (c_target - I1_plus - I2_plus - base * I2_minus - S34) / (base * I1_minus)

        # If g_I1 = g_baseline, solve for g_I2 to hit target
        # c = I1_plus + g_baseline*base*I1_minus + I2_plus + g_I2*base*I2_minus + S34 = c_target
        # g_I2 = (c_target - I1_plus - g_baseline*base*I1_minus - I2_plus - S34) / (base * I2_minus)
        g_I2_needed = (c_target - I1_plus - g_baseline * base * I1_minus - I2_plus - S34) / (base * I2_minus)

        print()
        print(f"{name} (R={R}):")
        print(f"  g_needed (uniform):         {g_needed_uniform:.6f} (vs baseline {g_baseline:.6f})")
        print(f"  g_I1_needed (if g_I2=1.0):  {g_I1_needed:.6f}")
        print(f"  g_I2_needed (if g_I1=g_b):  {g_I2_needed:.6f}")
        print(f"  I1_minus/S12_minus:         {I1_minus/S12_minus:.4f} (I1 fraction)")
        print(f"  I2_minus/S12_minus:         {I2_minus/S12_minus:.4f} (I2 fraction)")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print("Best configuration for eliminating residual:")

    # Find best configuration
    all_configs = [
        ("Config 1 (baseline)", results_1),
        ("Config 2 (g_I2=1)", results_2),
        ("Config 3 (g_I1=1)", results_3),
        ("Config 4 (no corr)", results_4),
        ("Config 5 (high/low)", results_5),
        ("Config 6 (low/high)", results_6),
    ]

    best_avg_gap = float('inf')
    best_config = None

    for name, results in all_configs:
        avg_gap = sum(abs(r.c_gap_pct) for r in results) / len(results)
        if avg_gap < best_avg_gap:
            best_avg_gap = avg_gap
            best_config = name

    print(f"  {best_config}: average |gap| = {best_avg_gap:.4f}%")


if __name__ == "__main__":
    main()
