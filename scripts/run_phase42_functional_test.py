#!/usr/bin/env python3
"""
Phase 42: Test Derived Functional m_functional = g_total × [exp(R) + (2K-1)]

This tests whether using the per-benchmark g_total from the M/C/g decomposition
reduces the residual compared to using the universal g_baseline.

Created: 2025-12-27 (Phase 42)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_transform_paper_exact import compute_S12_paper_sum
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


# Import the MCG computation function
from scripts.run_phase42_mcg_decomposition import compute_mcg_decomposition


@dataclass
class FunctionalTestResult:
    """Result of functional m test."""
    benchmark: str
    R: float
    c_target: float

    # Components
    S12_plus: float
    S12_minus: float
    S34: float

    # Multipliers
    g_baseline: float
    g_total: float
    m_baseline: float
    m_functional: float

    # Results
    c_baseline: float
    c_functional: float

    # Accuracy
    gap_baseline_pct: float
    gap_functional_pct: float


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

        # I₃ and I₄ are indices 2 and 3
        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += full_norm * result.value

    return S34


def test_functional(
    benchmark: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> FunctionalTestResult:
    """Test the functional m vs baseline m."""
    # Compute S12 at +R and -R
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S12_minus = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n_quad)

    # Compute S34
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # Compute MCG decomposition
    mcg = compute_mcg_decomposition(benchmark, R, theta, polynomials, K, n_quad)

    g_baseline = mcg.g_baseline  # 1 + θ/(2K(2K+1))
    g_total = mcg.g_total        # From MCG decomposition

    # Compute multipliers
    base = math.exp(R) + (2 * K - 1)  # exp(R) + 5 for K=3
    m_baseline = g_baseline * base
    m_functional = g_total * base

    # Assemble c
    c_baseline = S12_plus + m_baseline * S12_minus + S34
    c_functional = S12_plus + m_functional * S12_minus + S34

    # Compute gaps
    gap_baseline_pct = (c_baseline / c_target - 1) * 100
    gap_functional_pct = (c_functional / c_target - 1) * 100

    return FunctionalTestResult(
        benchmark=benchmark,
        R=R,
        c_target=c_target,
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        S34=S34,
        g_baseline=g_baseline,
        g_total=g_total,
        m_baseline=m_baseline,
        m_functional=m_functional,
        c_baseline=c_baseline,
        c_functional=c_functional,
        gap_baseline_pct=gap_baseline_pct,
        gap_functional_pct=gap_functional_pct,
    )


def print_results(results):
    """Print comparison table."""
    print()
    print("=" * 90)
    print("PHASE 42: DERIVED FUNCTIONAL TEST")
    print("=" * 90)
    print()
    print("Formula: m_functional = g_total × [exp(R) + (2K-1)]")
    print("where g_total = 1 + (C_1 + C_2) / (M_1 + M_2) from MCG decomposition")
    print()

    print("G-VALUE COMPARISON")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'g_baseline':<12} | {'g_total':<12} | {'delta_g':<12} | {'delta %':<10}")
    print("-" * 90)
    for r in results:
        delta_g = r.g_total - r.g_baseline
        delta_pct = (r.g_total / r.g_baseline - 1) * 100
        print(f"{r.benchmark:<12} | {r.g_baseline:<12.6f} | {r.g_total:<12.6f} | {delta_g:+12.6f} | {delta_pct:+10.4f}%")
    print()

    print("MULTIPLIER COMPARISON")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'m_baseline':<12} | {'m_functional':<14} | {'delta_m':<12}")
    print("-" * 90)
    for r in results:
        delta_m = r.m_functional - r.m_baseline
        print(f"{r.benchmark:<12} | {r.m_baseline:<12.6f} | {r.m_functional:<14.6f} | {delta_m:+12.6f}")
    print()

    print("ACCURACY COMPARISON (c values)")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'c_target':<12} | {'c_baseline':<12} | {'c_functional':<14}")
    print("-" * 90)
    for r in results:
        print(f"{r.benchmark:<12} | {r.c_target:<12.6f} | {r.c_baseline:<12.6f} | {r.c_functional:<14.6f}")
    print()

    print("GAP ANALYSIS")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'gap_baseline':<14} | {'gap_functional':<16} | {'improvement':<12}")
    print("-" * 90)
    for r in results:
        # Improvement = reduction in |gap|
        improvement = abs(r.gap_baseline_pct) - abs(r.gap_functional_pct)
        print(f"{r.benchmark:<12} | {r.gap_baseline_pct:+14.4f}% | {r.gap_functional_pct:+16.4f}% | {improvement:+12.4f}%")
    print()

    # Summary
    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()
    baseline_avg = sum(abs(r.gap_baseline_pct) for r in results) / len(results)
    functional_avg = sum(abs(r.gap_functional_pct) for r in results) / len(results)

    print(f"Average |gap| with baseline:   {baseline_avg:.4f}%")
    print(f"Average |gap| with functional: {functional_avg:.4f}%")
    print()

    if functional_avg < baseline_avg:
        print("✓ Derived functional IMPROVES accuracy over baseline")
        print(f"  Improvement: {baseline_avg - functional_avg:.4f}% absolute")
    else:
        print("✗ Derived functional does NOT improve (or worsens) accuracy")
        print(f"  Degradation: {functional_avg - baseline_avg:.4f}% absolute")
    print()


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    # c_target values
    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    print()
    print("Computing functional test for both benchmarks...")

    results = []

    # Kappa benchmark
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R_kappa = 1.3036

    result_kappa = test_functional("kappa", R_kappa, c_target_kappa, polynomials_kappa, theta, K, n_quad)
    results.append(result_kappa)

    # Kappa* benchmark
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    R_kappa_star = 1.1167

    result_kappa_star = test_functional("kappa*", R_kappa_star, c_target_kappa_star, polynomials_kappa_star, theta, K, n_quad)
    results.append(result_kappa_star)

    print_results(results)


if __name__ == "__main__":
    main()
