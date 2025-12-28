#!/usr/bin/env python3
"""
Phase 41.3: I1/I2 Component-Level Attribution

Breaks down S12 into I1 and I2 contributions to determine which
component dominates the residual.

GPT's Question: Does one integral type (I1 vs I2) contribute more
to the benchmark-specific residual?

Created: 2025-12-27 (Phase 41)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, List

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper
from src.evaluator.decomposition import compute_mirror_multiplier


@dataclass
class I1I2Breakdown:
    """Breakdown of S12 into I1 and I2 components."""

    benchmark: str
    R: float
    sign: str  # "+R" or "-R"

    # Totals
    I1_total: float
    I2_total: float
    S12_total: float

    # Per-pair breakdowns
    I1_by_pair: Dict[str, float]
    I2_by_pair: Dict[str, float]

    @property
    def I1_fraction(self) -> float:
        return self.I1_total / self.S12_total if abs(self.S12_total) > 1e-15 else 0

    @property
    def I2_fraction(self) -> float:
        return self.I2_total / self.S12_total if abs(self.S12_total) > 1e-15 else 0


def compute_I1_I2_breakdown(
    R: float,
    theta: float,
    polynomials: Dict,
    benchmark_name: str,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> I1I2Breakdown:
    """Compute separate I1 and I2 totals for S12."""

    # Factorial normalization
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }

    # Symmetry factors
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    pairs = ["11", "22", "33", "12", "13", "23"]

    I1_total = 0.0
    I2_total = 0.0
    I1_by_pair = {}
    I2_by_pair = {}

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm

        # I1
        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=True, apply_factorial_norm=True,
        )
        I1_contrib = I1_result.I1_value * norm * sym
        I1_total += I1_contrib
        I1_by_pair[pair_key] = I1_contrib

        # I2
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=True,
        )
        I2_contrib = I2_result.I2_value * norm * sym
        I2_total += I2_contrib
        I2_by_pair[pair_key] = I2_contrib

    sign = "+R" if R > 0 else "-R"

    return I1I2Breakdown(
        benchmark=benchmark_name,
        R=R,
        sign=sign,
        I1_total=I1_total,
        I2_total=I2_total,
        S12_total=I1_total + I2_total,
        I1_by_pair=I1_by_pair,
        I2_by_pair=I2_by_pair,
    )


def print_breakdown_table(breakdowns: List[I1I2Breakdown]) -> None:
    """Print I1/I2 breakdown table."""
    print()
    print("I1/I2 COMPONENT BREAKDOWN")
    print("=" * 80)
    print()

    # Summary table
    print(f"{'Benchmark':<12} | {'Sign':<5} | {'I1_total':<12} | {'I2_total':<12} | {'S12_total':<12} | {'I1%':<8} | {'I2%':<8}")
    print("-" * 80)
    for b in breakdowns:
        print(f"{b.benchmark:<12} | {b.sign:<5} | {b.I1_total:<12.6f} | {b.I2_total:<12.6f} | {b.S12_total:<12.6f} | {b.I1_fraction*100:<8.2f} | {b.I2_fraction*100:<8.2f}")
    print()


def print_per_pair_breakdown(breakdowns: List[I1I2Breakdown]) -> None:
    """Print per-pair I1/I2 breakdown."""
    print()
    print("PER-PAIR BREAKDOWN")
    print("=" * 80)

    pairs = ["11", "22", "33", "12", "13", "23"]

    for b in breakdowns:
        print(f"\n{b.benchmark} ({b.sign}):")
        print("-" * 60)
        print(f"{'Pair':<8} | {'I1':<12} | {'I2':<12} | {'I1+I2':<12} | {'I1%':<8}")
        print("-" * 60)
        for pair in pairs:
            i1 = b.I1_by_pair[pair]
            i2 = b.I2_by_pair[pair]
            total = i1 + i2
            i1_pct = (i1 / total * 100) if abs(total) > 1e-15 else 0
            print(f"{pair:<8} | {i1:<12.6f} | {i2:<12.6f} | {total:<12.6f} | {i1_pct:<8.2f}")


def analyze_residual_attribution(
    kappa_plus: I1I2Breakdown,
    kappa_minus: I1I2Breakdown,
    kappa_star_plus: I1I2Breakdown,
    kappa_star_minus: I1I2Breakdown,
    theta: float,
) -> None:
    """Analyze which component (I1 vs I2) contributes most to the residual difference."""
    print()
    print("=" * 80)
    print("RESIDUAL ATTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Get mirror multipliers
    m_kappa, _ = compute_mirror_multiplier(kappa_plus.R, K=3, formula="derived", theta=theta)
    m_kappa_star, _ = compute_mirror_multiplier(kappa_star_plus.R, K=3, formula="derived", theta=theta)

    # Compute c contributions from I1 and I2 separately
    # c = S12+ + m * S12- = (I1+ + I2+) + m * (I1- + I2-)
    #   = (I1+ + m*I1-) + (I2+ + m*I2-)
    #   = c_I1 + c_I2

    c_I1_kappa = kappa_plus.I1_total + m_kappa * kappa_minus.I1_total
    c_I2_kappa = kappa_plus.I2_total + m_kappa * kappa_minus.I2_total
    c_S12_kappa = c_I1_kappa + c_I2_kappa

    c_I1_kappa_star = kappa_star_plus.I1_total + m_kappa_star * kappa_star_minus.I1_total
    c_I2_kappa_star = kappa_star_plus.I2_total + m_kappa_star * kappa_star_minus.I2_total
    c_S12_kappa_star = c_I1_kappa_star + c_I2_kappa_star

    print("CONTRIBUTION TO c (S12 only, excludes S34)")
    print("-" * 60)
    print(f"{'Component':<12} | {'kappa':<14} | {'kappa*':<14} | {'Ratio':<10}")
    print("-" * 60)
    print(f"{'c_I1':<12} | {c_I1_kappa:<14.6f} | {c_I1_kappa_star:<14.6f} | {c_I1_kappa/c_I1_kappa_star:<10.4f}")
    print(f"{'c_I2':<12} | {c_I2_kappa:<14.6f} | {c_I2_kappa_star:<14.6f} | {c_I2_kappa/c_I2_kappa_star:<10.4f}")
    print(f"{'c_S12':<12} | {c_S12_kappa:<14.6f} | {c_S12_kappa_star:<14.6f} | {c_S12_kappa/c_S12_kappa_star:<10.4f}")
    print()

    # Fraction of S12 from I1 vs I2
    print("FRACTION OF c_S12 FROM I1 vs I2")
    print("-" * 60)
    print(f"{'Benchmark':<12} | {'I1 fraction':<14} | {'I2 fraction':<14}")
    print("-" * 60)
    print(f"{'kappa':<12} | {c_I1_kappa/c_S12_kappa*100:<14.2f}% | {c_I2_kappa/c_S12_kappa*100:<14.2f}%")
    print(f"{'kappa*':<12} | {c_I1_kappa_star/c_S12_kappa_star*100:<14.2f}% | {c_I2_kappa_star/c_S12_kappa_star*100:<14.2f}%")
    print()

    # Compare I1 and I2 ratios between benchmarks
    print("I1 vs I2 RATIO COMPARISON")
    print("-" * 60)
    I1_ratio = c_I1_kappa / c_I1_kappa_star
    I2_ratio = c_I2_kappa / c_I2_kappa_star
    S12_ratio = c_S12_kappa / c_S12_kappa_star
    print(f"c_I1(kappa) / c_I1(kappa*) = {I1_ratio:.4f}")
    print(f"c_I2(kappa) / c_I2(kappa*) = {I2_ratio:.4f}")
    print(f"c_S12(kappa) / c_S12(kappa*) = {S12_ratio:.4f}")
    print()

    if abs(I1_ratio - I2_ratio) > 0.05:
        print("OBSERVATION: I1 and I2 scale differently between benchmarks!")
        if I1_ratio > I2_ratio:
            print(f"  I1 ratio ({I1_ratio:.4f}) > I2 ratio ({I2_ratio:.4f})")
            print("  This suggests I1 dominates the benchmark-specific behavior")
        else:
            print(f"  I2 ratio ({I2_ratio:.4f}) > I1 ratio ({I1_ratio:.4f})")
            print("  This suggests I2 dominates the benchmark-specific behavior")
    else:
        print("OBSERVATION: I1 and I2 scale similarly between benchmarks")
        print("  The residual is not dominated by one component")


def main():
    """Main entry point."""
    theta = 4 / 7
    n_quad = 60

    print("=" * 80)
    print("PHASE 41.3: I1/I2 COMPONENT-LEVEL ATTRIBUTION")
    print("=" * 80)
    print()
    print(f"Computing I1/I2 breakdown for both benchmarks...")
    print()

    # Kappa benchmark
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R_kappa = 1.3036

    kappa_plus = compute_I1_I2_breakdown(R_kappa, theta, polynomials, "kappa", n_quad)
    kappa_minus = compute_I1_I2_breakdown(-R_kappa, theta, polynomials, "kappa", n_quad)

    # Kappa* benchmark
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    R_kappa_star = 1.1167

    kappa_star_plus = compute_I1_I2_breakdown(R_kappa_star, theta, polynomials_star, "kappa*", n_quad)
    kappa_star_minus = compute_I1_I2_breakdown(-R_kappa_star, theta, polynomials_star, "kappa*", n_quad)

    # Print breakdowns
    all_breakdowns = [kappa_plus, kappa_minus, kappa_star_plus, kappa_star_minus]
    print_breakdown_table(all_breakdowns)
    print_per_pair_breakdown(all_breakdowns)

    # Analyze residual attribution
    analyze_residual_attribution(kappa_plus, kappa_minus, kappa_star_plus, kappa_star_minus, theta)

    print()
    print("=" * 80)
    print("IMPLICATIONS FOR POLYNOMIAL-AWARE FUNCTIONAL")
    print("=" * 80)
    print()
    print("If I1 and I2 scale differently between benchmarks, then:")
    print("  g(P,Q,R,K,theta) should weight I1 and I2 contributions differently")
    print("  based on polynomial structure.")
    print()
    print("If they scale similarly, then:")
    print("  The residual is structural to the whole S12, not to a specific component.")
    print()


if __name__ == "__main__":
    main()
