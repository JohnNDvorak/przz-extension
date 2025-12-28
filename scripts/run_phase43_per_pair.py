#!/usr/bin/env python3
"""
Phase 43: Per-Pair Decomposition Analysis

Analyzes the m_needed for each (ℓ₁, ℓ₂) pair to identify which pairs
contribute most to the residual.

Hypothesis: Different pairs may need different mirror multipliers due
to polynomial degree differences.

Created: 2025-12-27 (Phase 43)
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
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


@dataclass
class PairResult:
    """Result for a single (ℓ₁, ℓ₂) pair."""
    pair_key: str
    ell1: int
    ell2: int

    # I1 and I2 at +R and -R
    I1_plus: float
    I1_minus: float
    I2_plus: float
    I2_minus: float
    I3I4: float  # No mirror for I3+I4

    # Normalized contribution
    S12_plus: float
    S12_minus: float
    c_pair: float  # Contribution to c with baseline m

    # What m would hit target contribution?
    m_baseline: float
    c_target_fraction: float  # Expected c contribution
    m_needed: float  # m to hit target fraction
    delta_m_pct: float  # (m_needed - m_baseline) / m_baseline * 100


def compute_pair_I1I2(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> tuple:
    """Compute I1 and I2 for a single pair at given R."""
    # I1
    I1_result = compute_I1_unified_paper(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )
    I1 = I1_result.I1_value

    # I2
    I2_result = compute_I2_unified_paper(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        include_Q=True,
    )
    I2 = I2_result.I2_value

    return I1, I2


def compute_pair_I3I4(
    pair_key: str,
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I3+I4 for a single pair."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    terms = all_terms[pair_key]

    I3I4 = 0.0
    for term in terms[2:4]:  # I3 and I4
        result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
        I3I4 += result.value

    return I3I4


def analyze_pair(
    pair_key: str,
    R: float,
    theta: float,
    polynomials: Dict,
    c_target: float,
    K: int = 3,
    n_quad: int = 60,
) -> PairResult:
    """Analyze a single pair."""
    ell1 = int(pair_key[0])
    ell2 = int(pair_key[1])

    # Normalization factors
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    norm = f_norm[pair_key] * symmetry[pair_key]

    # Compute I1 and I2 at +R and -R
    I1_plus, I2_plus = compute_pair_I1I2(R, theta, ell1, ell2, polynomials, n_quad)
    I1_minus, I2_minus = compute_pair_I1I2(-R, theta, ell1, ell2, polynomials, n_quad)

    # Apply normalization
    I1_plus *= norm
    I1_minus *= norm
    I2_plus *= norm
    I2_minus *= norm

    # Compute I3+I4
    I3I4 = compute_pair_I3I4(pair_key, theta, R, polynomials, n_quad) * norm

    # S12 at +R and -R
    S12_plus = I1_plus + I2_plus
    S12_minus = I1_minus + I2_minus

    # Baseline multiplier
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    base = math.exp(R) + (2 * K - 1)
    m_baseline = g_baseline * base

    # Contribution to c with baseline
    c_pair = S12_plus + m_baseline * S12_minus + I3I4

    # What fraction of c_target is this pair?
    # We'll compute what m would make this pair's contribution
    # match its "fair share" of c_target

    # First, we need to know what the total c is with baseline
    # For this pair, if we had the right m, the contribution would be:
    # c_pair_target = (pair's fraction of c_target)
    # But we don't know the fraction a priori...

    # Instead, solve for m that would make current contribution match target
    # If total c = c_target, and this pair is the only one that needs adjustment,
    # what m would this pair need?

    # We'll compute m_needed such that:
    # S12_plus + m_needed * S12_minus + I3I4 = c_pair * (c_target / c_total)
    # But we don't know c_total here...

    # Simpler approach: just report what m would make c_pair hit c_target
    # as if this pair were 100% of c
    # m_needed = (c_target - S12_plus - I3I4) / S12_minus

    # Actually, let's compute the "local" m for this pair
    # If each pair could have its own m, what would it be to hit the target
    # contribution proportionally?

    # For now, compute m_needed assuming this pair's contribution should scale
    # proportionally to c_target / c_computed_total
    # We'll need the total c from all pairs...

    # Let's just return the raw values and compute m_needed in post-processing
    return PairResult(
        pair_key=pair_key,
        ell1=ell1,
        ell2=ell2,
        I1_plus=I1_plus,
        I1_minus=I1_minus,
        I2_plus=I2_plus,
        I2_minus=I2_minus,
        I3I4=I3I4,
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        c_pair=c_pair,
        m_baseline=m_baseline,
        c_target_fraction=0.0,  # Will compute in post
        m_needed=0.0,           # Will compute in post
        delta_m_pct=0.0,        # Will compute in post
    )


def analyze_benchmark(
    benchmark: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> List[PairResult]:
    """Analyze all pairs for a benchmark."""
    pairs = ["11", "22", "33", "12", "13", "23"]

    results = []
    for pair_key in pairs:
        result = analyze_pair(pair_key, R, theta, polynomials, c_target, K, n_quad)
        results.append(result)

    # Post-process: compute c_total and m_needed for each pair
    c_total = sum(r.c_pair for r in results)
    scale_factor = c_target / c_total

    for r in results:
        # Target contribution for this pair
        r.c_target_fraction = r.c_pair * scale_factor

        # m_needed to hit target (if we could adjust this pair independently)
        # c_pair_target = S12_plus + m_needed * S12_minus + I3I4
        # m_needed = (c_pair_target - S12_plus - I3I4) / S12_minus
        if abs(r.S12_minus) > 1e-15:
            r.m_needed = (r.c_target_fraction - r.S12_plus - r.I3I4) / r.S12_minus
            r.delta_m_pct = (r.m_needed - r.m_baseline) / r.m_baseline * 100
        else:
            r.m_needed = r.m_baseline
            r.delta_m_pct = 0.0

    return results


def print_pair_analysis(benchmark: str, results: List[PairResult], c_target: float):
    """Print pair analysis table."""
    print()
    print(f"{'Pair':<6} | {'I1_minus':<12} | {'I2_minus':<12} | {'S12_minus':<12} | {'c_pair':<12} | {'m_needed':<10} | {'delta_m%':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r.pair_key:<6} | {r.I1_minus:<12.6f} | {r.I2_minus:<12.6f} | {r.S12_minus:<12.6f} | {r.c_pair:<12.6f} | {r.m_needed:<10.4f} | {r.delta_m_pct:+10.4f}%")

    print()
    c_total = sum(r.c_pair for r in results)
    print(f"Total c: {c_total:.6f} (target: {c_target:.6f}, gap: {(c_total/c_target-1)*100:+.4f}%)")


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
    print("PHASE 43: PER-PAIR DECOMPOSITION ANALYSIS")
    print("=" * 90)
    print()
    print(f"Baseline g = {g_baseline:.6f}")
    print()

    # Kappa benchmark
    print("=" * 90)
    print("KAPPA BENCHMARK (R=1.3036)")
    print("=" * 90)

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    results_kappa = analyze_benchmark("kappa", 1.3036, c_target_kappa, polynomials_kappa, theta, K, n_quad)
    print_pair_analysis("kappa", results_kappa, c_target_kappa)

    # Kappa* benchmark
    print()
    print("=" * 90)
    print("KAPPA* BENCHMARK (R=1.1167)")
    print("=" * 90)

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    results_kappa_star = analyze_benchmark("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star, theta, K, n_quad)
    print_pair_analysis("kappa*", results_kappa_star, c_target_kappa_star)

    # Compare pair-by-pair
    print()
    print("=" * 90)
    print("PAIR-BY-PAIR COMPARISON: delta_m% difference between κ and κ*")
    print("=" * 90)
    print()
    print(f"{'Pair':<6} | {'κ delta_m%':<12} | {'κ* delta_m%':<12} | {'difference':<12} | {'same sign?':<10}")
    print("-" * 70)

    for r_k, r_ks in zip(results_kappa, results_kappa_star):
        diff = r_k.delta_m_pct - r_ks.delta_m_pct
        same_sign = "YES" if (r_k.delta_m_pct * r_ks.delta_m_pct > 0) else "NO"
        print(f"{r_k.pair_key:<6} | {r_k.delta_m_pct:+12.4f}% | {r_ks.delta_m_pct:+12.4f}% | {diff:+12.4f}% | {same_sign:<10}")

    # Summary statistics
    print()
    print("=" * 90)
    print("SUMMARY: Which pairs drive the residual?")
    print("=" * 90)
    print()

    # Find pairs with largest |delta_m| difference
    max_diff = 0
    max_diff_pair = None
    for r_k, r_ks in zip(results_kappa, results_kappa_star):
        diff = abs(r_k.delta_m_pct - r_ks.delta_m_pct)
        if diff > max_diff:
            max_diff = diff
            max_diff_pair = r_k.pair_key

    print(f"Pair with largest κ/κ* difference: {max_diff_pair} (diff = {max_diff:.4f}%)")

    # Check if all pairs have same-sign delta_m within each benchmark
    kappa_signs = [r.delta_m_pct > 0 for r in results_kappa]
    kappa_star_signs = [r.delta_m_pct > 0 for r in results_kappa_star]

    print(f"κ: all pairs need m increase? {all(kappa_signs)}")
    print(f"κ*: all pairs need m decrease? {all(not s for s in kappa_star_signs)}")


if __name__ == "__main__":
    main()
