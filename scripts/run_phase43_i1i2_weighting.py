#!/usr/bin/env python3
"""
Phase 43: I1/I2 Fraction-Weighted Correction Analysis

Explores whether the ±0.15% residual correlates with the I1/I2 fraction
in the mirror channel (-R).

From Phase 42 and 43:
- κ: I1 fraction = 23.3%, needs g increased
- κ*: I1 fraction = 32.6%, needs g decreased

Hypothesis: Higher I1 fraction → baseline over-corrects → needs lower g

This could be first-principles if we can derive the correction from the
different cross-term contributions of I1 vs I2.

Created: 2025-12-27 (Phase 43)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Dict

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.mirror_transform_paper_exact import compute_S12_paper_sum
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


def compute_S34(theta: float, R: float, polynomials: Dict, n_quad: int = 60) -> float:
    """Compute S34."""
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


def analyze_i1i2_weighting(
    benchmark: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> Dict:
    """Analyze I1/I2 weighting effect on correction."""
    # Compute I1 and I2 at -R (mirror channel)
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)
    S12_minus = I1_minus + I2_minus

    # I1/I2 fractions
    f_I1 = I1_minus / S12_minus
    f_I2 = I2_minus / S12_minus

    # Compute S12 at +R and S34
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # m needed to hit target
    m_needed = (c_target - S12_plus - S34) / S12_minus

    # Baseline
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    base = math.exp(R) + (2 * K - 1)
    m_baseline = g_baseline * base

    # g needed
    g_needed = m_needed / base

    # Delta from baseline
    delta_g = g_needed - g_baseline
    delta_g_pct = (g_needed / g_baseline - 1) * 100

    return {
        "benchmark": benchmark,
        "R": R,
        "I1_minus": I1_minus,
        "I2_minus": I2_minus,
        "f_I1": f_I1,
        "f_I2": f_I2,
        "m_needed": m_needed,
        "m_baseline": m_baseline,
        "g_needed": g_needed,
        "g_baseline": g_baseline,
        "delta_g": delta_g,
        "delta_g_pct": delta_g_pct,
    }


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    print()
    print("=" * 90)
    print("PHASE 43: I1/I2 FRACTION-WEIGHTED CORRECTION ANALYSIS")
    print("=" * 90)
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Analyze both benchmarks
    results = [
        analyze_i1i2_weighting("kappa", 1.3036, c_target_kappa, polynomials_kappa, theta, K, n_quad),
        analyze_i1i2_weighting("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star, theta, K, n_quad),
    ]

    # Print results
    print("I1/I2 FRACTIONS AT -R (MIRROR CHANNEL)")
    print("-" * 70)
    print(f"{'Benchmark':<12} | {'I1_minus':<12} | {'I2_minus':<12} | {'f_I1':<10} | {'f_I2':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['benchmark']:<12} | {r['I1_minus']:<12.6f} | {r['I2_minus']:<12.6f} | {r['f_I1']:<10.4f} | {r['f_I2']:<10.4f}")

    print()
    print("G-VALUE ANALYSIS")
    print("-" * 70)
    print(f"{'Benchmark':<12} | {'g_needed':<12} | {'g_baseline':<12} | {'delta_g':<12} | {'delta_g%':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['benchmark']:<12} | {r['g_needed']:<12.6f} | {r['g_baseline']:<12.6f} | {r['delta_g']:+12.6f} | {r['delta_g_pct']:+10.4f}%")

    print()
    print("=" * 90)
    print("CORRELATION ANALYSIS")
    print("=" * 90)
    print()

    # Extract values
    f_I1_kappa = results[0]["f_I1"]
    f_I1_kappa_star = results[1]["f_I1"]
    delta_g_kappa = results[0]["delta_g"]
    delta_g_kappa_star = results[1]["delta_g"]

    print(f"κ:  f_I1 = {f_I1_kappa:.4f}, delta_g = {delta_g_kappa:+.6f}")
    print(f"κ*: f_I1 = {f_I1_kappa_star:.4f}, delta_g = {delta_g_kappa_star:+.6f}")
    print()

    # Correlation direction
    if delta_g_kappa > delta_g_kappa_star and f_I1_kappa < f_I1_kappa_star:
        print("PATTERN: Higher f_I1 → needs LOWER g (confirms hypothesis)")
    elif delta_g_kappa < delta_g_kappa_star and f_I1_kappa > f_I1_kappa_star:
        print("PATTERN: Higher f_I1 → needs HIGHER g (opposite of hypothesis)")
    else:
        print("PATTERN: No clear correlation between f_I1 and delta_g")

    print()

    # Linear fit: delta_g = a * f_I1 + b
    a = (delta_g_kappa - delta_g_kappa_star) / (f_I1_kappa - f_I1_kappa_star)
    b = delta_g_kappa - a * f_I1_kappa

    print("LINEAR FIT: delta_g = a × f_I1 + b")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print()
    print(f"  delta_g(κ):  {a * f_I1_kappa + b:+.6f} (actual: {delta_g_kappa:+.6f})")
    print(f"  delta_g(κ*): {a * f_I1_kappa_star + b:+.6f} (actual: {delta_g_kappa_star:+.6f})")
    print()

    # Derive correction formula
    print("=" * 90)
    print("PROPOSED CORRECTION FORMULA")
    print("=" * 90)
    print()
    print("If the correction is proportional to I1 fraction:")
    print()
    print(f"  g(f_I1) = g_baseline + {a:.6f} × (f_I1 - {f_I1_kappa:.4f}) + {delta_g_kappa:.6f}")
    print()
    print("Simplified:")
    print(f"  g(f_I1) = {results[0]['g_baseline'] + b:.6f} + {a:.6f} × f_I1")
    print()

    # Alternative: weighted g from I1 and I2 corrections
    print("=" * 90)
    print("ALTERNATIVE: WEIGHTED G FROM I1/I2 CORRECTIONS")
    print("=" * 90)
    print()
    print("From Phase 42, we know:")
    print("  g_I1 > g_baseline (I1 has cross-terms)")
    print("  g_I2 = 1.0 (I2 has no cross-terms)")
    print()
    print("If effective g is a weighted average:")
    print("  g_eff = f_I1 × g_I1 + f_I2 × g_I2")
    print("        = f_I1 × g_I1 + f_I2 × 1.0")
    print()

    # From Phase 42 MCG results
    g_I1_kappa = 1.098611
    g_I1_kappa_star = 1.145763

    g_eff_kappa = f_I1_kappa * g_I1_kappa + (1 - f_I1_kappa) * 1.0
    g_eff_kappa_star = f_I1_kappa_star * g_I1_kappa_star + (1 - f_I1_kappa_star) * 1.0

    print(f"  κ:  g_eff = {f_I1_kappa:.4f} × {g_I1_kappa:.4f} + {1-f_I1_kappa:.4f} × 1.0 = {g_eff_kappa:.6f}")
    print(f"  κ*: g_eff = {f_I1_kappa_star:.4f} × {g_I1_kappa_star:.4f} + {1-f_I1_kappa_star:.4f} × 1.0 = {g_eff_kappa_star:.6f}")
    print()
    print(f"  g_needed (κ):  {results[0]['g_needed']:.6f}")
    print(f"  g_needed (κ*): {results[1]['g_needed']:.6f}")
    print()
    print("  NOTE: g_eff is much larger than g_needed, confirming Phase 42 finding")
    print("        that MCG g values cannot be used directly as corrections.")

    print()
    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()
    print("The correlation between f_I1 and delta_g is:")
    print(f"  slope = {a:.6f} (per unit f_I1)")
    print()
    print("Interpretation:")
    if a < 0:
        print("  Higher I1 fraction → needs LESS g correction")
        print("  This makes sense because I1 has cross-terms that the baseline")
        print("  already accounts for. More I1 → baseline is more accurate.")
    else:
        print("  Higher I1 fraction → needs MORE g correction")
        print("  This is the opposite of expected.")
    print()
    print("But this is still empirical - we haven't derived WHY the slope is", f"{a:.4f}")


if __name__ == "__main__":
    main()
