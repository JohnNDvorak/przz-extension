#!/usr/bin/env python3
"""
Phase 43: Base Term Investigation

Investigates alternative base terms for the mirror multiplier:
- exp(R) + (2K-1) [current]
- exp(2R) + (2K-1) [theoretical?]
- exp(R/θ) + (2K-1)
- exp(2R/θ) + (2K-1)
- Various R-dependent corrections

The exp(R) vs exp(2R) discrepancy is unresolved:
- Theory suggests T^{-(α+β)} at α=β=-R/L gives exp(2R/θ)
- But empirically exp(R) + 5 works much better

This script explores what base term would eliminate the ±0.15% residual.

Created: 2025-12-27 (Phase 43)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Dict, List, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_transform_paper_exact import compute_S12_paper_sum
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


def compute_S34(theta: float, R: float, polynomials: Dict, n_quad: int = 60) -> float:
    """Compute S34 = I3 + I4."""
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


def compute_m_needed(S12_plus: float, S12_minus: float, S34: float, c_target: float) -> float:
    """Compute the m value needed to hit c_target."""
    # c = S12_plus + m * S12_minus + S34
    # m = (c_target - S12_plus - S34) / S12_minus
    return (c_target - S12_plus - S34) / S12_minus


def test_base_formulas(
    R: float,
    c_target: float,
    polynomials: Dict,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> Dict:
    """Test different base formulas."""
    # Compute components
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S12_minus = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n_quad)
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # m needed to hit target
    m_needed = compute_m_needed(S12_plus, S12_minus, S34, c_target)

    # Current baseline
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # Test different base formulas
    results = {}

    # Formula 1: exp(R) + (2K-1) [current]
    base_1 = math.exp(R) + (2 * K - 1)
    results["exp(R)+(2K-1)"] = {
        "base": base_1,
        "m": g_baseline * base_1,
        "gap_pct": (g_baseline * base_1 / m_needed - 1) * 100,
    }

    # Formula 2: exp(2R) + (2K-1)
    base_2 = math.exp(2 * R) + (2 * K - 1)
    results["exp(2R)+(2K-1)"] = {
        "base": base_2,
        "m": g_baseline * base_2,
        "gap_pct": (g_baseline * base_2 / m_needed - 1) * 100,
    }

    # Formula 3: exp(R/θ) + (2K-1)
    base_3 = math.exp(R / theta) + (2 * K - 1)
    results["exp(R/θ)+(2K-1)"] = {
        "base": base_3,
        "m": g_baseline * base_3,
        "gap_pct": (g_baseline * base_3 / m_needed - 1) * 100,
    }

    # Formula 4: exp(2R/θ) + (2K-1) [theory suggests this]
    base_4 = math.exp(2 * R / theta) + (2 * K - 1)
    results["exp(2R/θ)+(2K-1)"] = {
        "base": base_4,
        "m": g_baseline * base_4,
        "gap_pct": (g_baseline * base_4 / m_needed - 1) * 100,
    }

    # Formula 5: What g would make exp(R)+(2K-1) work?
    g_needed_1 = m_needed / (math.exp(R) + (2 * K - 1))
    results["g_needed for exp(R)+(2K-1)"] = {
        "g": g_needed_1,
        "g_baseline": g_baseline,
        "delta_g_pct": (g_needed_1 / g_baseline - 1) * 100,
    }

    # Formula 6: What base would make g_baseline work?
    base_needed = m_needed / g_baseline
    results["base_needed for g_baseline"] = {
        "base_needed": base_needed,
        "exp(R)+(2K-1)": math.exp(R) + (2 * K - 1),
        "delta_base_pct": (base_needed / (math.exp(R) + (2 * K - 1)) - 1) * 100,
    }

    # Store derived values
    results["m_needed"] = m_needed
    results["S12_plus"] = S12_plus
    results["S12_minus"] = S12_minus
    results["S34"] = S34
    results["c_target"] = c_target

    return results


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    print()
    print("=" * 90)
    print("PHASE 43: BASE TERM INVESTIGATION")
    print("=" * 90)
    print()
    print(f"θ = {theta:.6f}")
    print(f"K = {K}")
    print(f"g_baseline = 1 + θ/(2K(2K+1)) = {1 + theta / (2 * K * (2 * K + 1)):.6f}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Test kappa
    print("=" * 90)
    print("KAPPA BENCHMARK (R=1.3036)")
    print("=" * 90)

    R_kappa = 1.3036
    results_kappa = test_base_formulas(R_kappa, c_target_kappa, polynomials_kappa, theta, K, n_quad)

    print()
    print(f"m_needed = {results_kappa['m_needed']:.6f}")
    print()
    print("Base formula comparison:")
    print("-" * 70)
    for name in ["exp(R)+(2K-1)", "exp(2R)+(2K-1)", "exp(R/θ)+(2K-1)", "exp(2R/θ)+(2K-1)"]:
        r = results_kappa[name]
        print(f"  {name:<20}: base={r['base']:<10.4f}, m={r['m']:<10.4f}, gap={r['gap_pct']:+8.4f}%")

    print()
    g_info = results_kappa["g_needed for exp(R)+(2K-1)"]
    print(f"If using exp(R)+(2K-1): g_needed = {g_info['g']:.6f} (baseline={g_info['g_baseline']:.6f}, delta={g_info['delta_g_pct']:+.4f}%)")

    base_info = results_kappa["base_needed for g_baseline"]
    print(f"If using g_baseline: base_needed = {base_info['base_needed']:.4f} (exp(R)+5={base_info['exp(R)+(2K-1)']:.4f}, delta={base_info['delta_base_pct']:+.4f}%)")

    # Test kappa*
    print()
    print("=" * 90)
    print("KAPPA* BENCHMARK (R=1.1167)")
    print("=" * 90)

    R_kappa_star = 1.1167
    results_kappa_star = test_base_formulas(R_kappa_star, c_target_kappa_star, polynomials_kappa_star, theta, K, n_quad)

    print()
    print(f"m_needed = {results_kappa_star['m_needed']:.6f}")
    print()
    print("Base formula comparison:")
    print("-" * 70)
    for name in ["exp(R)+(2K-1)", "exp(2R)+(2K-1)", "exp(R/θ)+(2K-1)", "exp(2R/θ)+(2K-1)"]:
        r = results_kappa_star[name]
        print(f"  {name:<20}: base={r['base']:<10.4f}, m={r['m']:<10.4f}, gap={r['gap_pct']:+8.4f}%")

    print()
    g_info = results_kappa_star["g_needed for exp(R)+(2K-1)"]
    print(f"If using exp(R)+(2K-1): g_needed = {g_info['g']:.6f} (baseline={g_info['g_baseline']:.6f}, delta={g_info['delta_g_pct']:+.4f}%)")

    base_info = results_kappa_star["base_needed for g_baseline"]
    print(f"If using g_baseline: base_needed = {base_info['base_needed']:.4f} (exp(R)+5={base_info['exp(R)+(2K-1)']:.4f}, delta={base_info['delta_base_pct']:+.4f}%)")

    # Cross-benchmark analysis
    print()
    print("=" * 90)
    print("CROSS-BENCHMARK ANALYSIS")
    print("=" * 90)
    print()

    g_kappa = results_kappa["g_needed for exp(R)+(2K-1)"]["g"]
    g_kappa_star = results_kappa_star["g_needed for exp(R)+(2K-1)"]["g"]
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print(f"g_needed (κ):  {g_kappa:.6f} (delta from baseline: {(g_kappa - g_baseline) * 100:.4f}%)")
    print(f"g_needed (κ*): {g_kappa_star:.6f} (delta from baseline: {(g_kappa_star - g_baseline) * 100:.4f}%)")
    print(f"Difference:    {(g_kappa - g_kappa_star) * 100:.4f}%")
    print()

    # Check if g correlates with R
    print("R-dependent g fitting:")
    R1, R2 = 1.3036, 1.1167
    g1, g2 = g_kappa, g_kappa_star

    # Linear fit: g = a*R + b
    a = (g1 - g2) / (R1 - R2)
    b = g1 - a * R1

    print(f"  Linear fit: g(R) = {a:.6f} * R + {b:.6f}")
    print(f"  g(κ):  {a * R1 + b:.6f} (actual: {g1:.6f})")
    print(f"  g(κ*): {a * R2 + b:.6f} (actual: {g2:.6f})")
    print()

    # Check if base correlates with R
    print("R-dependent base fitting:")
    base_kappa = results_kappa["base_needed for g_baseline"]["base_needed"]
    base_kappa_star = results_kappa_star["base_needed for g_baseline"]["base_needed"]

    print(f"base_needed (κ):  {base_kappa:.4f}")
    print(f"base_needed (κ*): {base_kappa_star:.4f}")
    print(f"exp(R) + 5 (κ):  {math.exp(R1) + 5:.4f}")
    print(f"exp(R) + 5 (κ*): {math.exp(R2) + 5:.4f}")
    print()

    # Try alternative formula: exp(R) + (2K-1) + f(R)
    # where f(R) is a small R-dependent correction
    delta_base_kappa = base_kappa - (math.exp(R1) + 5)
    delta_base_kappa_star = base_kappa_star - (math.exp(R2) + 5)

    print("Base correction analysis:")
    print(f"  δbase (κ):  {delta_base_kappa:.6f}")
    print(f"  δbase (κ*): {delta_base_kappa_star:.6f}")

    # Linear fit: δbase = a*R + b
    a_base = (delta_base_kappa - delta_base_kappa_star) / (R1 - R2)
    b_base = delta_base_kappa - a_base * R1

    print(f"  Linear fit: δbase(R) = {a_base:.6f} * R + {b_base:.6f}")

    # Reconstruct corrected formula
    print()
    print("PROPOSED CORRECTED FORMULA:")
    print("  m = g_baseline × [exp(R) + (2K-1) + δbase(R)]")
    print(f"  where δbase(R) = {a_base:.4f} × R + {b_base:.4f}")


if __name__ == "__main__":
    main()
