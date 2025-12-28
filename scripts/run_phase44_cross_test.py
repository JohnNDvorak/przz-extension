#!/usr/bin/env python3
"""
Phase 44: Cross-Polynomial Test

Tests whether the correction is R-dependent or polynomial-dependent by
evaluating κ polynomials at κ*'s R value and vice versa.

This breaks the correlation between R and f_I1 to determine:
- Hypothesis A: delta_g depends ONLY on f_I1 (polynomial structure)
- Hypothesis B: delta_g depends ONLY on R

Created: 2025-12-27 (Phase 44)
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


def evaluate_at_R(
    polynomials: Dict,
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> Dict:
    """Evaluate a polynomial set at given R."""
    # Compute I1/I2 at -R
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)
    S12_minus = I1_minus + I2_minus
    f_I1 = I1_minus / S12_minus if abs(S12_minus) > 1e-15 else float('nan')

    # Compute S12 at +R and S34
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # Baseline
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    base = math.exp(R) + (2 * K - 1)
    m_baseline = g_baseline * base

    # Compute c with baseline
    c_baseline = S12_plus + m_baseline * S12_minus + S34

    return {
        "R": R,
        "f_I1": f_I1,
        "S12_plus": S12_plus,
        "S12_minus": S12_minus,
        "S34": S34,
        "c_baseline": c_baseline,
        "g_baseline": g_baseline,
        "base": base,
    }


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    R_kappa = 1.3036
    R_kappa_star = 1.1167

    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    print()
    print("=" * 90)
    print("PHASE 44: CROSS-POLYNOMIAL TEST")
    print("=" * 90)
    print()
    print("Goal: Determine if correction depends on f_I1 (polynomials) or R")
    print()

    # Standard evaluations (known benchmarks)
    print("STANDARD EVALUATIONS (known benchmarks)")
    print("-" * 90)

    result_kappa_at_kappa = evaluate_at_R(polys_kappa, R_kappa, theta, K, n_quad)
    gap_kappa = (result_kappa_at_kappa["c_baseline"] / c_target_kappa - 1) * 100

    result_kappa_star_at_kappa_star = evaluate_at_R(polys_kappa_star, R_kappa_star, theta, K, n_quad)
    gap_kappa_star = (result_kappa_star_at_kappa_star["c_baseline"] / c_target_kappa_star - 1) * 100

    print(f"κ polys at R={R_kappa}:     f_I1={result_kappa_at_kappa['f_I1']:.4f}, c={result_kappa_at_kappa['c_baseline']:.6f}, gap={gap_kappa:+.4f}%")
    print(f"κ* polys at R={R_kappa_star}: f_I1={result_kappa_star_at_kappa_star['f_I1']:.4f}, c={result_kappa_star_at_kappa_star['c_baseline']:.6f}, gap={gap_kappa_star:+.4f}%")

    # Cross evaluations (polynomial at "wrong" R)
    print()
    print("CROSS EVALUATIONS (polynomial at 'wrong' R)")
    print("-" * 90)

    result_kappa_at_kappa_star_R = evaluate_at_R(polys_kappa, R_kappa_star, theta, K, n_quad)
    result_kappa_star_at_kappa_R = evaluate_at_R(polys_kappa_star, R_kappa, theta, K, n_quad)

    print(f"κ polys at R={R_kappa_star}: f_I1={result_kappa_at_kappa_star_R['f_I1']:.4f}, c={result_kappa_at_kappa_star_R['c_baseline']:.6f}")
    print(f"κ* polys at R={R_kappa}:    f_I1={result_kappa_star_at_kappa_R['f_I1']:.4f}, c={result_kappa_star_at_kappa_R['c_baseline']:.6f}")

    # Predictions from both hypotheses
    print()
    print("=" * 90)
    print("HYPOTHESIS TESTING")
    print("=" * 90)

    # Our two competing models:
    # Hypothesis A: delta_g = -0.0185 * f_I1 + 0.00585
    # Hypothesis B: delta_g = 0.00927 * R - 0.01055

    a_f, b_f = -0.018544, 0.005848  # f_I1 model
    a_R, b_R = 0.009267, -0.010551  # R model

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print()
    print("For κ polys at R=1.1167 (κ* R value):")
    print("-" * 70)
    f_I1 = result_kappa_at_kappa_star_R["f_I1"]
    R = R_kappa_star

    pred_delta_g_A = a_f * f_I1 + b_f  # f_I1 model prediction
    pred_delta_g_B = a_R * R + b_R      # R model prediction

    pred_g_A = g_baseline + pred_delta_g_A
    pred_g_B = g_baseline + pred_delta_g_B

    base = result_kappa_at_kappa_star_R["base"]
    pred_m_A = pred_g_A * base
    pred_m_B = pred_g_B * base

    S12_plus = result_kappa_at_kappa_star_R["S12_plus"]
    S12_minus = result_kappa_at_kappa_star_R["S12_minus"]
    S34 = result_kappa_at_kappa_star_R["S34"]

    pred_c_A = S12_plus + pred_m_A * S12_minus + S34
    pred_c_B = S12_plus + pred_m_B * S12_minus + S34

    print(f"  f_I1 = {f_I1:.4f}, R = {R:.4f}")
    print(f"  Hypothesis A (f_I1 model): delta_g = {pred_delta_g_A:+.6f}, g = {pred_g_A:.6f}, c = {pred_c_A:.6f}")
    print(f"  Hypothesis B (R model):    delta_g = {pred_delta_g_B:+.6f}, g = {pred_g_B:.6f}, c = {pred_c_B:.6f}")
    print(f"  Baseline:                  delta_g = 0, g = {g_baseline:.6f}, c = {result_kappa_at_kappa_star_R['c_baseline']:.6f}")

    print()
    print("For κ* polys at R=1.3036 (κ R value):")
    print("-" * 70)
    f_I1 = result_kappa_star_at_kappa_R["f_I1"]
    R = R_kappa

    pred_delta_g_A = a_f * f_I1 + b_f
    pred_delta_g_B = a_R * R + b_R

    pred_g_A = g_baseline + pred_delta_g_A
    pred_g_B = g_baseline + pred_delta_g_B

    base = result_kappa_star_at_kappa_R["base"]
    pred_m_A = pred_g_A * base
    pred_m_B = pred_g_B * base

    S12_plus = result_kappa_star_at_kappa_R["S12_plus"]
    S12_minus = result_kappa_star_at_kappa_R["S12_minus"]
    S34 = result_kappa_star_at_kappa_R["S34"]

    pred_c_A = S12_plus + pred_m_A * S12_minus + S34
    pred_c_B = S12_plus + pred_m_B * S12_minus + S34

    print(f"  f_I1 = {f_I1:.4f}, R = {R:.4f}")
    print(f"  Hypothesis A (f_I1 model): delta_g = {pred_delta_g_A:+.6f}, g = {pred_g_A:.6f}, c = {pred_c_A:.6f}")
    print(f"  Hypothesis B (R model):    delta_g = {pred_delta_g_B:+.6f}, g = {pred_g_B:.6f}, c = {pred_c_B:.6f}")
    print(f"  Baseline:                  delta_g = 0, g = {g_baseline:.6f}, c = {result_kappa_star_at_kappa_R['c_baseline']:.6f}")

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print()
    print("Without target c values for cross-evaluations, we cannot definitively")
    print("determine which hypothesis is correct.")
    print()
    print("However, we CAN observe:")
    print("  - How f_I1 changes with R for the SAME polynomials")
    print("  - Whether the pattern is consistent")
    print()

    # f_I1 variation with R
    print("f_I1 variation with R (same polynomials):")
    print(f"  κ polys:  f_I1 at R=1.3036: {result_kappa_at_kappa['f_I1']:.4f}")
    print(f"            f_I1 at R=1.1167: {result_kappa_at_kappa_star_R['f_I1']:.4f}")
    print(f"            delta: {result_kappa_at_kappa_star_R['f_I1'] - result_kappa_at_kappa['f_I1']:+.4f}")
    print()
    print(f"  κ* polys: f_I1 at R=1.1167: {result_kappa_star_at_kappa_star['f_I1']:.4f}")
    print(f"            f_I1 at R=1.3036: {result_kappa_star_at_kappa_R['f_I1']:.4f}")
    print(f"            delta: {result_kappa_star_at_kappa_R['f_I1'] - result_kappa_star_at_kappa_star['f_I1']:+.4f}")


if __name__ == "__main__":
    main()
