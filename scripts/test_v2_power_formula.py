#!/usr/bin/env python3
"""
Test V2's (1-u) power formula in the unified evaluator.

V2 DSL uses: one_minus_u_power = max(0, (ℓ₁-1) + (ℓ₂-1))
OLD DSL uses: one_minus_u_power = ℓ₁ + ℓ₂

This script computes unified I1 with BOTH formulas and compares to V2 DSL.
"""

import sys
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.quadrature import gauss_legendre_01
from src.series import TruncatedSeries
from src.polynomials import load_przz_polynomials
from src.unified_s12_evaluator_v3 import build_unified_bracket_series
from src.terms_k3_d1 import make_all_terms_k3_v2
from src.evaluate import evaluate_term


def get_v2_dsl_actual_power(ell1: int, ell2: int) -> int:
    """
    Get the ACTUAL (1-u) power used by V2 DSL.

    V2 DSL has hardcoded exceptions for (1,1) and (2,2).
    """
    # Hardcoded in make_I1_11_v2 and make_I1_22_v2
    if (ell1, ell2) == (1, 1):
        return 2  # hardcoded
    elif (ell1, ell2) == (2, 2):
        return 2  # hardcoded, see line 1300 comment says "ℓ₁=ℓ₂=1"
    else:
        # Generic formula for (3,3), (1,2), (1,3), (2,3)
        return max(0, (ell1 - 1) + (ell2 - 1))


def compute_I1_unified_with_power(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: dict,
    power_formula: str = "OLD",  # "OLD", "V2_generic", "V2_actual"
    n_quad: int = 60,
) -> float:
    """
    Compute I1 with specified (1-u) power formula.

    power_formula:
        - "OLD": one_minus_u_power = ell1 + ell2
        - "V2_generic": one_minus_u_power = max(0, (ell1-1) + (ell2-1))
        - "V2_actual": Uses the actual powers from V2 DSL (with hardcoded exceptions)
    """
    var_names = ("x", "y")
    xy_mask = (1 << 0) | (1 << 1)

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Choose power formula
    if power_formula == "OLD":
        one_minus_u_power = ell1 + ell2
    elif power_formula == "V2_generic":
        one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))
    elif power_formula == "V2_actual":
        one_minus_u_power = get_v2_dsl_actual_power(ell1, ell2)
    else:
        raise ValueError(f"Unknown power_formula: {power_formula}")

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power if one_minus_u_power > 0 else 1.0

        for t, t_w in zip(t_nodes, t_weights):
            series = build_unified_bracket_series(
                u, t, theta, R, ell1, ell2, polynomials, var_names, include_Q=True
            )
            xy_coeff = series.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)
            total += xy_coeff * one_minus_u_factor * u_w * t_w

    return total


def compute_I1_empirical_v2(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: dict,
    n_quad: int = 60,
) -> float:
    """Compute I1 using V2 DSL at +R only (for comparison)."""
    all_terms_v2 = make_all_terms_k3_v2(theta, R, kernel_regime="paper")
    pair_key = f"{ell1}{ell2}"
    term_key = f"I1_{ell1}{ell2}"

    if pair_key not in all_terms_v2:
        raise ValueError(f"Pair {pair_key} not found in V2 terms")

    for term in all_terms_v2[pair_key]:
        if term.name == term_key:
            result = evaluate_term(term, polynomials, n=n_quad, R=R, theta=theta)
            # result is a TermResult, extract the value
            return result.value if hasattr(result, 'value') else float(result)

    raise ValueError(f"Term {term_key} not found in V2 terms for pair {pair_key}")


def main():
    theta = 4/7
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("V2 (1-u) Power Formula Comparison")
    print("=" * 70)
    print(f"θ = {theta:.6f}, R = {R}")
    print()

    # Load polynomials - returns tuple (P1, P2, P3, Q)
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # All pairs
    pairs = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]

    print("Power formulas:")
    print("  OLD: (1-u)^{ℓ₁+ℓ₂}")
    print("  V2:  (1-u)^{max(0, (ℓ₁-1)+(ℓ₂-1))}")
    print()

    print("Powers by pair:")
    print("-" * 60)
    print(f"{'Pair':<8} {'OLD':<8} {'V2 generic':<12} {'V2 actual':<12}")
    print("-" * 60)
    for ell1, ell2 in pairs:
        old_power = ell1 + ell2
        v2_generic = max(0, (ell1 - 1) + (ell2 - 1))
        v2_actual = get_v2_dsl_actual_power(ell1, ell2)
        note = " (hardcoded)" if v2_actual != v2_generic else ""
        print(f"({ell1},{ell2}){'':<4} {old_power:<8} {v2_generic:<12} {v2_actual}{note}")
    print()

    print("I1 Comparison (unified with V2 ACTUAL power vs V2 DSL):")
    print("-" * 70)
    print(f"{'Pair':<8} {'Unified(V2 act)':<18} {'V2 DSL':<18} {'Ratio':<12} {'Match?'}")
    print("-" * 70)

    results = []
    for ell1, ell2 in pairs:
        # Unified with V2 ACTUAL power formula (including hardcoded exceptions)
        unified_v2 = compute_I1_unified_with_power(
            R, theta, ell1, ell2, polynomials, power_formula="V2_actual", n_quad=n_quad
        )

        # V2 DSL (at +R only)
        empirical_v2 = compute_I1_empirical_v2(theta, R, ell1, ell2, polynomials, n_quad=n_quad)

        # Ratio
        if abs(empirical_v2) > 1e-10:
            ratio = unified_v2 / empirical_v2
        else:
            ratio = float('inf') if abs(unified_v2) > 1e-10 else 1.0

        match = abs(ratio - 1.0) < 0.05  # 5% tolerance

        results.append({
            'pair': f"({ell1},{ell2})",
            'unified_v2': unified_v2,
            'empirical_v2': empirical_v2,
            'ratio': ratio,
            'match': match,
        })

        match_str = "✓" if match else "✗"
        print(f"({ell1},{ell2}){'':<4} {unified_v2:>14.6e}     {empirical_v2:>14.6e}     {ratio:>8.4f}     {match_str}")

    print("-" * 70)

    # Also compare OLD power unified vs V2 DSL
    print()
    print("I1 Comparison (unified with OLD power vs V2 DSL):")
    print("-" * 70)
    print(f"{'Pair':<8} {'Unified(OLD)':<18} {'V2 DSL':<18} {'Ratio':<12} {'Match?'}")
    print("-" * 70)

    results_old = []
    for ell1, ell2 in pairs:
        # Unified with OLD power formula
        unified_old = compute_I1_unified_with_power(
            R, theta, ell1, ell2, polynomials, power_formula="OLD", n_quad=n_quad
        )

        # V2 DSL (at +R only)
        empirical_v2 = compute_I1_empirical_v2(theta, R, ell1, ell2, polynomials, n_quad=n_quad)

        # Ratio
        if abs(empirical_v2) > 1e-10:
            ratio = unified_old / empirical_v2
        else:
            ratio = float('inf') if abs(unified_old) > 1e-10 else 1.0

        match = abs(ratio - 1.0) < 0.05
        results_old.append({'pair': f"({ell1},{ell2})", 'ratio': ratio, 'match': match})
        match_str = "✓" if match else "✗"
        print(f"({ell1},{ell2}){'':<4} {unified_old:>14.6e}     {empirical_v2:>14.6e}     {ratio:>8.4f}     {match_str}")

    print("-" * 70)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    matches_v2 = sum(1 for r in results if r['match'])
    matches_old = sum(1 for r in results_old if r['match'])
    total = len(results)

    print(f"Pairs matching with V2 ACTUAL power: {matches_v2}/{total}")
    print(f"Pairs matching with OLD power:       {matches_old}/{total}")
    print()

    if matches_v2 > matches_old:
        print("V2 ACTUAL power formula works better!")
    elif matches_old > matches_v2:
        print("OLD power formula works better!")
    else:
        print("Both formulas have same match count.")

    print()
    print("Non-matching pairs:")
    print("  V2 ACTUAL:", [r['pair'] for r in results if not r['match']])
    print("  OLD:      ", [r['pair'] for r in results_old if not r['match']])


if __name__ == "__main__":
    main()
