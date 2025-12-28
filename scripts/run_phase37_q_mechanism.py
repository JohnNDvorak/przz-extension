#!/usr/bin/env python3
"""
Phase 37: Determine Q deviation mechanism by computing correction ratios.

The frozen-Q experiment on I1 shows Q reweighting is massive (~85%).
But what matters for the correction factor is the RATIO:

    correction = m_needed / m_base

where m_needed = (c_target - S12_plus - S34) / S12_minus

This script computes the correction ratio for three Q modes:
1. Q=1 (no Q)
2. frozen-Q (Q(t)² only)
3. normal Q (full Q(Arg) with x,y)

The key insight: If the RATIO changes between modes, then Q affects the correction.
If the ratio stays constant, Q is just an overall scale factor.

Created: 2025-12-26 (Phase 37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from src.polynomials import load_przz_polynomials
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode


def compute_S12_with_Q_mode(
    R: float,
    theta: float,
    polynomials: dict,
    q_mode: str,
    n_quad: int = 60,
) -> float:
    """Compute S12 sum with specified Q mode."""
    pairs = ["11", "22", "33", "12", "13", "23"]
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1 = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode=q_mode, n_quad_u=n_quad,
        )
        norm = f_norm[pair_key] * symmetry[pair_key]
        total += I1 * norm

    return total


def compute_correction_ratio_analysis(
    R: float,
    theta: float,
    polynomials: dict,
    n_quad: int = 60,
) -> dict:
    """
    Compute correction ratio for all three Q modes.

    The correction ratio is: m_formula / m_empirical
    where:
    - m_empirical = (c_target - S12_plus - S34) / S12_minus (from PRZZ benchmark)
    - m_formula = exp(R) + (2K-1) = exp(R) + 5 for K=3

    We compute this for Q=1, frozen-Q, and normal-Q to see how Q affects it.
    """
    K = 3
    m_base = math.exp(R) + (2 * K - 1)  # exp(R) + 5
    corr_beta = 1 + theta / (2 * K * (2 * K + 1))  # 1 + θ/42

    results = {}

    for q_mode in ["none", "frozen", "normal"]:
        # Compute S12 at +R and -R
        S12_plus = compute_S12_with_Q_mode(R, theta, polynomials, q_mode, n_quad)
        S12_minus = compute_S12_with_Q_mode(-R, theta, polynomials, q_mode, n_quad)

        # m_formula with correction
        m_derived = m_base * corr_beta

        # c using derived formula
        # Note: S34 computed separately without Q mode variation for simplicity
        # (I3/I4 don't use the same Q structure as I1/I2)
        c_derived = S12_plus + m_derived * S12_minus

        # Ratio c_derived / c_empirical (where c_empirical uses m = exp(R)+5)
        m_empirical = m_base
        c_empirical = S12_plus + m_empirical * S12_minus

        ratio = c_derived / c_empirical if abs(c_empirical) > 1e-15 else float('inf')

        results[q_mode] = {
            "S12_plus": S12_plus,
            "S12_minus": S12_minus,
            "c_empirical": c_empirical,
            "c_derived": c_derived,
            "ratio": ratio,
            "gap_from_beta": (ratio - 1) * 100,  # as percentage
        }

    return results


def main():
    print("=" * 70)
    print("PHASE 37: Q MECHANISM - CORRECTION RATIO ANALYSIS")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4 / 7
    R = 1.3036
    K = 3

    corr_beta = 1 + theta / (2 * K * (2 * K + 1))
    print(f"Parameters: θ={theta:.6f}, R={R}, K={K}")
    print(f"Beta moment correction = 1 + θ/(2K(2K+1)) = {corr_beta:.8f}")
    print()

    print("Computing S12 and correction ratios for each Q mode...")
    print()

    results = compute_correction_ratio_analysis(R, theta, polynomials, n_quad=60)

    print("S12 VALUES (I1 only, no I2/S34)")
    print("-" * 70)
    print(f"{'Q mode':<12} | {'S12(+R)':<14} | {'S12(-R)':<14} | {'S12(+R)/S12(-R)':<12}")
    print("-" * 70)

    for q_mode, data in results.items():
        ratio_S12 = data["S12_plus"] / data["S12_minus"] if abs(data["S12_minus"]) > 1e-15 else float('inf')
        print(f"{q_mode:<12} | {data['S12_plus']:+.8f}   | {data['S12_minus']:+.8f}   | {ratio_S12:.4f}")

    print()
    print("KEY INSIGHT: Look at the S12(+R)/S12(-R) ratio")
    print("-" * 70)

    # Compute the ratio of ratios to see if Q affects it
    ratio_none = results["none"]["S12_plus"] / results["none"]["S12_minus"]
    ratio_frozen = results["frozen"]["S12_plus"] / results["frozen"]["S12_minus"]
    ratio_normal = results["normal"]["S12_plus"] / results["normal"]["S12_minus"]

    print(f"  S12(+R)/S12(-R) with Q=1:      {ratio_none:.6f}")
    print(f"  S12(+R)/S12(-R) with frozen-Q: {ratio_frozen:.6f}")
    print(f"  S12(+R)/S12(-R) with normal-Q: {ratio_normal:.6f}")
    print()

    # The ratio of S12(+R)/S12(-R) determines m_needed
    # If this ratio is CONSTANT across Q modes, then Q doesn't affect m
    delta_frozen = (ratio_frozen / ratio_none - 1) * 100
    delta_normal = (ratio_normal / ratio_none - 1) * 100

    print(f"  Change in ratio (frozen vs Q=1): {delta_frozen:+.4f}%")
    print(f"  Change in ratio (normal vs Q=1): {delta_normal:+.4f}%")
    print()

    if abs(delta_normal - delta_frozen) > 0.1:
        print("  Q-DERIVATIVE EFFECT on ratio: significant")
        print(f"    → (normal - frozen) = {delta_normal - delta_frozen:+.4f}%")
    else:
        print("  Q-DERIVATIVE EFFECT on ratio: negligible")

    if abs(delta_frozen) > 0.1:
        print("  Q-REWEIGHT EFFECT on ratio: significant")
        print(f"    → frozen vs Q=1 = {delta_frozen:+.4f}%")
    else:
        print("  Q-REWEIGHT EFFECT on ratio: negligible")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if abs(delta_normal) < 0.1:
        print("  The S12(+R)/S12(-R) ratio is INVARIANT under Q mode changes!")
        print("  → Q affects the SCALE of S12, not the RATIO.")
        print("  → The correction factor is unaffected by Q polynomial.")
        print("  → The ±0.13% residual must come from elsewhere (I2, S34, or interaction).")
    else:
        print(f"  Q changes the S12(+R)/S12(-R) ratio by {delta_normal:+.4f}%.")
        print("  → This directly affects the correction factor.")
        print("  → Need to derive Q-dependent correction for full accuracy.")


if __name__ == "__main__":
    main()
