#!/usr/bin/env python3
"""
Test the Q=1 Hypothesis for First-Principles g Derivation

HYPOTHESIS:
- g_I1 ≈ 1.0 because I1's log factor cross-terms self-correct
- g_I2 ≈ g_baseline because I2 lacks log factor, needs external correction

TEST:
With Q=1 (constant polynomial), the Q asymmetry vanishes.
If hypothesis is correct:
  g_I1(Q=1) ≈ 1.0
  g_I2(Q=1) ≈ g_baseline = 1 + θ/(2K(2K+1))

VALIDATION APPROACH:
Since we don't have c_target for Q=1, we:
1. Compute I1/I2 with Q=1 for both benchmarks
2. Solve for g_I1(Q=1) and g_I2(Q=1) that minimize c gap across R values
3. Check if g_I1(Q=1) ≈ 1.0 and g_I2(Q=1) ≈ g_baseline

Created: 2025-12-27 (Phase 45 investigation)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.evaluator.g_first_principles import compute_S34, G_I1_CALIBRATED, G_I2_CALIBRATED


def compute_components_with_q_mode(R: float, theta: float, polynomials: dict,
                                    n_quad: int, q_mode: str = "real"):
    """
    Compute I1, I2, S34 components with specified Q mode.

    q_mode: "real" (use actual Q polynomial) or "unity" (Q=1)
    """
    if q_mode == "unity":
        # Create Q=1 polynomial (constant 1) using the project's Polynomial class
        from src.polynomials import Polynomial
        Q_unity = Polynomial(np.array([1.0]))  # Q(x) = 1
        polys_modified = {
            "P1": polynomials["P1"],
            "P2": polynomials["P2"],
            "P3": polynomials["P3"],
            "Q": Q_unity
        }
    else:
        polys_modified = polynomials

    # Compute I1 and I2 at +R and -R
    I1_plus, I2_plus = compute_I1_I2_totals(R, theta, polys_modified, n_quad)
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polys_modified, n_quad)

    # Compute S34
    S34 = compute_S34(theta, R, polys_modified, n_quad)

    return {
        "I1_plus": I1_plus,
        "I1_minus": I1_minus,
        "I2_plus": I2_plus,
        "I2_minus": I2_minus,
        "S34": S34,
    }


def compute_c_with_g(components: dict, g_I1: float, g_I2: float, R: float, K: int = 3):
    """Compute c given components and g values."""
    base = math.exp(R) + (2 * K - 1)

    c = (components["I1_plus"] + g_I1 * base * components["I1_minus"] +
         components["I2_plus"] + g_I2 * base * components["I2_minus"] +
         components["S34"])

    return c


def solve_g_from_ratio(comp_kappa: dict, comp_kappa_star: dict,
                        R_kappa: float, R_kappa_star: float, K: int = 3):
    """
    Solve for g_I1 and g_I2 that make the ratio c(κ)/c(κ*) consistent.

    Since we don't have c_target for Q=1, we use a different approach:
    - Assume the RATIO c(κ)/c(κ*) should equal the real-Q ratio
    - Solve for g values that achieve this
    """
    # Real Q ratio from calibrated values
    c_kappa_real = 2.13745440613217263636
    c_kappa_star_real = 1.9379524112
    target_ratio = c_kappa_real / c_kappa_star_real

    base_kappa = math.exp(R_kappa) + (2 * K - 1)
    base_kappa_star = math.exp(R_kappa_star) + (2 * K - 1)

    # Set up the system:
    # c_κ = I1p_κ + g1*base_κ*I1m_κ + I2p_κ + g2*base_κ*I2m_κ + S34_κ
    # c_κ* = I1p_κ* + g1*base_κ**I1m_κ* + I2p_κ* + g2*base_κ**I2m_κ* + S34_κ*
    # Constraint: c_κ / c_κ* = target_ratio

    # Define coefficients
    # c_κ = A_κ + g1*B1_κ + g2*B2_κ
    A_kappa = comp_kappa["I1_plus"] + comp_kappa["I2_plus"] + comp_kappa["S34"]
    B1_kappa = base_kappa * comp_kappa["I1_minus"]
    B2_kappa = base_kappa * comp_kappa["I2_minus"]

    A_kappa_star = comp_kappa_star["I1_plus"] + comp_kappa_star["I2_plus"] + comp_kappa_star["S34"]
    B1_kappa_star = base_kappa_star * comp_kappa_star["I1_minus"]
    B2_kappa_star = base_kappa_star * comp_kappa_star["I2_minus"]

    # We need another constraint. Use: g_I1 + g_I2 = 1.0 + g_baseline (sum constraint)
    # Or: minimize |g_I1 - 1| + |g_I2 - g_baseline|

    # Let's just solve for what g values give c ratio = target_ratio
    # and separately what give c ratio = 1 (equal)

    # For a grid search:
    theta = 4/7
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    best_g1, best_g2, best_error = None, None, float('inf')

    for g1 in np.linspace(0.9, 1.1, 201):
        for g2 in np.linspace(0.9, 1.1, 201):
            c_kappa = A_kappa + g1 * B1_kappa + g2 * B2_kappa
            c_kappa_star = A_kappa_star + g1 * B1_kappa_star + g2 * B2_kappa_star

            if c_kappa_star > 0:
                ratio = c_kappa / c_kappa_star
                error = abs(ratio - target_ratio)

                if error < best_error:
                    best_error = error
                    best_g1, best_g2 = g1, g2

    return best_g1, best_g2, best_error


def main():
    print("=" * 80)
    print("Q=1 HYPOTHESIS TEST: First-Principles g Derivation")
    print("=" * 80)
    print()

    theta = 4 / 7
    K = 3
    n_quad = 60

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print(f"Parameters:")
    print(f"  θ = {theta:.6f}")
    print(f"  K = {K}")
    print(f"  g_baseline = 1 + θ/(2K(2K+1)) = {g_baseline:.6f}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    R_kappa = 1.3036
    R_kappa_star = 1.1167

    # ==== PART 1: Compute components with real Q ====
    print("=" * 80)
    print("PART 1: Components with REAL Q polynomial")
    print("=" * 80)

    comp_kappa_real = compute_components_with_q_mode(R_kappa, theta, polys_kappa, n_quad, "real")
    comp_kappa_star_real = compute_components_with_q_mode(R_kappa_star, theta, polys_kappa_star, n_quad, "real")

    for name, comp, R in [("κ", comp_kappa_real, R_kappa), ("κ*", comp_kappa_star_real, R_kappa_star)]:
        print(f"\n{name} (R={R}):")
        print(f"  I1_plus:  {comp['I1_plus']:+.6f}")
        print(f"  I1_minus: {comp['I1_minus']:+.6f}")
        print(f"  I2_plus:  {comp['I2_plus']:+.6f}")
        print(f"  I2_minus: {comp['I2_minus']:+.6f}")
        print(f"  S34:      {comp['S34']:+.6f}")

        # Compute f_I1
        S12_minus = comp['I1_minus'] + comp['I2_minus']
        f_I1 = comp['I1_minus'] / S12_minus if abs(S12_minus) > 1e-15 else 0
        print(f"  f_I1:     {f_I1:.4f}")

    # Verify calibrated g values work
    c_kappa_computed = compute_c_with_g(comp_kappa_real, G_I1_CALIBRATED, G_I2_CALIBRATED, R_kappa)
    c_kappa_star_computed = compute_c_with_g(comp_kappa_star_real, G_I1_CALIBRATED, G_I2_CALIBRATED, R_kappa_star)

    print(f"\nVerify calibrated g values (g_I1={G_I1_CALIBRATED:.5f}, g_I2={G_I2_CALIBRATED:.5f}):")
    print(f"  c_κ computed:   {c_kappa_computed:.10f}")
    print(f"  c_κ target:     2.1374544061")
    print(f"  c_κ* computed:  {c_kappa_star_computed:.10f}")
    print(f"  c_κ* target:    1.9379524112")

    # ==== PART 2: Compute components with Q=1 ====
    print("\n" + "=" * 80)
    print("PART 2: Components with Q=1 (constant)")
    print("=" * 80)

    comp_kappa_q1 = compute_components_with_q_mode(R_kappa, theta, polys_kappa, n_quad, "unity")
    comp_kappa_star_q1 = compute_components_with_q_mode(R_kappa_star, theta, polys_kappa_star, n_quad, "unity")

    for name, comp, R in [("κ", comp_kappa_q1, R_kappa), ("κ*", comp_kappa_star_q1, R_kappa_star)]:
        print(f"\n{name} (R={R}, Q=1):")
        print(f"  I1_plus:  {comp['I1_plus']:+.6f}")
        print(f"  I1_minus: {comp['I1_minus']:+.6f}")
        print(f"  I2_plus:  {comp['I2_plus']:+.6f}")
        print(f"  I2_minus: {comp['I2_minus']:+.6f}")
        print(f"  S34:      {comp['S34']:+.6f}")

        S12_minus = comp['I1_minus'] + comp['I2_minus']
        f_I1 = comp['I1_minus'] / S12_minus if abs(S12_minus) > 1e-15 else 0
        print(f"  f_I1:     {f_I1:.4f}")

    # ==== PART 3: Test the hypothesis ====
    print("\n" + "=" * 80)
    print("PART 3: TEST THE HYPOTHESIS")
    print("=" * 80)
    print()
    print("HYPOTHESIS: With Q=1, g_I1 ≈ 1.0 and g_I2 ≈ g_baseline")
    print()

    # Test 1: What c values do we get with hypothesized g values?
    g_I1_hyp = 1.0
    g_I2_hyp = g_baseline

    c_kappa_hyp = compute_c_with_g(comp_kappa_q1, g_I1_hyp, g_I2_hyp, R_kappa)
    c_kappa_star_hyp = compute_c_with_g(comp_kappa_star_q1, g_I1_hyp, g_I2_hyp, R_kappa_star)

    print(f"Using hypothesized g values (g_I1={g_I1_hyp:.5f}, g_I2={g_I2_hyp:.5f}):")
    print(f"  c_κ (Q=1):   {c_kappa_hyp:.6f}")
    print(f"  c_κ* (Q=1):  {c_kappa_star_hyp:.6f}")
    print(f"  Ratio:       {c_kappa_hyp / c_kappa_star_hyp:.6f}")

    # Test 2: What g values minimize the ratio error?
    print(f"\nSolving for g values that match real-Q ratio...")

    g1_solved, g2_solved, error = solve_g_from_ratio(
        comp_kappa_q1, comp_kappa_star_q1, R_kappa, R_kappa_star)

    c_kappa_solved = compute_c_with_g(comp_kappa_q1, g1_solved, g2_solved, R_kappa)
    c_kappa_star_solved = compute_c_with_g(comp_kappa_star_q1, g1_solved, g2_solved, R_kappa_star)

    print(f"\nSolved g values (ratio-matching):")
    print(f"  g_I1 (Q=1): {g1_solved:.5f}  (hypothesis: 1.0)")
    print(f"  g_I2 (Q=1): {g2_solved:.5f}  (hypothesis: {g_baseline:.5f})")
    print(f"  Ratio error: {error:.6f}")
    print()
    print(f"  c_κ (Q=1):   {c_kappa_solved:.6f}")
    print(f"  c_κ* (Q=1):  {c_kappa_star_solved:.6f}")
    print(f"  Ratio:       {c_kappa_solved / c_kappa_star_solved:.6f}")
    print(f"  Target ratio: {2.137454 / 1.937952:.6f}")

    # ==== PART 4: Compare Q=1 vs Q=real g values ====
    print("\n" + "=" * 80)
    print("PART 4: COMPARISON")
    print("=" * 80)

    print("\n| Quantity | Q=real (calibrated) | Q=1 (solved) | Q=1 (hypothesis) |")
    print("|----------|---------------------|--------------|------------------|")
    print(f"| g_I1     | {G_I1_CALIBRATED:.5f}             | {g1_solved:.5f}        | 1.00000          |")
    print(f"| g_I2     | {G_I2_CALIBRATED:.5f}             | {g2_solved:.5f}        | {g_baseline:.5f}          |")
    print()

    # Compute differences
    delta_g1 = G_I1_CALIBRATED - g1_solved
    delta_g2 = G_I2_CALIBRATED - g2_solved

    print(f"Q contribution to g values (Q=real - Q=1):")
    print(f"  Δg_I1 = {delta_g1:+.6f}")
    print(f"  Δg_I2 = {delta_g2:+.6f}")
    print()

    # ==== PART 5: Verdict ====
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    g1_match = abs(g1_solved - 1.0) < 0.01
    g2_match = abs(g2_solved - g_baseline) < 0.01

    if g1_match and g2_match:
        print("\n✓ HYPOTHESIS SUPPORTED!")
        print(f"  With Q=1: g_I1 ≈ 1.0 ({g1_solved:.4f})")
        print(f"  With Q=1: g_I2 ≈ g_baseline ({g2_solved:.4f} vs {g_baseline:.4f})")
        print("\nThis suggests:")
        print("  1. I1's log factor cross-terms self-correct → g_I1 ≈ 1.0")
        print("  2. I2 lacks cross-terms, needs external correction → g_I2 ≈ g_baseline")
        print("  3. Q polynomial adds differential corrections ε_1, ε_2")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED")
        print(f"  g_I1 (Q=1) = {g1_solved:.4f}, expected ≈ 1.0")
        print(f"  g_I2 (Q=1) = {g2_solved:.4f}, expected ≈ {g_baseline:.4f}")
        print("\nNeed different approach to understand g values.")


if __name__ == "__main__":
    main()
