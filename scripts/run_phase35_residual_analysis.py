#!/usr/bin/env python3
"""
Phase 35: Comprehensive analysis of the ±0.15% residual.

This script implements GPT's recommended experiments:
1. Direction analysis: Does the formula overcount or undercount κ?
2. Experiment A: R-sweep with polynomials held fixed
3. Experiment B: Polynomial swap at fixed R
4. Microcase ladder: P=Q=1, P=real/Q=1, P=1/Q=real, P=Q=real

Goal: Determine if the ±0.15% is:
- True R-dependence
- Polynomial-set dependence
- Higher-order correction

Created: 2025-12-26 (Phase 35)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)
from src.mirror_transform_paper_exact import compute_S12_paper_sum


def compute_c_with_mirror(R, theta, polynomials, n_quad=60, m_formula="derived"):
    """Compute c using mirror assembly with specified m formula."""
    from src.evaluator.decomposition import compute_mirror_multiplier
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    K = 3

    # S12 at +R and -R
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S12_minus = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n_quad)

    # S34
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key] * symmetry_factor[pair_key]
        for term in terms[2:4]:  # I3 and I4
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += norm * result.value

    # Mirror multiplier
    m, _ = compute_mirror_multiplier(R, K, formula=m_formula)

    # Assemble c
    c = S12_plus + m * S12_minus + S34
    return c


def analyze_error_direction():
    """Analyze whether the formula overcounts or undercounts κ."""
    print("=" * 70)
    print("PHASE 35: ERROR DIRECTION ANALYSIS")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3

    # Benchmarks
    benchmarks = [
        {"name": "κ", "R": 1.3036, "c_target": 2.137454406132173, "kappa_target": 0.417293962},
        {"name": "κ*", "R": 1.1167, "c_target": 1.938, "kappa_target": 0.419},
    ]

    # Derived correction prediction
    predicted_corr = 1 + theta / (2 * K * (2 * K + 1))

    print("CORRECTION FACTOR ANALYSIS")
    print("-" * 50)
    print(f"  Derived prediction: 1 + θ/(2K(2K+1)) = {predicted_corr:.8f}")
    print()

    for bm in benchmarks:
        name = bm["name"]
        R = bm["R"]
        c_target = bm["c_target"]
        kappa_target = bm["kappa_target"]

        # Load appropriate polynomials
        if name == "κ":
            P1, P2, P3, Q = load_przz_polynomials()
        else:
            P1, P2, P3, Q = load_przz_polynomials_kappa_star()

        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # Compute c with different formulas
        c_empirical = compute_c_with_mirror(R, theta, polynomials, m_formula="empirical")
        c_derived = compute_c_with_mirror(R, theta, polynomials, m_formula="derived")

        # Compute κ from c
        kappa_empirical = 1 - math.log(c_empirical) / R
        kappa_derived = 1 - math.log(c_derived) / R

        # Gaps
        c_gap_empirical = (c_empirical / c_target - 1) * 100
        c_gap_derived = (c_derived / c_target - 1) * 100
        kappa_gap_empirical = (kappa_empirical - kappa_target) * 100  # In percentage points
        kappa_gap_derived = (kappa_derived - kappa_target) * 100

        print(f"Benchmark {name} (R={R}):")
        print(f"  c_target = {c_target:.6f}, κ_target = {kappa_target:.6f}")
        print()
        print(f"  Empirical (m = exp(R) + 5):")
        print(f"    c = {c_empirical:.6f}, gap = {c_gap_empirical:+.4f}%")
        print(f"    κ = {kappa_empirical:.6f}, gap = {kappa_gap_empirical:+.4f} pp")
        if c_gap_empirical < 0:
            print(f"    → UNDERCOUNTS c (κ too high)")
        else:
            print(f"    → OVERCOUNTS c (κ too low)")
        print()
        print(f"  Derived (m = [1+θ/42] × [exp(R)+5]):")
        print(f"    c = {c_derived:.6f}, gap = {c_gap_derived:+.4f}%")
        print(f"    κ = {kappa_derived:.6f}, gap = {kappa_gap_derived:+.4f} pp")
        if c_gap_derived < 0:
            print(f"    → UNDERCOUNTS c (κ too high)")
        else:
            print(f"    → OVERCOUNTS c (κ too low)")
        print()

        # What correction would be needed?
        # c_needed = c_target
        # c_computed = S12_plus + m * S12_minus + S34
        # We have c_derived / c_target, so the needed additional correction is:
        correction_needed = c_target / c_derived
        print(f"  Correction needed to match target: ×{correction_needed:.6f}")
        print(f"  As adjustment to (1+θ/42): {(correction_needed * predicted_corr):.8f}")
        print()

    print("SUMMARY")
    print("-" * 50)
    print("""
  The derived formula m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]:

  - At κ (R=1.3036): UNDERCOUNTS c → κ computed is TOO HIGH
    Need slightly LARGER m to increase c

  - At κ* (R=1.1167): OVERCOUNTS c → κ computed is TOO LOW
    Need slightly SMALLER m to decrease c

  This pattern suggests:
  - Higher R needs larger correction
  - Lower R needs smaller correction
  - Could be true R-dependence OR polynomial-set effects
""")


def run_r_sweep_experiment():
    """Experiment A: R-sweep with polynomials held fixed."""
    print()
    print("=" * 70)
    print("EXPERIMENT A: R-SWEEP WITH FIXED POLYNOMIALS")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3

    # Use κ polynomials (fixed)
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Target c at R=1.3036 is 2.137454
    c_target_kappa = 2.137454406132173

    print("Using κ polynomial set (fixed)")
    print("Sweeping R from 0.9 to 1.5")
    print()

    print("R       | c_empirical | c_derived | ratio (d/e) | needed_corr")
    print("-" * 65)

    R_values = [0.9, 1.0, 1.1, 1.1167, 1.2, 1.3, 1.3036, 1.4, 1.5]
    predicted_corr = 1 + theta / (2 * K * (2 * K + 1))

    for R in R_values:
        c_emp = compute_c_with_mirror(R, theta, polynomials, m_formula="empirical")
        c_der = compute_c_with_mirror(R, theta, polynomials, m_formula="derived")
        ratio = c_der / c_emp

        # At R=1.3036, we know the target, so compute needed correction
        if abs(R - 1.3036) < 0.01:
            needed = c_target_kappa / c_der
            needed_str = f"{needed:.6f}"
        else:
            needed_str = "-"

        marker = " <-- κ" if abs(R - 1.3036) < 0.01 else ""
        marker = " <-- κ*" if abs(R - 1.1167) < 0.01 else marker

        print(f"{R:.4f}  | {c_emp:.6f}   | {c_der:.6f}  | {ratio:.6f}    | {needed_str}{marker}")

    print()
    print("Analysis:")
    print(f"  The ratio c_derived/c_empirical should be constant = {predicted_corr:.6f}")
    print("  If it varies with R, there's R-dependence in the correction.")
    print()


def run_polynomial_swap_experiment():
    """Experiment B: Polynomial swap at fixed R."""
    print()
    print("=" * 70)
    print("EXPERIMENT B: POLYNOMIAL SWAP AT FIXED R")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3
    R_fixed = 1.20  # Midpoint between κ and κ*

    # Load both polynomial sets
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials()
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()

    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    print(f"Fixed R = {R_fixed}")
    print()

    predicted_corr = 1 + theta / (2 * K * (2 * K + 1))

    for name, polys in [("κ polynomials", polys_kappa), ("κ* polynomials", polys_kappa_star)]:
        c_emp = compute_c_with_mirror(R_fixed, theta, polys, m_formula="empirical")
        c_der = compute_c_with_mirror(R_fixed, theta, polys, m_formula="derived")
        ratio = c_der / c_emp

        print(f"{name}:")
        print(f"  c_empirical = {c_emp:.6f}")
        print(f"  c_derived = {c_der:.6f}")
        print(f"  ratio = {ratio:.6f} (predicted: {predicted_corr:.6f})")
        print()

    print("Analysis:")
    print("  If the ratio differs between polynomial sets, the correction has")
    print("  polynomial dependence (not just R-dependence).")
    print()


def run_microcase_ladder():
    """Phase 35C: Microcase ladder to locate the ±0.15% source."""
    print()
    print("=" * 70)
    print("PHASE 35C: MICROCASE LADDER")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3
    R = 1.3036
    n_quad = 60

    # Create constant polynomial P(x) = 1, Q(x) = 1
    one_poly = Polynomial([1.0])

    # Load real polynomials
    P1_real, P2_real, P3_real, Q_real = load_przz_polynomials()

    # Define microcase configurations
    microcases = [
        ("P=Q=1", {"P1": one_poly, "P2": one_poly, "P3": one_poly, "Q": one_poly}),
        ("P=real, Q=1", {"P1": P1_real, "P2": P2_real, "P3": P3_real, "Q": one_poly}),
        ("P=1, Q=real", {"P1": one_poly, "P2": one_poly, "P3": one_poly, "Q": Q_real}),
        ("P=Q=real", {"P1": P1_real, "P2": P2_real, "P3": P3_real, "Q": Q_real}),
    ]

    predicted_corr = 1 + theta / (2 * K * (2 * K + 1))

    print(f"R = {R}, θ = {theta:.4f}, K = {K}")
    print(f"Predicted correction: {predicted_corr:.6f}")
    print()

    print("Microcase       | c_empirical | c_derived | ratio    | gap from pred")
    print("-" * 70)

    for name, polys in microcases:
        try:
            c_emp = compute_c_with_mirror(R, theta, polys, m_formula="empirical", n_quad=n_quad)
            c_der = compute_c_with_mirror(R, theta, polys, m_formula="derived", n_quad=n_quad)
            ratio = c_der / c_emp
            gap = (ratio / predicted_corr - 1) * 100

            print(f"{name:15} | {c_emp:11.6f} | {c_der:9.6f} | {ratio:.6f} | {gap:+.4f}%")
        except Exception as e:
            print(f"{name:15} | ERROR: {e}")

    print()
    print("Interpretation:")
    print("  - If gap appears in P=Q=1: kernel/log-factor/measure issue")
    print("  - If gap appears only with P: polynomial derivative/weight interaction")
    print("  - If gap appears only with Q: eigenvalue/Q coupling")
    print("  - If gap appears only in P=Q=real: interaction effect")
    print()


def main():
    analyze_error_direction()
    run_r_sweep_experiment()
    run_polynomial_swap_experiment()
    run_microcase_ladder()

    print("=" * 70)
    print("PHASE 35 ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps based on results:")
    print("1. If ratio varies with R (Exp A): True R-dependence exists")
    print("2. If ratio varies with polys (Exp B): Polynomial-set dependence")
    print("3. Microcase ladder shows WHERE the effect originates")
    print()


if __name__ == "__main__":
    main()
