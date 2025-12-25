#!/usr/bin/env python3
"""
run_phase9_a_coefficient_sweep.py
Phase 9.4A: R-sweep analysis to understand the "a" coefficient.

This script investigates how the implied "a" coefficient varies with R,
where the empirical formula is:
    m1 = a × exp(R) + 5

From Phase 8 fitting:
    a ≈ 1.037 (fitted to minimize c gap)
    a = 1.0   (empirical formula m1 = exp(R) + 5)

The question: Is a(R) constant or R-dependent?

We compute:
1. m1_derived = S12_mirror_derived / S12_minus_basis
2. a = (m1_derived - 5) / exp(R)
3. Fit to candidate families

Candidate families:
1. m₁ = exp(R) + 5                  (current empirical)
2. m₁ = a × exp(R) + 5              (fitted a)
3. m₁ = exp(R) × (1 + c·θ) + 5      (theta correction)
4. m₁ = exp(R·(1+ε)) + 5            (effective R shift)
"""

import sys
import math
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, "src")

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from mirror_exact import (
    compute_S12_mirror_derived,
    compute_S12_minus_basis,
    compute_I1_with_shifted_Q,
    _compute_I2_with_shifted_Q,
)


def compute_S12_plus_std(theta: float, R: float, n: int, polynomials: Dict) -> float:
    """Compute S12(+R) with standard Q (no shift)."""
    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        full_norm = symmetry[pair_key] * factorial_norm[pair_key]

        I1 = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )
        I2 = _compute_I2_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )
        total += full_norm * (I1 + I2)

    return total


def sweep_R_values(
    R_values: List[float],
    polynomials: Dict,
    theta: float = 4/7,
    n: int = 40,
) -> List[Dict]:
    """
    Sweep over R values and compute derived quantities.

    Returns list of dicts with computed values at each R.
    """
    results = []

    for R in R_values:
        # Compute channels
        S12_plus = compute_S12_plus_std(theta, R, n, polynomials)
        S12_minus = compute_S12_minus_basis(theta=theta, R=R, n=n, polynomials=polynomials)
        S12_derived = compute_S12_mirror_derived(theta=theta, R=R, n=n, polynomials=polynomials)

        exp_R = math.exp(R)
        exp_2R = math.exp(2 * R)

        # Implied m1 values
        m1_derived = S12_derived / S12_minus if abs(S12_minus) > 1e-15 else float('nan')
        m1_empirical = exp_R + 5

        # "a" coefficient: m1_derived = a × exp(R) + 5
        a_derived = (m1_derived - 5) / exp_R if math.isfinite(m1_derived) else float('nan')

        # Alternative: from exp(2R) × S12(+R, std Q) / S12(-R)
        m1_from_std = (exp_2R * S12_plus) / S12_minus if abs(S12_minus) > 1e-15 else float('nan')
        a_from_std = (m1_from_std - 5) / exp_R if math.isfinite(m1_from_std) else float('nan')

        # Plus/minus ratio
        ratio_plus_minus = S12_plus / S12_minus if abs(S12_minus) > 1e-15 else float('nan')

        results.append({
            "R": R,
            "exp_R": exp_R,
            "exp_2R": exp_2R,
            "S12_plus": S12_plus,
            "S12_minus": S12_minus,
            "S12_derived": S12_derived,
            "ratio_plus_minus": ratio_plus_minus,
            "m1_empirical": m1_empirical,
            "m1_derived": m1_derived,
            "m1_from_std": m1_from_std,
            "a_derived": a_derived,
            "a_from_std": a_from_std,
            "m1_ratio": m1_derived / m1_empirical if math.isfinite(m1_derived) else float('nan'),
        })

    return results


def print_sweep_table(results: List[Dict], title: str):
    """Print sweep results as a table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    print(f"\n{'R':>8} {'exp(R)':>10} {'S12+':>12} {'S12-':>12} {'ratio':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['R']:>8.4f} {r['exp_R']:>10.4f} {r['S12_plus']:>12.6f} "
              f"{r['S12_minus']:>12.6f} {r['ratio_plus_minus']:>10.4f}")

    print(f"\n{'R':>8} {'m1_emp':>10} {'m1_std':>12} {'m1_der':>12} {'a_der':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['R']:>8.4f} {r['m1_empirical']:>10.4f} {r['m1_from_std']:>12.4f} "
              f"{r['m1_derived']:>12.4f} {r['a_derived']:>10.6f}")


def analyze_a_coefficient(results: List[Dict]):
    """Analyze the "a" coefficient across R values."""
    Rs = np.array([r["R"] for r in results])
    a_derived = np.array([r["a_derived"] for r in results])
    a_from_std = np.array([r["a_from_std"] for r in results])

    # Filter out NaN
    valid_der = np.isfinite(a_derived)
    valid_std = np.isfinite(a_from_std)

    print("\n" + "=" * 60)
    print("  Analysis of 'a' coefficient (m1 = a × exp(R) + 5)")
    print("=" * 60)

    if np.any(valid_der):
        print(f"\na_derived (from Q(1+·) shifted):")
        print(f"  Mean:   {np.mean(a_derived[valid_der]):.6f}")
        print(f"  Std:    {np.std(a_derived[valid_der]):.6f}")
        print(f"  Min:    {np.min(a_derived[valid_der]):.6f}")
        print(f"  Max:    {np.max(a_derived[valid_der]):.6f}")
        print(f"  Range:  {np.max(a_derived[valid_der]) - np.min(a_derived[valid_der]):.6f}")

    if np.any(valid_std):
        print(f"\na_from_std (from exp(2R) × S12+/S12-):")
        print(f"  Mean:   {np.mean(a_from_std[valid_std]):.6f}")
        print(f"  Std:    {np.std(a_from_std[valid_std]):.6f}")
        print(f"  Min:    {np.min(a_from_std[valid_std]):.6f}")
        print(f"  Max:    {np.max(a_from_std[valid_std]):.6f}")
        print(f"  Range:  {np.max(a_from_std[valid_std]) - np.min(a_from_std[valid_std]):.6f}")

    # Check if a(R) is constant or varies
    if np.any(valid_der) and len(Rs[valid_der]) > 1:
        # Simple linear regression: a = b0 + b1*R
        X = Rs[valid_der]
        Y = a_derived[valid_der]
        n = len(X)
        if n > 1:
            b1 = (n * np.sum(X * Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X**2) - np.sum(X)**2)
            b0 = (np.sum(Y) - b1 * np.sum(X)) / n
            print(f"\n  Linear fit: a = {b0:.6f} + {b1:.6f} × R")
            print(f"  (a ≈ 1.037 from Phase 8 fitting)")


def fit_candidate_families(results: List[Dict], c_targets: Dict[float, float]):
    """
    Fit the data to candidate m1 families.

    Candidate families:
    1. m₁ = exp(R) + 5           (empirical)
    2. m₁ = a × exp(R) + 5       (fitted a)
    3. m₁ = exp(R·(1+ε)) + 5     (effective R shift)
    """
    print("\n" + "=" * 60)
    print("  Candidate Family Fitting")
    print("=" * 60)

    Rs = np.array([r["R"] for r in results])
    m1_derived = np.array([r["m1_derived"] for r in results])

    valid = np.isfinite(m1_derived)
    Rs_valid = Rs[valid]
    m1_valid = m1_derived[valid]

    if len(Rs_valid) < 2:
        print("Not enough valid points for fitting")
        return

    # Family 1: m1 = exp(R) + 5 (no fitting)
    m1_family1 = np.exp(Rs_valid) + 5
    sse1 = np.sum((m1_valid - m1_family1)**2)
    print(f"\n1. m₁ = exp(R) + 5 (empirical)")
    print(f"   SSE = {sse1:.4f}")

    # Family 2: m1 = a × exp(R) + 5
    # Minimize sum((m1_derived - (a × exp(R) + 5))^2)
    # d/da = 0 => a = sum((m1_der - 5) * exp(R)) / sum(exp(R)^2)
    exp_Rs = np.exp(Rs_valid)
    a_opt = np.sum((m1_valid - 5) * exp_Rs) / np.sum(exp_Rs**2)
    m1_family2 = a_opt * exp_Rs + 5
    sse2 = np.sum((m1_valid - m1_family2)**2)
    print(f"\n2. m₁ = a × exp(R) + 5 (fitted)")
    print(f"   Optimal a = {a_opt:.6f}")
    print(f"   SSE = {sse2:.4f}")

    # Family 3: m1 = exp(R × (1+ε)) + 5
    # This is nonlinear; use simple grid search
    best_eps = 0.0
    best_sse3 = float('inf')
    for eps in np.linspace(-0.5, 0.5, 101):
        m1_test = np.exp(Rs_valid * (1 + eps)) + 5
        sse = np.sum((m1_valid - m1_test)**2)
        if sse < best_sse3:
            best_sse3 = sse
            best_eps = eps
    m1_family3 = np.exp(Rs_valid * (1 + best_eps)) + 5
    print(f"\n3. m₁ = exp(R × (1+ε)) + 5 (effective R shift)")
    print(f"   Optimal ε = {best_eps:.6f}")
    print(f"   SSE = {best_sse3:.4f}")

    print("\n" + "-" * 40)
    print("Note: SSE values are relative to m1_derived, which is ~100× larger")
    print("than m1_empirical. The families are fitted to the wrong target.")


def main():
    """Run R-sweep analysis."""
    print("=" * 80)
    print("  PHASE 9.4A: R-Sweep Analysis for 'a' Coefficient")
    print("  Investigating how a(R) varies in m1 = a × exp(R) + 5")
    print("=" * 80)

    # Load polynomials
    print("\nLoading κ benchmark polynomials...")
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    print("Loading κ* benchmark polynomials...")
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # R values to sweep
    R_values = [0.8, 0.9, 1.0, 1.1167, 1.2, 1.3036, 1.4, 1.5]

    n = 40  # Use modest quadrature for speed

    # Sweep for κ polynomials
    print(f"\nSweeping R ∈ {R_values}")
    print("Using κ benchmark polynomials...")

    results_kappa = sweep_R_values(R_values, polynomials_kappa, n=n)
    print_sweep_table(results_kappa, "R-Sweep with κ polynomials")
    analyze_a_coefficient(results_kappa)

    # Sweep for κ* polynomials
    print("\n" + "=" * 80)
    print("Using κ* benchmark polynomials...")
    results_kappa_star = sweep_R_values(R_values, polynomials_kappa_star, n=n)
    print_sweep_table(results_kappa_star, "R-Sweep with κ* polynomials")
    analyze_a_coefficient(results_kappa_star)

    # Compare the two at their respective benchmark R values
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPARISON")
    print("=" * 80)

    kappa_at_bench = [r for r in results_kappa if abs(r["R"] - 1.3036) < 0.01][0]
    kappa_star_at_bench = [r for r in results_kappa_star if abs(r["R"] - 1.1167) < 0.01][0]

    print(f"\nκ benchmark (R = 1.3036, κ polynomials):")
    print(f"  m1_empirical = {kappa_at_bench['m1_empirical']:.4f}")
    print(f"  m1_derived   = {kappa_at_bench['m1_derived']:.4f}")
    print(f"  a_derived    = {kappa_at_bench['a_derived']:.6f}")

    print(f"\nκ* benchmark (R = 1.1167, κ* polynomials):")
    print(f"  m1_empirical = {kappa_star_at_bench['m1_empirical']:.4f}")
    print(f"  m1_derived   = {kappa_star_at_bench['m1_derived']:.4f}")
    print(f"  a_derived    = {kappa_star_at_bench['a_derived']:.6f}")

    # Key insight
    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("-" * 60)
    print(f"""
The Q(1+·) shift creates a LARGE amplification factor that doesn't
match the empirical m1 = exp(R) + 5 formula.

Derived 'a' coefficients:
  κ benchmark:  a = {kappa_at_bench['a_derived']:.1f}  (expected ~1.037)
  κ* benchmark: a = {kappa_star_at_bench['a_derived']:.1f}  (expected ~1.037)

The shift identity Q(D)[T^{{-s}}F] = T^{{-s}}Q(1+D)[F] is mathematically correct,
but applying it naively to the mirror term gives wrong numerical values.

This suggests either:
1. The PRZZ mirror assembly uses a different transformation
2. There's a normalization factor we're missing
3. The empirical formula is capturing something more subtle

The '+5' term (= 2K-1 for K=3) absorbs polynomial-dependent effects
that our derived mirror doesn't correctly account for.
""")

    print("\nDone.")
    return results_kappa, results_kappa_star


if __name__ == "__main__":
    main()
