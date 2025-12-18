"""
Comprehensive analysis of ALL pairs to understand the ratio reversal.

This expands beyond just (2,2) to check if other pairs contribute differently.
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from przz_22_exact_oracle import przz_oracle_22


def compute_pair(label, P, Q, R, theta):
    """Compute oracle results for a single pair."""
    result = przz_oracle_22(P, Q, theta, R, n_quad=80, debug=False)
    return {
        'label': label,
        'I1': result.I1,
        'I2': result.I2,
        'I3': result.I3,
        'I4': result.I4,
        'total': result.total,
        'deriv_sum': result.I1 + result.I3 + result.I4,
    }


def main():
    theta = 4.0 / 7.0

    # Load both sets
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167

    print("="*70)
    print("COMPREHENSIVE PAIR ANALYSIS: κ vs κ*")
    print("="*70)

    # Define all pairs
    pairs_k = [
        ('(1,1)', P1_k, Q_k, R_kappa),
        ('(2,2)', P2_k, Q_k, R_kappa),
        ('(3,3)', P3_k, Q_k, R_kappa),
    ]

    pairs_ks = [
        ('(1,1)', P1_ks, Q_ks, R_kappa_star),
        ('(2,2)', P2_ks, Q_ks, R_kappa_star),
        ('(3,3)', P3_ks, Q_ks, R_kappa_star),
    ]

    # Compute all
    results_k = [compute_pair(label, P, Q, R, theta) for label, P, Q, R in pairs_k]
    results_ks = [compute_pair(label, P, Q, R, theta) for label, P, Q, R in pairs_ks]

    # Display table
    print("\nκ BENCHMARK (R=1.3036):")
    print(f"{'Pair':<8} {'I₁':>10} {'I₂':>10} {'I₃':>10} {'I₄':>10} {'Deriv':>10} {'Total':>10}")
    print("-"*70)
    for r in results_k:
        print(f"{r['label']:<8} {r['I1']:>10.4f} {r['I2']:>10.4f} {r['I3']:>10.4f} "
              f"{r['I4']:>10.4f} {r['deriv_sum']:>10.4f} {r['total']:>10.4f}")

    print("\nκ* BENCHMARK (R=1.1167):")
    print(f"{'Pair':<8} {'I₁':>10} {'I₂':>10} {'I₃':>10} {'I₄':>10} {'Deriv':>10} {'Total':>10}")
    print("-"*70)
    for r in results_ks:
        print(f"{r['label']:<8} {r['I1']:>10.4f} {r['I2']:>10.4f} {r['I3']:>10.4f} "
              f"{r['I4']:>10.4f} {r['deriv_sum']:>10.4f} {r['total']:>10.4f}")

    # Compute ratios
    print("\nRATIO ANALYSIS (κ / κ*):")
    print(f"{'Pair':<8} {'I₂ ratio':>10} {'Deriv ratio':>10} {'Total ratio':>10}")
    print("-"*70)
    for rk, rks in zip(results_k, results_ks):
        i2_ratio = rk['I2'] / rks['I2']
        deriv_ratio = rk['deriv_sum'] / rks['deriv_sum'] if rks['deriv_sum'] != 0 else float('inf')
        total_ratio = rk['total'] / rks['total']
        print(f"{rk['label']:<8} {i2_ratio:>10.4f} {deriv_ratio:>10.4f} {total_ratio:>10.4f}")

    # Aggregate totals
    total_I2_k = sum(r['I2'] for r in results_k)
    total_I2_ks = sum(r['I2'] for r in results_ks)
    total_deriv_k = sum(r['deriv_sum'] for r in results_k)
    total_deriv_ks = sum(r['deriv_sum'] for r in results_ks)
    total_c_k = sum(r['total'] for r in results_k)
    total_c_ks = sum(r['total'] for r in results_ks)

    print("\nAGGREGATE (diagonal pairs only):")
    print(f"  I₂ total:    κ={total_I2_k:.4f}, κ*={total_I2_ks:.4f}, ratio={total_I2_k/total_I2_ks:.4f}")
    print(f"  Deriv total: κ={total_deriv_k:.4f}, κ*={total_deriv_ks:.4f}, ratio={total_deriv_k/total_deriv_ks:.4f}")
    print(f"  c total:     κ={total_c_k:.4f}, κ*={total_c_ks:.4f}, ratio={total_c_k/total_c_ks:.4f}")

    # Derivative as percentage of I₂
    print("\nDERIVATIVE CONTRIBUTION AS % OF I₂:")
    print(f"{'Pair':<8} {'κ %':>10} {'κ* %':>10}")
    print("-"*70)
    for rk, rks in zip(results_k, results_ks):
        pct_k = 100 * rk['deriv_sum'] / rk['I2'] if rk['I2'] != 0 else 0
        pct_ks = 100 * rks['deriv_sum'] / rks['I2'] if rks['I2'] != 0 else 0
        print(f"{rk['label']:<8} {pct_k:>9.2f}% {pct_ks:>9.2f}%")

    # KEY INSIGHT
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print("\n1. I₂-only ratios (naive formula):")
    for rk, rks in zip(results_k, results_ks):
        ratio = rk['I2'] / rks['I2']
        print(f"   {rk['label']}: {ratio:.4f}")

    print("\n2. Derivative terms as fraction of I₂:")
    print("   κ benchmark:")
    for r in results_k:
        frac = r['deriv_sum'] / r['I2'] if r['I2'] != 0 else 0
        print(f"     {r['label']}: {frac:.4f} ({100*frac:.1f}%)")
    print("   κ* benchmark:")
    for r in results_ks:
        frac = r['deriv_sum'] / r['I2'] if r['I2'] != 0 else 0
        print(f"     {r['label']}: {frac:.4f} ({100*frac:.1f}%)")

    print("\n3. The derivative contribution is SMALLER for κ than κ*:")
    print(f"   κ:  {total_deriv_k/total_I2_k:.4f} ({100*total_deriv_k/total_I2_k:.1f}%)")
    print(f"   κ*: {total_deriv_ks/total_I2_ks:.4f} ({100*total_deriv_ks/total_I2_ks:.1f}%)")

    print("\n4. This means derivatives INCREASE the ratio, not decrease it!")
    print("   The hypothesis was BACKWARDS.")

    # What would explain the ratio?
    print("\n" + "="*70)
    print("WHAT WOULD EXPLAIN THE RATIO REVERSAL?")
    print("="*70)

    target_ratio = 0.94
    # For full c (including off-diagonal pairs), PRZZ gives:
    # c(κ) / c(κ*) = 2.137 / 1.939 = 1.102

    actual_ratio_diag = total_c_k / total_c_ks
    print(f"\nDiagonal pairs give ratio: {actual_ratio_diag:.4f}")
    print(f"Target (full c) ratio: ~1.10")
    print(f"Target (κ) ratio: 0.94")

    print("\nPossible explanations:")
    print("  1. Off-diagonal pairs (1,2), (1,3), (2,3) have different ratios")
    print("  2. I₅ arithmetic correction differs between κ and κ*")
    print("  3. The polynomial degree difference affects I₂ more than derivatives")
    print("  4. Missing normalization factor that depends on polynomial degree")

    # Degree analysis
    print("\n" + "="*70)
    print("POLYNOMIAL DEGREE ANALYSIS")
    print("="*70)

    print("\nκ benchmark:")
    print(f"  P₁: degree {P1_k.to_monomial().degree}")
    print(f"  P₂: degree {P2_k.to_monomial().degree}")
    print(f"  P₃: degree {P3_k.to_monomial().degree}")
    print(f"  Q:  degree {Q_k.to_monomial().degree}")

    print("\nκ* benchmark:")
    print(f"  P₁: degree {P1_ks.to_monomial().degree}")
    print(f"  P₂: degree {P2_ks.to_monomial().degree}")
    print(f"  P₃: degree {P3_ks.to_monomial().degree}")
    print(f"  Q:  degree {Q_ks.to_monomial().degree}")

    print("\nHigher degree polynomials have:")
    print("  ✓ Larger I₂ (polynomial squared in integrand)")
    print("  ✓ Larger derivatives")
    print("  ? Net effect depends on balance between I₂ growth and derivative subtraction")


if __name__ == "__main__":
    main()
