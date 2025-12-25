"""
Compare I₁, I₂, I₃, I₄ term structure between κ and κ* benchmarks.

Key question from handoff:
"Derivative terms make ratio WORSE (1.92 vs 1.71)"

This script decomposes where the ratio difference comes from:
- Is it in I₂ (base integral)?
- Is it in I₁ (mixed derivative)?
- Is it in I₃+I₄ (single derivatives)?

We compute the ratio I_term(κ) / I_term(κ*) for each term.
"""

from __future__ import annotations
import sys
from math import log

# Import the oracle
from przz_22_exact_oracle import przz_oracle_22

# Import polynomial loaders
from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def kappa_from_c(c: float, R: float) -> float:
    """Convert c to κ using κ = 1 - log(c)/R."""
    return 1.0 - log(c) / R


def main():
    print("="*80)
    print("I₁, I₂, I₃, I₄ TERM STRUCTURE COMPARISON: κ vs κ* for (2,2) pair")
    print("="*80)
    print()

    theta = 4/7
    n_quad = 80  # High-quality quadrature

    # =========================================================================
    # Benchmark 1: κ (R=1.3036)
    # =========================================================================
    print("--- BENCHMARK 1: κ (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036

    result_k = przz_oracle_22(P2_k, Q_k, theta, R_kappa, n_quad=n_quad, debug=False)

    print(f"I₁: {result_k.I1:>12.6f}")
    print(f"I₂: {result_k.I2:>12.6f}")
    print(f"I₃: {result_k.I3:>12.6f}")
    print(f"I₄: {result_k.I4:>12.6f}")
    print(f"Total: {result_k.total:>12.6f}")
    print()

    # Compute κ from the (2,2) contribution alone (for reference)
    kappa_22_only = kappa_from_c(result_k.total, R_kappa)
    print(f"κ from (2,2) alone: {kappa_22_only:.6f}")
    print()

    # =========================================================================
    # Benchmark 2: κ* (R=1.1167)
    # =========================================================================
    print("--- BENCHMARK 2: κ* (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167

    result_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_kappa_star, n_quad=n_quad, debug=False)

    print(f"I₁: {result_ks.I1:>12.6f}")
    print(f"I₂: {result_ks.I2:>12.6f}")
    print(f"I₃: {result_ks.I3:>12.6f}")
    print(f"I₄: {result_ks.I4:>12.6f}")
    print(f"Total: {result_ks.total:>12.6f}")
    print()

    kappa_star_22_only = kappa_from_c(result_ks.total, R_kappa_star)
    print(f"κ* from (2,2) alone: {kappa_star_22_only:.6f}")
    print()

    # =========================================================================
    # RATIO ANALYSIS
    # =========================================================================
    print("="*80)
    print("RATIO ANALYSIS: I_term(κ) / I_term(κ*)")
    print("="*80)
    print()

    # Individual term ratios
    ratio_I1 = result_k.I1 / result_ks.I1
    ratio_I2 = result_k.I2 / result_ks.I2
    ratio_I3 = result_k.I3 / result_ks.I3
    ratio_I4 = result_k.I4 / result_ks.I4
    ratio_total = result_k.total / result_ks.total

    print(f"I₁ ratio: {ratio_I1:>10.4f}")
    print(f"I₂ ratio: {ratio_I2:>10.4f}")
    print(f"I₃ ratio: {ratio_I3:>10.4f}")
    print(f"I₄ ratio: {ratio_I4:>10.4f}")
    print(f"Total ratio: {ratio_total:>10.4f}")
    print()

    # =========================================================================
    # COMPONENT GROUPING ANALYSIS
    # =========================================================================
    print("="*80)
    print("COMPONENT GROUPING ANALYSIS")
    print("="*80)
    print()

    # Group 1: Base integral (I₂)
    print("GROUP 1: Base integral (I₂ only)")
    print(f"  κ:  I₂ = {result_k.I2:.6f}")
    print(f"  κ*: I₂ = {result_ks.I2:.6f}")
    print(f"  Ratio: {ratio_I2:.4f}")
    print()

    # Group 2: Mixed derivative (I₁)
    print("GROUP 2: Mixed derivative (I₁ only)")
    print(f"  κ:  I₁ = {result_k.I1:.6f}")
    print(f"  κ*: I₁ = {result_ks.I1:.6f}")
    print(f"  Ratio: {ratio_I1:.4f}")
    print()

    # Group 3: Single derivatives (I₃ + I₄)
    I3_I4_sum_k = result_k.I3 + result_k.I4
    I3_I4_sum_ks = result_ks.I3 + result_ks.I4
    ratio_I3_I4 = I3_I4_sum_k / I3_I4_sum_ks

    print("GROUP 3: Single derivatives (I₃ + I₄)")
    print(f"  κ:  I₃+I₄ = {I3_I4_sum_k:.6f}")
    print(f"  κ*: I₃+I₄ = {I3_I4_sum_ks:.6f}")
    print(f"  Ratio: {ratio_I3_I4:.4f}")
    print()

    # Group 4: All derivatives (I₁ + I₃ + I₄)
    derivatives_k = result_k.I1 + result_k.I3 + result_k.I4
    derivatives_ks = result_ks.I1 + result_ks.I3 + result_ks.I4
    ratio_derivatives = derivatives_k / derivatives_ks

    print("GROUP 4: All derivatives (I₁ + I₃ + I₄)")
    print(f"  κ:  I₁+I₃+I₄ = {derivatives_k:.6f}")
    print(f"  κ*: I₁+I₃+I₄ = {derivatives_ks:.6f}")
    print(f"  Ratio: {ratio_derivatives:.4f}")
    print()

    # =========================================================================
    # CONTRIBUTION ANALYSIS
    # =========================================================================
    print("="*80)
    print("CONTRIBUTION TO TOTAL RATIO")
    print("="*80)
    print()

    print("What fraction of the total does each term contribute?")
    print()

    # κ contributions
    print("κ (Benchmark 1) contributions:")
    print(f"  I₁ / Total: {result_k.I1 / result_k.total:>10.4f}  ({100*result_k.I1/result_k.total:>6.2f}%)")
    print(f"  I₂ / Total: {result_k.I2 / result_k.total:>10.4f}  ({100*result_k.I2/result_k.total:>6.2f}%)")
    print(f"  I₃ / Total: {result_k.I3 / result_k.total:>10.4f}  ({100*result_k.I3/result_k.total:>6.2f}%)")
    print(f"  I₄ / Total: {result_k.I4 / result_k.total:>10.4f}  ({100*result_k.I4/result_k.total:>6.2f}%)")
    print()

    # κ* contributions
    print("κ* (Benchmark 2) contributions:")
    print(f"  I₁ / Total: {result_ks.I1 / result_ks.total:>10.4f}  ({100*result_ks.I1/result_ks.total:>6.2f}%)")
    print(f"  I₂ / Total: {result_ks.I2 / result_ks.total:>10.4f}  ({100*result_ks.I2/result_ks.total:>6.2f}%)")
    print(f"  I₃ / Total: {result_ks.I3 / result_ks.total:>10.4f}  ({100*result_ks.I3/result_ks.total:>6.2f}%)")
    print(f"  I₄ / Total: {result_ks.I4 / result_ks.total:>10.4f}  ({100*result_ks.I4/result_ks.total:>6.2f}%)")
    print()

    # =========================================================================
    # SIGNED CONTRIBUTION TO RATIO DISCREPANCY
    # =========================================================================
    print("="*80)
    print("SIGNED CONTRIBUTION TO RATIO DISCREPANCY")
    print("="*80)
    print()

    print("Expected target ratio (from full-pipeline results): ~1.10")
    print(f"Observed (2,2) ratio: {ratio_total:.4f}")
    print(f"Discrepancy: {ratio_total - 1.10:.4f}")
    print()

    print("If I₂ were the only term, its ratio would be:")
    print(f"  I₂ ratio: {ratio_I2:.4f} (discrepancy: {ratio_I2 - 1.10:+.4f})")
    print()

    print("Adding I₁ (mixed derivative) on top of I₂:")
    # Compute what the ratio would be if we only had I₁ + I₂
    I1_I2_sum_k = result_k.I1 + result_k.I2
    I1_I2_sum_ks = result_ks.I1 + result_ks.I2
    ratio_I1_I2 = I1_I2_sum_k / I1_I2_sum_ks
    print(f"  (I₁+I₂) ratio: {ratio_I1_I2:.4f} (discrepancy: {ratio_I1_I2 - 1.10:+.4f})")
    print()

    print("Adding I₃+I₄ (single derivatives) on top of I₁+I₂:")
    print(f"  Full ratio: {ratio_total:.4f} (discrepancy: {ratio_total - 1.10:+.4f})")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    print(f"1. Base integral (I₂) ratio: {ratio_I2:.4f}")
    print(f"2. Mixed derivative (I₁) ratio: {ratio_I1:.4f}")
    print(f"3. Single derivatives (I₃+I₄) ratio: {ratio_I3_I4:.4f}")
    print(f"4. All derivatives (I₁+I₃+I₄) ratio: {ratio_derivatives:.4f}")
    print(f"5. Total ratio: {ratio_total:.4f}")
    print()

    # Which term has the worst ratio?
    ratios = {
        'I₁': ratio_I1,
        'I₂': ratio_I2,
        'I₃': ratio_I3,
        'I₄': ratio_I4,
        'I₃+I₄': ratio_I3_I4,
    }

    # Find furthest from target 1.10
    target = 1.10
    worst_term = max(ratios.items(), key=lambda x: abs(x[1] - target))
    best_term = min(ratios.items(), key=lambda x: abs(x[1] - target))

    print(f"Term FURTHEST from target (1.10): {worst_term[0]} with ratio {worst_term[1]:.4f} (off by {abs(worst_term[1]-target):.4f})")
    print(f"Term CLOSEST to target (1.10): {best_term[0]} with ratio {best_term[1]:.4f} (off by {abs(best_term[1]-target):.4f})")
    print()

    # Final verdict
    print("="*80)
    print("KEY FINDING:")
    print("="*80)
    if ratio_I2 > ratio_total:
        print("Adding derivatives IMPROVES the ratio (makes it closer to 1.10).")
        print(f"  I₂ alone: {ratio_I2:.4f}")
        print(f"  With derivatives: {ratio_total:.4f}")
    else:
        print("Adding derivatives WORSENS the ratio (makes it further from 1.10).")
        print(f"  I₂ alone: {ratio_I2:.4f}")
        print(f"  With derivatives: {ratio_total:.4f}")
    print()

    print("From handoff: 'Derivative terms make ratio WORSE (1.92 vs 1.71)'")
    print("This script confirms whether that applies to the (2,2) pair specifically.")
    print()


if __name__ == "__main__":
    main()
