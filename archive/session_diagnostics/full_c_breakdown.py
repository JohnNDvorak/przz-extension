#!/usr/bin/env python3
"""
Compute full c breakdown for both κ and κ* to understand the overall ratio.

This uses the V2 DSL to compute I₁, I₂, I₃, I₄ for each pair and sums them up.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from terms_k3_d1 import make_all_terms_k3_v2
from evaluate import evaluate_term
from quadrature import tensor_grid_2d

def compute_full_c(benchmark="kappa"):
    """Compute full c value with I₁+I₂+I₃+I₄ breakdown"""

    if benchmark == "kappa":
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        R = 1.3036
        theta = 4.0/7.0
        label = "κ"
    else:  # kappa_star
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
        R = 1.1167
        theta = 4.0/7.0
        label = "κ*"

    polys = [P1, P2, P3]

    print(f"\n{'=' * 80}")
    print(f"{label} FULL c COMPUTATION (R={R})")
    print(f"{'=' * 80}\n")

    # Quadrature setup
    n_quad = 80
    u_vals, t_vals, weights_2d = tensor_grid_2d(n_quad)

    # Get all terms
    all_terms_dict = make_all_terms_k3_v2(theta, R)

    total_I1 = 0.0
    total_I2 = 0.0
    total_I3 = 0.0
    total_I4 = 0.0

    print(f"{'Pair':<8} {'I₁':<15} {'I₂':<15} {'I₃':<15} {'I₄':<15} {'Sum':<15}")
    print("-" * 88)

    pair_keys = ['11', '12', '13', '22', '23', '33']

    for pair_key in pair_keys:
        # Get terms for this pair
        terms = all_terms_dict[pair_key]

        # Group by type (use name prefix)
        I1_terms = [t for t in terms if t.name.startswith("I1_")]
        I2_terms = [t for t in terms if t.name.startswith("I2_")]
        I3_terms = [t for t in terms if t.name.startswith("I3_")]
        I4_terms = [t for t in terms if t.name.startswith("I4_")]

        # Evaluate each type
        I1_val = sum(evaluate_term(t, polys, Q, R, theta, u_vals, t_vals, weights_2d, mode="main")
                    for t in I1_terms)
        I2_val = sum(evaluate_term(t, polys, Q, R, theta, u_vals, t_vals, weights_2d, mode="main")
                    for t in I2_terms)
        I3_val = sum(evaluate_term(t, polys, Q, R, theta, u_vals, t_vals, weights_2d, mode="main")
                    for t in I3_terms)
        I4_val = sum(evaluate_term(t, polys, Q, R, theta, u_vals, t_vals, weights_2d, mode="main")
                    for t in I4_terms)

        pair_sum = I1_val + I2_val + I3_val + I4_val

        print(f"({pair_key[0]},{pair_key[1]})     {I1_val:>14.8f} {I2_val:>14.8f} {I3_val:>14.8f} {I4_val:>14.8f} {pair_sum:>14.8f}")

        total_I1 += I1_val
        total_I2 += I2_val
        total_I3 += I3_val
        total_I4 += I4_val

    print("-" * 88)
    total = total_I1 + total_I2 + total_I3 + total_I4
    print(f"{'TOTAL':<8} {total_I1:>14.8f} {total_I2:>14.8f} {total_I3:>14.8f} {total_I4:>14.8f} {total:>14.8f}")
    print()

    print(f"Breakdown by component:")
    print(f"  I₁: {total_I1:>14.8f}  ({100*total_I1/total:>6.2f}%)")
    print(f"  I₂: {total_I2:>14.8f}  ({100*total_I2/total:>6.2f}%)")
    print(f"  I₃: {total_I3:>14.8f}  ({100*total_I3/total:>6.2f}%)")
    print(f"  I₄: {total_I4:>14.8f}  ({100*total_I4/total:>6.2f}%)")
    print(f"  Total c: {total:>14.8f}")
    print()

    kappa_val = 1 - np.log(total) / R
    print(f"Computed {label}: {kappa_val:.9f}")
    print()

    return {
        'I1': total_I1,
        'I2': total_I2,
        'I3': total_I3,
        'I4': total_I4,
        'c': total,
        'kappa': kappa_val
    }

def main():
    print("=" * 80)
    print("FULL c BREAKDOWN: κ vs κ*")
    print("=" * 80)

    # Compute both
    results_kappa = compute_full_c("kappa")
    results_star = compute_full_c("kappa_star")

    # Compare
    print("=" * 80)
    print("RATIO ANALYSIS: κ/κ*")
    print("=" * 80)
    print()

    components = ['I1', 'I2', 'I3', 'I4', 'c']

    print(f"{'Component':<12} {'κ value':<15} {'κ* value':<15} {'Ratio κ/κ*':<15}")
    print("-" * 60)

    for comp in components:
        val_k = results_kappa[comp]
        val_s = results_star[comp]
        ratio = val_k / val_s if abs(val_s) > 1e-15 else float('inf')

        print(f"{comp:<12} {val_k:>14.8f}  {val_s:>14.8f}  {ratio:>14.6f}")

    print()

    # Target comparison
    c_target_kappa = 2.13745440613217263636
    c_target_star = 1.9379524124677437
    target_ratio = c_target_kappa / c_target_star

    print(f"Target c ratio (from PRZZ): {target_ratio:.6f}")
    print(f"Computed c ratio:           {results_kappa['c']/results_star['c']:.6f}")
    print()

    ratio_error = (results_kappa['c']/results_star['c'] - target_ratio) / target_ratio
    print(f"Relative error in c ratio: {100*ratio_error:+.2f}%")
    print()

    # Key insight
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("If the computed c ratio matches the target ratio, then:")
    print("  - The I₂ component has one ratio")
    print("  - The derivative terms (I₁+I₃+I₄) must compensate to achieve the target")
    print("  - This is EXPECTED behavior, not a bug")
    print()
    print("If the computed c ratio DIFFERS from the target ratio, then:")
    print("  - There is a systematic error in the formulas")
    print("  - This is the source of the mystery")
    print()

if __name__ == "__main__":
    main()
