#!/usr/bin/env python3
"""
Phase 37B: I1 vs I2 Breakdown with Q Modes

Key insight from unified_i2_paper.py:
- I2 ALWAYS uses Q(t)² (frozen eigenvalues) - see line 8
- I1 uses full Q(Arg_α)×Q(Arg_β) with x,y dependence

This explains why microcase experiments showed only ~0.4% Q effect:
I2 dominates numerically and is already "frozen-Q".

This script computes:
1. I1 and I2 separately
2. Their relative contributions to S12
3. How the Q mode affects the I1/I2 ratio

Created: 2025-12-26 (Phase 37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from src.polynomials import load_przz_polynomials
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode


def main():
    print("=" * 70)
    print("PHASE 37B: I1 vs I2 BREAKDOWN WITH Q MODES")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4 / 7
    R = 1.3036
    n_quad = 60

    # Factorial normalization
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}
    pairs = ["11", "22", "33", "12", "13", "23"]

    print("KEY INSIGHT: I2 always uses Q(t)² (frozen eigenvalue)")
    print("             I1 uses full Q(Arg_α)×Q(Arg_β) with x,y dependence")
    print()

    # Compute I1 and I2 for each pair
    print("PER-PAIR BREAKDOWN (R=+1.3036)")
    print("-" * 70)
    print(f"{'Pair':<6} | {'I1(normal)':<12} | {'I2 (frozen)':<12} | {'I1/I2':<10} | {'I1 share':<10}")
    print("-" * 70)

    total_I1 = 0.0
    total_I2 = 0.0

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        # I1 with normal Q (full x,y dependence)
        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True, apply_factorial_norm=True,
        )

        # I2 (always uses Q(t)²)
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )

        norm = f_norm[pair_key] * symmetry[pair_key]
        I1_normed = I1_result.I1_value * norm
        I2_normed = I2_result.I2_value * norm

        ratio = I1_normed / I2_normed if abs(I2_normed) > 1e-15 else float('inf')
        share = abs(I1_normed) / (abs(I1_normed) + abs(I2_normed)) * 100 if abs(I1_normed) + abs(I2_normed) > 1e-15 else 0

        print(f"({ell1},{ell2})  | {I1_normed:+.6f}   | {I2_normed:+.6f}   | {ratio:+.4f}   | {share:.1f}%")

        total_I1 += I1_normed
        total_I2 += I2_normed

    print("-" * 70)
    total_S12 = total_I1 + total_I2
    I1_share = abs(total_I1) / (abs(total_I1) + abs(total_I2)) * 100
    print(f"TOTAL  | {total_I1:+.6f}   | {total_I2:+.6f}   | {total_I1/total_I2:+.4f}   | {I1_share:.1f}%")
    print()

    print("S12 SUMMARY")
    print("-" * 70)
    print(f"  Total I1 (has Q derivatives): {total_I1:+.8f}")
    print(f"  Total I2 (frozen Q):          {total_I2:+.8f}")
    print(f"  S12 = I1 + I2:                {total_S12:+.8f}")
    print(f"  I1 share of S12:              {I1_share:.2f}%")
    print()

    # Now compute I1 with different Q modes to see the derivative effect
    print("I1 WITH DIFFERENT Q MODES (summed over all pairs)")
    print("-" * 70)

    I1_modes = {"none": 0.0, "frozen": 0.0, "normal": 0.0}

    for q_mode in ["none", "frozen", "normal"]:
        for pair_key in pairs:
            ell1 = int(pair_key[0])
            ell2 = int(pair_key[1])

            I1 = compute_I1_with_Q_mode(
                R, theta, ell1, ell2, polynomials,
                q_mode=q_mode, n_quad_u=n_quad,
            )
            norm = f_norm[pair_key] * symmetry[pair_key]
            I1_modes[q_mode] += I1 * norm

    print(f"  I1(Q=1):      {I1_modes['none']:+.8f}")
    print(f"  I1(frozen):   {I1_modes['frozen']:+.8f}")
    print(f"  I1(normal):   {I1_modes['normal']:+.8f}")
    print()

    # Q derivative effect on I1
    deriv_effect = (I1_modes['normal'] - I1_modes['frozen']) / abs(total_S12) * 100
    print(f"  Q derivative effect on I1: {deriv_effect:+.4f}% of S12")
    print()

    print("CONCLUSION")
    print("-" * 70)
    print(f"  I1 is only {I1_share:.1f}% of S12, so Q derivative effects are diluted.")
    print(f"  The Q derivative effect on I1 alone is only {deriv_effect:+.4f}% of total S12.")
    print()
    print("  This explains why the microcase experiments showed only ~0.4% Q effect:")
    print("  → I2 dominates and always uses Q(t)² (frozen)")
    print("  → I1's Q derivative effects are diluted by the larger I2 contribution")
    print()

    # Check if this explains the ±0.13% residual
    print("RELATING TO THE ±0.13% RESIDUAL")
    print("-" * 70)
    print("  The Beta moment correction = 1 + θ/42 = 1.01361 applies to m (mirror multiplier).")
    print("  The m formula multiplies S12(-R), not I1 alone.")
    print()
    print("  For the correction to be accurate, we need to understand how")
    print("  Q affects the RATIO S12(+R)/S12(-R), not just the absolute values.")


if __name__ == "__main__":
    main()
