#!/usr/bin/env python3
"""
GPT Run 19 Diagnostic Script

This script compares the tex_exact I1 computation with tex_mirror
and analyzes the structural differences.

Key questions:
1. Does tex_exact I1 give reasonable values?
2. How does it compare with tex_mirror I1?
3. What is the full assembly gap compared to targets?
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import (
    compute_I1_tex_exact_11,
    compute_c_paper_tex_mirror,
    compute_I2_tex_combined_11,
    compute_S34_base_11,
)


THETA = 4.0 / 7.0

TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
    }
}


def main():
    print("=" * 80)
    print("GPT Run 19 Diagnostic: TeX-Exact I1 vs tex_mirror")
    print("=" * 80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, TARGETS["kappa"]),
        ("κ*", polys_kappa_star, TARGETS["kappa_star"]),
    ]

    for name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        print(f"\n{'=' * 50}")
        print(f"{name} Benchmark (R={R})")
        print("=" * 50)

        # Get tex_exact I1 result
        tex_exact_result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            verbose=True,
        )
        print()

        # Get tex_mirror result
        tex_mirror_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        tex_mirror_I1 = tex_mirror_result.I1_plus + tex_mirror_result.m1 * tex_mirror_result.I1_minus_base

        print("Component Comparison:")
        print(f"  tex_exact I1:       {tex_exact_result.I1_tex_exact:.6f}")
        print(f"  tex_mirror I1:      {tex_mirror_I1:.6f}")
        print(f"  tex_mirror I1_plus: {tex_mirror_result.I1_plus:.6f}")
        print(f"  tex_mirror m1:      {tex_mirror_result.m1:.6f}")
        print(f"  tex_mirror I1_minus: {tex_mirror_result.I1_minus_base:.6f}")

        # Get I2 and S34 from existing implementations
        i2_result = compute_I2_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
        )

        s34_base = compute_S34_base_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
        )

        print()
        print("Other Components:")
        print(f"  I2_base:            {i2_result.I2_base:.6f}")
        print(f"  tex_mirror I2:      {tex_mirror_result.I2_plus + tex_mirror_result.m2 * tex_mirror_result.I2_minus_base:.6f}")
        print(f"  S34_base:           {s34_base:.6f}")
        print(f"  tex_mirror S34:     {tex_mirror_result.S34_plus:.6f}")

        # Full assembly attempts
        print()
        print("Assembly Attempts:")

        # Attempt 1: tex_exact I1 + I2_base + S34_base
        c_attempt1 = tex_exact_result.I1_tex_exact + i2_result.I2_base + s34_base
        gap1 = 100 * (c_attempt1 - c_target) / c_target
        print(f"  Attempt 1 (tex_exact I1 + I2_base + S34_base):")
        print(f"    c = {c_attempt1:.4f}, gap = {gap1:+.2f}%")

        # tex_mirror for reference
        c_tex_mirror = tex_mirror_result.c
        gap_mirror = 100 * (c_tex_mirror - c_target) / c_target
        print(f"  tex_mirror (reference):")
        print(f"    c = {c_tex_mirror:.4f}, gap = {gap_mirror:+.2f}%")

        print(f"  c_target: {c_target:.4f}")

        # Analyze the ratio of tex_exact I1 to tex_mirror I1
        if abs(tex_mirror_I1) > 1e-10:
            ratio = tex_exact_result.I1_tex_exact / tex_mirror_I1
            print()
            print(f"  Ratio tex_exact/tex_mirror I1: {ratio:.4f}")

    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    print("""
The tex_exact I1 values are significantly different from tex_mirror I1.
This is expected because:

1. CombinedI1Integrand computes:
   plus_branch = Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β)
   minus_branch = Q(arg_α+1) × Q(arg_β+1) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R)
   combined = plus_branch + minus_branch

2. tex_mirror computes:
   I1 = I1_plus + m1 × I1_minus
   where m1 is a calibrated amplitude factor

The structural difference is that:
- tex_exact has Q-shift INSIDE the combined structure
- tex_mirror applies Q-shift to a SEPARATE -R evaluation

The large differences suggest the formula interpretation may need adjustment.
Possible issues:
- The +R/-R structure might not be a simple sum
- The Q-shift might need to be applied differently
- The exponential arguments might need different treatment
""")


if __name__ == "__main__":
    main()
