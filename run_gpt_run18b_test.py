#!/usr/bin/env python3
"""
GPT Run 18B: Test I1 Channel with TeX Combined Structure

This script tests the compute_I1_tex_combined_11 function to verify
that the combined integral structure produces reasonable values.

Usage:
    python run_gpt_run18b_test.py
"""

import numpy as np

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import (
    compute_I1_tex_combined_11,
    compute_I1_tex_combined_11_replace,
    compute_I2_tex_combined_11,
    compute_S34_tex_combined_11,
    compute_S34_base_11,
    compute_c_paper_tex_mirror,
)


THETA = 4.0 / 7.0

# PRZZ Targets
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
    print("GPT Run 18B: I1 Channel with TeX Combined Structure")
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

        print(f"\n{name} Benchmark (R={R}):")
        print("-" * 50)

        # Get tex_mirror I1 for comparison
        tex_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        # Compute I1 using combined structure (multiply)
        combined_result = compute_I1_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            n_quad_s=20,
            verbose=True,
        )

        # Compute I1 using combined structure (replace)
        combined_replace = compute_I1_tex_combined_11_replace(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            n_quad_s=20,
            verbose=True,
        )

        print()
        print("Comparison:")
        print(f"  c_target:                {c_target:.4f}")
        print(f"  tex_mirror c:            {tex_result.c:.4f}")
        print(f"  tex_mirror I1_total:     {tex_result.I1_plus + tex_result.m1 * tex_result.I1_minus_base:.4f}")
        print(f"    I1_plus:               {tex_result.I1_plus:.4f}")
        print(f"    I1_mirror:             {tex_result.m1 * tex_result.I1_minus_base:.4f}")
        print(f"  combined I1 (multiply):  {combined_result.I1_combined:.4f}")
        print(f"  combined I1 (replace):   {combined_replace.I1_combined:.4f}")
        print(f"  scalar_limit:            {combined_result.scalar_limit:.4f}")

        # Compute I2 using combined structure
        i2_result = compute_I2_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            verbose=True,
        )

        # Totals - try different combinations
        total_with_i2_combined = combined_replace.I1_combined + i2_result.I2_combined
        total_with_i2_base = combined_replace.I1_combined + i2_result.I2_base

        print()
        print("I2 Component:")
        print(f"  tex_mirror I2_plus:      {tex_result.I2_plus:.4f}")
        print(f"  I2_base:                 {i2_result.I2_base:.4f}")
        print(f"  I2_combined:             {i2_result.I2_combined:.4f}")
        print()
        print("Combined Totals (I1 + I2, no S34):")
        print(f"  I1_replace + I2_combined: {total_with_i2_combined:.4f}")
        print(f"  I1_replace + I2_base:     {total_with_i2_base:.4f}")
        print(f"  tex_mirror I1+I2 total:   {tex_result.I1_plus + tex_result.m1 * tex_result.I1_minus_base + tex_result.I2_plus + tex_result.m2 * tex_result.I2_minus_base:.4f}")
        print(f"  c_target:                 {c_target:.4f}")
        print()
        print(f"  Gap (I1_rep + I2_base):   {100 * (total_with_i2_base - c_target) / c_target:+.2f}%")

        # Compute S34 using combined structure (with factor)
        s34_result = compute_S34_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            n_quad_s=20,
            verbose=True,
        )

        # Compute S34 base (no combined factor, per PRZZ TRUTH_SPEC)
        s34_base = compute_S34_base_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            verbose=True,
        )

        print()
        print("S34 Component:")
        print(f"  tex_mirror S34_plus:     {tex_result.S34_plus:.4f}")
        print(f"  S34_combined:            {s34_result.S34_combined:.4f}")
        print(f"  S34_base (no factor):    {s34_base:.4f}")

        # Full assembly with S34_base (correct interpretation)
        c_full_with_s34_base = combined_replace.I1_combined + i2_result.I2_base + s34_base

        print()
        print("=" * 50)
        print("FULL ASSEMBLY (I1_replace + I2_base + S34_base):")
        print(f"  c_tex_combined:          {c_full_with_s34_base:.4f}")
        print(f"  c_tex_mirror:            {tex_result.c:.4f}")
        print(f"  c_target:                {c_target:.4f}")
        print(f"  Gap (combined vs target): {100 * (c_full_with_s34_base - c_target) / c_target:+.2f}%")
        print(f"  Gap (mirror vs target):   {100 * (tex_result.c - c_target) / c_target:+.2f}%")
        print("=" * 50)

        # Check if combined is reasonable
        is_finite_mult = np.isfinite(combined_result.I1_combined)
        is_finite_repl = np.isfinite(combined_replace.I1_combined)
        is_finite_i2 = np.isfinite(i2_result.I2_combined)
        print()
        print(f"Gate checks:")
        print(f"  I1 (multiply) is finite: {'PASS' if is_finite_mult else 'FAIL'}")
        print(f"  I1 (replace) is finite:  {'PASS' if is_finite_repl else 'FAIL'}")
        print(f"  I2 combined is finite:   {'PASS' if is_finite_i2 else 'FAIL'}")
        print(f"  I2 combined is positive: {'PASS' if i2_result.I2_combined > 0 else 'FAIL'}")

    # Quadrature convergence test
    print("\n" + "=" * 80)
    print("Quadrature Convergence Test (κ benchmark)")
    print("=" * 80)

    R = TARGETS["kappa"]["R"]
    for n_quad in [20, 40, 60]:
        result = compute_I1_tex_combined_11(
            theta=THETA,
            R=R,
            n=n_quad,
            polynomials=polys_kappa,
            n_quad_s=20,
        )
        print(f"  n={n_quad}: I1_combined = {result.I1_combined:.6f}")

    print("\n" + "=" * 80)
    print("s-Quadrature Convergence Test (κ benchmark)")
    print("=" * 80)

    for n_s in [10, 20, 40]:
        result = compute_I1_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys_kappa,
            n_quad_s=n_s,
        )
        print(f"  n_s={n_s}: I1_combined = {result.I1_combined:.6f}")


if __name__ == "__main__":
    main()
