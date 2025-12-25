#!/usr/bin/env python3
"""
run_operator_post_identity_golden.py
Golden diagnostic script for Post-Identity Operator validation.

This is the "one-command sanity check" before touching K>3.
Run it to verify that the post-identity operator approach is working correctly.

Usage:
    python3 run_operator_post_identity_golden.py

Outputs diagnostic information in stages:
1. Affine coefficients (θt-θ cross-terms)
2. QQexp coefficients {c00, cx, cy, cxy}
3. I1 comparison for all K=3 pairs (both benchmarks)
"""

import sys
import numpy as np

from src.operator_post_identity import (
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
    apply_QQexp_post_identity_composition,
    compute_I1_operator_post_identity_pair,
)
from src.polynomials import load_przz_polynomials
from src.terms_k3_d1 import (
    make_all_terms_11_v2, make_all_terms_12_v2, make_all_terms_13_v2,
    make_all_terms_22_v2, make_all_terms_23_v2, make_all_terms_33_v2,
)
from src.evaluate import evaluate_terms


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def stage_1_affine_coefficients(theta: float):
    """Stage 1: Show A_α, A_β affine coefficients (θt-θ cross-terms)."""
    print_header("Stage 1: Affine Coefficients (θt-θ cross-terms)")

    print(f"\nθ = {theta:.10f}")
    print(f"\nThe affine forms are:")
    print(f"  A_α = t + θ(t-1)·x + θt·y")
    print(f"  A_β = t + θt·x + θ(t-1)·y")
    print()

    for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
        u0_a, ax_a, ay_a = get_A_alpha_affine_coeffs(t, theta)
        u0_b, ax_b, ay_b = get_A_beta_affine_coeffs(t, theta)
        print(f"  t={t:.1f}: A_α = {u0_a:.3f} + ({ax_a:+.4f})x + ({ay_a:+.4f})y")
        print(f"        A_β = {u0_b:.3f} + ({ax_b:+.4f})x + ({ay_b:+.4f})y")
        print()


def stage_2_qqexp_coefficients(Q_poly, theta: float, R: float):
    """Stage 2: QQexp coefficients at sample t values."""
    print_header("Stage 2: QQexp Coefficients {c00, cx, cy, cxy}")

    print(f"\nR = {R}, θ = {theta:.10f}")
    print(f"\nQ×Q×exp core series coefficients:")
    print()

    var_names = ("x", "y")
    for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
        core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
        c00 = core_series.extract(())
        cx = core_series.extract(("x",))
        cy = core_series.extract(("y",))
        cxy = core_series.extract(("x", "y"))

        print(f"  t={t:.1f}: c00={c00:+.8f}, cx={cx:+.8f}, cy={cy:+.8f}, cxy={cxy:+.8f}")


def stage_3_i1_comparison(polys: dict, theta: float, R: float, benchmark_name: str, n: int = 40):
    """Stage 3: I1 comparison for all K=3 pairs."""
    print_header(f"Stage 3: I1(ℓ₁,ℓ₂) Post-Identity vs DSL ({benchmark_name}, R={R})")

    terms_builders = {
        (1, 1): make_all_terms_11_v2,
        (1, 2): make_all_terms_12_v2,
        (1, 3): make_all_terms_13_v2,
        (2, 2): make_all_terms_22_v2,
        (2, 3): make_all_terms_23_v2,
        (3, 3): make_all_terms_33_v2,
    }

    print(f"\nQuadrature: n={n}")
    print(f"{'Pair':<8} {'Post-Identity':>15} {'DSL':>15} {'Diff':>12} {'Match':>8}")
    print("-" * 60)

    all_match = True
    for (ell1, ell2), terms_fn in terms_builders.items():
        # Post-identity
        result = compute_I1_operator_post_identity_pair(theta, R, ell1, ell2, n, polys)
        I1_post = result.I1_value

        # DSL
        terms = terms_fn(theta, R, kernel_regime='paper')
        i1_terms = [t for t in terms if 'I1' in t.name]
        dsl_result = evaluate_terms(i1_terms, polys, n, return_breakdown=True, R=R, theta=theta)
        I1_dsl = dsl_result.total

        diff = abs(I1_post - I1_dsl)
        match = diff < 1e-10
        all_match = all_match and match
        status = "OK" if match else "MISMATCH"

        print(f"({ell1},{ell2}){' ':<5} {I1_post:>+15.10f} {I1_dsl:>+15.10f} {diff:>12.2e} {status:>8}")

    return all_match


def main():
    print("=" * 70)
    print("POST-IDENTITY OPERATOR GOLDEN DIAGNOSTIC")
    print("=" * 70)
    print()
    print("This script validates the post-identity operator approach for I1 computation.")
    print("It compares against the DSL (paper regime) which uses the same affine forms.")
    print()

    # Load polynomials
    print("Loading PRZZ polynomials...")
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    print("  Q(0) =", Q.eval(np.array([0.0]))[0])
    print("  Q(0.5) =", Q.eval(np.array([0.5]))[0])
    print()

    theta = 4.0 / 7.0

    # Stage 1: Affine coefficients
    stage_1_affine_coefficients(theta)

    # Stage 2: QQexp coefficients (κ benchmark)
    stage_2_qqexp_coefficients(Q, theta, R=1.3036)

    # Stage 3: I1 comparison - κ benchmark
    match_kappa = stage_3_i1_comparison(polys, theta, R=1.3036, benchmark_name="κ")

    # Stage 3: I1 comparison - κ* benchmark
    match_kappa_star = stage_3_i1_comparison(polys, theta, R=1.1167, benchmark_name="κ*")

    # Summary
    print_header("SUMMARY")
    print()
    print(f"  κ benchmark (R=1.3036):  {'PASS' if match_kappa else 'FAIL'}")
    print(f"  κ* benchmark (R=1.1167): {'PASS' if match_kappa_star else 'FAIL'}")
    print()

    if match_kappa and match_kappa_star:
        print("  All tests PASSED - Post-identity operator is validated!")
        print("  Safe to proceed with K>3 extension.")
        return 0
    else:
        print("  FAILED - Some tests did not match.")
        print("  Investigate before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
