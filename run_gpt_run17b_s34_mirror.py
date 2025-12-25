#!/usr/bin/env python3
"""
GPT Run 17B: S34 Mirror Contribution Investigation

This script investigates whether I3/I4 (S34) should have mirror contribution.
Currently tex_mirror uses plus-only for S34.

Key insight from TeX lines 1553-1570:
    I₃ involves: (N^{αx} - T^{-α-β}N^{-βx}) / (α+β)

This suggests I3 DOES have mirror structure! Same for I4.

Goal: Determine if S34 mirror contribution explains part of the ~1% gap.

Method:
1. Build I3 series at +R (current tex_mirror approach)
2. Build I3 series at -R (mirror contribution)
3. Combine with exp(2R) prefactor
4. Compare against plus-only approach
5. Same for I4

Usage:
    python run_gpt_run17b_s34_mirror.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import tensor_grid_2d
from src.series import TruncatedSeries
from src.term_dsl import AffineExpr, SeriesContext
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.evaluate import compute_c_paper_tex_mirror


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

# K=3 pairs
K3_PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


@dataclass
class S34MirrorResult:
    """Result of S34 mirror investigation for one pair."""
    ell1: int
    ell2: int
    I3_plus: float       # I3 at +R only
    I3_minus: float      # I3 at -R only
    I3_combined: float   # I3_plus + exp(2R) × I3_minus
    I4_plus: float
    I4_minus: float
    I4_combined: float
    S34_plus_only: float      # Current tex_mirror approach
    S34_with_mirror: float    # With mirror contribution
    delta: float
    delta_pct: float


def get_poly_for_ell(ell: int, polynomials: Dict):
    """Get polynomial P_ℓ from dictionary."""
    if ell == 1: return polynomials["P1"]
    if ell == 2: return polynomials["P2"]
    if ell == 3: return polynomials["P3"]


def _make_Q_arg_alpha_x_only(theta: float) -> AffineExpr:
    """Q argument α with only x: t + θtx (for I₃, y set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x": lambda U, T, th=theta: th * T}
    )


def _make_Q_arg_beta_x_only(theta: float) -> AffineExpr:
    """Q argument β with only x: t + θ(t-1)x (for I₃, y set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x": lambda U, T, th=theta: th * (T - 1)}
    )


def _make_Q_arg_alpha_y_only(theta: float) -> AffineExpr:
    """Q argument α with only y: t + θ(t-1)y (for I₄, x set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y": lambda U, T, th=theta: th * (T - 1)}
    )


def _make_Q_arg_beta_y_only(theta: float) -> AffineExpr:
    """Q argument β with only y: t + θty (for I₄, x set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y": lambda U, T, th=theta: th * T}
    )


def _make_P_argument_x() -> AffineExpr:
    """P argument: x + u"""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"x": 1.0})


def _make_P_argument_y() -> AffineExpr:
    """P argument: y + u"""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"y": 1.0})


def _make_algebraic_prefactor_x_only(theta: float) -> AffineExpr:
    """Algebraic prefactor with only x: (1+θx)/θ = 1/θ + x"""
    return AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0})


def _make_algebraic_prefactor_y_only(theta: float) -> AffineExpr:
    """Algebraic prefactor with only y: (1+θy)/θ = 1/θ + y"""
    return AffineExpr(a0=1.0 / theta, var_coeffs={"y": 1.0})


def build_I3_series(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    U: np.ndarray,
    T: np.ndarray,
    use_negative_R: bool = False,
) -> TruncatedSeries:
    """
    Build the I3 integrand series for a single pair at +R or -R.

    I3 has d/dx derivative only (x-variable), y is evaluated at 0.
    """
    var_names = ("x",)  # I3 only has x variable
    ctx = SeriesContext(var_names=var_names)

    R_eff = -R if use_negative_R else R

    integrand = ctx.scalar_series(np.ones_like(U))

    # Q arguments for I3 (x-only)
    Q_arg_alpha = _make_Q_arg_alpha_x_only(theta)
    Q_arg_beta = _make_Q_arg_beta_x_only(theta)

    # P arguments - both use x+u for I3 (but P_ell2 is evaluated at u only since y=0)
    P_arg_x = _make_P_argument_x()
    P_arg_u = AffineExpr(a0=lambda U, T: U, var_coeffs={})  # Just u, no formal vars

    # P_ell1(x+u)
    a0_vals = P_arg_x.evaluate_a0(U, T)
    lin_vals = {"x": P_arg_x.evaluate_coeff("x", U, T)}
    poly_series = compose_polynomial_on_affine(P_ell1, a0_vals, lin_vals, var_names)
    integrand = integrand * poly_series

    # P_ell2(u) - no formal variable dependence
    P2_vals = P_ell2.eval(U.flatten()).reshape(U.shape)
    integrand = integrand * ctx.scalar_series(P2_vals)

    # Q factors
    for arg in [Q_arg_alpha, Q_arg_beta]:
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {"x": arg.evaluate_coeff("x", U, T)}
        poly_series = compose_polynomial_on_affine(Q, a0_vals, lin_vals, var_names)
        integrand = integrand * poly_series

    # Exponential factors
    for arg in [Q_arg_alpha, Q_arg_beta]:
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {"x": arg.evaluate_coeff("x", U, T)}
        exp_series = compose_exp_on_affine(R_eff, a0_vals, lin_vals, var_names)
        integrand = integrand * exp_series

    # Algebraic prefactor: 1/θ + x
    alg_pref = _make_algebraic_prefactor_x_only(theta)
    a0_pref = alg_pref.evaluate_a0(U, T)
    x_coeff = alg_pref.evaluate_coeff("x", U, T)
    prefactor_series = ctx.scalar_series(a0_pref) + ctx.variable_series("x") * x_coeff
    integrand = integrand * prefactor_series

    return integrand


def build_I4_series(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    U: np.ndarray,
    T: np.ndarray,
    use_negative_R: bool = False,
) -> TruncatedSeries:
    """
    Build the I4 integrand series for a single pair at +R or -R.

    I4 has d/dy derivative only (y-variable), x is evaluated at 0.
    """
    var_names = ("y",)  # I4 only has y variable
    ctx = SeriesContext(var_names=var_names)

    R_eff = -R if use_negative_R else R

    integrand = ctx.scalar_series(np.ones_like(U))

    # Q arguments for I4 (y-only)
    Q_arg_alpha = _make_Q_arg_alpha_y_only(theta)
    Q_arg_beta = _make_Q_arg_beta_y_only(theta)

    # P arguments
    P_arg_u = AffineExpr(a0=lambda U, T: U, var_coeffs={})  # Just u
    P_arg_y = _make_P_argument_y()

    # P_ell1(u) - no formal variable dependence
    P1_vals = P_ell1.eval(U.flatten()).reshape(U.shape)
    integrand = integrand * ctx.scalar_series(P1_vals)

    # P_ell2(y+u)
    a0_vals = P_arg_y.evaluate_a0(U, T)
    lin_vals = {"y": P_arg_y.evaluate_coeff("y", U, T)}
    poly_series = compose_polynomial_on_affine(P_ell2, a0_vals, lin_vals, var_names)
    integrand = integrand * poly_series

    # Q factors
    for arg in [Q_arg_alpha, Q_arg_beta]:
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {"y": arg.evaluate_coeff("y", U, T)}
        poly_series = compose_polynomial_on_affine(Q, a0_vals, lin_vals, var_names)
        integrand = integrand * poly_series

    # Exponential factors
    for arg in [Q_arg_alpha, Q_arg_beta]:
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {"y": arg.evaluate_coeff("y", U, T)}
        exp_series = compose_exp_on_affine(R_eff, a0_vals, lin_vals, var_names)
        integrand = integrand * exp_series

    # Algebraic prefactor: 1/θ + y
    alg_pref = _make_algebraic_prefactor_y_only(theta)
    a0_pref = alg_pref.evaluate_a0(U, T)
    y_coeff = alg_pref.evaluate_coeff("y", U, T)
    prefactor_series = ctx.scalar_series(a0_pref) + ctx.variable_series("y") * y_coeff
    integrand = integrand * prefactor_series

    return integrand


def evaluate_I3_combined(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> Tuple[float, float, float]:
    """
    Evaluate I3 with combined +R/-R.

    Returns:
        (I3_combined, I3_plus, I3_minus)
    """
    U, T, W = tensor_grid_2d(n_quad)

    series_plus = build_I3_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, False)
    series_minus = build_I3_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, True)

    prefactor = np.exp(2 * R)
    series_combined = series_plus + series_minus * prefactor

    # Extract d/dx coefficient
    coeff_combined = series_combined.extract(("x",))
    coeff_plus = series_plus.extract(("x",))
    coeff_minus = series_minus.extract(("x",))

    # Poly prefactor: (1-u)^(ell1+ell2-1) for I3 (one less power)
    power = ell1 + ell2 - 1
    poly_pref = (1 - U) ** power

    I3_combined = float(np.sum(W * coeff_combined * poly_pref))
    I3_plus = float(np.sum(W * coeff_plus * poly_pref))
    I3_minus = float(np.sum(W * coeff_minus * poly_pref))

    return I3_combined, I3_plus, I3_minus


def evaluate_I4_combined(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> Tuple[float, float, float]:
    """
    Evaluate I4 with combined +R/-R.
    """
    U, T, W = tensor_grid_2d(n_quad)

    series_plus = build_I4_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, False)
    series_minus = build_I4_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, True)

    prefactor = np.exp(2 * R)
    series_combined = series_plus + series_minus * prefactor

    # Extract d/dy coefficient
    coeff_combined = series_combined.extract(("y",))
    coeff_plus = series_plus.extract(("y",))
    coeff_minus = series_minus.extract(("y",))

    # Poly prefactor: (1-u)^(ell1+ell2-1) for I4
    power = ell1 + ell2 - 1
    poly_pref = (1 - U) ** power

    I4_combined = float(np.sum(W * coeff_combined * poly_pref))
    I4_plus = float(np.sum(W * coeff_plus * poly_pref))
    I4_minus = float(np.sum(W * coeff_minus * poly_pref))

    return I4_combined, I4_plus, I4_minus


def compute_S34_results_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    c_target: float,
    n_quad: int = 60,
) -> S34MirrorResult:
    """
    Compute S34 with and without mirror for a single pair.
    """
    P_ell1 = get_poly_for_ell(ell1, polynomials)
    P_ell2 = get_poly_for_ell(ell2, polynomials)
    Q = polynomials["Q"]

    I3_combined, I3_plus, I3_minus = evaluate_I3_combined(
        theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad
    )

    I4_combined, I4_plus, I4_minus = evaluate_I4_combined(
        theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad
    )

    S34_plus_only = I3_plus + I4_plus
    S34_with_mirror = I3_combined + I4_combined

    delta = S34_with_mirror - S34_plus_only
    delta_pct = 100 * delta / c_target if c_target != 0 else 0.0

    return S34MirrorResult(
        ell1=ell1,
        ell2=ell2,
        I3_plus=I3_plus,
        I3_minus=I3_minus,
        I3_combined=I3_combined,
        I4_plus=I4_plus,
        I4_minus=I4_minus,
        I4_combined=I4_combined,
        S34_plus_only=S34_plus_only,
        S34_with_mirror=S34_with_mirror,
        delta=delta,
        delta_pct=delta_pct,
    )


def main():
    print("=" * 90)
    print("GPT Run 17B: S34 Mirror Contribution Investigation")
    print("=" * 90)
    print()
    print("TeX lines 1553-1570 suggest I3/I4 have mirror structure:")
    print("  I₃ involves: (N^{αx} - T^{-α-β}N^{-βx}) / (α+β)")
    print()
    print("Goal: Determine if S34 mirror contribution explains part of ~1% gap.")
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

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        # Get tex_mirror reference
        tex_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        print("=" * 90)
        print(f"BENCHMARK: {bench_name} (R={R})")
        print("=" * 90)
        print()
        print(f"tex_mirror S34_plus total: {tex_result.S34_plus:.6f}")
        print(f"exp(2R) prefactor = {np.exp(2*R):.4f}")
        print()

        print("S34 with and without mirror (per pair):")
        print("-" * 90)
        print(f"{'Pair':<8} {'S34_plus':>12} {'S34_mirror':>12} {'delta':>12} {'delta%':>10}")
        print("-" * 90)

        results = []
        total_plus = 0.0
        total_mirror = 0.0

        for ell1, ell2 in K3_PAIRS:
            result = compute_S34_results_for_pair(
                THETA, R, ell1, ell2, polys, c_target, n_quad=60
            )
            results.append(result)

            mult = 2 if ell1 != ell2 else 1
            total_plus += mult * result.S34_plus_only
            total_mirror += mult * result.S34_with_mirror

            print(f"({ell1},{ell2}){' ' * 4} {result.S34_plus_only:12.6f} "
                  f"{result.S34_with_mirror:12.6f} {result.delta:12.6f} "
                  f"{result.delta_pct:+10.4f}%")

        print("-" * 90)
        print()

        total_delta = total_mirror - total_plus
        total_delta_pct = 100 * total_delta / c_target

        print("S34 TOTAL:")
        print(f"  Plus only (current): {total_plus:.6f}")
        print(f"  With mirror:         {total_mirror:.6f}")
        print(f"  Delta:               {total_delta:.6f} ({total_delta_pct:+.4f}%)")
        print(f"  tex_mirror S34_plus: {tex_result.S34_plus:.6f}")
        print()

    # Analysis
    print()
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print("""
KEY FINDINGS:

1. S34 Mirror Contribution:
   - If delta > 0.5% of c_target: S34 mirror explains part of the gap
   - If delta < 0.1%: Current plus-only approach is correct

2. TeX Structure (lines 1553-1570):
   - I3 involves (N^{αx} - T^{-α-β}N^{-βx}) / (α+β)
   - This IS a mirror structure, similar to I1

3. Implications:
   - If S34 has significant mirror: tex_mirror needs to include it
   - If S34 mirror is small: The ~1% gap comes entirely from I1/I2

NEXT STEP: Agent 17C will aggregate these findings into a residual table.
""")


if __name__ == "__main__":
    main()
