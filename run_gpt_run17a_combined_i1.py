#!/usr/bin/env python3
"""
GPT Run 17A: Combined Mirror for I1 with Derivatives Intact

This script implements the TeX-combined mirror evaluation for I1 using
the series machinery, with derivatives extracted AFTER combining +R/-R.

Key insight from Run 17A0:
- Correct prefactor is exp(2R) ≈ 13.56 (NOT exp(2R/θ) ≈ 95.83)
- BUT naive prefactor gives c ≈ 12.3 (475% off!)
- The derivative structure (d²/dxdy) fundamentally changes the effective weight

Goal: Understand how derivative extraction modifies the mirror contribution.

Method:
1. Build +R series using existing machinery
2. Build -R series (mirror) using same machinery with scale=-R
3. Combine: F_total = F_plus + exp(2R) × F_minus
4. Extract d²/dxdy AFTER combining
5. Compare against tex_mirror's factorized result

Usage:
    python run_gpt_run17a_combined_i1.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import tensor_grid_2d
from src.series import TruncatedSeries
from src.term_dsl import AffineExpr, PolyFactor, ExpFactor, SeriesContext
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
class CombinedMirrorResult:
    """Result of combined mirror evaluation for one pair."""
    ell1: int
    ell2: int
    I1_combined: float      # I1 with +R/-R combined BEFORE derivatives
    I1_plus: float          # I1 at +R only
    I1_minus: float         # I1 at -R only (with derivatives)
    I1_factorized: float    # I1_plus + tex_m × I1_minus
    delta: float            # I1_combined - I1_factorized
    delta_pct: float        # 100 × delta / c_target
    m_implied: float        # (I1_combined - I1_plus) / I1_minus


def get_poly_for_ell(ell: int, polynomials: Dict):
    """Get polynomial P_ℓ from dictionary."""
    if ell == 1: return polynomials["P1"]
    if ell == 2: return polynomials["P2"]
    if ell == 3: return polynomials["P3"]


def _make_Q_arg_alpha(theta: float) -> AffineExpr:
    """Q argument α: t + θtx + θ(t-1)y"""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={
            "x": lambda U, T, th=theta: th * T,
            "y": lambda U, T, th=theta: th * (T - 1)
        }
    )


def _make_Q_arg_beta(theta: float) -> AffineExpr:
    """Q argument β: t + θ(t-1)x + θty"""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={
            "x": lambda U, T, th=theta: th * (T - 1),
            "y": lambda U, T, th=theta: th * T
        }
    )


def _make_P_argument_x() -> AffineExpr:
    """P argument: x + u"""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"x": 1.0})


def _make_P_argument_y() -> AffineExpr:
    """P argument: y + u"""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"y": 1.0})


def _make_algebraic_prefactor(theta: float) -> AffineExpr:
    """Algebraic prefactor: (1+θ(x+y))/θ = 1/θ + x + y"""
    return AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0, "y": 1.0})


def build_I1_series(
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
    Build the I1 integrand series for a single pair at +R or -R.

    The series contains all formal variable terms (x, y).
    Derivative extraction happens LATER.

    Args:
        use_negative_R: If True, use -R for the exponential factors

    Returns:
        TruncatedSeries representing the integrand before derivative extraction
    """
    var_names = ("x", "y")
    ctx = SeriesContext(var_names=var_names)

    # The effective R for exponentials
    R_eff = -R if use_negative_R else R

    # Build the series integrand
    integrand = ctx.scalar_series(np.ones_like(U))

    # Q arguments
    Q_arg_alpha = _make_Q_arg_alpha(theta)
    Q_arg_beta = _make_Q_arg_beta(theta)

    # P arguments
    P_arg_x = _make_P_argument_x()
    P_arg_y = _make_P_argument_y()

    # Polynomial factors: P_ell1(x+u), P_ell2(y+u), Q(arg_α), Q(arg_β)
    for P, arg in [(P_ell1, P_arg_x), (P_ell2, P_arg_y), (Q, Q_arg_alpha), (Q, Q_arg_beta)]:
        # Evaluate a0 and linear coefficients
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {v: arg.evaluate_coeff(v, U, T) for v in var_names}

        # Compose polynomial on affine
        poly_series = compose_polynomial_on_affine(P, a0_vals, lin_vals, var_names)
        integrand = integrand * poly_series

    # Exponential factors: exp(R_eff × arg_α), exp(R_eff × arg_β)
    for arg in [Q_arg_alpha, Q_arg_beta]:
        a0_vals = arg.evaluate_a0(U, T)
        lin_vals = {v: arg.evaluate_coeff(v, U, T) for v in var_names}
        exp_series = compose_exp_on_affine(R_eff, a0_vals, lin_vals, var_names)
        integrand = integrand * exp_series

    # Algebraic prefactor: 1/θ + x + y
    # Build manually: scalar + x_coeff*x + y_coeff*y
    alg_pref = _make_algebraic_prefactor(theta)
    a0_pref = alg_pref.evaluate_a0(U, T)  # 1/θ
    x_coeff = alg_pref.evaluate_coeff("x", U, T)  # 1.0
    y_coeff = alg_pref.evaluate_coeff("y", U, T)  # 1.0

    # Build prefactor series: a0 + x_coeff*x + y_coeff*y
    prefactor_series = ctx.scalar_series(a0_pref)
    prefactor_series = prefactor_series + ctx.variable_series("x") * x_coeff
    prefactor_series = prefactor_series + ctx.variable_series("y") * y_coeff
    integrand = integrand * prefactor_series

    return integrand


def evaluate_I1_combined(
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
    Evaluate I1 with combined +R/-R BEFORE derivative extraction.

    Returns:
        (I1_combined, I1_plus, I1_minus)
    """
    U, T, W = tensor_grid_2d(n_quad)

    # Build +R series
    series_plus = build_I1_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, use_negative_R=False)

    # Build -R series
    series_minus = build_I1_series(theta, R, ell1, ell2, P_ell1, P_ell2, Q, U, T, use_negative_R=True)

    # The correct prefactor from 17A0: exp(2R)
    prefactor = np.exp(2 * R)

    # Combined series: F_plus + prefactor × F_minus
    series_combined = series_plus + series_minus * prefactor

    # Extract d²/dxdy coefficient (xy term) from each
    coeff_combined = series_combined.extract(("x", "y"))
    coeff_plus = series_plus.extract(("x", "y"))
    coeff_minus = series_minus.extract(("x", "y"))

    # Poly prefactor: (1-u)^(ell1+ell2)
    power = ell1 + ell2
    poly_pref = (1 - U) ** power

    # Integrate
    I1_combined = float(np.sum(W * coeff_combined * poly_pref))
    I1_plus = float(np.sum(W * coeff_plus * poly_pref))
    I1_minus = float(np.sum(W * coeff_minus * poly_pref))

    return I1_combined, I1_plus, I1_minus


def compute_I1_results_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    tex_m1: float,
    c_target: float,
    n_quad: int = 60,
) -> CombinedMirrorResult:
    """
    Compute combined and factorized I1 for a single pair.
    """
    P_ell1 = get_poly_for_ell(ell1, polynomials)
    P_ell2 = get_poly_for_ell(ell2, polynomials)
    Q = polynomials["Q"]

    I1_combined, I1_plus, I1_minus = evaluate_I1_combined(
        theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad
    )

    # tex_mirror's factorized result
    I1_factorized = I1_plus + tex_m1 * I1_minus

    delta = I1_combined - I1_factorized
    delta_pct = 100 * delta / c_target if c_target != 0 else 0.0

    # Implied mirror weight: what m makes I1_combined = I1_plus + m × I1_minus?
    if abs(I1_minus) > 1e-12:
        m_implied = (I1_combined - I1_plus) / I1_minus
    else:
        m_implied = float('inf')

    return CombinedMirrorResult(
        ell1=ell1,
        ell2=ell2,
        I1_combined=I1_combined,
        I1_plus=I1_plus,
        I1_minus=I1_minus,
        I1_factorized=I1_factorized,
        delta=delta,
        delta_pct=delta_pct,
        m_implied=m_implied,
    )


def main():
    print("=" * 90)
    print("GPT Run 17A: Combined Mirror for I1 with Derivatives Intact")
    print("=" * 90)
    print()
    print("Key insight from 17A0:")
    print("  - Correct prefactor exp(2R) ≈ 13.56, but naive use gives c +475% off")
    print("  - The derivative structure (d²/dxdy) modifies the effective weight")
    print()
    print("Method:")
    print("  - Build +R and -R series")
    print("  - Combine: F_total = F_plus + exp(2R) × F_minus")
    print("  - Extract d²/dxdy AFTER combining")
    print("  - Compare against tex_mirror's factorized approach")
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
        print(f"tex_mirror reference: m1={tex_result.m1:.4f}, m2={tex_result.m2:.4f}")
        print(f"exp(2R) prefactor = {np.exp(2*R):.4f}")
        print()

        print("I1 Combined vs Factorized (per pair):")
        print("-" * 90)
        print(f"{'Pair':<8} {'I1_comb':>12} {'I1_plus':>12} {'I1_minus':>12} "
              f"{'m_implied':>12} {'delta':>12} {'delta%':>10}")
        print("-" * 90)

        results = []
        total_combined = 0.0
        total_factorized = 0.0

        for ell1, ell2 in K3_PAIRS:
            result = compute_I1_results_for_pair(
                THETA, R, ell1, ell2, polys, tex_result.m1, c_target, n_quad=60
            )
            results.append(result)

            # Multiplicity
            mult = 2 if ell1 != ell2 else 1
            total_combined += mult * result.I1_combined
            total_factorized += mult * result.I1_factorized

            print(f"({ell1},{ell2}){' ' * 4} {result.I1_combined:12.6f} "
                  f"{result.I1_plus:12.6f} {result.I1_minus:12.6f} "
                  f"{result.m_implied:12.4f} {result.delta:12.6f} "
                  f"{result.delta_pct:+10.4f}%")

        print("-" * 90)
        print()

        # Summary
        total_delta = total_combined - total_factorized
        total_delta_pct = 100 * total_delta / c_target

        print("I1 TOTAL:")
        print(f"  Combined (exp(2R) prefactor): {total_combined:.6f}")
        print(f"  Factorized (tex_mirror m1):   {total_factorized:.6f}")
        print(f"  Delta:                        {total_delta:.6f} ({total_delta_pct:+.4f}%)")
        print()

        # Analyze m_implied
        m_implied_values = [r.m_implied for r in results if abs(r.m_implied) < 1e6]
        if m_implied_values:
            m_mean = np.mean(m_implied_values)
            m_std = np.std(m_implied_values)
            print(f"m_implied statistics:")
            print(f"  Mean: {m_mean:.4f}")
            print(f"  Std:  {m_std:.4f}")
            print(f"  CV:   {100*m_std/abs(m_mean):.1f}%")
            print(f"  vs exp(2R)={np.exp(2*R):.4f}: ratio = {m_mean/np.exp(2*R):.4f}")
            print(f"  vs tex_m1={tex_result.m1:.4f}: ratio = {m_mean/tex_result.m1:.4f}")
        print()

    # Final analysis
    print()
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print("""
KEY FINDINGS:

1. The m_implied from combined mirror shows how derivative extraction
   modifies the effective mirror weight.

2. If m_implied ≈ exp(2R):
   - The naive prefactor IS correct for I1
   - The huge c gap in 17A0 comes from I2/S34, not I1

3. If m_implied ≈ tex_m1:
   - tex_mirror's factorization is essentially exact for I1
   - The gap must come from elsewhere (S34 mirror?)

4. If m_implied differs significantly from both:
   - The derivative structure creates a NEW effective weight
   - This would require a different assembly formula

NEXT STEPS:
- Agent 17B: Check if S34 has mirror contribution
- Agent 17C: Build residual truth table from these findings
""")


if __name__ == "__main__":
    main()
