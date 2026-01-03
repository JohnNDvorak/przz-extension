#!/usr/bin/env python3
"""
Symbolic I₂ Computation in Paper Regime.

I₂ is the cleanest integral (no derivatives):
    I₂ = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × K_{ω₁}(u) × K_{ω₂}(u) × Q(t)² du dt

where K_ω uses Case C kernels for ω ≥ 1 (pieces ℓ ≥ 2).

The goal is to express I₂ in the form:
    I₂ = (A(R) e^{2R} + B(R)) / D(R)

where A, B, D are polynomials in R.

Usage:
    python -m src.path_a.i2_paper_symbolic
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, binomial, N, integrate, Poly
)
from typing import Dict, Tuple, List

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import (
    R, theta, R_star_approx,
    P1_tilde, P2_tilde, P3_tilde, Q_coeffs,
    build_P1, build_P2, build_P3, build_Q
)

# Additional symbols
u, t = symbols('u t', real=True, positive=True)


def compute_K_omega_symbolic(
    ell: int,
    u_sym,
    R_sym=R,
    theta_val=theta
) -> sp.Expr:
    """
    Compute K_ω(u) symbolically for piece ℓ.

    For ℓ=1 (ω=0): K = P₁(u) (raw polynomial)
    For ℓ≥2 (ω=ℓ-1): K = Case C kernel
    """
    omega = ell - 1

    # Get polynomial coefficients in standard basis
    from src.path_a.case_c_symbolic import get_polynomial_standard_coeffs

    poly_coeffs = get_polynomial_standard_coeffs(ell)

    if omega == 0:
        # Case B: raw polynomial P_ℓ(u)
        return sum(c * u_sym**k for k, c in enumerate(poly_coeffs))

    # Case C: K_ω(u) via J_n integrals
    lam = R_sym * theta_val * u_sym

    result = sp.Integer(0)

    for k, c_k in enumerate(poly_coeffs):
        if c_k == 0:
            continue

        # For polynomial term c_k * x^k at x = (1-a)u:
        # ∫₀¹ a^{ω-1} (1-a)^k exp(λa) da = Σ_j C(k,j)(-1)^j J_{ω-1+j}(λ)
        binomial_sum = sp.Integer(0)
        for j in range(k + 1):
            coeff = ((-1)**j) * binomial(k, j)
            J_val = J_n_closed_form(omega - 1 + j, lam)
            binomial_sum += coeff * J_val

        result += c_k * (u_sym**k) * binomial_sum

    # Multiply by u^ω / (ω-1)!
    result = result * (u_sym**omega) / factorial(omega - 1)

    return result


def compute_I2_paper_symbolic(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> sp.Expr:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂) in paper regime symbolically.

    I₂ = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × K_{ω₁}(u) × K_{ω₂}(u) × Q(t)² du dt
    """
    if verbose:
        print(f"Computing I₂^{{({ell1},{ell2})}} symbolically...")

    # Get Q(t)² symbolically
    Q_t = build_Q(t)
    Q_squared = expand(Q_t**2)

    # Get Case C kernels
    K1 = compute_K_omega_symbolic(ell1, u)
    K2 = compute_K_omega_symbolic(ell2, u)

    if verbose:
        print("  K₁(u) and K₂(u) computed")

    # The integrand is: exp(2Rt) × K₁(u) × K₂(u) × Q(t)²
    # Separate u and t integrals since they factor

    # u-integral: ∫₀¹ K₁(u) × K₂(u) du
    kernel_product = K1 * K2
    kernel_product = simplify(kernel_product)

    if verbose:
        print("  Computing u-integral...")

    # The u-integral is complicated due to K_ω structure
    # For symbolic computation, we'll keep it unevaluated for now
    # and focus on structure

    # t-integral: ∫₀¹ exp(2Rt) × Q(t)² dt
    # This uses J_n integrals with λ = 2R

    # Expand Q² and collect powers of t
    Q_sq_poly = Poly(Q_squared, t)
    Q_sq_coeffs = Q_sq_poly.all_coeffs()[::-1]  # [c₀, c₁, c₂, ...]

    if verbose:
        print(f"  Q² is degree {len(Q_sq_coeffs)-1} in t")

    # t-integral = Σ_n c_n × J_n(2R)
    t_integral = sp.Integer(0)
    lam_2R = 2 * R
    for n, c_n in enumerate(Q_sq_coeffs):
        if c_n != 0:
            t_integral += c_n * J_n_closed_form(n, lam_2R)

    t_integral = simplify(t_integral)

    if verbose:
        print("  t-integral computed")

    # For now, return a structure that shows the algebraic form
    # Full symbolic u-integration is complex

    return {
        't_integral': t_integral,
        'K1': K1,
        'K2': K2,
        'kernel_product': kernel_product,
        'prefactor': 1 / theta,
    }


def compute_I2_at_R_star(ell1: int, ell2: int) -> float:
    """
    Numerically evaluate I₂ at R* using symbolic formulas.
    """
    from scipy import integrate
    import numpy as np

    result = compute_I2_paper_symbolic(ell1, ell2, verbose=False)

    # t-integral at R*
    t_int_val = float(N(result['t_integral'].subs(R, R_star_approx), 30))

    # u-integral numerically
    K1 = result['K1']
    K2 = result['K2']
    kernel = result['kernel_product']

    def u_integrand(u_val):
        return float(N(kernel.subs({R: R_star_approx, u: u_val}), 20))

    u_int_val, _ = integrate.quad(u_integrand, 0.01, 0.99, limit=100)

    I2 = float(result['prefactor']) * u_int_val * t_int_val

    return I2


def main():
    print("=" * 70)
    print("SYMBOLIC I₂ IN PAPER REGIME")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")

    # Test (1,1) pair first - should match symbolic exactly (no Case C)
    print("\n" + "=" * 60)
    print("I₂^{(1,1)} - No Case C (baseline)")
    print("=" * 60)

    result_11 = compute_I2_paper_symbolic(1, 1)

    # Show t-integral structure
    print(f"\nt-integral structure:")
    t_int = result_11['t_integral']
    print(f"  {simplify(t_int)}")

    # Evaluate at R*
    t_int_val = float(N(t_int.subs(R, R_star_approx), 20))
    print(f"  At R* = {R_star_approx}: {t_int_val:.10f}")

    # Compare with numeric engine
    from src.unified_i2_paper import compute_I2_unified_paper
    from src.path_a.unit_test_symbolic import get_optimal_polynomials

    polys = get_optimal_polynomials()
    numeric_result = compute_I2_unified_paper(
        R_star_approx, 4/7, 1, 1, polys, include_Q=True
    )
    print(f"\nNumeric I₂^{{(1,1)}} = {numeric_result.I2_value:.10f}")

    # Symbolic I2 at R*
    I2_sym = compute_I2_at_R_star(1, 1)
    print(f"Symbolic I₂^{{(1,1)}} = {I2_sym:.10f}")

    print("\n" + "=" * 60)
    print("I₂^{(2,2)} - With Case C")
    print("=" * 60)

    numeric_22 = compute_I2_unified_paper(
        R_star_approx, 4/7, 2, 2, polys, include_Q=True
    )
    print(f"Numeric I₂^{{(2,2)}} = {numeric_22.I2_value:.10f}")

    I2_sym_22 = compute_I2_at_R_star(2, 2)
    print(f"Symbolic I₂^{{(2,2)}} = {I2_sym_22:.10f}")

    print("\n" + "=" * 70)
    print("I₂ PAPER SYMBOLIC MODULE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
