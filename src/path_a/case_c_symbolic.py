#!/usr/bin/env python3
"""
Symbolic Case C Kernel Implementation for Path A.

The Case C kernel for piece ℓ ≥ 2 is:
    K_ω(u; R, θ) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da

where ω = ℓ - 1.

This module computes K_ω symbolically using J_n integrals:
    J_n(λ) = ∫₀¹ t^n exp(λt) dt = (e^λ P_n(λ) - n!) / λ^{n+1}

Key derivation:
    P((1-a)u) = Σ_k c_k (1-a)^k u^k
    (1-a)^k = Σ_j C(k,j) (-1)^j a^j

    So K_ω(u) = u^ω/(ω-1)! × Σ_{k,j} c_k u^k C(k,j) (-1)^j × J_{ω-1+j}(Rθu)

The result is algebraic in u and exp(Rθu).

Usage:
    python -m src.path_a.case_c_symbolic
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, binomial, N
)
from typing import List, Tuple

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import R, theta, R_star_approx

# Symbols for Case C
u = symbols('u', real=True, positive=True)


def compute_K_omega_symbolic(
    omega: int,
    poly_coeffs: List[sp.Rational],
    R_sym=R,
    theta_val=theta,
    simplify_result: bool = True
) -> sp.Expr:
    """
    Compute K_ω(u) symbolically for a polynomial P(x) = Σ_k c_k x^k.

    K_ω(u) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da

    Args:
        omega: Kernel order (1 for P₂, 2 for P₃)
        poly_coeffs: Standard basis coefficients [c₀, c₁, c₂, ...]
        R_sym: Symbolic R variable
        theta_val: Value of θ (4/7)
        simplify_result: Whether to simplify the result

    Returns:
        Symbolic expression for K_ω(u)
    """
    if omega < 1:
        raise ValueError("omega must be >= 1 for Case C kernels")

    # Exponential argument: λ = R*θ*u
    lam = R_sym * theta_val * u

    # Compute the sum over polynomial terms
    result = sp.Integer(0)

    for k, c_k in enumerate(poly_coeffs):
        if c_k == 0:
            continue

        # For each polynomial term c_k * x^k, we have c_k * ((1-a)u)^k
        # The (1-a)^k expands via binomial:
        # (1-a)^k = Σ_j C(k,j) (-1)^j a^j

        binomial_sum = sp.Integer(0)
        for j in range(k + 1):
            coeff = ((-1)**j) * binomial(k, j)
            # Integral: ∫₀¹ a^{ω-1+j} exp(λa) da = J_{ω-1+j}(λ)
            J_val = J_n_closed_form(omega - 1 + j, lam)
            binomial_sum += coeff * J_val

        result += c_k * (u**k) * binomial_sum

    # Multiply by u^ω / (ω-1)!
    result = result * (u**omega) / factorial(omega - 1)

    if simplify_result:
        result = simplify(result)

    return result


def get_polynomial_standard_coeffs(ell: int) -> List[sp.Rational]:
    """
    Get standard basis coefficients for P_ℓ(x).

    For ℓ=1: P₁(x) = x + x(1-x)·P̃₁(1-x) - returns monomial coefficients
    For ℓ≥2: P_ℓ(x) = x·P̃_ℓ(x) - returns monomial coefficients
    """
    from src.path_a.optimal_coeffs import build_P1, build_P2, build_P3

    x = symbols('x')

    if ell == 1:
        P_expr = build_P1(x)
    elif ell == 2:
        P_expr = build_P2(x)
    elif ell == 3:
        P_expr = build_P3(x)
    else:
        raise ValueError(f"Invalid piece index: {ell}")

    # Expand and extract coefficients
    P_expanded = expand(P_expr)

    # Get coefficients in standard basis
    from sympy import Poly
    poly = Poly(P_expanded, x)

    # Return as list [c0, c1, c2, ...]
    coeffs_dict = poly.as_dict()
    max_deg = max(k[0] for k in coeffs_dict.keys())

    result = []
    for i in range(max_deg + 1):
        coeff = coeffs_dict.get((i,), sp.Integer(0))
        result.append(sp.Rational(coeff))

    return result


def eval_K_omega_at_R(
    omega: int,
    ell: int,
    u_val: float,
    R_val: float = R_star_approx,
    theta_val: float = 4/7,
) -> float:
    """
    Numerically evaluate K_ω(u) at specific R and u values.
    """
    poly_coeffs = get_polynomial_standard_coeffs(ell)
    K_expr = compute_K_omega_symbolic(omega, poly_coeffs)

    # Substitute values
    K_numeric = K_expr.subs({R: R_val, u: u_val})
    return float(N(K_numeric, 30))


def compare_case_c_vs_raw(ell: int, u_val: float, R_val: float = R_star_approx) -> Tuple[float, float, float]:
    """
    Compare Case C kernel vs raw polynomial at given u.

    Returns:
        (raw_value, case_c_value, ratio)
    """
    from src.path_a.optimal_coeffs import get_P

    omega = ell - 1

    # Raw: P(u)
    x_sym = symbols('x')
    P_expr = get_P(ell, x_sym)
    raw_val = float(N(P_expr.subs(x_sym, u_val), 30))

    if omega == 0:
        # Case B: no attenuation
        return raw_val, raw_val, 1.0

    # Case C: K_ω(u)
    case_c_val = eval_K_omega_at_R(omega, ell, u_val, R_val)

    ratio = raw_val / case_c_val if abs(case_c_val) > 1e-15 else float('inf')

    return raw_val, case_c_val, ratio


def main():
    print("=" * 70)
    print("SYMBOLIC CASE C KERNEL COMPUTATION")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")

    # Test Case C for P₂ (ω=1)
    print("\n" + "=" * 60)
    print("CASE C FOR P₂ (ω=1)")
    print("=" * 60)

    p2_coeffs = get_polynomial_standard_coeffs(2)
    print(f"\nP₂ standard coefficients: {[float(c) for c in p2_coeffs]}")

    print("\nComputing K₁(u) symbolically...")
    K1 = compute_K_omega_symbolic(1, p2_coeffs, simplify_result=False)
    print(f"K₁(u) structure: {type(K1)}")

    # Evaluate at some u values
    print("\nNumerical comparison P₂(u) vs K₁(u):")
    print(f"{'u':>8} {'P₂(u)':>15} {'K₁(u)':>15} {'Ratio':>10}")
    print("-" * 55)

    for u_val in [0.2, 0.4, 0.5, 0.6, 0.8]:
        raw, case_c, ratio = compare_case_c_vs_raw(2, u_val)
        print(f"{u_val:>8.2f} {raw:>15.8f} {case_c:>15.8f} {ratio:>10.4f}")

    # Test Case C for P₃ (ω=2)
    print("\n" + "=" * 60)
    print("CASE C FOR P₃ (ω=2)")
    print("=" * 60)

    p3_coeffs = get_polynomial_standard_coeffs(3)
    print(f"\nP₃ standard coefficients: {[float(c) for c in p3_coeffs]}")

    print("\nNumerical comparison P₃(u) vs K₂(u):")
    print(f"{'u':>8} {'P₃(u)':>15} {'K₂(u)':>15} {'Ratio':>10}")
    print("-" * 55)

    for u_val in [0.2, 0.4, 0.5, 0.6, 0.8]:
        raw, case_c, ratio = compare_case_c_vs_raw(3, u_val)
        print(f"{u_val:>8.2f} {raw:>15.8f} {case_c:>15.8f} {ratio:>10.4f}")

    print("\n" + "=" * 70)
    print("CASE C SYMBOLIC MODULE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
