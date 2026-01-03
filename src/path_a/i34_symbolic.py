#!/usr/bin/env python3
"""
Step 3b: Symbolic I₃ and I₄ in z-Basis

Computes I₃ and I₄ for all pairs (ℓ₁, ℓ₂) symbolically in z-basis.

CRITICAL: Using z = e^{R/7} basis (NOT y = e^{2R/7})
  - Reason: Mirror multiplier M₀ = e^R + 5 requires e^R = z⁷ (integer power)

z-Power Structure (same as I₁):
  z⁰  = 1
  z¹⁴ = exp(2R)

From PRZZ TeX lines 1562-1563 (I₃):
    I₃ = -(1+θx)/θ × (d/dx)|_{x=0}
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁} P_{ℓ₁}(x+u) P_{ℓ₂}(u)
         × exp(R[A₃_α + A₃_β]) × Q(A₃_α) × Q(A₃_β) du dt

From PRZZ TeX lines 1568-1569 (I₄):
    I₄ = -(1+θy)/θ × (d/dy)|_{y=0}
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₂} P_{ℓ₁}(u) P_{ℓ₂}(y+u)
         × exp(R[A₄_α + A₄_β]) × Q(A₄_α) × Q(A₄_β) du dt

Where:
    A₃_α = t + θtx, A₃_β = t - θx(1-t)
    A₄_α = t + θty, A₄_β = t - θy(1-t)

Key properties:
- Both have NEGATIVE sign
- Single derivatives (d/dx or d/dy)
- I₃ = I₄ for diagonal pairs (symmetry)
- z-powers: {0, 14} for all pairs

Usage:
    python -m src.path_a.i34_symbolic
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, N, integrate, diff
)
from typing import Dict, Tuple, Optional

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import (
    R, theta, R_star_approx,
    build_P1, build_P2, build_P3, build_Q
)

# Symbols
u = symbols('u', real=True, positive=True)
t = symbols('t', real=True, positive=True)
x_sym = symbols('x', real=True)
y_sym = symbols('y', real=True)
z_basis = symbols('z', positive=True)  # z = e^{R/7}


def get_polynomial_expr(ell: int, var) -> sp.Expr:
    """Get P_ℓ(var) as a symbolic expression."""
    if ell == 1:
        return build_P1(var)
    elif ell == 2:
        return build_P2(var)
    elif ell == 3:
        return build_P3(var)
    else:
        raise ValueError(f"Invalid piece index: {ell}")


def compute_I3_x_coefficient(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> sp.Expr:
    """
    Compute the x coefficient of the I₃ integrand symbolically.

    The integrand is:
        [(1+θx)/θ] × (1-u)^{ℓ₁} × P_{ℓ₁}(x+u) × P_{ℓ₂}(u)
        × exp(R[A₃_α + A₃_β]) × Q(A₃_α) × Q(A₃_β)

    Where:
        A₃_α = t + θtx = t(1 + θx)
        A₃_β = t - θx(1-t)

    Returns:
        Symbolic expression for the x coefficient (function of u, t, R)
    """
    if verbose:
        print(f"Computing I₃ x-coefficient for pair ({ell1},{ell2})...")

    # Build polynomials
    P1_u = get_polynomial_expr(ell1, u)
    P1_deriv = diff(P1_u, u)
    P2_u = get_polynomial_expr(ell2, u)  # P_{ℓ₂}(u), no shift

    Q_t = build_Q(t)
    Q_deriv = diff(Q_t, t)

    # (1-u)^{ℓ₁} factor
    one_minus_u_power = (1 - u) ** ell1

    # Affine eigenvalue coefficients (at x=0, both reduce to t)
    # A₃_α = t + θt·x
    # A₃_β = t - θ(1-t)·x
    A3_alpha_x = theta * t
    A3_beta_x = -theta * (1 - t)

    # Exponential: exp(R[A₃_α + A₃_β])
    # Sum: 2t + θ(2t-1)·x at x=0
    exp_0 = exp(2 * R * t)
    exp_x_coeff = R * theta * (2*t - 1)

    # Prefactor: (1+θx)/θ = 1/θ + x
    pref_0 = 1 / theta
    pref_x = sp.Integer(1)

    # P expansions:
    # P_{ℓ₁}(x+u) = P₁(u) + P₁'(u)·x
    # P_{ℓ₂}(u) = constant in x
    P_0 = P1_u * P2_u
    P_x = P1_deriv * P2_u

    # Q expansions:
    # Q(A₃_α) = Q(t) + Q'(t)·θt·x
    # Q(A₃_β) = Q(t) + Q'(t)·(-θ(1-t))·x
    Q_0 = Q_t * Q_t  # Q(t)²
    Q_alpha_shift_x = Q_deriv * A3_alpha_x
    Q_beta_shift_x = Q_deriv * A3_beta_x
    # Q product x-coefficient: Q(t)·Q'(t)·(θt - θ(1-t)) = Q·Q'·θ(2t-1)
    Q_x = Q_t * (Q_alpha_shift_x + Q_beta_shift_x)

    # Build symbolic expressions and collect x coefficient
    # F1 = (1/θ + x)
    # F2 = (P₁(u)P₂(u) + P₁'(u)P₂(u)·x)
    # F3 = exp(2Rt)·(1 + exp_x_coeff·x)
    # F4 = Q² + Q_x·x
    F1 = pref_0 + pref_x * x_sym
    F2 = P_0 + P_x * x_sym
    F3 = exp_0 * (1 + exp_x_coeff * x_sym)
    F4 = Q_0 + Q_x * x_sym

    product = expand(F1 * F2 * F3 * F4)

    # Extract x coefficient
    x_coeff = product.coeff(x_sym)

    # Multiply by (1-u)^{ℓ₁}
    I3_integrand = one_minus_u_power * x_coeff

    if verbose:
        print(f"  Integrand computed (length: {len(str(I3_integrand))} chars)")

    return I3_integrand


def compute_I4_y_coefficient(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> sp.Expr:
    """
    Compute the y coefficient of the I₄ integrand symbolically.

    The integrand is:
        [(1+θy)/θ] × (1-u)^{ℓ₂} × P_{ℓ₁}(u) × P_{ℓ₂}(y+u)
        × exp(R[A₄_α + A₄_β]) × Q(A₄_α) × Q(A₄_β)

    Where:
        A₄_α = t + θty
        A₄_β = t - θy(1-t)

    Returns:
        Symbolic expression for the y coefficient (function of u, t, R)
    """
    if verbose:
        print(f"Computing I₄ y-coefficient for pair ({ell1},{ell2})...")

    # Build polynomials
    P1_u = get_polynomial_expr(ell1, u)  # P_{ℓ₁}(u), no shift
    P2_u = get_polynomial_expr(ell2, u)
    P2_deriv = diff(P2_u, u)

    Q_t = build_Q(t)
    Q_deriv = diff(Q_t, t)

    # (1-u)^{ℓ₂} factor
    one_minus_u_power = (1 - u) ** ell2

    # Affine eigenvalue coefficients
    # A₄_α = t + θt·y
    # A₄_β = t - θ(1-t)·y
    A4_alpha_y = theta * t
    A4_beta_y = -theta * (1 - t)

    # Exponential: exp(R[A₄_α + A₄_β])
    exp_0 = exp(2 * R * t)
    exp_y_coeff = R * theta * (2*t - 1)

    # Prefactor: (1+θy)/θ = 1/θ + y
    pref_0 = 1 / theta
    pref_y = sp.Integer(1)

    # P expansions:
    # P_{ℓ₁}(u) = constant in y
    # P_{ℓ₂}(y+u) = P₂(u) + P₂'(u)·y
    P_0 = P1_u * P2_u
    P_y = P1_u * P2_deriv

    # Q expansions:
    Q_0 = Q_t * Q_t
    Q_alpha_shift_y = Q_deriv * A4_alpha_y
    Q_beta_shift_y = Q_deriv * A4_beta_y
    Q_y = Q_t * (Q_alpha_shift_y + Q_beta_shift_y)

    # Build symbolic expressions and collect y coefficient
    F1 = pref_0 + pref_y * y_sym
    F2 = P_0 + P_y * y_sym
    F3 = exp_0 * (1 + exp_y_coeff * y_sym)
    F4 = Q_0 + Q_y * y_sym

    product = expand(F1 * F2 * F3 * F4)

    # Extract y coefficient
    y_coeff = product.coeff(y_sym)

    # Multiply by (1-u)^{ℓ₂}
    I4_integrand = one_minus_u_power * y_coeff

    if verbose:
        print(f"  Integrand computed (length: {len(str(I4_integrand))} chars)")

    return I4_integrand


def compute_I3_symbolic(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> Tuple[sp.Expr, Dict, sp.Expr]:
    """
    Compute I₃ for a pair (ℓ₁, ℓ₂) symbolically.

    I₃ has a NEGATIVE sign.

    Returns:
        (I3_expr, z_coeffs, denominator)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"I₃ Symbolic for pair ({ell1},{ell2})")
        print("=" * 60)

    # Get the x coefficient of the integrand
    x_coeff = compute_I3_x_coefficient(ell1, ell2, verbose=verbose)

    # Integrate over u ∈ [0,1]
    if verbose:
        print("  Integrating over u...")
    I3_after_u = integrate(x_coeff, (u, 0, 1))
    I3_after_u = simplify(I3_after_u)

    if verbose:
        print(f"  After u-integration: {len(str(I3_after_u))} chars")

    # Integrate over t ∈ [0,1]
    if verbose:
        print("  Integrating over t...")
    I3_expr = integrate(I3_after_u, (t, 0, 1))
    I3_expr = simplify(I3_expr)

    # Apply negative sign
    I3_expr = -I3_expr

    if verbose:
        print(f"  After t-integration (with -sign): {len(str(I3_expr))} chars")

    # Extract z-basis coefficients
    z_coeffs, denominator = extract_z_basis_coefficients(I3_expr, verbose=verbose)

    return I3_expr, z_coeffs, denominator


def compute_I4_symbolic(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> Tuple[sp.Expr, Dict, sp.Expr]:
    """
    Compute I₄ for a pair (ℓ₁, ℓ₂) symbolically.

    I₄ has a NEGATIVE sign.

    Returns:
        (I4_expr, z_coeffs, denominator)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"I₄ Symbolic for pair ({ell1},{ell2})")
        print("=" * 60)

    # Get the y coefficient of the integrand
    y_coeff = compute_I4_y_coefficient(ell1, ell2, verbose=verbose)

    # Integrate over u ∈ [0,1]
    if verbose:
        print("  Integrating over u...")
    I4_after_u = integrate(y_coeff, (u, 0, 1))
    I4_after_u = simplify(I4_after_u)

    if verbose:
        print(f"  After u-integration: {len(str(I4_after_u))} chars")

    # Integrate over t ∈ [0,1]
    if verbose:
        print("  Integrating over t...")
    I4_expr = integrate(I4_after_u, (t, 0, 1))
    I4_expr = simplify(I4_expr)

    # Apply negative sign
    I4_expr = -I4_expr

    if verbose:
        print(f"  After t-integration (with -sign): {len(str(I4_expr))} chars")

    # Extract z-basis coefficients
    z_coeffs, denominator = extract_z_basis_coefficients(I4_expr, verbose=verbose)

    return I4_expr, z_coeffs, denominator


def extract_z_basis_coefficients(
    expr: sp.Expr,
    verbose: bool = True
) -> Tuple[Optional[Dict], sp.Expr]:
    """
    Extract coefficients in z-basis where z = e^{R/7}.

    For I₃ and I₄, z-powers are {0, 14} (same as I₁).

    Returns (coeffs_dict, denominator)
    """
    from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch

    # Handle Piecewise
    main_expr = _extract_piecewise_main_branch(expr)

    # Combine into single fraction
    main_expr = together(main_expr)
    num, den = fraction(main_expr)

    # Expand numerator
    num = expand(num)

    z = symbols('z', positive=True)

    # Substitute key exponentials
    num_sub = num.subs(exp(2*R), z**14)
    num_sub = num_sub.subs(exp(R), z**7)

    try:
        if num_sub.has(z):
            num_sub = expand(num_sub)
            coeffs = {}

            # Extract coefficients for expected z-powers: 0 and 14
            for p in [0, 7, 14]:
                if p == 0:
                    coeff = num_sub.subs(z, 0)
                else:
                    coeff = num_sub.coeff(z, p)
                if coeff != 0:
                    coeffs[p] = coeff

            if verbose:
                powers_present = sorted(coeffs.keys())
                print(f"  z-basis: powers present = {powers_present}")

            return coeffs, den
        else:
            return {0: num}, den

    except Exception as e:
        if verbose:
            print(f"  Could not extract z-basis: {e}")

    return None, den


def compute_I34_all_pairs(verbose: bool = True) -> Dict:
    """Compute I₃ and I₄ symbolically for all 6 pairs in z-basis."""
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    results = {}

    for ell1, ell2 in pairs:
        # Compute I₃
        I3_expr, I3_z_coeffs, I3_den = compute_I3_symbolic(ell1, ell2, verbose=verbose)

        from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch
        I3_main = _extract_piecewise_main_branch(I3_expr)
        I3_numeric = float(N(I3_main.subs(R, R_star_approx), 20))

        # Compute I₄
        I4_expr, I4_z_coeffs, I4_den = compute_I4_symbolic(ell1, ell2, verbose=verbose)

        I4_main = _extract_piecewise_main_branch(I4_expr)
        I4_numeric = float(N(I4_main.subs(R, R_star_approx), 20))

        results[(ell1, ell2)] = {
            'I3_expr': I3_expr,
            'I3_z_coeffs': I3_z_coeffs,
            'I3_denominator': I3_den,
            'I3_numeric': I3_numeric,
            'I4_expr': I4_expr,
            'I4_z_coeffs': I4_z_coeffs,
            'I4_denominator': I4_den,
            'I4_numeric': I4_numeric,
        }

        if verbose:
            print(f"\n  I₃(R*) = {I3_numeric:.10f}")
            print(f"  I₄(R*) = {I4_numeric:.10f}")
            print(f"  I₃+I₄  = {I3_numeric + I4_numeric:.10f}")
            if ell1 == ell2:
                ratio = I3_numeric / I4_numeric if abs(I4_numeric) > 1e-15 else float('inf')
                print(f"  I₃/I₄ (diagonal) = {ratio:.6f} (expected: 1.0)")

    return results


def validate_against_numeric(results: Dict, verbose: bool = True) -> None:
    """Validate symbolic I₃/I₄ against numeric implementation."""
    from src.przz_exact_i34 import compute_I34_all_pairs as compute_I34_numeric
    import numpy as np
    from sympy import N as sympy_N, Float, diff as sym_diff, symbols

    # Create polynomial wrappers using optimal_coeffs (same as symbolic)
    class SymbolicPolyWrapper:
        def __init__(self, build_fn):
            self.build_fn = build_fn

        def eval(self, x_arr):
            result = []
            for x in x_arr:
                val = float(sympy_N(self.build_fn(Float(x)), 20))
                result.append(val)
            return np.array(result)

        def eval_deriv(self, x_arr, k=1):
            u_s = symbols('u')
            expr = self.build_fn(u_s)
            deriv_expr = expr
            for _ in range(k):
                deriv_expr = sym_diff(deriv_expr, u_s)
            result = []
            for x in x_arr:
                val = float(sympy_N(deriv_expr.subs(u_s, Float(x)), 20))
                result.append(val)
            return np.array(result)

    polynomials = {
        "P1": SymbolicPolyWrapper(build_P1),
        "P2": SymbolicPolyWrapper(build_P2),
        "P3": SymbolicPolyWrapper(build_P3),
        "Q": SymbolicPolyWrapper(build_Q),
    }

    theta_val = float(theta)

    # Compute numeric I₃/I₄ at R*
    numeric_results = compute_I34_numeric(theta_val, R_star_approx, polynomials, n_quad=80)

    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION: Symbolic vs Numeric I₃/I₄")
        print("=" * 60)
        print("\n| Pair | I₃_sym | I₃_num | Ratio | I₄_sym | I₄_num | Ratio |")
        print("|------|--------|--------|-------|--------|--------|-------|")

    for (ell1, ell2), r in results.items():
        I3_sym = r['I3_numeric']
        I4_sym = r['I4_numeric']
        key = f"{ell1}{ell2}"
        I3_num = numeric_results["I3"][key].value
        I4_num = numeric_results["I4"][key].value

        r3 = I3_sym / I3_num if abs(I3_num) > 1e-15 else float('inf')
        r4 = I4_sym / I4_num if abs(I4_num) > 1e-15 else float('inf')

        if verbose:
            print(f"| ({ell1},{ell2}) | {I3_sym:+.4f} | {I3_num:+.4f} | {r3:.4f} | {I4_sym:+.4f} | {I4_num:+.4f} | {r4:.4f} |")


def main():
    print("=" * 70)
    print("STEP 3b: SYMBOLIC I₃ AND I₄ IN z-BASIS (z = e^{R/7})")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")
    print("\nz-Power Mapping:")
    print("  z⁰ = 1, z⁷ = e^R (mirror), z¹⁴ = e^{2R}")

    # Compute I₃/I₄ for all pairs
    results = compute_I34_all_pairs(verbose=True)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: I₃ AND I₄ z-BASIS DECOMPOSITION")
    print("=" * 60)

    print("\n| Pair | I₃(R*) | I₄(R*) | I₃+I₄ |")
    print("|------|--------|--------|-------|")
    for (ell1, ell2), r in results.items():
        I3 = r['I3_numeric']
        I4 = r['I4_numeric']
        print(f"| ({ell1},{ell2}) | {I3:+.6f} | {I4:+.6f} | {I3+I4:+.6f} |")

    # Verify I₃ = I₄ for diagonal pairs
    print("\n--- Diagonal Symmetry Check (I₃ = I₄?) ---")
    for (ell1, ell2), r in results.items():
        if ell1 == ell2:
            I3 = r['I3_numeric']
            I4 = r['I4_numeric']
            diff = abs(I3 - I4)
            print(f"  ({ell1},{ell2}): I₃={I3:.8f}, I₄={I4:.8f}, |diff|={diff:.2e}")

    # Validate against numeric
    validate_against_numeric(results, verbose=True)

    print("\n" + "=" * 70)
    print("STEP 3b COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
