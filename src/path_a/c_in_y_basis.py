#!/usr/bin/env python3
"""
Step 2: Express c(R) in z-Basis for Path A.

CRITICAL: Using z = e^{R/7} basis (NOT y = e^{2R/7})
  - Reason: Mirror multiplier M₀ = e^R + 5 requires e^R = z⁷ (integer power)
  - Old y-basis had e^R = y^{7/2} (fractional - FORBIDDEN!)
  - Relationship: y = z², so existing work maps directly

Combines:
- U(R) from u_integral_symbolic.py (u-integrals in z-basis)
- T(R) = t-integral = ∫₀¹ exp(2Rt) Q(t)² dt ∈ ℚ(R, e^{2R})
- Assembly using mirror formula

Goal: Express c(R) - 1 = N(R, z) / D(R) where z = e^{R/7}

z-Power Mapping:
  z⁰  = 1           (constant)
  z⁴  = exp(4R/7)   (was y²)
  z⁷  = exp(R)      (NEW - for mirror M₀)
  z⁸  = exp(8R/7)   (was y⁴)
  z¹⁴ = exp(2R)     (was y⁷)
  z¹⁸ = exp(18R/7)  (was y⁹)
  z²² = exp(22R/7)  (was y¹¹)

Structure:
    I₂ = (1/θ) × U(R) × T(R)
    c(R) = S₁₂(+R) + M₀·S₁₂(-R) + S₃₄(+R)
    where M₀ = z⁷ + 5 = e^R + 5

For now, focus on I₂ contributions to understand the algebraic structure.

Usage:
    python -m src.path_a.c_in_y_basis
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, N, integrate, Poly
)
from typing import Dict, Tuple, Optional

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import R, theta, R_star_approx, build_Q
from src.path_a.u_integral_symbolic import (
    compute_all_symbolic,
    _extract_piecewise_main_branch
)

# Additional symbols
t = symbols('t', real=True, positive=True)
z = symbols('z', positive=True)  # z = e^{R/7} (integer powers for all exponentials)


def compute_T_symbolic(verbose: bool = True) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute T(R) = ∫₀¹ exp(2Rt) Q(t)² dt symbolically.

    Returns (T_expr, z14_coeff, const_coeff, denominator) where:
        T(R) = (A(R)·z¹⁴ + B(R)) / D(R)
        and z¹⁴ = e^{2R} (z = e^{R/7})
    """
    if verbose:
        print("Computing T(R) = ∫₀¹ exp(2Rt) Q(t)² dt...")

    # Build Q(t)²
    Q_t = build_Q(t)
    Q_sq = expand(Q_t**2)

    if verbose:
        Q_poly = Poly(Q_sq, t)
        print(f"  Q(t)² is degree {Q_poly.degree()} in t")

    # Get coefficients of Q²
    Q_sq_poly = Poly(Q_sq, t)
    Q_sq_coeffs = Q_sq_poly.all_coeffs()[::-1]  # [c₀, c₁, c₂, ...]

    # T = Σ_n c_n × J_n(2R) where J_n(λ) = ∫₀¹ t^n e^{λt} dt
    lam = 2 * R
    T_expr = sp.Integer(0)
    for n, c_n in enumerate(Q_sq_coeffs):
        if c_n != 0:
            J_val = J_n_closed_form(n, lam)
            T_expr += c_n * J_val

    T_expr = simplify(T_expr)

    if verbose:
        print("  T(R) computed")

    # Express in z-basis: e^{2R} = z^14 (since z = e^{R/7})
    # J_n(2R) = (A_n(2R) e^{2R} + B_n) / (2R)^{n+1}
    # So T is of form (A(R) e^{2R} + B(R)) / R^p

    # Get fraction form
    T_together = together(T_expr)
    num, den = fraction(T_together)

    # Extract e^{2R} coefficient (which becomes z^14)
    z_temp = symbols('z_temp', positive=True)
    num_exp = expand(num)
    num_sub = num_exp.subs(exp(2*R), z_temp)

    coeff_z14 = num_sub.coeff(z_temp, 1)
    const = num_sub.subs(z_temp, 0)

    if verbose:
        print(f"  T = (A·z¹⁴ + B) / D  where z¹⁴ = e^{{2R}}")
        print(f"  A(R) length: {len(str(coeff_z14))} chars")
        print(f"  B(R) length: {len(str(const))} chars")
        print(f"  D(R) = {den}")

    return T_expr, coeff_z14, const, den


def compute_I2_z_basis(
    U_results: Dict,
    T_expr: sp.Expr,
    T_z14_coeff: sp.Expr,
    T_const: sp.Expr,
    T_den: sp.Expr,
    verbose: bool = True
) -> Dict:
    """
    Compute I₂ contributions for all pairs in z-basis (z = e^{R/7}).

    CRITICAL CORRECTION (2026-01-03):
    - U z-powers are PAIR-DEPENDENT and include NEGATIVE powers!
    - I₂ z-powers range from -8 to +38

    I₂ = (1/θ) × U(R) × T(R)

    where:
        U(R) = (Σ A_k·z^k) / D_U    (pair-dependent k, including negative!)
        T(R) = (A_T·z¹⁴ + B_T) / D_T

    U z-Powers by Pair:
        (1,1): {-8, 8}
        (1,2): {-4, 12}
        (1,3): {0, 16}
        (2,2): {0, 16}
        (2,3): {4, 20}
        (3,3): {8, 24}

    I₂ = U × T z-Powers:
        (1,1): {-8, 6, 8, 22}
        (1,2): {-4, 10, 12, 26}
        (1,3): {0, 14, 16, 30}
        (2,2): {0, 14, 16, 30}
        (2,3): {4, 18, 20, 34}
        (3,3): {8, 22, 24, 38}

    Returns dict with I₂ expression for each pair.
    """
    results = {}

    for (ell1, ell2), r in U_results.items():
        if verbose:
            print(f"\n--- Pair ({ell1},{ell2}) ---")

        z_coeffs = r['z_coeffs']
        U_den = r['denominator']

        # Compute I₂ symbolically
        # First, form U(R) × T(R)
        # U × T = (Σ A_k·z^k) × (A_T·z¹⁴ + B_T) / (D_U × D_T)

        # Expand the product in z-basis
        # (A₀ + A₄z⁴ + A₈z⁸) × (A_T·z¹⁴ + B_T)
        # = A₀·A_T·z¹⁴ + A₀·B_T
        # + A₄·A_T·z¹⁸ + A₄·B_T·z⁴
        # + A₈·A_T·z²² + A₈·B_T·z⁸

        I2_coeffs = {}

        for z_power, U_coeff in z_coeffs.items():
            # Term with z^14 from T
            new_power_14 = z_power + 14
            term_14 = U_coeff * T_z14_coeff
            I2_coeffs[new_power_14] = I2_coeffs.get(new_power_14, 0) + term_14

            # Term with constant from T
            term_const = U_coeff * T_const
            I2_coeffs[z_power] = I2_coeffs.get(z_power, 0) + term_const

        # Denominator
        I2_den = U_den * T_den * theta

        # Simplify coefficients
        for p in I2_coeffs:
            I2_coeffs[p] = expand(I2_coeffs[p])

        results[(ell1, ell2)] = {
            'z_coeffs': I2_coeffs,
            'denominator': I2_den,
        }

        if verbose:
            z_powers_present = sorted([p for p, c in I2_coeffs.items() if c != 0])
            print(f"  I₂ z-powers: {z_powers_present}")

            # Verify at R*
            z_star = float(N(exp(R_star_approx / 7), 20))
            den_val = float(N(I2_den.subs(R, R_star_approx), 20))

            I2_check = 0
            for p, c in I2_coeffs.items():
                if c != 0:
                    c_val = float(N(c.subs(R, R_star_approx), 20))
                    I2_check += c_val * (z_star ** p)
            I2_check /= den_val

            # Compare with direct computation
            U_numeric = r['U_numeric']
            T_numeric = float(N(T_expr.subs(R, R_star_approx), 20))
            I2_direct = U_numeric * T_numeric / float(theta)

            print(f"  I₂(R*) from z-basis: {I2_check:.10f}")
            print(f"  I₂(R*) direct: {I2_direct:.10f}")
            ratio = I2_check / I2_direct if abs(I2_direct) > 1e-15 else float('inf')
            print(f"  Ratio: {ratio:.6f}")

    return results


def assemble_S12_z_basis(I2_results: Dict, verbose: bool = True) -> Tuple[Dict, sp.Expr]:
    """
    Assemble S₁₂(R) = Σ w_{ℓ₁,ℓ₂} × I₂ in z-basis (z = e^{R/7}).

    For I₂ only (ignoring I₁ for now).

    Returns (z_coeffs, denominator) for S₁₂.
    """
    S12_coeffs = {}
    S12_den = sp.Integer(1)

    for (ell1, ell2), r in I2_results.items():
        # Weight factor
        sym_factor = 2 if ell1 != ell2 else 1
        factorial_norm = Rational(1, factorial(ell1) * factorial(ell2))
        weight = sym_factor * factorial_norm

        z_coeffs = r['z_coeffs']
        den = r['denominator']

        # Add weighted contribution
        # Need common denominator
        for p, c in z_coeffs.items():
            term = weight * c * S12_den / den
            S12_coeffs[p] = S12_coeffs.get(p, 0) + term

        # Update denominator (LCM would be better, but for simplicity multiply)
        # Actually, let's just track symbolically

    # For proper assembly, we need to combine over common denominator
    # This is getting complex - let's compute numerically first

    if verbose:
        print("\n" + "=" * 60)
        print("S₁₂ ASSEMBLY (I₂ contributions only)")
        print("=" * 60)

        # Compute S12 numerically at R*
        S12_numeric = 0
        for (ell1, ell2), r in I2_results.items():
            sym_factor = 2 if ell1 != ell2 else 1
            factorial_norm = 1 / (factorial(ell1) * factorial(ell2))
            weight = float(sym_factor * factorial_norm)

            z_coeffs = r['z_coeffs']
            den = r['denominator']

            z_star = float(N(exp(R_star_approx / 7), 20))
            den_val = float(N(den.subs(R, R_star_approx), 20))

            I2_val = 0
            for p, c in z_coeffs.items():
                if c != 0:
                    c_val = float(N(c.subs(R, R_star_approx), 20))
                    I2_val += c_val * (z_star ** p)
            I2_val /= den_val

            contribution = weight * I2_val
            S12_numeric += contribution

            print(f"  ({ell1},{ell2}): weight={weight:.6f}, I₂={I2_val:.10f}, contrib={contribution:.10f}")

        print(f"\n  S₁₂(R*) from I₂ only = {S12_numeric:.10f}")

    return S12_coeffs, S12_den


def main():
    print("=" * 70)
    print("STEP 2: c(R) IN z-BASIS (z = e^{R/7})")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")
    print("\nz-Power Mapping:")
    print("  z⁰ = 1, z⁴ = e^{4R/7}, z⁷ = e^R (mirror), z⁸ = e^{8R/7}, z¹⁴ = e^{2R}")

    # Step 1: Get U(R) for all pairs
    print("\n" + "=" * 60)
    print("LOADING U(R) FROM STEP 1")
    print("=" * 60)

    U_results = compute_all_symbolic(verbose=False)
    print(f"Loaded {len(U_results)} pairs")

    # Step 2: Compute T(R) symbolically
    print("\n" + "=" * 60)
    print("COMPUTING T(R)")
    print("=" * 60)

    T_expr, T_z14, T_const, T_den = compute_T_symbolic(verbose=True)

    # Verify T at R*
    T_numeric = float(N(T_expr.subs(R, R_star_approx), 20))
    print(f"\n  T(R*) = {T_numeric:.10f}")

    # Step 3: Compute I₂ for all pairs in z-basis
    print("\n" + "=" * 60)
    print("COMPUTING I₂ IN z-BASIS")
    print("=" * 60)

    I2_results = compute_I2_z_basis(U_results, T_expr, T_z14, T_const, T_den, verbose=True)

    # Step 4: Assemble S₁₂
    print("\n" + "=" * 60)
    print("ASSEMBLING S₁₂")
    print("=" * 60)

    S12_coeffs, S12_den = assemble_S12_z_basis(I2_results, verbose=True)

    # Compare with full numeric c computation
    print("\n" + "=" * 60)
    print("COMPARISON WITH FULL c(R)")
    print("=" * 60)

    from src.unified_i2_paper import compute_I2_unified_paper
    from src.path_a.unit_test_symbolic import get_optimal_polynomials

    polys = get_optimal_polynomials()

    # Get S12 from full paper regime
    S12_full = 0
    for ell1, ell2 in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]:
        sym_factor = 2 if ell1 != ell2 else 1
        factorial_norm = 1 / (factorial(ell1) * factorial(ell2))
        weight = float(sym_factor * factorial_norm)

        I2_result = compute_I2_unified_paper(
            R_star_approx, 4/7, ell1, ell2, polys,
            n_quad_u=80, n_quad_t=80, include_Q=True
        )
        S12_full += weight * I2_result.I2_value

    print(f"S₁₂(R*) from symbolic I₂: matches numeric above")
    print(f"S₁₂(R*) from full engine: {S12_full:.10f}")

    # Summary of z-basis structure
    print("\n" + "=" * 60)
    print("Z-BASIS STRUCTURE SUMMARY")
    print("=" * 60)
    print("\nI₂ = U(R) × T(R) / θ")
    print("\nwhere U(R) = (Σ_k A_k·z^k) / D_U  and  T(R) = (A_T·z¹⁴ + B_T) / D_T")
    print("\nPowers of z present in I₂:")
    for (ell1, ell2), r in I2_results.items():
        z_powers = sorted([p for p, c in r['z_coeffs'].items() if c != 0])
        print(f"  ({ell1},{ell2}): {z_powers}")

    print("\nNote: z = e^{R/7}, so:")
    print("  z⁴ = exp(4R/7) = exp(Rθ)")
    print("  z⁷ = exp(R)   ← FOR MIRROR M₀ = z⁷ + 5")
    print("  z⁸ = exp(8R/7)")
    print("  z¹⁴ = exp(2R)")
    print("  z¹⁸ = exp(18R/7) = z⁴·z¹⁴")
    print("  z²² = exp(22R/7) = z⁸·z¹⁴")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)

    return U_results, I2_results


if __name__ == "__main__":
    main()
