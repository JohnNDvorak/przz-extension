#!/usr/bin/env python3
"""
Path A Phase 2: Express c(R) - 1 in Algebraic Normal Form

Goal: Derive c(R) - 1 = N(R, e^{2R}) / D(R) where N, D are polynomials,
and show that N(R*) = 0 for R* ≈ 1.14976.

The PRZZ assembly formula is:
    c(R) = S₁₂(+R) + M(R) × S₁₂(-R) + S₃₄(+R)

where:
    S₁₂ = Σ w_{ℓ₁,ℓ₂} × (I₁ + I₂)
    S₃₄ = Σ w_{ℓ₁,ℓ₂} × (I₃ + I₄)
    M(R) = G × (exp(R) + 5)  for K=3
    G ≈ 1.015 (correction factor)

Each integral has form: (A·e^{2R} + B) / (C·R^{11})

So c(R) lives in ℚ(R, e^R, e^{2R}), and c(R) - 1 has a root at R*.

Usage:
    python -m src.path_a.phase2_c_minus_1
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, Poly, N, factor, lcm, gcd
)
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.path_a.optimal_coeffs import R, theta, R_star_approx
from src.path_a.symbolic_pairs import compute_pair

# =============================================================================
# Symbolic symbols
# =============================================================================
z = symbols('z')  # placeholder for exp(R)
w = symbols('w')  # placeholder for exp(2R)

# Correction factor G (from first-principles derivation)
# G = f_I1 * g_I1 + (1 - f_I1) * g_I2
# For our optimal polynomials: G ≈ 1.015
G_approx = Rational(1015, 1000)

# Mirror base: M₀ = exp(R) + 5 for K=3
K = 3


def compute_S12_S34(all_results: Dict, verbose: bool = True) -> Tuple[sp.Expr, sp.Expr]:
    """
    Compute S₁₂ and S₃₄ as weighted sums over all pairs.

    S₁₂ = Σ w_{ℓ₁,ℓ₂} × (I₁ + I₂)
    S₃₄ = Σ w_{ℓ₁,ℓ₂} × (I₃ + I₄)
    """
    from sympy import factorial

    S12 = sp.Integer(0)
    S34 = sp.Integer(0)

    for (ell1, ell2), results in all_results.items():
        # Weight factor: (2 if off-diagonal else 1) × 1/(ℓ₁!ℓ₂!)
        sym_factor = 2 if ell1 != ell2 else 1
        factorial_norm = Rational(1, factorial(ell1) * factorial(ell2))
        weight = sym_factor * factorial_norm

        # I₁ + I₂
        I1 = results.get('I1', {}).get('expr', sp.Integer(0))
        I2 = results.get('I2', {}).get('expr', sp.Integer(0))
        S12 += weight * (I1 + I2)

        # I₃ + I₄
        I3 = results.get('I3', {}).get('expr', sp.Integer(0))
        I4 = results.get('I4', {}).get('expr', sp.Integer(0))
        S34 += weight * (I3 + I4)

        if verbose:
            I12_val = float(N((I1 + I2).subs(R, R_star_approx), 20))
            I34_val = float(N((I3 + I4).subs(R, R_star_approx), 20))
            print(f"  ({ell1},{ell2}): weight={float(weight):.6f}, "
                  f"I1+I2={I12_val:.10f}, I3+I4={I34_val:.10f}")

    return S12, S34


def compute_c_algebraic(S12: sp.Expr, S34: sp.Expr, verbose: bool = True) -> sp.Expr:
    """
    Compute c(R) using the mirror formula:

    c(R) = S₁₂(+R) + M(R) × S₁₂(-R) + S₃₄(+R)

    where M(R) = G × (exp(R) + 2K - 1) = G × (exp(R) + 5)
    """
    # Mirror multiplier
    M = G_approx * (exp(R) + (2*K - 1))

    # S₁₂ at -R
    S12_minus = S12.subs(R, -R)

    # Assembly
    c_expr = S12 + M * S12_minus + S34

    if verbose:
        M_val = float(N(M.subs(R, R_star_approx), 20))
        S12_plus_val = float(N(S12.subs(R, R_star_approx), 20))
        S12_minus_val = float(N(S12_minus.subs(R, R_star_approx), 20))
        S34_val = float(N(S34.subs(R, R_star_approx), 20))
        c_val = float(N(c_expr.subs(R, R_star_approx), 20))

        print(f"\nMirror assembly at R* = {R_star_approx}:")
        print(f"  M(R*)     = {M_val:.10f}")
        print(f"  S₁₂(+R*)  = {S12_plus_val:.10f}")
        print(f"  S₁₂(-R*)  = {S12_minus_val:.10f}")
        print(f"  S₃₄(+R*)  = {S34_val:.10f}")
        print(f"  c(R*)     = {c_val:.15f}")

    return c_expr


def express_in_normal_form(expr: sp.Expr, verbose: bool = True) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Express expr in the form (A·e^{2R} + B·e^R + C) / D

    Returns (numerator, denominator, degree info)
    """
    # Combine to single fraction
    expr_together = together(expand(expr))
    num, den = fraction(expr_together)
    num = expand(num)

    if verbose:
        print(f"\nNormal form analysis:")
        print(f"  Denominator: {factor(den)}")

    # Substitute z = exp(R), w = exp(2R) to analyze structure
    num_sub = num.subs({exp(2*R): w, exp(R): z})

    try:
        # Try as polynomial in z and w
        num_sub = expand(num_sub)

        # Check if we have both z and w terms
        has_z = z in num_sub.free_symbols
        has_w = w in num_sub.free_symbols

        if verbose:
            print(f"  Contains exp(R): {has_z}")
            print(f"  Contains exp(2R): {has_w}")

        # Get coefficient structure
        if has_w:
            poly_w = Poly(num_sub, w)
            if verbose:
                print(f"  Degree in exp(2R): {poly_w.degree()}")

        if has_z:
            poly_z = Poly(num_sub, z)
            if verbose:
                print(f"  Degree in exp(R): {poly_z.degree()}")

    except Exception as e:
        if verbose:
            print(f"  Polynomial analysis failed: {e}")

    return num, den, num_sub


def analyze_c_minus_1(c_expr: sp.Expr, verbose: bool = True) -> Dict:
    """
    Analyze c(R) - 1 to find the algebraic structure.
    """
    diff = c_expr - 1

    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS OF c(R) - 1")
        print("=" * 60)

    # Evaluate at R*
    diff_at_Rstar = float(N(diff.subs(R, R_star_approx), 30))
    if verbose:
        print(f"\nc(R*) - 1 = {diff_at_Rstar:.15e}")

    # Get normal form
    num, den, num_sub = express_in_normal_form(diff, verbose)

    # Verify: does numerator vanish at R*?
    num_at_Rstar = float(N(num.subs(R, R_star_approx), 30))
    den_at_Rstar = float(N(den.subs(R, R_star_approx), 30))

    if verbose:
        print(f"\n  Numerator at R*:   {num_at_Rstar:.10e}")
        print(f"  Denominator at R*: {den_at_Rstar:.10e}")
        print(f"  Ratio:             {num_at_Rstar/den_at_Rstar:.15e}")

    return {
        'diff': diff,
        'numerator': num,
        'denominator': den,
        'num_substituted': num_sub,
        'diff_at_Rstar': diff_at_Rstar,
        'num_at_Rstar': num_at_Rstar,
    }


def main():
    print("=" * 70)
    print("PATH A PHASE 2: c(R) - 1 IN ALGEBRAIC NORMAL FORM")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")
    print(f"G ≈ {G_approx} = {float(G_approx):.10f}")
    print(f"K = {K}")

    # Step 1: Compute all pairs
    print("\n" + "=" * 60)
    print("STEP 1: COMPUTING ALL PAIRS")
    print("=" * 60)

    all_results = {}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    for ell1, ell2 in pairs:
        print(f"\n  Computing pair ({ell1},{ell2})...")
        results = compute_pair(ell1, ell2, verbose=False)
        all_results[(ell1, ell2)] = results

    # Step 2: Compute S₁₂ and S₃₄
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING S₁₂ AND S₃₄")
    print("=" * 60)

    S12, S34 = compute_S12_S34(all_results, verbose=True)

    S12_val = float(N(S12.subs(R, R_star_approx), 20))
    S34_val = float(N(S34.subs(R, R_star_approx), 20))
    print(f"\n  Total S₁₂(+R*) = {S12_val:.10f}")
    print(f"  Total S₃₄(+R*) = {S34_val:.10f}")

    # Step 3: Compute c(R) with mirror formula
    print("\n" + "=" * 60)
    print("STEP 3: APPLYING MIRROR FORMULA")
    print("=" * 60)

    c_expr = compute_c_algebraic(S12, S34, verbose=True)

    # Step 4: Analyze c(R) - 1
    analysis = analyze_c_minus_1(c_expr, verbose=True)

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    c_val = float(N(c_expr.subs(R, R_star_approx), 30))
    print(f"\nc(R*) = {c_val:.15f}")
    print(f"c(R*) - 1 = {analysis['diff_at_Rstar']:.15e}")

    if abs(analysis['diff_at_Rstar']) < 0.1:
        print("\n✓ c(R*) is close to 1!")
        print("  The numerator of c(R) - 1 nearly vanishes at R*.")
    else:
        print("\n✗ c(R*) differs significantly from 1.")
        print("  Check normalization or polynomial coefficients.")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    return c_expr, analysis


if __name__ == "__main__":
    c_expr, analysis = main()
