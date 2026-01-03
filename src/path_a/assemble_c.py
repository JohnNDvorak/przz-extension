#!/usr/bin/env python3
"""
Path A: Assemble full c(R) from all pair integrals.

The PRZZ formula for c(R) combines all 6 pairs with:
- Symmetry factor: 2 for off-diagonal pairs
- Factorial normalization: 1/(ℓ₁! ℓ₂!)
- Sign convention from paper

For K=3, d=1:
    c(R) = Σ_{ℓ₁≤ℓ₂} w_{ℓ₁,ℓ₂} × (I₁ + I₂ + I₃ + I₄)_{ℓ₁,ℓ₂}

where w_{ℓ₁,ℓ₂} = (2 if ℓ₁≠ℓ₂ else 1) × 1/(ℓ₁! ℓ₂!)

With the mirror formula applied for full assembly.

Usage:
    python -m src.path_a.assemble_c
"""
import sympy as sp
from sympy import N, factorial, simplify, together, expand
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.path_a.optimal_coeffs import R, theta, R_star_approx
from src.path_a.symbolic_pairs import compute_pair, split_exp2R


def compute_all_pairs(verbose: bool = True) -> Dict:
    """Compute all 6 pairs and return their symbolic expressions."""
    pairs = [
        (1, 1), (1, 2), (1, 3),
        (2, 2), (2, 3), (3, 3)
    ]

    all_results = {}

    for ell1, ell2 in pairs:
        if verbose:
            print(f"\n--- Computing pair ({ell1},{ell2}) ---")
        results = compute_pair(ell1, ell2, verbose=verbose)
        all_results[(ell1, ell2)] = results

    return all_results


def assemble_c_simple(all_results: Dict, verbose: bool = True) -> sp.Expr:
    """
    Assemble c(R) using simple sum over all pairs.

    c(R) = Σ_{ℓ₁≤ℓ₂} w_{ℓ₁,ℓ₂} × (I₁ + I₂ + I₃ + I₄)_{ℓ₁,ℓ₂}
    """
    c_expr = sp.Integer(0)

    for (ell1, ell2), results in all_results.items():
        # Weight factor
        sym_factor = 2 if ell1 != ell2 else 1
        factorial_norm = 1 / (factorial(ell1) * factorial(ell2))
        weight = sym_factor * factorial_norm

        # Sum I1 + I2 + I3 + I4
        pair_sum = sp.Integer(0)
        for name in ['I1', 'I2', 'I3', 'I4']:
            if name in results and 'expr' in results[name]:
                pair_sum += results[name]['expr']

        contribution = weight * pair_sum

        if verbose:
            val = float(N(contribution.subs(R, R_star_approx), 20))
            print(f"  ({ell1},{ell2}): weight={float(weight):.6f}, contribution at R*={val:.10f}")

        c_expr += contribution

    return simplify(c_expr)


def assemble_c_with_mirror(all_results: Dict, verbose: bool = True) -> sp.Expr:
    """
    Assemble c(R) using mirror formula from PRZZ.

    The mirror formula separates I₁+I₂ (which need mirror) from I₃+I₄ (no mirror):

    c(R) = S₁₂(+R) + m(R)·S₁₂(-R) + S₃₄(+R)

    where:
    - S₁₂(R) = Σ w_{ℓ₁,ℓ₂} × (I₁ + I₂)
    - S₃₄(R) = Σ w_{ℓ₁,ℓ₂} × (I₃ + I₄)
    - m(R) = exp(R) + (2K - 1) = exp(R) + 5  for K=3
    """
    K = 3
    m = sp.exp(R) + (2*K - 1)

    S12 = sp.Integer(0)
    S34 = sp.Integer(0)

    for (ell1, ell2), results in all_results.items():
        # Weight factor
        sym_factor = 2 if ell1 != ell2 else 1
        factorial_norm = 1 / (factorial(ell1) * factorial(ell2))
        weight = sym_factor * factorial_norm

        # I1 + I2
        I1_I2 = sp.Integer(0)
        for name in ['I1', 'I2']:
            if name in results and 'expr' in results[name]:
                I1_I2 += results[name]['expr']

        # I3 + I4
        I3_I4 = sp.Integer(0)
        for name in ['I3', 'I4']:
            if name in results and 'expr' in results[name]:
                I3_I4 += results[name]['expr']

        S12 += weight * I1_I2
        S34 += weight * I3_I4

    # Mirror formula
    S12_plus = S12
    S12_minus = S12.subs(R, -R)

    c_expr = S12_plus + m * S12_minus + S34

    if verbose:
        S12_val = float(N(S12_plus.subs(R, R_star_approx), 20))
        S12_neg_val = float(N(S12_minus.subs(R, R_star_approx), 20))
        S34_val = float(N(S34.subs(R, R_star_approx), 20))
        m_val = float(N(m.subs(R, R_star_approx), 20))

        print(f"\nMirror assembly at R* = {R_star_approx}:")
        print(f"  S₁₂(+R*) = {S12_val:.10f}")
        print(f"  S₁₂(-R*) = {S12_neg_val:.10f}")
        print(f"  S₃₄(+R*) = {S34_val:.10f}")
        print(f"  m(R*)    = {m_val:.10f}")

    return simplify(c_expr)


def analyze_c_minus_1(c_expr: sp.Expr, verbose: bool = True):
    """
    Analyze c(R) - 1 to understand the algebraic structure.

    Goal: Express c(R) - 1 = N(R, e^{2R}) / D(R) and check if N(R*) = 0.
    """
    diff_expr = c_expr - 1

    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS OF c(R) - 1")
        print("=" * 60)

    # Evaluate at R*
    diff_at_Rstar = float(N(diff_expr.subs(R, R_star_approx), 30))
    if verbose:
        print(f"\nc(R*) - 1 = {diff_at_Rstar:.15e}")

    # Try to get normal form
    try:
        A2, A0, den = split_exp2R(diff_expr)
        if verbose:
            print(f"\nNormal form of c(R) - 1:")
            print(f"  Denominator: {den}")
            if isinstance(A2, sp.Expr):
                try:
                    from sympy import Poly
                    poly_A2 = Poly(A2, R)
                    poly_A0 = Poly(A0, R)
                    print(f"  A₂ degree: {poly_A2.degree()}")
                    print(f"  A₀ degree: {poly_A0.degree()}")
                except:
                    pass
    except Exception as e:
        if verbose:
            print(f"  Could not get normal form: {e}")

    return diff_at_Rstar


def main():
    print("=" * 70)
    print("PATH A: FULL c(R) ASSEMBLY")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")

    # Compute all pairs
    print("\n" + "=" * 60)
    print("COMPUTING ALL PAIRS")
    print("=" * 60)
    all_results = compute_all_pairs(verbose=True)

    # Simple assembly (for comparison)
    print("\n" + "=" * 60)
    print("SIMPLE ASSEMBLY (no mirror)")
    print("=" * 60)
    c_simple = assemble_c_simple(all_results, verbose=True)
    c_simple_val = float(N(c_simple.subs(R, R_star_approx), 30))
    print(f"\nc_simple(R*) = {c_simple_val:.15f}")

    # With mirror formula
    print("\n" + "=" * 60)
    print("MIRROR ASSEMBLY")
    print("=" * 60)
    c_mirror = assemble_c_with_mirror(all_results, verbose=True)
    c_mirror_val = float(N(c_mirror.subs(R, R_star_approx), 30))
    print(f"\nc_mirror(R*) = {c_mirror_val:.15f}")

    # Analyze c - 1
    analyze_c_minus_1(c_simple, verbose=True)

    print("\n" + "=" * 70)
    print("ASSEMBLY COMPLETE")
    print("=" * 70)

    return c_simple, c_mirror, all_results


if __name__ == "__main__":
    main()
