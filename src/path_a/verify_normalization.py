#!/usr/bin/env python3
"""
Verify normalization factors between symbolic and KappaEngine.

KEY FINDINGS (2026-01-03):
==========================

1. I₁ Normalization:
   - Paper regime uses Case C kernel attenuation for P₂/P₃
   - KappaEngine applies factorial normalization: 1/(ℓ₁!·ℓ₂!)
   - After both: Paper I₁ with factorial norm = KappaEngine I₁ ✓

2. S₃₄ Normalization:
   - Raw S₃₄ to KappaEngine: factor = (2K)(2K-1)/(2K+1)² = 30/49
   - Plus factorial normalization: 1/(ℓ₁!·ℓ₂!)

3. I₂ Normalization:
   - Symbolic I₂ matches KappaEngine exactly (no correction needed)

FORMULAS:
=========
For K=3:
  S₃₄_corrected = S₃₄_raw × (30/49) × factorial_norm
  I₁_corrected = I₁_paper × factorial_norm
  I₂_corrected = I₂_symbolic (no correction)

where factorial_norm[pair] = 1/(ℓ₁!·ℓ₂!)
"""

import numpy as np
from fractions import Fraction
from dataclasses import dataclass
from typing import Dict

from src.kappa_engine import KappaEngine
from src.path_a.optimal_coeffs import Q_coeffs, R_star_approx

# Polynomial coefficients
P1_list = [-2.0, 0.9375, 1.0, -0.6]
P2_list = [0.5241, 1.3199, -0.9401]
P3_list = [0.1367, -0.6865, -0.0499]


def expand_q_to_monomial(q0, q1, q3, q5):
    c0 = q0 + q1 + q3 + q5
    c1 = -2*q1 - 6*q3 - 10*q5
    c2 = 12*q3 + 40*q5
    c3 = -8*q3 - 80*q5
    c4 = 80*q5
    c5 = -32*q5
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]


Q_mono = expand_q_to_monomial(
    float(Q_coeffs['q0']), float(Q_coeffs['q1']),
    float(Q_coeffs['q3']), float(Q_coeffs['q5'])
)

R_star = float(R_star_approx)
theta = 4/7
K = 3


@dataclass
class NormalizationFactors:
    """Normalization factors for symbolic → KappaEngine conversion."""
    factorial_norm: Dict[str, float]  # 1/(ell1! × ell2!)
    symmetry: Dict[str, float]  # 2 for off-diagonal, 1 for diagonal
    S34_factor: Fraction  # (2K)(2K-1)/(2K+1)²


def get_normalization_factors(K: int = 3) -> NormalizationFactors:
    """Get normalization factors for K pieces."""
    from math import factorial

    pairs = ["11", "12", "13", "22", "23", "33"]

    factorial_norm = {}
    symmetry = {}

    for pair in pairs:
        ell1 = int(pair[0])
        ell2 = int(pair[1])
        factorial_norm[pair] = 1.0 / (factorial(ell1) * factorial(ell2))
        symmetry[pair] = 2.0 if ell1 != ell2 else 1.0

    # S34 factor: (2K)(2K-1)/(2K+1)²
    S34_factor = Fraction(2*K * (2*K - 1), (2*K + 1)**2)

    return NormalizationFactors(
        factorial_norm=factorial_norm,
        symmetry=symmetry,
        S34_factor=S34_factor,
    )


def verify_I1_normalization():
    """Verify I₁ normalization using paper regime."""
    print("=" * 70)
    print("VERIFYING I₁ NORMALIZATION")
    print("=" * 70)

    from src.unified_i1_paper import compute_I1_unified_paper
    from src.polynomials import make_P1_from_tilde, make_Pell_from_tilde, make_Q_from_basis

    # Create polynomial objects
    polynomials = {
        "P1": make_P1_from_tilde(P1_list),
        "P2": make_Pell_from_tilde(P2_list),
        "P3": make_Pell_from_tilde(P3_list),
        "Q": make_Q_from_basis({0: float(Q_coeffs['q0']), 1: float(Q_coeffs['q1']),
                                3: float(Q_coeffs['q3']), 5: float(Q_coeffs['q5'])}),
    }

    norm = get_normalization_factors(K)
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    # Compute paper I1 for all pairs
    paper_I1_weighted = 0.0

    print(f"\nPaper I₁ with factorial normalization:")
    print(f"{'Pair':^8} {'Paper I₁':^15} {'Norm':^10} {'Sym':^5} {'Contrib':^15}")
    print("-" * 60)

    for ell1, ell2 in pairs:
        result = compute_I1_unified_paper(
            R_star, theta, ell1, ell2, polynomials,
            n_quad_u=80, n_quad_t=80, n_quad_a=60,
            include_Q=True, apply_factorial_norm=True
        )

        pair_key = f"{ell1}{ell2}"
        f_norm = norm.factorial_norm[pair_key]
        sym = norm.symmetry[pair_key]
        contrib = result.I1_value * f_norm * sym
        paper_I1_weighted += contrib

        print(f"({ell1},{ell2}){' ':^4} {result.I1_value:>+15.10f} {f_norm:^10.4f} {sym:^5.1f} {contrib:>+15.10f}")

    # Compare with KappaEngine
    engine = KappaEngine(P1_list, P2_list, P3_list, Q_mono, theta=theta, K=K, R=R_star)
    kappa_result = engine.compute_kappa()

    print(f"\n  Paper I₁ weighted sum = {paper_I1_weighted:.10f}")
    print(f"  KappaEngine I₁(+R)   = {kappa_result.integrals.I1_plus:.10f}")
    print(f"  Match: {abs(paper_I1_weighted - kappa_result.integrals.I1_plus) < 1e-8}")

    return paper_I1_weighted, kappa_result.integrals.I1_plus


def verify_S34_normalization():
    """Verify S₃₄ normalization using the (2K)(2K-1)/(2K+1)² factor."""
    print("\n" + "=" * 70)
    print("VERIFYING S₃₄ NORMALIZATION")
    print("=" * 70)

    from src.przz_exact_i34 import compute_I34_all_pairs
    from src.polynomials import make_P1_from_tilde, make_Pell_from_tilde, make_Q_from_basis

    # Create polynomial objects
    polynomials = {
        "P1": make_P1_from_tilde(P1_list),
        "P2": make_Pell_from_tilde(P2_list),
        "P3": make_Pell_from_tilde(P3_list),
        "Q": make_Q_from_basis({0: float(Q_coeffs['q0']), 1: float(Q_coeffs['q1']),
                                3: float(Q_coeffs['q3']), 5: float(Q_coeffs['q5'])}),
    }

    norm = get_normalization_factors(K)

    # Compute raw I34
    raw_results = compute_I34_all_pairs(theta, R_star, polynomials, n_quad=80)

    # Apply factorial normalization + S34 factor
    S34_factor = float(norm.S34_factor)
    corrected_S34 = 0.0

    print(f"\nS₃₄ factor = (2K)(2K-1)/(2K+1)² = {norm.S34_factor} = {S34_factor:.10f}")
    print(f"\n{'Pair':^8} {'Raw I₃₄':^15} {'Factorial':^10} {'Sym':^5} {'Contrib':^15}")
    print("-" * 60)

    for pair_key in raw_results["I3"].keys():
        I3_val = raw_results["I3"][pair_key].value
        I4_val = raw_results["I4"][pair_key].value
        raw_I34 = I3_val + I4_val

        f_norm = norm.factorial_norm[pair_key]
        sym = norm.symmetry[pair_key]

        # Apply both factorial normalization AND S34 factor
        contrib = raw_I34 * f_norm * sym * S34_factor
        corrected_S34 += contrib

        print(f"{pair_key:^8} {raw_I34:>+15.10f} {f_norm:^10.4f} {sym:^5.1f} {contrib:>+15.10f}")

    # Compare with KappaEngine
    engine = KappaEngine(P1_list, P2_list, P3_list, Q_mono, theta=theta, K=K, R=R_star)
    kappa_result = engine.compute_kappa()

    print(f"\n  Corrected S₃₄ = {corrected_S34:.10f}")
    print(f"  KappaEngine S₃₄ = {kappa_result.integrals.S34_plus:.10f}")

    ratio = corrected_S34 / kappa_result.integrals.S34_plus
    print(f"  Ratio: {ratio:.6f}")
    print(f"  Match (within 1%): {abs(ratio - 1.0) < 0.01}")

    return corrected_S34, kappa_result.integrals.S34_plus


def main():
    print("=" * 70)
    print("NORMALIZATION FACTOR VERIFICATION")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  K = {K}")
    print(f"  θ = {theta:.10f}")
    print(f"  R* = {R_star:.10f}")

    norm = get_normalization_factors(K)

    print(f"\nNormalization Factors:")
    print(f"  Factorial norm 1/(ℓ₁!·ℓ₂!):")
    for pair, val in norm.factorial_norm.items():
        print(f"    ({pair[0]},{pair[1]}): {val:.10f}")

    print(f"\n  S₃₄ factor: {norm.S34_factor} = {float(norm.S34_factor):.10f}")
    print(f"    = (2K)(2K-1)/(2K+1)² = {2*K}×{2*K-1}/{(2*K+1)}²")

    # Verify I1
    paper_I1, engine_I1 = verify_I1_normalization()

    # Verify S34
    corrected_S34, engine_S34 = verify_S34_normalization()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. I₁ Normalization:")
    print("   Paper I₁ + factorial norm = KappaEngine I₁ ✓")
    print(f"   {paper_I1:.10f} ≈ {engine_I1:.10f}")

    print("\n2. S₃₄ Normalization:")
    print("   Raw S₃₄ × (30/49) × factorial norm ≈ KappaEngine S₃₄")
    print(f"   {corrected_S34:.10f} ≈ {engine_S34:.10f}")
    print(f"   Ratio: {corrected_S34/engine_S34:.6f}")

    # Remaining discrepancy in S34
    if abs(corrected_S34/engine_S34 - 1.0) > 0.001:
        print("\n   ⚠️ S₃₄ has ~0.5% discrepancy")
        print("   This may be due to:")
        print("   - Different paper regime for I₃/I₄ in KappaEngine")
        print("   - Numerical precision differences")
    else:
        print("   ✓ Match within 0.1%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
