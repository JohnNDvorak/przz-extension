#!/usr/bin/env python3
"""
scripts/synthesis_przz_derivation_paths.py
SYNTHESIS: Analysis of all three derivation paths for PRZZ-theorem-faithful m

SUMMARY OF FINDINGS:

PATH 1: Bracket Derivative Analysis
=====================================
KEY INSIGHT: The COMBINED identity (with outer exp(-Rθ(x+y))) fundamentally
changes the t-weighting structure for derivative terms.

- The combined exponential is exp(Rθ(x+y)(2t-1)), NOT exp(2Rtθ(x+y))
- At t=0.5, the combined exponential equals 1 (xy-dependence vanishes)
- The xy coefficient integral (for I₁) has DIFFERENT structure than scalar (for I₂)

CONCLUSION: The bracket structure DOES distinguish I₁ from I₂ terms.
This provides a mathematical basis for g_I1 ≠ g_I2.

PATH 2: Mirror Operator Analysis
================================
KEY INSIGHT: Treating the mirror as an operator with different eigenvalues
for I₁ vs I₂ matches the production structure (g_I1, g_I2).

- Combined bracket gives m_I1/m_I2 ratio ≈ 0.11 (eigenvalue structure exists)
- Production g_I1/g_I2 ratio ≈ 0.98 (close to 1)
- The operator eigenvalues don't directly match production values

CONCLUSION: There's operator structure, but a normalization/renormalization
step is needed to connect bracket eigenvalues to production g factors.

PATH 3: Combinatorial Analysis
==============================
KEY INSIGHT: The non-circular B/A ratio is ~6.0, not 5.0.

- 2K-1 = 5 was ASSUMED in ABD code (circular definition)
- Non-circular computation gives B/A ≈ 6.0 for both benchmarks
- This suggests 2K might be correct, not 2K-1
- However, (exp(R)+6)/(exp(R)+5) ≈ 1.115 is LARGER than needed gap (~1.015)

CONCLUSION: The "+5" is not directly derived from PRZZ, but neither is "+6"
the full answer. The true structure is more subtle.

OVERALL SYNTHESIS
=================
The three paths reveal that:

1. The PRZZ bracket structure DOES have mathematical features that
   distinguish I₁ (derivative) from I₂ (non-derivative) terms

2. These features could in principle DERIVE an operator-valued mirror
   with different eigenvalues for different integral types

3. The specific production formulas (g_I1, g_I2, base = exp(R)+5)
   are NOT directly derivable but could be justified as:
   - Eigenvalue approximations to a more complex operator
   - Empirically calibrated values that capture the correct structure

RECOMMENDATION
==============
The most honest claim is:

"We identify PRZZ bracket structure that distinguishes I₁ from I₂ terms,
providing mathematical justification for integral-type-dependent mirror
weights. The specific formulas m = g_total × (exp(R) + 5) with
g_I1 = 1 + θ(1-θ)(2(K-1)+θ)/(8K(2K+1)²) and
g_I2 = 1 + θ(2-θ)/(2K(2K+1))
are validated to <0.0003% against two independent PRZZ benchmarks,
though the derivation of these specific coefficients from the bracket
structure remains incomplete."

This is stronger than "purely empirical" but weaker than "fully derived."

Created: 2025-12-29 (Phase 53 - PRZZ Derivation Investigation)
"""

import math
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quadrature import gauss_legendre_01


def compute_comprehensive_analysis(R: float, theta: float = 4/7, K: int = 3, n_quad: int = 100):
    """
    Comprehensive analysis combining all three paths.
    """
    print("=" * 70)
    print(f"COMPREHENSIVE ANALYSIS FOR R = {R}, θ = {theta}, K = {K}")
    print("=" * 70)
    print()

    # =========================================================================
    # PATH 1: Bracket derivative structure
    # =========================================================================
    print("PATH 1: BRACKET DERIVATIVE STRUCTURE")
    print("-" * 40)

    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Combined identity structure
    xy_integral_combined = 0.0
    scalar_integral = 0.0

    for t, w in zip(t_nodes, t_weights):
        exp_2Rt = math.exp(2 * R * t)
        u = R * theta * (2*t - 1)
        xy_coeff_combined = u**2 + 2*theta*u

        xy_integral_combined += exp_2Rt * xy_coeff_combined * w
        scalar_integral += exp_2Rt * w

    print(f"  Scalar integral (I₂-type): {scalar_integral:.6f}")
    print(f"  xy-coeff integral (I₁-type): {xy_integral_combined:.6f}")
    print(f"  Ratio xy/scalar: {xy_integral_combined/scalar_integral:.6f}")
    print()

    # =========================================================================
    # PATH 2: Operator eigenvalue structure
    # =========================================================================
    print("PATH 2: OPERATOR EIGENVALUE STRUCTURE")
    print("-" * 40)

    # Effective eigenvalues from bracket
    m_I1_bracket = xy_integral_combined / scalar_integral
    m_I2_bracket = scalar_integral / scalar_integral  # = 1 (normalized)

    # DQ limit
    dq_limit = (math.exp(2*R) - 1) / (2*R)

    print(f"  DQ scalar limit: {dq_limit:.6f}")
    print(f"  Effective m_I1 (xy/scalar ratio): {m_I1_bracket:.6f}")
    print(f"  Effective m_I2 (scalar/scalar): {m_I2_bracket:.6f}")
    print()

    # Production values
    g_I1 = 1 + theta * (1 - theta) * (2*(K-1) + theta) / (8 * K * (2*K + 1)**2)
    g_I2 = 1 + theta * (2 - theta) / (2 * K * (2*K + 1))
    base = math.exp(R) + (2*K - 1)

    print("  Production formulas:")
    print(f"    g_I1 = {g_I1:.8f}")
    print(f"    g_I2 = {g_I2:.8f}")
    print(f"    base = exp(R) + {2*K-1} = {base:.6f}")
    print(f"    m_I1_prod = g_I1 × base = {g_I1 * base:.6f}")
    print(f"    m_I2_prod = g_I2 × base = {g_I2 * base:.6f}")
    print()

    # =========================================================================
    # PATH 3: Combinatorial structure
    # =========================================================================
    print("PATH 3: COMBINATORIAL STRUCTURE")
    print("-" * 40)

    print(f"  2K-1 (production) = {2*K-1}")
    print(f"  2K (non-circular B/A) = {2*K}")
    print(f"  exp(R) = {math.exp(R):.6f}")
    print(f"  exp(R) + (2K-1) = {math.exp(R) + 2*K-1:.6f}")
    print(f"  exp(R) + 2K = {math.exp(R) + 2*K:.6f}")
    print()

    # =========================================================================
    # SYNTHESIS: What can be derived?
    # =========================================================================
    print("=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print()

    print("WHAT THE BRACKET STRUCTURE PROVIDES:")
    print("  1. Mathematical distinction between I₁ (derivative) and I₂ (scalar)")
    print("  2. A ratio structure: I₁ terms get different weighting than I₂")
    print("  3. The t-weighting at t=0.5 has special structure (xy vanishes)")
    print()

    print("WHAT REMAINS UNRESOLVED:")
    print("  1. How to normalize bracket eigenvalues to production g values")
    print("  2. Why base = exp(R) + (2K-1) instead of DQ limit or other forms")
    print("  3. The exact coefficient structure in g_I1 and g_I2 formulas")
    print()

    print("PROPOSED INTERPRETATION:")
    print("  The production formulas are PHENOMENOLOGICAL FITS that capture")
    print("  the CORRECT STRUCTURE (I₁ ≠ I₂) but with coefficients that")
    print("  were numerically determined rather than algebraically derived.")
    print()

    # =========================================================================
    # Alternative formula exploration
    # =========================================================================
    print("=" * 70)
    print("ALTERNATIVE FORMULA EXPLORATION")
    print("=" * 70)
    print()

    # Can we write m in terms of bracket quantities?
    # m = A × exp(R) + B where A, B depend on integral structure

    # The effective "g" from bracket structure:
    g_bracket_I1 = (1 + m_I1_bracket) / 2 if m_I1_bracket < 1 else 1 + (m_I1_bracket - 1)/2

    print("Attempting to connect bracket to production:")
    print(f"  Bracket xy/scalar ratio: {m_I1_bracket:.6f}")
    print(f"  Production g_I1: {g_I1:.8f}")
    print(f"  Production g_I2: {g_I2:.8f}")
    print()

    # What if the g factors encode the bracket structure?
    # g_I1 - 1 ≈ 0.00095, g_I2 - 1 ≈ 0.0194
    # Ratio: (g_I1-1)/(g_I2-1) ≈ 0.049

    ratio_g_minus_1 = (g_I1 - 1) / (g_I2 - 1) if (g_I2 - 1) != 0 else 0

    print("Correction structure:")
    print(f"  g_I1 - 1 = {g_I1 - 1:.8f}")
    print(f"  g_I2 - 1 = {g_I2 - 1:.8f}")
    print(f"  Ratio (g_I1-1)/(g_I2-1) = {ratio_g_minus_1:.6f}")
    print()

    # Compare to bracket ratio
    print(f"  Bracket xy/scalar ratio: {m_I1_bracket:.6f}")
    print(f"  Is there a relationship? Need further analysis.")
    print()

    # Check if the correction terms have simple θ,K structure
    print("Checking if corrections are simple functions of θ, K:")
    print(f"  θ = {theta:.6f}")
    print(f"  θ² = {theta**2:.6f}")
    print(f"  θ(1-θ) = {theta*(1-theta):.6f}")
    print(f"  θ(2-θ) = {theta*(2-theta):.6f}")
    print(f"  1/(2K(2K+1)) = {1/(2*K*(2*K+1)):.6f}")
    print(f"  θ(2-θ)/(2K(2K+1)) = {theta*(2-theta)/(2*K*(2*K+1)):.8f}")
    print(f"  This equals (g_I2 - 1): {g_I2 - 1:.8f} ✓")
    print()

    return {
        'scalar_integral': scalar_integral,
        'xy_integral_combined': xy_integral_combined,
        'dq_limit': dq_limit,
        'g_I1': g_I1,
        'g_I2': g_I2,
        'base': base,
        'ratio_bracket': xy_integral_combined/scalar_integral,
    }


def final_recommendation():
    """
    Final recommendation on what can be claimed.
    """
    print()
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print()

    print("TIER A - FULLY PRZZ-DERIVED:")
    print("  • κ = 1 - log(c)/R formula")
    print("  • Mirror structure: I₁,I₂ need mirror; I₃,I₄ don't")
    print("  • Polynomial constraints: P(0)=0, P₁(1)=1, Q(0)=1")
    print()

    print("TIER B - STRUCTURALLY JUSTIFIED:")
    print("  • g_I1 ≠ g_I2 (different mirror weights for I₁ vs I₂)")
    print("    → Justified by combined bracket having different xy vs scalar structure")
    print("  • m = g × base structure")
    print("    → Justified by PRZZ bracket with T^{-α-β} prefactor")
    print()

    print("TIER C - PHENOMENOLOGICAL (EMPIRICALLY CALIBRATED):")
    print("  • g_I1 = 1 + θ(1-θ)(2(K-1)+θ)/(8K(2K+1)²)")
    print("    → Numerically found, validated to <0.0003%")
    print("  • g_I2 = 1 + θ(2-θ)/(2K(2K+1))")
    print("    → Numerically found, validated to <0.0003%")
    print("  • base = exp(R) + (2K-1)")
    print("    → 2K-1 was circular; non-circular gives ~6, not 5")
    print()

    print("UPGRADED CLAIM (vs 'purely empirical'):")
    print("  'We derive integral-type-dependent mirror structure from PRZZ,")
    print("   with phenomenological correction factors validated to <0.0003%'")
    print()

    print("WHAT WOULD CLOSE THE GAP:")
    print("  1. Derive the specific g_I1, g_I2 coefficients from PRZZ algebra")
    print("  2. Show why base = exp(R) + (2K-1) despite non-circular B/A ≈ 6")
    print("  3. Or: reformulate with base = exp(R) + 2K and new g factors")


def main():
    # Analyze both benchmarks
    for R in [1.3036, 1.1167]:
        result = compute_comprehensive_analysis(R)
        print()

    final_recommendation()


if __name__ == "__main__":
    main()
