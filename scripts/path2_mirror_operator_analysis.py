#!/usr/bin/env python3
"""
scripts/path2_mirror_operator_analysis.py
Path 2: Investigate mirror as operator with eigenvalue structure

HYPOTHESIS:
The mirror multiplier m should NOT be a scalar, but an OPERATOR that acts
differently on I₁ (derivative terms) vs I₂ (non-derivative terms).

EVIDENCE:
1. Production code uses g_I1 ≠ g_I2, effectively giving different weights
2. Non-circular B/A ≈ 6.0 (not 5.0), suggesting the "+5" is wrong
3. The combined identity (Path 1) shows different t-weighting for xy vs scalar

STRUCTURAL INSIGHT:
If the mirror is an operator M acting on the pair-space:
    M × |I₁⟩ = m₁ × |I₁⟩
    M × |I₂⟩ = m₂ × |I₂⟩

Then the scalar approximation m = f_I1 × m₁ + (1-f_I1) × m₂ would explain
why the production g_I1/g_I2 formulas work.

GOAL:
Derive m₁ and m₂ from the PRZZ bracket structure, showing they are
eigenvalues of a derivable mirror operator.

Created: 2025-12-29 (Phase 53 - PRZZ Derivation Investigation)
"""

import math
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quadrature import gauss_legendre_01


@dataclass
class MirrorOperatorResult:
    """Results from mirror operator analysis."""
    # Eigenvalues for I₁ and I₂
    m_I1: float  # Mirror eigenvalue for I₁ (derivative terms)
    m_I2: float  # Mirror eigenvalue for I₂ (non-derivative terms)

    # Derivation path
    source: str  # How was this derived?

    # Comparison with production
    g_I1_implied: float
    g_I2_implied: float


def compute_i1_mirror_eigenvalue(R: float, theta: float, n_quad: int = 100) -> Tuple[float, Dict]:
    """
    Compute the effective mirror eigenvalue for I₁ terms.

    I₁ terms involve ∂²/∂x∂y, so they "see" the xy coefficient of the
    combined bracket, not the scalar.

    From Path 1 analysis:
    - Combined xy_coeff uses exp(Rθ(x+y)(2t-1)) structure
    - At t=0.5, (2t-1)=0 so xy contribution vanishes

    The effective mirror weight for I₁ should be proportional to:
        ∫₀¹ [xy_coeff(t)] dt / [some normalization]
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # From Path 1: Combined xy coefficient is:
    # exp(2Rt) × [u² + 2θu] where u = Rθ(2t-1)
    # = exp(2Rt) × Rθ(2t-1) × [Rθ(2t-1) + 2θ]
    # = exp(2Rt) × θ × (2t-1) × [Rθ(2t-1) + 2θ] × R

    xy_integral = 0.0
    scalar_integral = 0.0

    for t, w in zip(t_nodes, t_weights):
        exp_2Rt = math.exp(2 * R * t)
        u = R * theta * (2*t - 1)
        xy_coeff = u**2 + 2*theta*u

        xy_integral += exp_2Rt * xy_coeff * w
        scalar_integral += exp_2Rt * w

    # The effective "weight" for I₁ terms
    # This is what multiplies I₁(-R) in the mirror assembly
    effective_weight_I1 = xy_integral / scalar_integral if scalar_integral != 0 else 0

    diagnostics = {
        'xy_integral': xy_integral,
        'scalar_integral': scalar_integral,
        'ratio': effective_weight_I1,
        'dq_limit': (math.exp(2*R) - 1) / (2*R),
    }

    return effective_weight_I1, diagnostics


def compute_i2_mirror_eigenvalue(R: float, theta: float, n_quad: int = 100) -> Tuple[float, Dict]:
    """
    Compute the effective mirror eigenvalue for I₂ terms.

    I₂ terms have NO derivatives (they evaluate at x=y=0), so they "see"
    the scalar limit of the combined bracket.

    The scalar limit is: ∫₀¹ exp(2Rt) dt = (exp(2R)-1)/(2R)

    But for I₂, we need to account for the PRZZ structure where the
    mirror term has T^{-α-β} = exp(2R) prefactor evaluated at x=y=0.
    """
    # The DQ scalar limit
    dq_limit = (math.exp(2*R) - 1) / (2*R)

    # For I₂ (no derivatives), the bracket at x=y=0 gives the DQ limit
    # The mirror contribution in PRZZ is T^{-α-β} × I(-β,-α)
    # At α=β=-R/L, this is exp(2R) × I(-R) where the exp(2R) is explicit

    # But the DQ identity COMBINES direct and mirror via the t-integral
    # So the effective scalar weight is the DQ limit itself

    diagnostics = {
        'dq_limit': dq_limit,
        'exp_2R': math.exp(2*R),
        'exp_R': math.exp(R),
    }

    return dq_limit, diagnostics


def compute_operator_eigenvalue_ratio(R: float, theta: float, n_quad: int = 100) -> Dict:
    """
    Compute the ratio of I₁ to I₂ eigenvalues.

    If the mirror is an operator, this ratio determines how much
    differently I₁ and I₂ terms are weighted.
    """
    m_I1_eff, diag_I1 = compute_i1_mirror_eigenvalue(R, theta, n_quad)
    m_I2_eff, diag_I2 = compute_i2_mirror_eigenvalue(R, theta, n_quad)

    ratio = m_I1_eff / m_I2_eff if m_I2_eff != 0 else 0

    return {
        'm_I1_effective': m_I1_eff,
        'm_I2_effective': m_I2_eff,
        'ratio_I1_to_I2': ratio,
        'diagnostics_I1': diag_I1,
        'diagnostics_I2': diag_I2,
    }


def analyze_production_g_factors(R: float, theta: float, K: int = 3):
    """
    Analyze how the production g_I1 and g_I2 factors relate to operator eigenvalues.

    Production formula:
        m = g_total × base
        g_total = f_I1 × g_I1 + (1-f_I1) × g_I2
        base = exp(R) + (2K-1)

    This effectively gives:
        m_I1 = g_I1 × base
        m_I2 = g_I2 × base

    Compare these to our derived eigenvalues.
    """
    # Production g formulas
    g_I1 = 1 + theta * (1 - theta) * (2*(K-1) + theta) / (8 * K * (2*K + 1)**2)
    g_I2 = 1 + theta * (2 - theta) / (2 * K * (2*K + 1))

    base = math.exp(R) + (2*K - 1)

    m_I1_prod = g_I1 * base
    m_I2_prod = g_I2 * base

    return {
        'g_I1': g_I1,
        'g_I2': g_I2,
        'base': base,
        'm_I1_production': m_I1_prod,
        'm_I2_production': m_I2_prod,
        'ratio_g': g_I1 / g_I2 if g_I2 != 0 else 0,
    }


def infer_base_from_operator_structure(
    R: float,
    theta: float,
    c_target: float,
    S12_plus: float,
    S12_minus: float,
    S34_plus: float,
    f_I1: float,
    n_quad: int = 100
) -> Dict:
    """
    Given the integral values and operator eigenvalue structure,
    infer what the "base" should be.

    Mirror assembly: c = S12(+R) + m × S12(-R) + S34(+R)

    If m = g_total × base where g_total is the weighted eigenvalue,
    then: base = (c_target - S12_plus - S34_plus) / (g_total × S12_minus)

    BUT we can also try to derive base from first principles using
    the operator eigenvalue structure.
    """
    # Get operator eigenvalue structure
    op_result = compute_operator_eigenvalue_ratio(R, theta, n_quad)

    # The eigenvalue ratio tells us how I₁ and I₂ are weighted differently
    ratio_I1_to_I2 = op_result['ratio_I1_to_I2']

    # If we assume the base is related to the DQ limit or exp(R),
    # what g factors would we need?

    m_I2_eff = op_result['m_I2_effective']  # This is the DQ limit for I₂
    m_I1_eff = op_result['m_I1_effective']  # This is the xy coefficient ratio for I₁

    # Compute what g_I1 and g_I2 would be if base = exp(R) + 5
    base_production = math.exp(R) + 5

    implied_g_I1 = m_I1_eff / base_production if base_production != 0 else 0
    implied_g_I2 = m_I2_eff / base_production if base_production != 0 else 0

    return {
        'operator_m_I1': m_I1_eff,
        'operator_m_I2': m_I2_eff,
        'ratio_I1_to_I2': ratio_I1_to_I2,
        'base_production': base_production,
        'implied_g_I1': implied_g_I1,
        'implied_g_I2': implied_g_I2,
    }


def check_eigenvalue_formula_hypotheses(R: float, theta: float, K: int = 3):
    """
    Test hypotheses for what the I₁ and I₂ eigenvalues should be.

    HYPOTHESIS A: m_I1 and m_I2 are related to the derivative structure
    HYPOTHESIS B: m_I1/m_I2 ratio is a simple function of θ
    HYPOTHESIS C: The eigenvalues involve exp(R) not exp(2R)
    """
    op_result = compute_operator_eigenvalue_ratio(R, theta)
    prod_result = analyze_production_g_factors(R, theta, K)

    print("EIGENVALUE ANALYSIS:")
    print(f"  From combined bracket structure:")
    print(f"    m_I1 (xy-coeff/scalar) = {op_result['m_I1_effective']:.6f}")
    print(f"    m_I2 (DQ limit) = {op_result['m_I2_effective']:.6f}")
    print(f"    Ratio m_I1/m_I2 = {op_result['ratio_I1_to_I2']:.6f}")
    print()

    print(f"  From production formulas:")
    print(f"    m_I1_prod = g_I1 × base = {prod_result['m_I1_production']:.6f}")
    print(f"    m_I2_prod = g_I2 × base = {prod_result['m_I2_production']:.6f}")
    print(f"    Ratio g_I1/g_I2 = {prod_result['ratio_g']:.6f}")
    print()

    # Test hypothesis: Is there a simple relationship?
    print("HYPOTHESIS TESTS:")

    # A: Is the ratio close to (1 - some function of θ)?
    ratio = op_result['ratio_I1_to_I2']
    print(f"  A1: Is ratio ≈ 1-θ? {1-theta:.6f} vs {ratio:.6f}")
    print(f"  A2: Is ratio ≈ θ? {theta:.6f} vs {ratio:.6f}")
    print(f"  A3: Is ratio ≈ 2θ? {2*theta:.6f} vs {ratio:.6f}")
    print()

    # B: What does the production g_I1/g_I2 ratio correspond to?
    g_ratio = prod_result['ratio_g']
    print(f"  B1: Production g_I1/g_I2 = {g_ratio:.6f}")
    print(f"  B2: g_I1 - 1 = {prod_result['g_I1'] - 1:.8f}")
    print(f"  B3: g_I2 - 1 = {prod_result['g_I2'] - 1:.8f}")
    print(f"  B4: Ratio of (g-1) terms = {(prod_result['g_I1']-1)/(prod_result['g_I2']-1):.6f}")
    print()

    # C: What's the gap between operator eigenvalues and production values?
    print("GAPS:")
    print(f"  m_I1 gap: operator/prod = {op_result['m_I1_effective']/prod_result['m_I1_production']:.6f}")
    print(f"  m_I2 gap: operator/prod = {op_result['m_I2_effective']/prod_result['m_I2_production']:.6f}")


def derive_m_from_combined_bracket(R: float, theta: float, K: int = 3):
    """
    Attempt to derive the full m formula from the combined bracket structure.

    The combined bracket (after DQ identity) has structure:
        B ~ exp(2Rt) × exp(Rθ(x+y)(2t-1)) × (1+θ(x+y))

    For I₁ (xy coefficient):
        contribution ~ ∫₀¹ exp(2Rt) × [u² + 2θu] dt  where u = Rθ(2t-1)

    For I₂ (scalar):
        contribution ~ ∫₀¹ exp(2Rt) dt = DQ_limit

    The RATIO of these is the key to understanding m.
    """
    print("=" * 70)
    print("DERIVING m FROM COMBINED BRACKET STRUCTURE")
    print("=" * 70)
    print()

    op_result = compute_operator_eigenvalue_ratio(R, theta)

    xy_integral = op_result['diagnostics_I1']['xy_integral']
    scalar_integral = op_result['diagnostics_I1']['scalar_integral']
    dq_limit = op_result['diagnostics_I2']['dq_limit']

    print("RAW INTEGRALS:")
    print(f"  ∫ exp(2Rt) × [xy_coeff(t)] dt = {xy_integral:.10f}")
    print(f"  ∫ exp(2Rt) dt (scalar) = {scalar_integral:.10f}")
    print(f"  DQ limit = (exp(2R)-1)/(2R) = {dq_limit:.10f}")
    print()

    # The key question: what normalization relates these to m_I1, m_I2?
    # The production uses m_I1 = g_I1 × (exp(R) + 5), m_I2 = g_I2 × (exp(R) + 5)

    exp_r = math.exp(R)
    base = exp_r + 5

    print("NORMALIZATION ANALYSIS:")
    print(f"  If base = exp(R) + 5 = {base:.6f}")
    print(f"  Then g_I1 would need: {xy_integral / base:.6f} × (norm factor)")
    print(f"  And g_I2 would need: {dq_limit / base:.6f} × (norm factor)")
    print()

    # What normalization factor makes g_I2 ≈ 1.0194?
    # That means: dq_limit / base × norm ≈ 1.0194
    # So: norm ≈ 1.0194 × base / dq_limit

    target_g_I2 = 1.0194
    implied_norm = target_g_I2 * base / dq_limit
    print(f"  To get g_I2 ≈ 1.0194:")
    print(f"    Normalization factor = {implied_norm:.6f}")
    print(f"    This would give g_I1 = {xy_integral / base * implied_norm:.6f}")
    print()

    # Compare with production g_I1
    prod = analyze_production_g_factors(R, theta, K)
    print("COMPARISON WITH PRODUCTION:")
    print(f"  Production g_I1 = {prod['g_I1']:.8f}")
    print(f"  Production g_I2 = {prod['g_I2']:.8f}")
    print(f"  Derived g_I1 (with norm) = {xy_integral / base * implied_norm:.8f}")


def main():
    theta = 4.0 / 7.0
    K = 3

    print("=" * 70)
    print("PATH 2: MIRROR OPERATOR ANALYSIS")
    print("=" * 70)
    print()

    for R in [1.3036, 1.1167]:
        print(f"\n{'='*70}")
        print(f"R = {R}")
        print("=" * 70)
        print()

        # Analyze operator eigenvalues
        check_eigenvalue_formula_hypotheses(R, theta, K)
        print()

        # Try to derive m
        derive_m_from_combined_bracket(R, theta, K)
        print()


if __name__ == "__main__":
    main()
