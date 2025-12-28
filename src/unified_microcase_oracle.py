"""
src/unified_microcase_oracle.py
Phase 26B: P=Q=1 Microcase Oracle for Independent Validation

With P=Q=1, the PRZZ bracket simplifies analytically:
- P_ℓ(u+x) = 1 for all u, x
- Q(A) = 1 for all A

The bracket becomes:
    exp(2Rt) × exp(Rθ(2t-1)(x+y)) × (1/θ + x + y)

This allows analytic derivation of the x^ℓ₁ y^ℓ₂ coefficient.

DERIVATION:
===========
exp(a(x+y)) = exp(ax) × exp(ay)
            = sum_{i≥0} (ax)^i/i! × sum_{j≥0} (ay)^j/j!

Coefficient of x^i y^j in exp(a(x+y)):
    = a^i/i! × a^j/j!

For (1/θ + x + y) × exp(a(x+y)):

1. Constant term (1/θ) contribution to [x^i y^j]:
   (1/θ) × a^i/i! × a^j/j!

2. x term contribution to [x^i y^j]:
   a^{i-1}/(i-1)! × a^j/j!  if i ≥ 1, else 0

3. y term contribution to [x^i y^j]:
   a^i/i! × a^{j-1}/(j-1)!  if j ≥ 1, else 0

Sum:
    [x^i y^j] = a^{i+j} × [1/(θ·i!·j!) + 1/((i-1)!·j!) + 1/(i!·(j-1)!)]

where terms with (-1)! are zero.
"""

from __future__ import annotations
import math
from typing import Tuple

from src.quadrature import gauss_legendre_01


def oracle_coeff_P1Q1(
    theta: float, R: float, ell1: int, ell2: int, t: float
) -> float:
    """
    Analytic coefficient of x^ℓ₁ y^ℓ₂ in the P=Q=1 bracket.

    Bracket = exp(2Rt) × exp(a(x+y)) × (1/θ + x + y)

    where a = Rθ(2t-1).

    Returns:
        exp(2Rt) × coefficient of x^ℓ₁ y^ℓ₂ in [exp(a(x+y)) × (1/θ + x + y)]
    """
    a = R * theta * (2 * t - 1)
    exp_2Rt = math.exp(2 * R * t)

    # Compute a^{ℓ₁+ℓ₂} once
    a_power = a ** (ell1 + ell2) if (ell1 + ell2) > 0 else 1.0

    # Term 1: (1/θ) × a^ℓ₁/ℓ₁! × a^ℓ₂/ℓ₂!
    term1 = (1.0 / theta) / (math.factorial(ell1) * math.factorial(ell2))

    # Term 2: x contribution → a^{ℓ₁-1}/(ℓ₁-1)! × a^ℓ₂/ℓ₂!  if ℓ₁ ≥ 1
    if ell1 >= 1:
        # This adds a^{-1} relative to a^{ℓ₁+ℓ₂}, but also ℓ₁ in denominator
        term2 = 1.0 / (math.factorial(ell1 - 1) * math.factorial(ell2))
        if a != 0:
            term2 /= a  # Adjust power from a^{ℓ₁+ℓ₂} to a^{ℓ₁+ℓ₂-1}
        else:
            # If a=0, only constant survives; this term is 0 unless ell1+ell2-1=0
            term2 = 0.0 if (ell1 + ell2 - 1) != 0 else 1.0 / (
                math.factorial(ell1 - 1) * math.factorial(ell2)
            )
    else:
        term2 = 0.0

    # Term 3: y contribution → a^ℓ₁/ℓ₁! × a^{ℓ₂-1}/(ℓ₂-1)!  if ℓ₂ ≥ 1
    if ell2 >= 1:
        term3 = 1.0 / (math.factorial(ell1) * math.factorial(ell2 - 1))
        if a != 0:
            term3 /= a
        else:
            term3 = 0.0 if (ell1 + ell2 - 1) != 0 else 1.0 / (
                math.factorial(ell1) * math.factorial(ell2 - 1)
            )
    else:
        term3 = 0.0

    # Combine: multiply by a^{ℓ₁+ℓ₂}
    coeff = a_power * (term1 + term2 + term3)

    return exp_2Rt * coeff


def oracle_I1_P1Q1(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    n_quad_t: int = 100,
    apply_factorial_norm: bool = True,
) -> float:
    """
    Compute I₁ with P=Q=1 using analytic oracle.

    I₁ = (1/(ℓ₁+ℓ₂+1)) × ∫₀¹ oracle_coeff_P1Q1(θ, R, ℓ₁, ℓ₂, t) dt

    The u integral gives:
        ∫₀¹ (1-u)^{ℓ₁+ℓ₂} du = 1/(ℓ₁+ℓ₂+1)

    Args:
        theta, R, ell1, ell2: PRZZ parameters
        n_quad_t: Quadrature points for t integral
        apply_factorial_norm: Whether to multiply by ℓ₁!ℓ₂!

    Returns:
        I₁ value for P=Q=1 microcase
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # u integral factor
    u_integral = 1.0 / (ell1 + ell2 + 1)

    # t integral
    t_integral = 0.0
    for t, t_w in zip(t_nodes, t_weights):
        t_integral += oracle_coeff_P1Q1(theta, R, ell1, ell2, t) * t_w

    total = u_integral * t_integral

    # Apply factorial normalization
    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    # Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return total


def validate_oracle_vs_unified(
    theta: float = 4 / 7, R: float = 1.3036, n_quad: int = 60
) -> dict:
    """
    Validate oracle against unified evaluator for P=Q=1.

    Returns dict with comparison results.
    """
    from src.unified_i1_general import compute_I1_unified_general_P1Q1

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    results = {}

    for ell1, ell2 in pairs:
        oracle_val = oracle_I1_P1Q1(theta, R, ell1, ell2, n_quad_t=n_quad)
        unified_val = compute_I1_unified_general_P1Q1(
            R, theta, ell1, ell2, n_quad_u=n_quad, n_quad_t=n_quad
        )

        if abs(oracle_val) > 1e-12:
            ratio = unified_val / oracle_val
            rel_err = abs(ratio - 1.0)
        else:
            ratio = float("inf") if abs(unified_val) > 1e-12 else 1.0
            rel_err = abs(unified_val - oracle_val)

        results[(ell1, ell2)] = {
            "oracle": oracle_val,
            "unified": unified_val,
            "ratio": ratio,
            "rel_err": rel_err,
            "match": rel_err < 1e-6,
        }

    return results
