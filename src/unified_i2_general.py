"""
src/unified_i2_general.py
Phase 26B: Unified I₂ Evaluator

I₂ is the simplest term:
- No variables (no derivatives)
- No (1-u) factor (PRZZ TeX line 1548)
- Uses Q(t)² not Q(A_α)×Q(A_β)

MATHEMATICAL STRUCTURE:
=====================
I₂_{ℓ₁,ℓ₂} = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × P_{ℓ₁}(u) × P_{ℓ₂}(u) × Q(t)² du dt

This is just direct quadrature with no coefficient extraction needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from src.quadrature import gauss_legendre_01


@dataclass
class UnifiedI2GeneralResult:
    """Result of unified I₂ evaluation."""

    ell1: int
    ell2: int
    I2_value: float

    n_quad_u: int
    n_quad_t: int
    include_Q: bool


def get_polynomial_coeffs(poly) -> List[float]:
    """Extract polynomial coefficients [c0, c1, c2, ...] from a polynomial object."""
    if hasattr(poly, 'tilde_coeffs') and poly.tilde_coeffs is not None:
        import numpy as np
        n_coeffs = len(poly.tilde_coeffs) + 2
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        coeffs = np.polyfit(u_points, y_points, len(poly.tilde_coeffs) + 1)
        return list(coeffs[::-1])

    if hasattr(poly, 'standard_coeffs'):
        return list(poly.standard_coeffs)

    if hasattr(poly, 'eval'):
        import numpy as np
        n_coeffs = 10
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        for deg in range(n_coeffs, 0, -1):
            coeffs = np.polyfit(u_points, y_points, deg)
            fitted = np.polyval(coeffs, u_points)
            if np.max(np.abs(fitted - y_points)) < 1e-10:
                return list(coeffs[::-1])
        coeffs = np.polyfit(u_points, y_points, 5)
        return list(coeffs[::-1])

    raise ValueError(f"Cannot extract coefficients from polynomial: {type(poly)}")


def eval_polynomial(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial at x using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def compute_I2_unified_general(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    include_Q: bool = True,
) -> UnifiedI2GeneralResult:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂).

    I₂ has no derivatives and no (1-u) factor.

    I₂ = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × P_{ℓ₁}(u) × P_{ℓ₂}(u) × Q(t)² du dt

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (typically 4/7)
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        n_quad_u: Number of quadrature points for u integral
        n_quad_t: Number of quadrature points for t integral
        include_Q: Whether to include Q factor (default True)

    Returns:
        UnifiedI2GeneralResult with I₂ value
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    P_ell1_coeffs = get_polynomial_coeffs(P_ell1)
    P_ell2_coeffs = get_polynomial_coeffs(P_ell2)
    Q_coeffs = get_polynomial_coeffs(Q) if Q is not None and include_Q else None

    total = 0.0

    for u, u_w in zip(u_nodes, u_weights):
        # Evaluate P_ℓ₁(u) × P_ℓ₂(u)
        P1_val = eval_polynomial(P_ell1_coeffs, u)
        P2_val = eval_polynomial(P_ell2_coeffs, u)
        P_product = P1_val * P2_val

        for t, t_w in zip(t_nodes, t_weights):
            # exp(2Rt)
            exp_val = math.exp(2 * R * t)

            # Q(t)² if enabled
            if include_Q and Q_coeffs is not None:
                Q_val = eval_polynomial(Q_coeffs, t)
                Q_sq = Q_val ** 2
            else:
                Q_sq = 1.0

            # Accumulate
            total += exp_val * P_product * Q_sq * u_w * t_w

    # Multiply by 1/θ prefactor
    total *= 1.0 / theta

    return UnifiedI2GeneralResult(
        ell1=ell1,
        ell2=ell2,
        I2_value=total,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
    )


def compute_I2_unified_general_P1Q1(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
) -> float:
    """
    Compute I₂ with P=Q=1 (constant polynomials).

    I₂ = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × 1 × 1 du dt
       = (1/θ) × 1 × (exp(2R)-1)/(2R)

    Args:
        R, theta, ell1, ell2: PRZZ parameters (ell1/ell2 unused but kept for API consistency)
        n_quad_u, n_quad_t: Quadrature points

    Returns:
        I₂ value for P=Q=1
    """
    # Analytic formula:
    # ∫₀¹ exp(2Rt) dt = (exp(2R)-1)/(2R)
    t_integral = (math.exp(2 * R) - 1) / (2 * R)

    # ∫₀¹ du = 1
    u_integral = 1.0

    # 1/θ prefactor
    return (1.0 / theta) * u_integral * t_integral
