"""
src/unified_i1_general.py
Phase 26B: Unified I₁ Evaluator with x^ℓ₁ y^ℓ₂ Coefficient Extraction

This module computes I₁ for any pair (ℓ₁, ℓ₂) using the unified bracket
structure with proper coefficient extraction.

KEY DIFFERENCE FROM unified_s12_evaluator_v3:
- v3 extracts xy coefficient (d²/dxdy) for ALL pairs
- This module extracts x^ℓ₁ y^ℓ₂ coefficient (matching OLD DSL derivative order)

MATHEMATICAL STRUCTURE:
=====================
Per quadrature point (u, t):
1. Build E = exp(2Rt) × exp(Rθ(2t-1)(x+y)) as bivariate series
2. Build L = (1/θ + x + y) as bivariate series
3. Build Pfac = P_ℓ₁(u+x) × P_ℓ₂(u+y) expanded to (dx=ℓ₁, dy=ℓ₂)
4. Build Qfac = Q(A_α) × Q(A_β) expanded to same degrees
5. Multiply: S = E × L × Pfac × Qfac
6. Extract coeff(ℓ₁, ℓ₂)
7. Multiply by (1-u)^{ℓ₁+ℓ₂}
8. Integrate over (u, t)

FACTORIAL NORMALIZATION:
=======================
OLD DSL extracts d^{ℓ₁+ℓ₂}/dx₁...dy_{ℓ₂} which equals:
    ℓ₁! × ℓ₂! × [x^ℓ₁ y^ℓ₂]

So to match OLD DSL, we multiply the extracted coefficient by ℓ₁! × ℓ₂!.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from src.quadrature import gauss_legendre_01
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_log_factor,
    build_P_factor,
    build_Q_factor,
)


@dataclass
class UnifiedI1GeneralResult:
    """Result of unified I₁ evaluation for a pair using x^ℓ₁y^ℓ₂ extraction."""

    ell1: int
    ell2: int
    I1_value: float

    # Diagnostics
    n_quad_u: int
    n_quad_t: int
    include_Q: bool
    factorial_norm_applied: bool


def get_polynomial_coeffs(poly) -> List[float]:
    """
    Extract polynomial coefficients [c0, c1, c2, ...] from a polynomial object.

    Works with both Chebyshev-based PRZZ polynomials and simple numpy polys.
    """
    # Check if it has tilde_coeffs (PRZZ polynomials)
    if hasattr(poly, 'tilde_coeffs') and poly.tilde_coeffs is not None:
        # Tilde coefficients are in (1-2u) basis, need to convert to u basis
        # For now, evaluate at sample points and fit
        import numpy as np
        n_coeffs = len(poly.tilde_coeffs) + 2  # Extra for safety
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        # Fit polynomial
        coeffs = np.polyfit(u_points, y_points, len(poly.tilde_coeffs) + 1)
        # Reverse to get [c0, c1, c2, ...]
        return list(coeffs[::-1])

    # Check if it has standard_coeffs
    if hasattr(poly, 'standard_coeffs'):
        return list(poly.standard_coeffs)

    # Check if it's a simple polynomial with eval
    if hasattr(poly, 'eval'):
        import numpy as np
        # Evaluate at sample points and fit
        n_coeffs = 10  # Assume max degree 10
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        # Fit polynomial with reasonable degree
        for deg in range(n_coeffs, 0, -1):
            coeffs = np.polyfit(u_points, y_points, deg)
            # Check if fit is good
            fitted = np.polyval(coeffs, u_points)
            if np.max(np.abs(fitted - y_points)) < 1e-10:
                return list(coeffs[::-1])
        # Fallback
        coeffs = np.polyfit(u_points, y_points, 5)
        return list(coeffs[::-1])

    raise ValueError(f"Cannot extract coefficients from polynomial: {type(poly)}")


def compute_I1_unified_general(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    include_Q: bool = True,
    apply_factorial_norm: bool = True,
) -> UnifiedI1GeneralResult:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) using x^ℓ₁ y^ℓ₂ coefficient extraction.

    This matches OLD DSL's derivative order by extracting the correct
    higher-order coefficient from the unified bracket.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (typically 4/7)
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        n_quad_u: Number of quadrature points for u integral
        n_quad_t: Number of quadrature points for t integral
        include_Q: Whether to include Q factors (default True)
        apply_factorial_norm: Whether to multiply by ℓ₁!ℓ₂! (default True)

    Returns:
        UnifiedI1GeneralResult with I₁ value
    """
    max_dx = ell1
    max_dy = ell2

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # Get polynomial coefficient lists
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    P_ell1_coeffs = get_polynomial_coeffs(P_ell1)
    P_ell2_coeffs = get_polynomial_coeffs(P_ell2)
    Q_coeffs = get_polynomial_coeffs(Q) if Q is not None and include_Q else None

    # PRZZ (1-u) power from Euler-Maclaurin: (1-u)^{ℓ₁+ℓ₂} for I₁
    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        # (1-u)^{ℓ₁+ℓ₂} prefactor
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # 1. Exp factor: exp(2Rt + Rθ(2t-1)(x+y))
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # 2. Log factor: 1/θ + x + y
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # 3. P factors: P_ℓ₁(u+x) × P_ℓ₂(u+y)
            P_x = build_P_factor(P_ell1_coeffs, u, "x", max_dx, max_dy)
            P_y = build_P_factor(P_ell2_coeffs, u, "y", max_dx, max_dy)

            # Build bracket
            bracket = exp_factor * log_factor * P_x * P_y

            # 4. Q factors (if enabled)
            if include_Q and Q_coeffs is not None:
                # Q eigenvalue arguments:
                # A_α = t + θ(t-1)x + θt·y
                # A_β = t + θt·x + θ(t-1)·y
                Q_alpha = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * (t - 1),
                    ay=theta * t,
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                Q_beta = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * t,
                    ay=theta * (t - 1),
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                bracket = bracket * Q_alpha * Q_beta

            # 5. Extract x^ℓ₁ y^ℓ₂ coefficient
            coeff = bracket.extract(ell1, ell2)

            # 6. Add to integral
            total += coeff * one_minus_u_factor * u_w * t_w

    # 7. Apply factorial normalization if requested
    # OLD DSL effectively computes ℓ₁!ℓ₂! × [x^ℓ₁ y^ℓ₂]
    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    # 8. Apply sign convention to match OLD DSL
    # OFF-diagonal pairs where ℓ₁ ≠ ℓ₂ have sign (-1)^{ℓ₁+ℓ₂}
    # This comes from the asymmetric derivative ordering in the residue calculus
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return UnifiedI1GeneralResult(
        ell1=ell1,
        ell2=ell2,
        I1_value=total,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
        factorial_norm_applied=apply_factorial_norm,
    )


def compute_I1_unified_general_P1Q1(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    apply_factorial_norm: bool = True,
) -> float:
    """
    Compute I₁ with P=Q=1 (constant polynomials) for microcase validation.

    With P=Q=1:
    - P_ℓ₁(u+x) = 1 for all u,x
    - P_ℓ₂(u+y) = 1 for all u,y
    - Q(A) = 1 for all A

    This simplifies the bracket to:
        exp(2Rt + Rθ(2t-1)(x+y)) × (1/θ + x + y)

    Args:
        R, theta, ell1, ell2: PRZZ parameters
        n_quad_u, n_quad_t: Quadrature points
        apply_factorial_norm: Whether to multiply by ℓ₁!ℓ₂!

    Returns:
        I₁ value for P=Q=1 microcase
    """
    max_dx = ell1
    max_dy = ell2

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # Exp factor
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # Log factor
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # Bracket with P=Q=1
            bracket = exp_factor * log_factor

            # Extract coefficient
            coeff = bracket.extract(ell1, ell2)

            total += coeff * one_minus_u_factor * u_w * t_w

    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    # Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return total
