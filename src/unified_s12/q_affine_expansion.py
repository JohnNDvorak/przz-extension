"""
src/unified_s12/q_affine_expansion.py
Phase 46A: Q-Derivative Analytic Kernel Module

This module implements the nilpotent Taylor expansion for Q(A) where A is an
affine function of nilpotent variables x, y (with x^2 = y^2 = 0).

MATHEMATICAL FOUNDATION:
========================

For x^2 = y^2 = 0 and A(t; x, y) = t + a*x + b*y:

    Q(A) = Q(t) + Q'(t)*(a*x + b*y) + Q''(t)*a*b*xy

Because (a*x + b*y)^2 = 2*a*b*xy when x^2 = y^2 = 0.

POST-IDENTITY EIGENVALUES (from PRZZ):
======================================

    A_alpha = t + theta*(t-1)*x + theta*t*y
    A_beta  = t + theta*t*x + theta*(t-1)*y

So:
    a_alpha = theta*(t-1),  b_alpha = theta*t
    a_beta  = theta*t,      b_beta  = theta*(t-1)

The [xy] coefficient of Q(A_alpha)*Q(A_beta) is:

    [xy] Q(A_alpha)Q(A_beta) = Q(t)*Q''(t)*(a_alpha*b_alpha + a_beta*b_beta)
                             + (Q'(t))^2*(a_alpha*b_beta + a_beta*b_alpha)

Computing the geometric terms:
    a_alpha*b_alpha + a_beta*b_beta = 2*theta^2*t*(t-1)
    a_alpha*b_beta + a_beta*b_alpha = theta^2*((t-1)^2 + t^2)

Final formula:
    [xy] Q(A_alpha)Q(A_beta) = theta^2 * [
        2*t*(t-1)*Q(t)*Q''(t) + (t^2 + (t-1)^2)*(Q'(t))^2
    ]

WHY THIS MATTERS:
=================

- I2 uses frozen Q(t)^2 - NO Q', Q'' contributions at xy extraction level
- I1 has affine dependence on (x,y), so it DOES have Q', Q'' contributions
- This asymmetry creates the small, stubborn residuals epsilon_I1, epsilon_I2

Created: 2025-12-27 (Phase 46A)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Dict
import numpy as np

from src.polynomials import Polynomial


@dataclass
class NilpotentSeriesCoeffs:
    """Coefficients of Q(t + a*x + b*y) in nilpotent series.

    Q(A) = c0 + cx*x + cy*y + cxy*xy

    where x^2 = y^2 = 0.
    """
    c0: float    # Constant term: Q(t)
    cx: float    # Coefficient of x: Q'(t)*a
    cy: float    # Coefficient of y: Q'(t)*b
    cxy: float   # Coefficient of xy: Q''(t)*a*b


def q_affine_series_at_xy(
    Q: Polynomial,
    t: float,
    a: float,
    b: float,
) -> NilpotentSeriesCoeffs:
    """
    Compute nilpotent series coefficients of Q(t + a*x + b*y).

    For nilpotent x, y (x^2 = y^2 = 0):
        Q(t + a*x + b*y) = Q(t) + Q'(t)*a*x + Q'(t)*b*y + Q''(t)*a*b*xy

    Args:
        Q: Polynomial (with eval, eval_deriv methods)
        t: Base point
        a: Coefficient of x in affine argument
        b: Coefficient of y in affine argument

    Returns:
        NilpotentSeriesCoeffs with c0, cx, cy, cxy
    """
    t_arr = np.array([t])

    # Q(t)
    Q_val = Q.eval(t_arr)[0]

    # Q'(t) - first derivative
    Q_prime_val = Q.eval_deriv(t_arr, 1)[0]

    # Q''(t) - second derivative
    Q_double_prime_val = Q.eval_deriv(t_arr, 2)[0]

    return NilpotentSeriesCoeffs(
        c0=Q_val,
        cx=Q_prime_val * a,
        cy=Q_prime_val * b,
        cxy=Q_double_prime_val * a * b,
    )


def q_product_xy_coeff(
    Q1_coeffs: NilpotentSeriesCoeffs,
    Q2_coeffs: NilpotentSeriesCoeffs,
) -> float:
    """
    Compute [xy] coefficient of (Q1 series) * (Q2 series).

    If Q1 = c0_1 + cx_1*x + cy_1*y + cxy_1*xy
    and Q2 = c0_2 + cx_2*x + cy_2*y + cxy_2*xy

    Then [xy](Q1*Q2) = c0_1*cxy_2 + cxy_1*c0_2 + cx_1*cy_2 + cy_1*cx_2

    (Using x^2 = y^2 = 0, xy = yx)
    """
    return (
        Q1_coeffs.c0 * Q2_coeffs.cxy +
        Q1_coeffs.cxy * Q2_coeffs.c0 +
        Q1_coeffs.cx * Q2_coeffs.cy +
        Q1_coeffs.cy * Q2_coeffs.cx
    )


def q_product_xy_coeff_post_identity(
    Q: Polynomial,
    t: float,
    theta: float,
) -> float:
    """
    Compute [xy] Q(A_alpha)*Q(A_beta) for post-identity eigenvalues.

    Post-identity eigenvalues (from PRZZ):
        A_alpha = t + theta*(t-1)*x + theta*t*y
        A_beta  = t + theta*t*x + theta*(t-1)*y

    The closed-form formula is:
        [xy] = theta^2 * [2*t*(t-1)*Q(t)*Q''(t) + (t^2 + (t-1)^2)*(Q'(t))^2]

    Args:
        Q: Polynomial
        t: Integration variable (in [0,1])
        theta: PRZZ theta parameter (typically 4/7)

    Returns:
        The [xy] coefficient value at this t
    """
    t_arr = np.array([t])

    # Evaluate Q and its derivatives
    Q_val = Q.eval(t_arr)[0]
    Q_prime_val = Q.eval_deriv(t_arr, 1)[0]
    Q_double_prime_val = Q.eval_deriv(t_arr, 2)[0]

    # Geometric coefficients
    geom_QQpp = 2 * t * (t - 1)           # Coefficient of Q*Q''
    geom_Qp2 = t**2 + (t - 1)**2          # Coefficient of (Q')^2

    # The formula
    result = theta**2 * (
        geom_QQpp * Q_val * Q_double_prime_val +
        geom_Qp2 * Q_prime_val**2
    )

    return result


def q_product_xy_coeff_post_identity_vectorized(
    Q: Polynomial,
    t_arr: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Vectorized version of q_product_xy_coeff_post_identity.

    Args:
        Q: Polynomial
        t_arr: Array of t values
        theta: PRZZ theta parameter

    Returns:
        Array of [xy] coefficient values
    """
    # Evaluate Q and its derivatives at all t points
    Q_vals = Q.eval(t_arr)
    Q_prime_vals = Q.eval_deriv(t_arr, 1)
    Q_double_prime_vals = Q.eval_deriv(t_arr, 2)

    # Geometric coefficients (vectorized)
    geom_QQpp = 2 * t_arr * (t_arr - 1)
    geom_Qp2 = t_arr**2 + (t_arr - 1)**2

    # The formula (vectorized)
    result = theta**2 * (
        geom_QQpp * Q_vals * Q_double_prime_vals +
        geom_Qp2 * Q_prime_vals**2
    )

    return result


@dataclass
class QMomentsResult:
    """Result of Q-moment computation under integration weight."""

    # Raw moments
    Q_squared_moment: float          # integral of Q(t)^2 * weight
    Q_Q_double_prime_moment: float   # integral of Q(t)*Q''(t) * weight
    Q_prime_squared_moment: float    # integral of (Q'(t))^2 * weight

    # Weighted geometric moments
    t_times_t_minus_1_Q_Qpp: float   # integral of t(t-1)*Q*Q'' * weight
    t2_plus_tm1_2_Qp2: float         # integral of (t^2+(t-1)^2)*(Q')^2 * weight

    # The full [xy] moment
    xy_coeff_moment: float           # integral of [xy] Q(A_alpha)Q(A_beta) * weight

    # Parameters
    theta: float
    R: float
    weight_type: str  # "exp_2R" or "exp_minus_2R" or "uniform"


def compute_q_moments_under_weight(
    Q: Polynomial,
    theta: float,
    R: float,
    n_quad: int = 60,
    weight_type: str = "exp_2R",
) -> QMomentsResult:
    """
    Compute Q-derivative moments under integration weight.

    This computes the key integrals needed to quantify the Q-derivative
    contribution to the epsilon correction.

    Args:
        Q: Polynomial
        theta: PRZZ theta parameter
        R: R parameter (for exp(+/-2R*t) weight)
        n_quad: Quadrature points
        weight_type: "exp_2R", "exp_minus_2R", or "uniform"

    Returns:
        QMomentsResult with all computed moments
    """
    from src.quadrature import gauss_legendre_01

    t_pts, t_wts = gauss_legendre_01(n_quad)

    # Compute weight function
    if weight_type == "exp_2R":
        weights = np.exp(2 * R * t_pts)
    elif weight_type == "exp_minus_2R":
        weights = np.exp(-2 * R * t_pts)
    elif weight_type == "uniform":
        weights = np.ones_like(t_pts)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Evaluate Q and derivatives
    Q_vals = Q.eval(t_pts)
    Q_prime_vals = Q.eval_deriv(t_pts, 1)
    Q_double_prime_vals = Q.eval_deriv(t_pts, 2)

    # Geometric terms
    t_tm1 = t_pts * (t_pts - 1)
    t2_tm12 = t_pts**2 + (t_pts - 1)**2

    # Compute moments
    effective_wts = t_wts * weights

    Q_squared_moment = np.sum(effective_wts * Q_vals**2)
    Q_Q_double_prime_moment = np.sum(effective_wts * Q_vals * Q_double_prime_vals)
    Q_prime_squared_moment = np.sum(effective_wts * Q_prime_vals**2)

    t_times_t_minus_1_Q_Qpp = np.sum(effective_wts * t_tm1 * Q_vals * Q_double_prime_vals)
    t2_plus_tm1_2_Qp2 = np.sum(effective_wts * t2_tm12 * Q_prime_vals**2)

    # Full [xy] moment
    xy_coeff_vals = q_product_xy_coeff_post_identity_vectorized(Q, t_pts, theta)
    xy_coeff_moment = np.sum(effective_wts * xy_coeff_vals)

    return QMomentsResult(
        Q_squared_moment=Q_squared_moment,
        Q_Q_double_prime_moment=Q_Q_double_prime_moment,
        Q_prime_squared_moment=Q_prime_squared_moment,
        t_times_t_minus_1_Q_Qpp=t_times_t_minus_1_Q_Qpp,
        t2_plus_tm1_2_Qp2=t2_plus_tm1_2_Qp2,
        xy_coeff_moment=xy_coeff_moment,
        theta=theta,
        R=R,
        weight_type=weight_type,
    )


def compute_frozen_vs_full_xy_ratio(
    Q: Polynomial,
    theta: float,
    R: float,
    n_quad: int = 60,
) -> Dict[str, float]:
    """
    Compare frozen Q(t)^2 to full Q(A_alpha)Q(A_beta) xy coefficient.

    This quantifies how much the Q-derivative terms contribute.

    For frozen Q: [xy] contribution = 0 (Q only depends on t, not x,y)
    For full Q(A): [xy] contribution = theta^2 * [...]

    Returns:
        Dict with:
        - frozen_Q2_integral: integral of Q(t)^2 * exp(2Rt)
        - xy_derivative_integral: integral of [xy] Q(A_alpha)Q(A_beta) * exp(2Rt)
        - ratio: xy_derivative / frozen_Q2 (the relative Q-derivative effect)
    """
    moments_plus = compute_q_moments_under_weight(Q, theta, R, n_quad, "exp_2R")
    moments_minus = compute_q_moments_under_weight(Q, theta, R, n_quad, "exp_minus_2R")

    return {
        "frozen_Q2_plus": moments_plus.Q_squared_moment,
        "frozen_Q2_minus": moments_minus.Q_squared_moment,
        "xy_deriv_plus": moments_plus.xy_coeff_moment,
        "xy_deriv_minus": moments_minus.xy_coeff_moment,
        "ratio_plus": moments_plus.xy_coeff_moment / moments_plus.Q_squared_moment if abs(moments_plus.Q_squared_moment) > 1e-15 else float('nan'),
        "ratio_minus": moments_minus.xy_coeff_moment / moments_minus.Q_squared_moment if abs(moments_minus.Q_squared_moment) > 1e-15 else float('nan'),
    }


def verify_formula_vs_series(
    Q: Polynomial,
    t: float,
    theta: float,
    tol: float = 1e-12,
) -> Dict[str, float]:
    """
    Verify the closed-form [xy] formula against brute-force series computation.

    This is a validation function to ensure the formula is correct.

    Args:
        Q: Polynomial
        t: Base point
        theta: PRZZ theta parameter
        tol: Tolerance for comparison

    Returns:
        Dict with formula value, series value, and absolute difference
    """
    # Post-identity eigenvalue coefficients
    a_alpha = theta * (t - 1)
    b_alpha = theta * t
    a_beta = theta * t
    b_beta = theta * (t - 1)

    # Method 1: Closed-form formula
    formula_value = q_product_xy_coeff_post_identity(Q, t, theta)

    # Method 2: Brute-force series multiplication
    Q_alpha_coeffs = q_affine_series_at_xy(Q, t, a_alpha, b_alpha)
    Q_beta_coeffs = q_affine_series_at_xy(Q, t, a_beta, b_beta)
    series_value = q_product_xy_coeff(Q_alpha_coeffs, Q_beta_coeffs)

    diff = abs(formula_value - series_value)

    return {
        "formula_value": formula_value,
        "series_value": series_value,
        "absolute_diff": diff,
        "match": diff < tol,
    }
