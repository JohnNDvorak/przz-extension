"""
src/evaluator/g_q_functional.py
Phase 46B: First-Principles Q-Derivative Functional for g Correction

This module derives epsilon_I1 and epsilon_I2 from first principles using
the Q-derivative analytic kernel, with NO c_target or fitted parameters.

CORE INSIGHT:
=============

The ~0.4% gap between first-principles g and calibrated g comes from Q-derivative
terms that appear differently in I1 vs I2:

- I1: Q evaluated at affine arguments A_alpha(x,y,t), A_beta(x,y,t)
      When extracting [xy] coefficient, Q' and Q'' contribute

- I2: Q evaluated as frozen Q(t)² (no x,y dependence in Q argument)
      NO Q-derivative contribution at [xy] extraction level

The epsilon corrections are:
    epsilon_I1^Q = contribution from Q-derivative terms in I1 (should be small)
    epsilon_I2^Q = contribution from Q-derivative terms in I2 (should be ~0)

DERIVATION APPROACH:
====================

For I1, the full [xy] coefficient of Q(A_alpha)Q(A_beta) is:
    [xy] = theta² × [2t(t-1)×Q×Q'' + (t²+(t-1)²)×(Q')²]

This modifies the effective "g" that I1 sees. The epsilon is:
    epsilon_I1 = (integral of Q-derivative term) / (integral of frozen term)

For I2, since Q is frozen at Q(t)², there's NO Q-derivative contribution.
    epsilon_I2 = 0 (by construction)

But wait - the calibrated values show epsilon_I2 ≈ 0.0058, not 0!
This means the gap for I2 comes from something ELSE - possibly the
interaction between the log factor and the Beta moment integral.

Created: 2025-12-27 (Phase 46B)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from src.polynomials import Polynomial, load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01
from src.unified_s12.q_affine_expansion import (
    q_product_xy_coeff_post_identity_vectorized,
    compute_q_moments_under_weight,
    QMomentsResult,
)


@dataclass
class QFunctionalResult:
    """Result of Q-functional epsilon derivation."""

    # Derived epsilon values from Q-derivative terms
    epsilon_I1_Q: float      # Q-derivative contribution to I1
    epsilon_I2_Q: float      # Q-derivative contribution to I2 (should be 0)

    # Baseline g values (without Q-derivative correction)
    g_I1_baseline: float     # 1.0 (from log factor self-correction)
    g_I2_baseline: float     # 1 + theta/(2K(2K+1))

    # Corrected g values
    g_I1_corrected: float    # g_I1_baseline + epsilon_I1_Q
    g_I2_corrected: float    # g_I2_baseline + epsilon_I2_Q

    # Diagnostic integrals
    frozen_Q2_minus: float   # integral of Q(t)² × exp(-2Rt)
    xy_deriv_minus: float    # integral of [xy] Q(A)Q(A) × exp(-2Rt)

    # Parameters
    theta: float
    K: int
    R: float


def compute_q_derivative_epsilon(
    Q: Polynomial,
    theta: float,
    R: float,
    K: int = 3,
    n_quad: int = 80,
) -> QFunctionalResult:
    """
    Compute the Q-derivative epsilon corrections from first principles.

    The key formula:
        epsilon_I1 = (Q-derivative moment) / (frozen Q² moment) × scaling_factor

    where the scaling factor accounts for how the [xy] extraction maps
    into the effective g correction.

    Args:
        Q: Q polynomial
        theta: PRZZ theta parameter
        R: R parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points

    Returns:
        QFunctionalResult with derived epsilon values
    """
    # Get quadrature points for t integration
    t_pts, t_wts = gauss_legendre_01(n_quad)

    # Compute weight: exp(-2Rt) for mirror term at -R
    weights_minus = np.exp(-2 * R * t_pts)

    # ========================================
    # FROZEN Q² INTEGRAL (baseline)
    # ========================================
    # This is what we'd get if Q didn't depend on (x,y)
    Q_vals = Q.eval(t_pts)
    frozen_Q2_minus = np.sum(t_wts * weights_minus * Q_vals**2)

    # ========================================
    # Q-DERIVATIVE [xy] INTEGRAL
    # ========================================
    # The [xy] coefficient of Q(A_alpha)Q(A_beta)
    xy_coeffs = q_product_xy_coeff_post_identity_vectorized(Q, t_pts, theta)
    xy_deriv_minus = np.sum(t_wts * weights_minus * xy_coeffs)

    # ========================================
    # EPSILON DERIVATION FOR I1
    # ========================================
    # The Q-derivative term modifies the [xy] extraction
    # The ratio tells us the relative size of this effect
    #
    # For the I1 integral structure:
    #   I1 ∝ integral of [d²/dxdy bracket] × Q(A_alpha)Q(A_beta) × weight
    #
    # The [xy] from Q(A)Q(A) adds to the [xy] from the log factor,
    # so it modifies the effective g correction.
    #
    # The epsilon is proportional to:
    #   (Q-derivative [xy]) / (frozen Q²)

    if abs(frozen_Q2_minus) > 1e-15:
        raw_ratio = xy_deriv_minus / frozen_Q2_minus
    else:
        raw_ratio = 0.0

    # The [xy] from Q derivatives adds to the I1 bracket's [xy] coefficient
    # The log factor already provides g_baseline - 1 correction
    # The Q-derivative term provides an additional correction
    #
    # Key insight: The [xy] coefficient formula has theta² prefactor,
    # and the baseline correction is theta/(2K(2K+1))
    #
    # So the epsilon scales as:
    #   epsilon_I1 ≈ raw_ratio × (normalization factor)
    #
    # The normalization maps the [xy] moment ratio into a g-correction

    # The raw_ratio is already the relative size of Q-derivative contribution
    # But we need to account for how it enters the full integral
    epsilon_I1_Q = raw_ratio  # First approximation

    # For I2, Q is frozen Q(t)², so no Q-derivative at [xy] level
    epsilon_I2_Q = 0.0

    # ========================================
    # BASELINE g VALUES
    # ========================================
    g_I1_baseline = 1.0  # Log factor self-correction
    g_I2_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # ========================================
    # CORRECTED g VALUES
    # ========================================
    # The epsilon adds to the baseline
    g_I1_corrected = g_I1_baseline + epsilon_I1_Q
    g_I2_corrected = g_I2_baseline + epsilon_I2_Q

    return QFunctionalResult(
        epsilon_I1_Q=epsilon_I1_Q,
        epsilon_I2_Q=epsilon_I2_Q,
        g_I1_baseline=g_I1_baseline,
        g_I2_baseline=g_I2_baseline,
        g_I1_corrected=g_I1_corrected,
        g_I2_corrected=g_I2_corrected,
        frozen_Q2_minus=frozen_Q2_minus,
        xy_deriv_minus=xy_deriv_minus,
        theta=theta,
        K=K,
        R=R,
    )


def compute_full_q_derivative_analysis(
    Q: Polynomial,
    theta: float,
    R: float,
    K: int = 3,
    n_quad: int = 80,
) -> Dict[str, float]:
    """
    Comprehensive Q-derivative analysis at both +R and -R.

    Returns detailed diagnostic information for understanding
    the Q-derivative contribution.
    """
    result = {}

    # Get moments at +R and -R
    moments_plus = compute_q_moments_under_weight(Q, theta, R, n_quad, "exp_2R")
    moments_minus = compute_q_moments_under_weight(Q, theta, R, n_quad, "exp_minus_2R")

    result["frozen_Q2_plus"] = moments_plus.Q_squared_moment
    result["frozen_Q2_minus"] = moments_minus.Q_squared_moment

    result["xy_deriv_plus"] = moments_plus.xy_coeff_moment
    result["xy_deriv_minus"] = moments_minus.xy_coeff_moment

    # Ratios
    result["ratio_plus"] = (
        moments_plus.xy_coeff_moment / moments_plus.Q_squared_moment
        if abs(moments_plus.Q_squared_moment) > 1e-15 else 0.0
    )
    result["ratio_minus"] = (
        moments_minus.xy_coeff_moment / moments_minus.Q_squared_moment
        if abs(moments_minus.Q_squared_moment) > 1e-15 else 0.0
    )

    # Component moments
    result["Q_Qpp_plus"] = moments_plus.Q_Q_double_prime_moment
    result["Q_Qpp_minus"] = moments_minus.Q_Q_double_prime_moment
    result["Qp2_plus"] = moments_plus.Q_prime_squared_moment
    result["Qp2_minus"] = moments_minus.Q_prime_squared_moment

    # Geometric-weighted moments (these enter the formula)
    result["t_tm1_Q_Qpp_plus"] = moments_plus.t_times_t_minus_1_Q_Qpp
    result["t_tm1_Q_Qpp_minus"] = moments_minus.t_times_t_minus_1_Q_Qpp
    result["t2tm12_Qp2_plus"] = moments_plus.t2_plus_tm1_2_Qp2
    result["t2tm12_Qp2_minus"] = moments_minus.t2_plus_tm1_2_Qp2

    return result


def analyze_przz_benchmarks() -> Dict[str, Dict]:
    """
    Analyze Q-derivative contributions for both PRZZ benchmarks.

    Returns analysis for kappa (R=1.3036) and kappa* (R=1.1167).
    """
    theta = 4/7
    K = 3

    # Load polynomials for each benchmark
    _, _, _, Q_kappa = load_przz_polynomials(enforce_Q0=False)
    _, _, _, Q_kappa_star = load_przz_polynomials_kappa_star(enforce_Q0=False)

    results = {}

    # Kappa benchmark
    R_kappa = 1.3036
    results["kappa"] = {
        "R": R_kappa,
        "epsilon_result": compute_q_derivative_epsilon(
            Q_kappa.to_monomial(), theta, R_kappa, K
        ),
        "full_analysis": compute_full_q_derivative_analysis(
            Q_kappa.to_monomial(), theta, R_kappa, K
        ),
    }

    # Kappa* benchmark
    R_kappa_star = 1.1167
    results["kappa_star"] = {
        "R": R_kappa_star,
        "epsilon_result": compute_q_derivative_epsilon(
            Q_kappa_star.to_monomial(), theta, R_kappa_star, K
        ),
        "full_analysis": compute_full_q_derivative_analysis(
            Q_kappa_star.to_monomial(), theta, R_kappa_star, K
        ),
    }

    return results


def compare_to_calibrated(
    epsilon_result: QFunctionalResult,
    g_I1_calibrated: float = 1.00091428,
    g_I2_calibrated: float = 1.01945154,
) -> Dict[str, float]:
    """
    Compare derived epsilon values to what calibration gives.

    The calibrated g values imply:
        epsilon_I1_calibrated = g_I1_calibrated - g_I1_baseline
        epsilon_I2_calibrated = g_I2_calibrated - g_I2_baseline
    """
    epsilon_I1_calibrated = g_I1_calibrated - epsilon_result.g_I1_baseline
    epsilon_I2_calibrated = g_I2_calibrated - epsilon_result.g_I2_baseline

    return {
        "epsilon_I1_derived": epsilon_result.epsilon_I1_Q,
        "epsilon_I1_calibrated": epsilon_I1_calibrated,
        "epsilon_I1_gap": epsilon_result.epsilon_I1_Q - epsilon_I1_calibrated,
        "epsilon_I1_gap_pct": (epsilon_result.epsilon_I1_Q / epsilon_I1_calibrated - 1) * 100 if abs(epsilon_I1_calibrated) > 1e-10 else float('nan'),

        "epsilon_I2_derived": epsilon_result.epsilon_I2_Q,
        "epsilon_I2_calibrated": epsilon_I2_calibrated,
        "epsilon_I2_gap": epsilon_result.epsilon_I2_Q - epsilon_I2_calibrated,
        "epsilon_I2_gap_pct": (epsilon_result.epsilon_I2_Q / epsilon_I2_calibrated - 1) * 100 if abs(epsilon_I2_calibrated) > 1e-10 else float('nan'),

        "g_I1_derived": epsilon_result.g_I1_corrected,
        "g_I1_calibrated": g_I1_calibrated,
        "g_I2_derived": epsilon_result.g_I2_corrected,
        "g_I2_calibrated": g_I2_calibrated,
    }
