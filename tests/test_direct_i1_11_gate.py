"""
tests/test_direct_i1_11_gate.py
Gate tests: Direct I1(1,1) must match DSL evaluation.

Run 8B1: This test verifies that the direct series-based I1(1,1) evaluation
matches the DSL-based evaluation within 2% tolerance.

For pair (1,1), P₁ is Case B (ω=0), so no Case C kernel derivatives are needed.
This validates the derivative extraction machinery for the simplest case.

Usage:
    pytest tests/test_direct_i1_11_gate.py -v
    pytest tests/test_direct_i1_11_gate.py -v -m calibration
"""

import pytest
import numpy as np
from typing import Dict

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import compute_operator_implied_weights
from src.quadrature import gauss_legendre_01


THETA = 4.0 / 7.0
TOLERANCE = 0.02  # 2% tolerance


def compute_i1_11_direct_series(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 60
) -> float:
    """
    Compute I1 for pair (1,1) using the series engine directly.

    This mirrors what the DSL does but constructs the series manually
    to validate the approach.

    Returns the I1(1,1) value.
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    total = 0.0

    for i, u in enumerate(u_nodes):
        for j, t in enumerate(t_nodes):
            weight = u_weights[i] * t_weights[j]

            # Initialize coefficients dict
            # key = bitset (0=const, 1=x, 2=y, 3=xy)

            # 1. Algebraic prefactor: (1/θ + x + y)
            alg_const = 1.0 / theta
            alg_x = 1.0
            alg_y = 1.0

            # 2. Poly prefactor: (1-u)² - pure scalar
            poly_pf = (1 - u) ** 2

            # 3. P₁(x+u) = P₁(u) + P₁'(u)·x + O(x²)
            P1_u = P1.eval(np.array([u]))[0]
            P1_deriv_u = P1.eval_deriv(np.array([u]), 1)[0]

            # P₁(x+u)·P₁(y+u) = [P₁(u) + P₁'(u)·x][P₁(u) + P₁'(u)·y]
            P_const = P1_u * P1_u
            P_x = P1_u * P1_deriv_u
            P_y = P1_u * P1_deriv_u
            P_xy = P1_deriv_u * P1_deriv_u

            # 4. Q factors: Q(Arg_α)·Q(Arg_β)
            Q_t = Q.eval(np.array([t]))[0]
            Q_deriv_t = Q.eval_deriv(np.array([t]), 1)[0]
            Q_deriv2_t = Q.eval_deriv(np.array([t]), 2)[0]

            # Coefficients for Q(Arg_α):
            Q_alpha_const = Q_t
            Q_alpha_x = Q_deriv_t * theta * t
            Q_alpha_y = Q_deriv_t * theta * (t - 1)
            Q_alpha_xy = Q_deriv2_t * theta * theta * t * (t - 1)

            # Coefficients for Q(Arg_β):
            Q_beta_const = Q_t
            Q_beta_x = Q_deriv_t * theta * (t - 1)
            Q_beta_y = Q_deriv_t * theta * t
            Q_beta_xy = Q_deriv2_t * theta * theta * (t - 1) * t

            # Q(Arg_α)·Q(Arg_β) product:
            QQ_xy = (Q_alpha_const * Q_beta_xy +
                     Q_beta_const * Q_alpha_xy +
                     Q_alpha_x * Q_beta_y +
                     Q_alpha_y * Q_beta_x)
            QQ_const = Q_alpha_const * Q_beta_const
            QQ_x = Q_alpha_const * Q_beta_x + Q_alpha_x * Q_beta_const
            QQ_y = Q_alpha_const * Q_beta_y + Q_alpha_y * Q_beta_const

            # 5. Exp factor: exp(R·(Arg_α + Arg_β))
            exp_2Rt = np.exp(2 * R * t)
            exp_coeff = R * theta * (2 * t - 1)

            E_const = exp_2Rt
            E_x = exp_2Rt * exp_coeff
            E_y = exp_2Rt * exp_coeff
            E_xy = exp_2Rt * exp_coeff * exp_coeff

            # Combine: alg_pf · poly_pf · P_prod · Q_prod · E
            AP_const = poly_pf * alg_const
            AP_x = poly_pf * alg_x
            AP_y = poly_pf * alg_y
            AP_xy = 0.0

            APP_xy = (AP_const * P_xy + AP_xy * P_const +
                      AP_x * P_y + AP_y * P_x)
            APP_const = AP_const * P_const
            APP_x = AP_const * P_x + AP_x * P_const
            APP_y = AP_const * P_y + AP_y * P_const

            APPQ_xy = (APP_const * QQ_xy + APP_xy * QQ_const +
                       APP_x * QQ_y + APP_y * QQ_x)
            APPQ_const = APP_const * QQ_const
            APPQ_x = APP_const * QQ_x + APP_x * QQ_const
            APPQ_y = APP_const * QQ_y + APP_y * QQ_const

            full_xy = (APPQ_const * E_xy + APPQ_xy * E_const +
                       APPQ_x * E_y + APPQ_y * E_x)

            total += weight * full_xy

    return total


@pytest.mark.calibration
class TestDirectI111Gate:
    """Gate tests: direct I1(1,1) must match DSL evaluation."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_i1_11_alignment_kappa(self, polys_kappa):
        """Direct I1(1,1) should match DSL within tolerance at κ benchmark."""
        R = 1.3036

        # Get DSL result
        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        # DSL I1(1,1) raw value (without factorial normalization, which is 1 for (1,1))
        dsl_value = implied.pair_breakdown["11"]["I1_plus_raw"]

        # Compute direct
        direct_value = compute_i1_11_direct_series(THETA, R, polys_kappa, n=60)

        if abs(dsl_value) > 1e-10:
            ratio = direct_value / dsl_value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ I1(1,1): ratio={ratio:.6f}, expected ≈1.0 (DSL={dsl_value:.8f}, direct={direct_value:.8f})"

    def test_i1_11_alignment_kappa_star(self, polys_kappa_star):
        """Direct I1(1,1) should match DSL within tolerance at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        dsl_value = implied.pair_breakdown["11"]["I1_plus_raw"]
        direct_value = compute_i1_11_direct_series(THETA, R, polys_kappa_star, n=60)

        if abs(dsl_value) > 1e-10:
            ratio = direct_value / dsl_value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ* I1(1,1): ratio={ratio:.6f}, expected ≈1.0"

    def test_i1_11_exact_match_kappa(self, polys_kappa):
        """Direct I1(1,1) should match DSL almost exactly at κ benchmark."""
        R = 1.3036

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        dsl_value = implied.pair_breakdown["11"]["I1_plus_raw"]
        direct_value = compute_i1_11_direct_series(THETA, R, polys_kappa, n=60)

        # For Case B (pair 1,1), the match should be exact (within quadrature precision)
        assert np.isclose(direct_value, dsl_value, rtol=1e-6), \
            f"κ I1(1,1) not exact: DSL={dsl_value:.10f}, direct={direct_value:.10f}"

    def test_i1_11_exact_match_kappa_star(self, polys_kappa_star):
        """Direct I1(1,1) should match DSL almost exactly at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        dsl_value = implied.pair_breakdown["11"]["I1_plus_raw"]
        direct_value = compute_i1_11_direct_series(THETA, R, polys_kappa_star, n=60)

        assert np.isclose(direct_value, dsl_value, rtol=1e-6), \
            f"κ* I1(1,1) not exact: DSL={dsl_value:.10f}, direct={direct_value:.10f}"
