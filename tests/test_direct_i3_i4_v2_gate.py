"""
tests/test_direct_i3_i4_v2_gate.py
Gate tests: Direct I3 and I4 computation must match V2 DSL evaluation.

Run 11: This test verifies that direct series-based I3 and I4 computation
matches V2 DSL evaluation for ALL 9 K=3 pairs at both benchmarks.

KEY INSIGHT (2025-12-20):
I3 Structure:
- Single variable: x only
- Left: P_ℓ₁(x+u) shifted
- Right: P_ℓ₂(u) UNSHIFTED
- (1-u) power: (1,1)=1 explicit, others max(0, ℓ₁ - 1)
- Q args: Q(t+θtx), Q(t+θ(t-1)x)

I4 Structure:
- Single variable: y only
- Left: P_ℓ₁(u) UNSHIFTED
- Right: P_ℓ₂(y+u) shifted
- (1-u) power: (1,1)=1 explicit, others max(0, ℓ₂ - 1)
- Q args: Q(t+θ(t-1)y), Q(t+θty)

Usage:
    pytest tests/test_direct_i3_i4_v2_gate.py -v
    pytest tests/test_direct_i3_i4_v2_gate.py -v -m calibration
"""

import pytest
import numpy as np
from typing import Dict

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import gauss_legendre_01
from src.terms_k3_d1 import make_all_terms_k3_ordered_v2
from src.evaluate import evaluate_term
from src.mollifier_profiles import case_b_taylor_coeffs, case_c_taylor_coeffs


THETA = 4.0 / 7.0
TOLERANCE = 0.002  # 0.2% tolerance (slightly relaxed for I3/I4)


def get_i3_one_minus_u_power(ell1: int, ell2: int) -> int:
    """Get the (1-u) power for I3."""
    if ell1 == 1 and ell2 == 1:
        return 1  # Explicit in V2 for (1,1)
    else:
        return max(0, ell1 - 1)


def get_i4_one_minus_u_power(ell1: int, ell2: int) -> int:
    """Get the (1-u) power for I4."""
    if ell1 == 1 and ell2 == 1:
        return 1  # Explicit in V2 for (1,1)
    else:
        return max(0, ell2 - 1)


def compute_i3_direct_v2(
    theta: float,
    R: float,
    polynomials: Dict,
    ell1: int,
    ell2: int,
    n: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I3 for pair (ell1, ell2) using V2-compatible structure.
    Returns the signed I3 value (includes numeric_prefactor = -1).
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q = polynomials["Q"]
    poly_map = {1: (polynomials["P1"], 0), 2: (polynomials["P2"], 1), 3: (polynomials["P3"], 2)}
    P_left, omega_left = poly_map[ell1]
    P_right, omega_right = poly_map[ell2]

    one_minus_u_power = get_i3_one_minus_u_power(ell1, ell2)
    total = 0.0

    for i, u in enumerate(u_nodes):
        # Left: P_ℓ₁(x+u) shifted with Taylor expansion
        if omega_left == 0:
            left_coeffs = case_b_taylor_coeffs(P_left, u, max_order=1)
        else:
            left_coeffs = case_c_taylor_coeffs(
                P_left, u, omega=omega_left, R=R, theta=theta,
                max_order=1, n_quad_a=n_quad_a
            )
        left_const, left_deriv = left_coeffs[0], left_coeffs[1]

        # Right: P_ℓ₂(u) UNSHIFTED
        if omega_right == 0:
            right_val = P_right.eval(np.array([u]))[0]
        else:
            right_coeffs = case_c_taylor_coeffs(
                P_right, u, omega=omega_right, R=R, theta=theta,
                max_order=0, n_quad_a=n_quad_a
            )
            right_val = right_coeffs[0]

        for j, t in enumerate(t_nodes):
            weight = u_weights[i] * t_weights[j]

            alg_const = 1.0 / theta
            alg_x = 1.0
            poly_pf = (1 - u) ** one_minus_u_power

            LR_const = left_const * right_val
            LR_x = left_deriv * right_val

            Q_t = Q.eval(np.array([t]))[0]
            Q_deriv_t = Q.eval_deriv(np.array([t]), 1)[0]

            Q_alpha_const = Q_t
            Q_alpha_x = Q_deriv_t * theta * t
            Q_beta_const = Q_t
            Q_beta_x = Q_deriv_t * theta * (t - 1)

            QQ_const = Q_alpha_const * Q_beta_const
            QQ_x = Q_alpha_const * Q_beta_x + Q_alpha_x * Q_beta_const

            exp_2Rt = np.exp(2 * R * t)
            exp_coeff = R * theta * (2 * t - 1)

            E_const = exp_2Rt
            E_x = exp_2Rt * exp_coeff

            AP_const = poly_pf * alg_const
            AP_x = poly_pf * alg_x

            APLR_const = AP_const * LR_const
            APLR_x = AP_const * LR_x + AP_x * LR_const

            APLRQ_const = APLR_const * QQ_const
            APLRQ_x = APLR_const * QQ_x + APLR_x * QQ_const

            full_x = APLRQ_x * E_const + APLRQ_const * E_x
            total += weight * full_x

    # Apply numeric_prefactor = -1
    return -1.0 * total


def compute_i4_direct_v2(
    theta: float,
    R: float,
    polynomials: Dict,
    ell1: int,
    ell2: int,
    n: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I4 for pair (ell1, ell2) using V2-compatible structure.
    Returns the signed I4 value (includes numeric_prefactor = -1).
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q = polynomials["Q"]
    poly_map = {1: (polynomials["P1"], 0), 2: (polynomials["P2"], 1), 3: (polynomials["P3"], 2)}
    P_left, omega_left = poly_map[ell1]
    P_right, omega_right = poly_map[ell2]

    one_minus_u_power = get_i4_one_minus_u_power(ell1, ell2)
    total = 0.0

    for i, u in enumerate(u_nodes):
        # Left: P_ℓ₁(u) UNSHIFTED
        if omega_left == 0:
            left_const = P_left.eval(np.array([u]))[0]
        else:
            left_coeffs = case_c_taylor_coeffs(
                P_left, u, omega=omega_left, R=R, theta=theta,
                max_order=0, n_quad_a=n_quad_a
            )
            left_const = left_coeffs[0]

        # Right: P_ℓ₂(y+u) shifted with Taylor expansion
        if omega_right == 0:
            right_coeffs = case_b_taylor_coeffs(P_right, u, max_order=1)
        else:
            right_coeffs = case_c_taylor_coeffs(
                P_right, u, omega=omega_right, R=R, theta=theta,
                max_order=1, n_quad_a=n_quad_a
            )
        right_const, right_deriv = right_coeffs[0], right_coeffs[1]

        for j, t in enumerate(t_nodes):
            weight = u_weights[i] * t_weights[j]

            alg_const = 1.0 / theta
            alg_y = 1.0
            poly_pf = (1 - u) ** one_minus_u_power if one_minus_u_power > 0 else 1.0

            LR_const = left_const * right_const
            LR_y = left_const * right_deriv

            Q_t = Q.eval(np.array([t]))[0]
            Q_deriv_t = Q.eval_deriv(np.array([t]), 1)[0]

            Q_alpha_const = Q_t
            Q_alpha_y = Q_deriv_t * theta * (t - 1)
            Q_beta_const = Q_t
            Q_beta_y = Q_deriv_t * theta * t

            QQ_const = Q_alpha_const * Q_beta_const
            QQ_y = Q_alpha_const * Q_beta_y + Q_alpha_y * Q_beta_const

            exp_2Rt = np.exp(2 * R * t)
            exp_coeff = R * theta * (2 * t - 1)

            E_const = exp_2Rt
            E_y = exp_2Rt * exp_coeff

            AP_const = poly_pf * alg_const
            AP_y = poly_pf * alg_y

            APLR_const = AP_const * LR_const
            APLR_y = AP_const * LR_y + AP_y * LR_const

            APLRQ_const = APLR_const * QQ_const
            APLRQ_y = APLR_const * QQ_y + APLR_y * QQ_const

            full_y = APLRQ_const * E_y + APLRQ_y * E_const
            total += weight * full_y

    # Apply numeric_prefactor = -1
    return -1.0 * total


@pytest.mark.calibration
class TestDirectI3V2Gate:
    """Gate tests: direct I3 must match V2 evaluation for all pairs."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i3_plus_kappa(self, polys_kappa, pair):
        """Direct I3 at +R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i3 = v2_terms[pair][2]  # I3 is index [2]
        v2_result = evaluate_term(v2_i3, polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i3_direct_v2(THETA, R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ I3+({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i3_minus_kappa(self, polys_kappa, pair):
        """Direct I3 at -R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, -R, kernel_regime="paper")
        v2_i3 = v2_terms[pair][2]
        v2_result = evaluate_term(v2_i3, polys_kappa, n=60, R=-R, theta=THETA, n_quad_a=40)

        direct_value = compute_i3_direct_v2(THETA, -R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i3_plus_kappa_star(self, polys_kappa_star, pair):
        """Direct I3 at +R should match V2 at κ* benchmark."""
        R = 1.1167
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i3 = v2_terms[pair][2]
        v2_result = evaluate_term(v2_i3, polys_kappa_star, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i3_direct_v2(THETA, R, polys_kappa_star, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE


@pytest.mark.calibration
class TestDirectI4V2Gate:
    """Gate tests: direct I4 must match V2 evaluation for all pairs."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i4_plus_kappa(self, polys_kappa, pair):
        """Direct I4 at +R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i4 = v2_terms[pair][3]  # I4 is index [3]
        v2_result = evaluate_term(v2_i4, polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i4_direct_v2(THETA, R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ I4+({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i4_minus_kappa(self, polys_kappa, pair):
        """Direct I4 at -R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, -R, kernel_regime="paper")
        v2_i4 = v2_terms[pair][3]
        v2_result = evaluate_term(v2_i4, polys_kappa, n=60, R=-R, theta=THETA, n_quad_a=40)

        direct_value = compute_i4_direct_v2(THETA, -R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i4_plus_kappa_star(self, polys_kappa_star, pair):
        """Direct I4 at +R should match V2 at κ* benchmark."""
        R = 1.1167
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i4 = v2_terms[pair][3]
        v2_result = evaluate_term(v2_i4, polys_kappa_star, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i4_direct_v2(THETA, R, polys_kappa_star, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE
