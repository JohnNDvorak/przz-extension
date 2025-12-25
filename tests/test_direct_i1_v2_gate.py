"""
tests/test_direct_i1_v2_gate.py
Gate tests: Direct I1 computation must match V2 DSL evaluation.

Run 9A: This test verifies that direct series-based I1 computation
matches V2 DSL evaluation for ALL 9 K=3 pairs at both benchmarks.

KEY INSIGHT (2025-12-20):
V2 uses different (1-u) power formula than OLD DSL:
- V2 (1,1): explicit power=2
- V2 others: max(0, (ℓ₁-1) + (ℓ₂-1))
- OLD: 2 + max(0, (ℓ₁-1) + (ℓ₂-1)) [includes grid base]

The direct computation matches V2 exactly for all pairs.

Usage:
    pytest tests/test_direct_i1_v2_gate.py -v
    pytest tests/test_direct_i1_v2_gate.py -v -m calibration
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
TOLERANCE = 0.001  # 0.1% tolerance (exact match expected)


def get_v2_one_minus_u_power(ell1: int, ell2: int) -> int:
    """Get the (1-u) power for V2 DSL structure."""
    if ell1 == 1 and ell2 == 1:
        return 2
    else:
        return max(0, (ell1 - 1) + (ell2 - 1))


def compute_i1_direct_v2(
    theta: float,
    R: float,
    polynomials: Dict,
    ell1: int,
    ell2: int,
    n: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I1 for pair (ell1, ell2) using V2-compatible structure.

    Returns the signed I1 value (includes (-1)^{ell1+ell2} factor).
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q = polynomials["Q"]
    poly_map = {
        1: (polynomials["P1"], 0),
        2: (polynomials["P2"], 1),
        3: (polynomials["P3"], 2)
    }
    P_left, omega_left = poly_map[ell1]
    P_right, omega_right = poly_map[ell2]

    one_minus_u_power = get_v2_one_minus_u_power(ell1, ell2)

    total = 0.0

    for i, u in enumerate(u_nodes):
        if omega_left == 0:
            left_coeffs = case_b_taylor_coeffs(P_left, u, max_order=1)
        else:
            left_coeffs = case_c_taylor_coeffs(
                P_left, u, omega=omega_left, R=R, theta=theta,
                max_order=1, n_quad_a=n_quad_a
            )
        left_const, left_deriv = left_coeffs[0], left_coeffs[1]

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
            alg_x, alg_y = 1.0, 1.0

            poly_pf = (1 - u) ** one_minus_u_power

            LR_const = left_const * right_const
            LR_x = left_deriv * right_const
            LR_y = left_const * right_deriv
            LR_xy = left_deriv * right_deriv

            Q_t = Q.eval(np.array([t]))[0]
            Q_deriv_t = Q.eval_deriv(np.array([t]), 1)[0]
            Q_deriv2_t = Q.eval_deriv(np.array([t]), 2)[0]

            Q_alpha_const = Q_t
            Q_alpha_x = Q_deriv_t * theta * t
            Q_alpha_y = Q_deriv_t * theta * (t - 1)
            Q_alpha_xy = Q_deriv2_t * theta * theta * t * (t - 1)

            Q_beta_const = Q_t
            Q_beta_x = Q_deriv_t * theta * (t - 1)
            Q_beta_y = Q_deriv_t * theta * t
            Q_beta_xy = Q_deriv2_t * theta * theta * (t - 1) * t

            QQ_const = Q_alpha_const * Q_beta_const
            QQ_x = Q_alpha_const * Q_beta_x + Q_alpha_x * Q_beta_const
            QQ_y = Q_alpha_const * Q_beta_y + Q_alpha_y * Q_beta_const
            QQ_xy = (Q_alpha_const * Q_beta_xy + Q_beta_const * Q_alpha_xy +
                     Q_alpha_x * Q_beta_y + Q_alpha_y * Q_beta_x)

            exp_2Rt = np.exp(2 * R * t)
            exp_coeff = R * theta * (2 * t - 1)

            E_const = exp_2Rt
            E_x = exp_2Rt * exp_coeff
            E_y = exp_2Rt * exp_coeff
            E_xy = exp_2Rt * exp_coeff * exp_coeff

            AP_const = poly_pf * alg_const
            AP_x = poly_pf * alg_x
            AP_y = poly_pf * alg_y
            AP_xy = 0.0

            APLR_const = AP_const * LR_const
            APLR_x = AP_const * LR_x + AP_x * LR_const
            APLR_y = AP_const * LR_y + AP_y * LR_const
            APLR_xy = AP_const * LR_xy + AP_x * LR_y + AP_y * LR_x

            APLRQ_const = APLR_const * QQ_const
            APLRQ_x = APLR_const * QQ_x + APLR_x * QQ_const
            APLRQ_y = APLR_const * QQ_y + APLR_y * QQ_const
            APLRQ_xy = (APLR_const * QQ_xy + APLR_xy * QQ_const +
                        APLR_x * QQ_y + APLR_y * QQ_x)

            full_xy = (APLRQ_const * E_xy + APLRQ_xy * E_const +
                       APLRQ_x * E_y + APLRQ_y * E_x)

            total += weight * full_xy

    # Apply sign factor
    sign = (-1) ** (ell1 + ell2)
    return sign * total


@pytest.mark.calibration
class TestDirectI1V2Gate:
    """Gate tests: direct I1 must match V2 evaluation for all pairs."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # =========================================================================
    # κ Benchmark Tests (R=1.3036)
    # =========================================================================

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i1_plus_kappa(self, polys_kappa, pair):
        """Direct I1 at +R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i1 = v2_terms[pair][0]
        v2_result = evaluate_term(v2_i1, polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i1_direct_v2(THETA, R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ I1+({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8, \
                f"κ I1+({pair}): V2≈0 but direct={direct_value:.8f}"

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i1_minus_kappa(self, polys_kappa, pair):
        """Direct I1 at -R should match V2 at κ benchmark."""
        R = 1.3036
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, -R, kernel_regime="paper")
        v2_i1 = v2_terms[pair][0]
        v2_result = evaluate_term(v2_i1, polys_kappa, n=60, R=-R, theta=THETA, n_quad_a=40)

        direct_value = compute_i1_direct_v2(THETA, -R, polys_kappa, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ I1-({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8, \
                f"κ I1-({pair}): V2≈0 but direct={direct_value:.8f}"

    # =========================================================================
    # κ* Benchmark Tests (R=1.1167)
    # =========================================================================

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i1_plus_kappa_star(self, polys_kappa_star, pair):
        """Direct I1 at +R should match V2 at κ* benchmark."""
        R = 1.1167
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        v2_i1 = v2_terms[pair][0]
        v2_result = evaluate_term(v2_i1, polys_kappa_star, n=60, R=R, theta=THETA, n_quad_a=40)

        direct_value = compute_i1_direct_v2(THETA, R, polys_kappa_star, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ* I1+({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8, \
                f"κ* I1+({pair}): V2≈0 but direct={direct_value:.8f}"

    @pytest.mark.parametrize("pair", ["11", "12", "21", "22", "13", "31", "23", "32", "33"])
    def test_i1_minus_kappa_star(self, polys_kappa_star, pair):
        """Direct I1 at -R should match V2 at κ* benchmark."""
        R = 1.1167
        ell1, ell2 = int(pair[0]), int(pair[1])

        v2_terms = make_all_terms_k3_ordered_v2(THETA, -R, kernel_regime="paper")
        v2_i1 = v2_terms[pair][0]
        v2_result = evaluate_term(v2_i1, polys_kappa_star, n=60, R=-R, theta=THETA, n_quad_a=40)

        direct_value = compute_i1_direct_v2(THETA, -R, polys_kappa_star, ell1, ell2, n=60, n_quad_a=40)

        if abs(v2_result.value) > 1e-10:
            ratio = direct_value / v2_result.value
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"κ* I1-({pair}): ratio={ratio:.6f}, expected ≈1.0"
        else:
            assert abs(direct_value) < 1e-8, \
                f"κ* I1-({pair}): V2≈0 but direct={direct_value:.8f}"


@pytest.mark.calibration
class TestV2VsOLDComparison:
    """Document V2 vs OLD DSL differences."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_v2_vs_old_difference_documented(self, polys_kappa):
        """V2 and OLD differ by 2-3% for non-diagonal pairs (expected)."""
        from src.terms_k3_d1 import make_all_terms_k3_ordered

        R = 1.3036

        v2_terms = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
        old_terms = make_all_terms_k3_ordered(THETA, R, kernel_regime="paper")

        # (1,1) should match exactly
        v2_11 = evaluate_term(v2_terms["11"][0], polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)
        old_11 = evaluate_term(old_terms["11"][0], polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)
        ratio_11 = v2_11.value / old_11.value
        assert abs(ratio_11 - 1.0) < 0.001, f"(1,1) should match: ratio={ratio_11}"

        # (1,2) should differ (documents expected behavior)
        v2_12 = evaluate_term(v2_terms["12"][0], polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)
        old_12 = evaluate_term(old_terms["12"][0], polys_kappa, n=60, R=R, theta=THETA, n_quad_a=40)
        ratio_12 = v2_12.value / old_12.value

        # V2/OLD ratio for (1,2) is approximately 1.025 (2.5% difference)
        # This is expected due to different (1-u) power formulas
        assert 1.01 < ratio_12 < 1.05, \
            f"(1,2) V2/OLD ratio should be ~1.025, got {ratio_12}"
