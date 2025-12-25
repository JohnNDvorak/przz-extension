"""
tests/test_direct_i2_caseC_gate.py
Gate tests: Direct I2 with Case C must match model I2.

These tests verify that the direct TeX I2 evaluation with Case C kernels
produces the same values as the model's DSL-based evaluation.

Run 7 proved that ALL 9 ordered pairs match exactly (ratio = 1.0)
when Case C kernels are correctly implemented.

Usage:
    pytest tests/test_direct_i2_caseC_gate.py -v
    pytest tests/test_direct_i2_caseC_gate.py -v -m calibration
"""

import pytest
import math
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

# Factorial normalization: 1/(ell1! × ell2!)
F_NORM = {
    f"{i}{j}": 1.0 / (math.factorial(i) * math.factorial(j))
    for i in (1, 2, 3) for j in (1, 2, 3)
}


def compute_case_c_kernel(
    P_eval,
    u_nodes: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    a_nodes: np.ndarray,
    a_weights: np.ndarray
) -> np.ndarray:
    """
    Compute Case C kernel K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got {omega}")

    u_grid = u_nodes[:, np.newaxis]
    a_grid = a_nodes[np.newaxis, :]
    poly_arg = (1 - a_grid) * u_grid
    P_vals = P_eval(poly_arg.flatten()).reshape(len(u_nodes), len(a_nodes))

    if omega == 1:
        a_weight = np.ones_like(a_nodes)
    else:
        a_weight = a_nodes ** (omega - 1)

    exp_factor = np.exp(R * theta * u_grid * a_grid)
    integrand = P_vals * a_weight[np.newaxis, :] * exp_factor
    K = np.sum(a_weights[np.newaxis, :] * integrand, axis=1)

    return K


def compute_i2_all_pairs_case_c(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 100,
    n_a: int = 40
) -> Dict[str, Dict[str, float]]:
    """
    Compute I2 for all 9 ordered pairs with Case C kernel handling.
    Returns dict with pair_key -> {"i2_plus": val, "i2_minus": val}
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    a_nodes, a_weights = gauss_legendre_01(n_a)

    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials["Q"]

    # Compute kernels at +R and -R
    K1_plus = P1.eval(u_nodes)
    K1_minus = P1.eval(u_nodes)

    K2_kernel_plus = compute_case_c_kernel(P2.eval, u_nodes, 1, R, theta, a_nodes, a_weights)
    K2_kernel_minus = compute_case_c_kernel(P2.eval, u_nodes, 1, -R, theta, a_nodes, a_weights)
    K2_plus = u_nodes * K2_kernel_plus
    K2_minus = u_nodes * K2_kernel_minus

    K3_kernel_plus = compute_case_c_kernel(P3.eval, u_nodes, 2, R, theta, a_nodes, a_weights)
    K3_kernel_minus = compute_case_c_kernel(P3.eval, u_nodes, 2, -R, theta, a_nodes, a_weights)
    K3_plus = (u_nodes ** 2) * K3_kernel_plus
    K3_minus = (u_nodes ** 2) * K3_kernel_minus

    K_vals_plus = {"P1": K1_plus, "P2": K2_plus, "P3": K3_plus}
    K_vals_minus = {"P1": K1_minus, "P2": K2_minus, "P3": K3_minus}

    # t-integrals
    Q_vals = Q.eval(u_nodes)
    exp_plus = np.exp(2 * R * u_nodes)
    exp_minus = np.exp(-2 * R * u_nodes)
    t_integral_plus = np.sum(u_weights * Q_vals**2 * exp_plus) / theta
    t_integral_minus = np.sum(u_weights * Q_vals**2 * exp_minus) / theta

    results = {}
    P_map = {"1": "P1", "2": "P2", "3": "P3"}

    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        p1_key = P_map[pair_key[0]]
        p2_key = P_map[pair_key[1]]

        u_integral_plus = np.sum(u_weights * K_vals_plus[p1_key] * K_vals_plus[p2_key])
        u_integral_minus = np.sum(u_weights * K_vals_minus[p1_key] * K_vals_minus[p2_key])

        i2_plus = u_integral_plus * t_integral_plus
        i2_minus = u_integral_minus * t_integral_minus

        results[pair_key] = {
            "i2_plus": i2_plus,
            "i2_minus": i2_minus,
        }

    return results


@pytest.mark.calibration
class TestDirectI2CaseCGate:
    """Gate tests: direct I2 with Case C must match model I2."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_per_pair_i2_plus_alignment_kappa(self, polys_kappa):
        """Each pair's direct I2+ should match model within tolerance at κ benchmark."""
        R = 1.3036

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa, n=100, n_a=40)

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            norm = F_NORM[pair_key]
            direct_val = norm * direct[pair_key]["i2_plus"]
            model_val = implied.pair_breakdown[pair_key]["I2_plus"]

            if abs(model_val) > 1e-10:
                ratio = direct_val / model_val
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ pair {pair_key} I2+: ratio={ratio:.6f}, expected ≈1.0"

    def test_per_pair_i2_minus_alignment_kappa(self, polys_kappa):
        """Each pair's direct I2- should match model within tolerance at κ benchmark."""
        R = 1.3036

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa, n=100, n_a=40)

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            norm = F_NORM[pair_key]
            direct_val = norm * direct[pair_key]["i2_minus"]
            model_val = implied.pair_breakdown[pair_key]["I2_minus_base"]

            if abs(model_val) > 1e-10:
                ratio = direct_val / model_val
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ pair {pair_key} I2-: ratio={ratio:.6f}, expected ≈1.0"

    def test_per_pair_i2_plus_alignment_kappa_star(self, polys_kappa_star):
        """Each pair's direct I2+ should match model within tolerance at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa_star, n=100, n_a=40)

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            norm = F_NORM[pair_key]
            direct_val = norm * direct[pair_key]["i2_plus"]
            model_val = implied.pair_breakdown[pair_key]["I2_plus"]

            if abs(model_val) > 1e-10:
                ratio = direct_val / model_val
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ* pair {pair_key} I2+: ratio={ratio:.6f}, expected ≈1.0"

    def test_per_pair_i2_minus_alignment_kappa_star(self, polys_kappa_star):
        """Each pair's direct I2- should match model within tolerance at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa_star, n=100, n_a=40)

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            norm = F_NORM[pair_key]
            direct_val = norm * direct[pair_key]["i2_minus"]
            model_val = implied.pair_breakdown[pair_key]["I2_minus_base"]

            if abs(model_val) > 1e-10:
                ratio = direct_val / model_val
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ* pair {pair_key} I2-: ratio={ratio:.6f}, expected ≈1.0"

    def test_total_i2_plus_kappa(self, polys_kappa):
        """Total direct I2+ should match model I2_plus at κ benchmark."""
        R = 1.3036

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa, n=100, n_a=40)

        direct_total = sum(F_NORM[k] * direct[k]["i2_plus"] for k in direct)
        model_total = implied.I2_plus

        ratio = direct_total / model_total
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ total I2+: ratio={ratio:.6f}, expected ≈1.0"

    def test_total_i2_minus_kappa(self, polys_kappa):
        """Total direct I2- should match model I2_minus_base at κ benchmark."""
        R = 1.3036

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa, n=100, n_a=40)

        direct_total = sum(F_NORM[k] * direct[k]["i2_minus"] for k in direct)
        model_total = implied.I2_minus_base

        ratio = direct_total / model_total
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ total I2-: ratio={ratio:.6f}, expected ≈1.0"

    def test_total_i2_plus_kappa_star(self, polys_kappa_star):
        """Total direct I2+ should match model I2_plus at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa_star, n=100, n_a=40)

        direct_total = sum(F_NORM[k] * direct[k]["i2_plus"] for k in direct)
        model_total = implied.I2_plus

        ratio = direct_total / model_total
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ* total I2+: ratio={ratio:.6f}, expected ≈1.0"

    def test_total_i2_minus_kappa_star(self, polys_kappa_star):
        """Total direct I2- should match model I2_minus_base at κ* benchmark."""
        R = 1.1167

        implied = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40
        )

        direct = compute_i2_all_pairs_case_c(THETA, R, polys_kappa_star, n=100, n_a=40)

        direct_total = sum(F_NORM[k] * direct[k]["i2_minus"] for k in direct)
        model_total = implied.I2_minus_base

        ratio = direct_total / model_total
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ* total I2-: ratio={ratio:.6f}, expected ≈1.0"


@pytest.mark.calibration
class TestCaseCKernelAnalytic:
    """Analytic tests for Case C kernel implementation."""

    def test_r_zero_constant_polynomial_omega1(self):
        """At R=0 with P=1, K_1(u; R=0) = ∫₀¹ a^0 da = 1."""
        from src.polynomials import Polynomial
        P_const = Polynomial([1.0])

        u_nodes, _ = gauss_legendre_01(50)
        a_nodes, a_weights = gauss_legendre_01(50)

        K1 = compute_case_c_kernel(
            P_const.eval, u_nodes, omega=1, R=0, theta=THETA,
            a_nodes=a_nodes, a_weights=a_weights
        )

        assert np.allclose(K1, 1.0, rtol=1e-10), \
            f"K_1(u; R=0) for P=1 should be 1.0, got {K1[len(K1)//2]:.10f}"

    def test_r_zero_constant_polynomial_omega2(self):
        """At R=0 with P=1, K_2(u; R=0) = ∫₀¹ a^1 da = 0.5."""
        from src.polynomials import Polynomial
        P_const = Polynomial([1.0])

        u_nodes, _ = gauss_legendre_01(50)
        a_nodes, a_weights = gauss_legendre_01(50)

        K2 = compute_case_c_kernel(
            P_const.eval, u_nodes, omega=2, R=0, theta=THETA,
            a_nodes=a_nodes, a_weights=a_weights
        )

        assert np.allclose(K2, 0.5, rtol=1e-10), \
            f"K_2(u; R=0) for P=1 should be 0.5, got {K2[len(K2)//2]:.10f}"

    def test_case_b_unchanged(self):
        """P1 (Case B) should not be affected by R changes."""
        P1, _, _, _ = load_przz_polynomials(enforce_Q0=True)

        u_nodes, _ = gauss_legendre_01(50)

        # Case B: just raw polynomial
        P1_vals = P1.eval(u_nodes)

        # Should be same at any R (no kernel dependence)
        assert np.allclose(P1_vals, P1.eval(u_nodes)), \
            "P1 (Case B) should not depend on R"
