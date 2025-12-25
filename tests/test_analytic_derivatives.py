"""
tests/test_analytic_derivatives.py
Unit tests for the analytic derivatives module.

Tests verify:
1. d^{n+m}/dα^n dβ^m [1/(α+β)] formulas are correct
2. Eigenvalue properties for exponential terms
3. Binomial coefficients for Leibniz rule
4. Numerical cross-checks where applicable
"""

import pytest
import numpy as np
from math import factorial

from src.analytic_derivatives import (
    deriv_inverse_sum,
    deriv_inverse_sum_at_point,
    deriv_exp_linear_alpha_PRZZ,
    deriv_exp_linear_beta_PRZZ,
    deriv_exp_linear_mixed_PRZZ,
    deriv_mirror_exp_factor_PRZZ,
    deriv_exp_linear_raw,
    deriv_mirror_exp_raw,
    binomial,
    leibniz_coefficients,
)


class TestDerivInverseSum:
    """Test derivatives of 1/(α+β)."""

    def test_zeroth_order(self):
        """d^0/dα^0 d^0/dβ^0 [1/(α+β)] = 1/(α+β)."""
        f = deriv_inverse_sum(0, 0)
        assert np.isclose(f(1.0, 1.0), 0.5)
        assert np.isclose(f(2.0, 3.0), 0.2)
        assert np.isclose(f(0.5, 0.5), 1.0)

    def test_first_order_alpha(self):
        """d/dα [1/(α+β)] = -1/(α+β)²."""
        f = deriv_inverse_sum(1, 0)
        # At α=β=1: -1/(1+1)² = -1/4 = -0.25
        assert np.isclose(f(1.0, 1.0), -0.25)
        # At α=β=0.5: -1/(0.5+0.5)² = -1
        assert np.isclose(f(0.5, 0.5), -1.0)

    def test_first_order_beta(self):
        """d/dβ [1/(α+β)] = -1/(α+β)²."""
        f = deriv_inverse_sum(0, 1)
        # Should be same as d/dα since symmetric
        assert np.isclose(f(1.0, 1.0), -0.25)
        assert np.isclose(f(0.5, 0.5), -1.0)

    def test_second_order_alpha(self):
        """d²/dα² [1/(α+β)] = 2/(α+β)³."""
        f = deriv_inverse_sum(2, 0)
        # At α=β=1: 2/(1+1)³ = 2/8 = 0.25
        assert np.isclose(f(1.0, 1.0), 0.25)
        # At α=β=0.5: 2/(0.5+0.5)³ = 2
        assert np.isclose(f(0.5, 0.5), 2.0)

    def test_second_order_beta(self):
        """d²/dβ² [1/(α+β)] = 2/(α+β)³."""
        f = deriv_inverse_sum(0, 2)
        assert np.isclose(f(1.0, 1.0), 0.25)
        assert np.isclose(f(0.5, 0.5), 2.0)

    def test_mixed_first_order(self):
        """d²/dαdβ [1/(α+β)] = 2/(α+β)³."""
        f = deriv_inverse_sum(1, 1)
        # Same as d²/dα² since ∂/∂α and ∂/∂β are equivalent on 1/(α+β)
        assert np.isclose(f(1.0, 1.0), 0.25)
        assert np.isclose(f(0.5, 0.5), 2.0)

    def test_general_formula(self):
        """d^{n+m}/dα^n dβ^m [1/(α+β)] = (-1)^{n+m} × (n+m)! / (α+β)^{n+m+1}."""
        for n, m in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (3, 0)]:
            total = n + m
            alpha, beta = 1.0, 2.0  # α+β = 3
            expected = ((-1) ** total) * factorial(total) / (3.0 ** (total + 1))
            actual = deriv_inverse_sum_at_point(n, m, alpha, beta)
            assert np.isclose(actual, expected), f"Failed for n={n}, m={m}"

    def test_at_przz_point(self):
        """Test at α=β=-R/L point (typical PRZZ evaluation)."""
        R = 1.3036
        L = 100.0
        alpha = beta = -R / L

        # α+β = -2R/L
        sum_val = alpha + beta
        assert np.isclose(sum_val, -2 * R / L)

        # 1/(α+β) = -L/(2R)
        f0 = deriv_inverse_sum(0, 0)
        assert np.isclose(f0(alpha, beta), -L / (2 * R))

        # d/dα [1/(α+β)] = -1/(α+β)² = -L²/(4R²)
        f1 = deriv_inverse_sum(1, 0)
        expected = -1.0 / (sum_val ** 2)
        assert np.isclose(f1(alpha, beta), expected)


class TestExpLinearDerivativesPRZZ:
    """Test exponential derivatives using PRZZ D convention."""

    @pytest.fixture
    def params(self):
        return {
            "theta": 4.0 / 7.0,
            "L": 100.0,
            "alpha": 0.1,
            "beta": 0.2,
            "x": 0.3,
            "y": 0.4,
        }

    def test_zeroth_order(self, params):
        """D^0 exp(θL(αx+βy)) = exp(θL(αx+βy))."""
        result = deriv_exp_linear_mixed_PRZZ(
            0, 0,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        expected = np.exp(
            params["theta"] * params["L"] *
            (params["alpha"] * params["x"] + params["beta"] * params["y"])
        )
        assert np.isclose(result, expected)

    def test_first_order_alpha(self, params):
        """D_α exp(θL(αx+βy)) = (-θx) × exp(θL(αx+βy))."""
        result = deriv_exp_linear_alpha_PRZZ(
            1,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exp_val = np.exp(
            params["theta"] * params["L"] *
            (params["alpha"] * params["x"] + params["beta"] * params["y"])
        )
        expected = (-params["theta"] * params["x"]) * exp_val
        assert np.isclose(result, expected)

    def test_first_order_beta(self, params):
        """D_β exp(θL(αx+βy)) = (-θy) × exp(θL(αx+βy))."""
        result = deriv_exp_linear_beta_PRZZ(
            1,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exp_val = np.exp(
            params["theta"] * params["L"] *
            (params["alpha"] * params["x"] + params["beta"] * params["y"])
        )
        expected = (-params["theta"] * params["y"]) * exp_val
        assert np.isclose(result, expected)

    def test_mixed_order(self, params):
        """D_α D_β exp(θL(αx+βy)) = (-θx)(-θy) × exp(θL(αx+βy))."""
        result = deriv_exp_linear_mixed_PRZZ(
            1, 1,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exp_val = np.exp(
            params["theta"] * params["L"] *
            (params["alpha"] * params["x"] + params["beta"] * params["y"])
        )
        expected = (-params["theta"] * params["x"]) * (-params["theta"] * params["y"]) * exp_val
        assert np.isclose(result, expected)

    def test_higher_order_alpha(self, params):
        """D_α^n exp(θL(αx+βy)) = (-θx)^n × exp(θL(αx+βy))."""
        for n in range(5):
            result = deriv_exp_linear_alpha_PRZZ(
                n,
                params["alpha"], params["beta"],
                params["x"], params["y"],
                params["theta"], params["L"]
            )
            exp_val = np.exp(
                params["theta"] * params["L"] *
                (params["alpha"] * params["x"] + params["beta"] * params["y"])
            )
            expected = ((-params["theta"] * params["x"]) ** n) * exp_val
            assert np.isclose(result, expected), f"Failed for n={n}"

    def test_at_x_equals_zero(self, params):
        """At x=0, D_α^n should give 0 for n>0."""
        for n in [1, 2, 3]:
            result = deriv_exp_linear_alpha_PRZZ(
                n,
                params["alpha"], params["beta"],
                0.0, params["y"],  # x=0
                params["theta"], params["L"]
            )
            # (-θ×0)^n × exp(...) = 0 for n>0
            assert result == 0.0, f"Failed for n={n}"

    def test_at_y_equals_zero(self, params):
        """At y=0, D_β^m should give 0 for m>0."""
        for m in [1, 2, 3]:
            result = deriv_exp_linear_beta_PRZZ(
                m,
                params["alpha"], params["beta"],
                params["x"], 0.0,  # y=0
                params["theta"], params["L"]
            )
            assert result == 0.0, f"Failed for m={m}"


class TestMirrorExpDerivativesPRZZ:
    """Test mirror exponential factor derivatives using PRZZ D convention."""

    @pytest.fixture
    def params(self):
        return {
            "theta": 4.0 / 7.0,
            "L": 100.0,
            "alpha": -0.01,  # Typical PRZZ: α = -R/L
            "beta": -0.01,
            "x": 0.3,
            "y": 0.4,
        }

    def test_zeroth_order(self, params):
        """D^0 [T^{-(α+β)} × N^{-βx-αy}] = exp(-L[(α+β) + θ(βx+αy)])."""
        result = deriv_mirror_exp_factor_PRZZ(
            0, 0,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exponent = -params["L"] * (
            (params["alpha"] + params["beta"]) +
            params["theta"] * (params["beta"] * params["x"] + params["alpha"] * params["y"])
        )
        expected = np.exp(exponent)
        assert np.isclose(result, expected)

    def test_first_order_alpha_eigenvalue(self, params):
        """D_α has eigenvalue (1+θy) on the mirror term."""
        result = deriv_mirror_exp_factor_PRZZ(
            1, 0,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exponent = -params["L"] * (
            (params["alpha"] + params["beta"]) +
            params["theta"] * (params["beta"] * params["x"] + params["alpha"] * params["y"])
        )
        exp_val = np.exp(exponent)
        eigenvalue = 1 + params["theta"] * params["y"]
        expected = eigenvalue * exp_val
        assert np.isclose(result, expected)

    def test_first_order_beta_eigenvalue(self, params):
        """D_β has eigenvalue (1+θx) on the mirror term."""
        result = deriv_mirror_exp_factor_PRZZ(
            0, 1,
            params["alpha"], params["beta"],
            params["x"], params["y"],
            params["theta"], params["L"]
        )
        exponent = -params["L"] * (
            (params["alpha"] + params["beta"]) +
            params["theta"] * (params["beta"] * params["x"] + params["alpha"] * params["y"])
        )
        exp_val = np.exp(exponent)
        eigenvalue = 1 + params["theta"] * params["x"]
        expected = eigenvalue * exp_val
        assert np.isclose(result, expected)

    def test_mixed_order_eigenvalues(self, params):
        """D_α^n D_β^m has eigenvalues (1+θy)^n × (1+θx)^m."""
        for n, m in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            result = deriv_mirror_exp_factor_PRZZ(
                n, m,
                params["alpha"], params["beta"],
                params["x"], params["y"],
                params["theta"], params["L"]
            )
            exponent = -params["L"] * (
                (params["alpha"] + params["beta"]) +
                params["theta"] * (params["beta"] * params["x"] + params["alpha"] * params["y"])
            )
            exp_val = np.exp(exponent)
            eigenvalue_alpha = (1 + params["theta"] * params["y"]) ** n
            eigenvalue_beta = (1 + params["theta"] * params["x"]) ** m
            expected = eigenvalue_alpha * eigenvalue_beta * exp_val
            assert np.isclose(result, expected), f"Failed for n={n}, m={m}"

    def test_at_przz_point(self):
        """Test at α=β=-R/L (typical PRZZ evaluation point)."""
        R = 1.3036
        L = 100.0
        theta = 4.0 / 7.0
        alpha = beta = -R / L
        x, y = 0.3, 0.4

        result = deriv_mirror_exp_factor_PRZZ(0, 0, alpha, beta, x, y, theta, L)

        # At α=β=-R/L: α+β = -2R/L
        # exponent = -L[-2R/L + θ(-R/L × x - R/L × y)]
        #          = -L × (-2R/L) × (1 + θ(x+y)/2)
        #          = 2R × (1 + θ(x+y)/2)
        #          = 2R + Rθ(x+y)
        sum_ab = alpha + beta
        cross = beta * x + alpha * y
        exponent = -L * (sum_ab + theta * cross)
        expected = np.exp(exponent)
        assert np.isclose(result, expected)

        # At x=y=0: exp(2R) — the famous mirror weight base
        result_xy0 = deriv_mirror_exp_factor_PRZZ(0, 0, alpha, beta, 0.0, 0.0, theta, L)
        expected_xy0 = np.exp(2 * R)
        assert np.isclose(result_xy0, expected_xy0, rtol=1e-3)


class TestBinomialCoefficients:
    """Test binomial coefficient computation."""

    def test_edge_cases(self):
        """C(n,0) = C(n,n) = 1."""
        for n in range(10):
            assert binomial(n, 0) == 1
            assert binomial(n, n) == 1

    def test_pascals_triangle(self):
        """C(n,k) = C(n-1,k-1) + C(n-1,k)."""
        for n in range(1, 10):
            for k in range(1, n):
                assert binomial(n, k) == binomial(n - 1, k - 1) + binomial(n - 1, k)

    def test_known_values(self):
        """Test known values."""
        assert binomial(5, 2) == 10
        assert binomial(6, 3) == 20
        assert binomial(10, 5) == 252

    def test_symmetry(self):
        """C(n,k) = C(n,n-k)."""
        for n in range(10):
            for k in range(n + 1):
                assert binomial(n, k) == binomial(n, n - k)


class TestLeibnizCoefficients:
    """Test Leibniz coefficient generation."""

    def test_order_0(self):
        """Order 0: [1]."""
        assert leibniz_coefficients(0) == [1]

    def test_order_1(self):
        """Order 1: [1, 1]."""
        assert leibniz_coefficients(1) == [1, 1]

    def test_order_2(self):
        """Order 2: [1, 2, 1]."""
        assert leibniz_coefficients(2) == [1, 2, 1]

    def test_order_3(self):
        """Order 3: [1, 3, 3, 1]."""
        assert leibniz_coefficients(3) == [1, 3, 3, 1]

    def test_order_4(self):
        """Order 4: [1, 4, 6, 4, 1]."""
        assert leibniz_coefficients(4) == [1, 4, 6, 4, 1]

    def test_sum_equals_2_to_n(self):
        """Sum of coefficients = 2^n."""
        for n in range(10):
            assert sum(leibniz_coefficients(n)) == 2 ** n


class TestRawVsPRZZDerivatives:
    """Verify relationship between raw and PRZZ derivatives."""

    def test_przz_equals_scaled_raw_for_exp_linear(self):
        """D_α^n = (-1/L)^n × d^n/dα^n, so D_α^n [exp] = (-1/L)^n × raw deriv."""
        theta = 4.0 / 7.0
        L = 100.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        for n in range(4):
            for m in range(4):
                przz = deriv_exp_linear_mixed_PRZZ(n, m, alpha, beta, x, y, theta, L)
                raw = deriv_exp_linear_raw(n, m, alpha, beta, x, y, theta, L)

                # D_α = -1/L × d/dα, so D_α^n = (-1/L)^n × d^n/dα^n
                # and D_α^n D_β^m = (-1/L)^{n+m} × d^{n+m}/dα^n dβ^m
                scale = (-1 / L) ** (n + m)
                expected_przz = scale * raw

                assert np.isclose(przz, expected_przz, rtol=1e-10), \
                    f"Failed for n={n}, m={m}"

    def test_przz_equals_scaled_raw_for_mirror(self):
        """Same relationship for mirror exponential."""
        theta = 4.0 / 7.0
        L = 100.0
        alpha, beta = -0.01, -0.01
        x, y = 0.3, 0.4

        for n in range(3):
            for m in range(3):
                przz = deriv_mirror_exp_factor_PRZZ(n, m, alpha, beta, x, y, theta, L)
                raw = deriv_mirror_exp_raw(n, m, alpha, beta, x, y, theta, L)

                scale = (-1 / L) ** (n + m)
                expected_przz = scale * raw

                assert np.isclose(przz, expected_przz, rtol=1e-10), \
                    f"Failed for n={n}, m={m}"


class TestNumericalCrossCheck:
    """Numerical cross-checks for analytic formulas."""

    def test_inverse_sum_numerical_first_derivative(self):
        """Cross-check d/dα [1/(α+β)] with numerical derivative."""
        alpha, beta = 1.0, 2.0
        epsilon = 1e-6

        # Analytic
        analytic = deriv_inverse_sum_at_point(1, 0, alpha, beta)

        # Numerical (central difference)
        f_plus = 1.0 / (alpha + epsilon + beta)
        f_minus = 1.0 / (alpha - epsilon + beta)
        numerical = (f_plus - f_minus) / (2 * epsilon)

        assert np.isclose(analytic, numerical, rtol=1e-5)

    def test_inverse_sum_numerical_second_derivative(self):
        """Cross-check d²/dα² [1/(α+β)] with numerical derivative."""
        alpha, beta = 1.0, 2.0
        epsilon = 1e-5

        # Analytic
        analytic = deriv_inverse_sum_at_point(2, 0, alpha, beta)

        # Numerical (central difference for second derivative)
        f_plus = 1.0 / (alpha + epsilon + beta)
        f_mid = 1.0 / (alpha + beta)
        f_minus = 1.0 / (alpha - epsilon + beta)
        numerical = (f_plus - 2 * f_mid + f_minus) / (epsilon ** 2)

        assert np.isclose(analytic, numerical, rtol=1e-4)

    def test_exp_eigenvalue_numerical(self):
        """Cross-check D_α [exp(θL(αx+βy))] with numerical derivative."""
        theta = 4.0 / 7.0
        L = 10.0  # Smaller L for numerical stability
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4
        epsilon = 1e-6

        # Analytic (PRZZ convention)
        analytic = deriv_exp_linear_alpha_PRZZ(1, alpha, beta, x, y, theta, L)

        # Numerical D_α = -1/L × d/dα
        def f(a):
            return np.exp(theta * L * (a * x + beta * y))

        numerical = -1 / L * (f(alpha + epsilon) - f(alpha - epsilon)) / (2 * epsilon)

        assert np.isclose(analytic, numerical, rtol=1e-5)
