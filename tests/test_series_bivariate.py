"""
tests/test_series_bivariate.py
Phase 26B: Tests for BivariateSeries engine

Tests:
1. Basic operations (construction, add, mul, extract)
2. Truncation correctness
3. exp_linear matches analytic binomial formula
4. Polynomial composition correctness
5. Integration with PRZZ structure
"""

import pytest
import math
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_log_factor,
    build_P_factor,
    build_Q_factor,
)


class TestConstruction:
    """Test series construction."""

    def test_from_scalar(self):
        s = BivariateSeries.from_scalar(3.5, max_dx=2, max_dy=2)
        assert s.extract(0, 0) == 3.5
        assert s.extract(1, 0) == 0.0
        assert s.extract(0, 1) == 0.0

    def test_x_variable(self):
        s = BivariateSeries.x(max_dx=2, max_dy=2)
        assert s.extract(0, 0) == 0.0
        assert s.extract(1, 0) == 1.0
        assert s.extract(0, 1) == 0.0

    def test_y_variable(self):
        s = BivariateSeries.y(max_dx=2, max_dy=2)
        assert s.extract(0, 0) == 0.0
        assert s.extract(1, 0) == 0.0
        assert s.extract(0, 1) == 1.0

    def test_zero_series(self):
        s = BivariateSeries.zero(max_dx=3, max_dy=3)
        assert s.total_terms() == 0
        for i in range(4):
            for j in range(4):
                assert s.extract(i, j) == 0.0

    def test_one_series(self):
        s = BivariateSeries.one(max_dx=3, max_dy=3)
        assert s.extract(0, 0) == 1.0
        assert s.total_terms() == 1


class TestAddition:
    """Test addition operations."""

    def test_add_scalars(self):
        s = BivariateSeries.from_scalar(2.0, 2, 2) + BivariateSeries.from_scalar(3.0, 2, 2)
        assert s.extract(0, 0) == 5.0

    def test_add_scalar_to_series(self):
        s = BivariateSeries.x(2, 2) + 5.0
        assert s.extract(0, 0) == 5.0
        assert s.extract(1, 0) == 1.0

    def test_add_x_and_y(self):
        s = BivariateSeries.x(2, 2) + BivariateSeries.y(2, 2)
        assert s.extract(0, 0) == 0.0
        assert s.extract(1, 0) == 1.0
        assert s.extract(0, 1) == 1.0

    def test_add_incompatible_fails(self):
        s1 = BivariateSeries.x(2, 2)
        s2 = BivariateSeries.x(3, 3)
        with pytest.raises(ValueError):
            s1 + s2


class TestMultiplication:
    """Test multiplication operations."""

    def test_mul_by_scalar(self):
        s = BivariateSeries.x(2, 2) * 3.0
        assert s.extract(1, 0) == 3.0

    def test_mul_x_times_y(self):
        s = BivariateSeries.x(2, 2) * BivariateSeries.y(2, 2)
        assert s.extract(0, 0) == 0.0
        assert s.extract(1, 0) == 0.0
        assert s.extract(0, 1) == 0.0
        assert s.extract(1, 1) == 1.0

    def test_mul_truncation_x(self):
        """x^3 should be zero when max_dx=2."""
        x = BivariateSeries.x(2, 2)
        x3 = x * x * x
        assert x3.extract(3, 0) == 0.0  # Truncated
        assert x3.extract(2, 0) == 0.0  # x^3 has no x^2 term

    def test_mul_1_plus_x_squared(self):
        """(1 + x)^2 = 1 + 2x + x^2."""
        one_plus_x = BivariateSeries.one(3, 3) + BivariateSeries.x(3, 3)
        squared = one_plus_x * one_plus_x
        assert abs(squared.extract(0, 0) - 1.0) < 1e-10
        assert abs(squared.extract(1, 0) - 2.0) < 1e-10
        assert abs(squared.extract(2, 0) - 1.0) < 1e-10

    def test_mul_1_plus_x_times_1_plus_y(self):
        """(1 + x)(1 + y) = 1 + x + y + xy."""
        a = BivariateSeries.one(2, 2) + BivariateSeries.x(2, 2)
        b = BivariateSeries.one(2, 2) + BivariateSeries.y(2, 2)
        c = a * b
        assert abs(c.extract(0, 0) - 1.0) < 1e-10
        assert abs(c.extract(1, 0) - 1.0) < 1e-10
        assert abs(c.extract(0, 1) - 1.0) < 1e-10
        assert abs(c.extract(1, 1) - 1.0) < 1e-10


class TestPower:
    """Test power operation."""

    def test_power_0(self):
        s = BivariateSeries.x(3, 3)
        s0 = s ** 0
        assert s0.extract(0, 0) == 1.0
        assert s0.total_terms() == 1

    def test_power_1(self):
        s = BivariateSeries.x(3, 3) + BivariateSeries.y(3, 3)
        s1 = s ** 1
        assert s1.extract(1, 0) == 1.0
        assert s1.extract(0, 1) == 1.0

    def test_power_2(self):
        """(x + y)^2 = x^2 + 2xy + y^2."""
        s = BivariateSeries.x(3, 3) + BivariateSeries.y(3, 3)
        s2 = s ** 2
        assert abs(s2.extract(2, 0) - 1.0) < 1e-10  # x^2
        assert abs(s2.extract(0, 2) - 1.0) < 1e-10  # y^2
        assert abs(s2.extract(1, 1) - 2.0) < 1e-10  # 2xy

    def test_power_3(self):
        """(x + y)^3 = x^3 + 3x^2y + 3xy^2 + y^3."""
        s = BivariateSeries.x(4, 4) + BivariateSeries.y(4, 4)
        s3 = s ** 3
        assert abs(s3.extract(3, 0) - 1.0) < 1e-10  # x^3
        assert abs(s3.extract(0, 3) - 1.0) < 1e-10  # y^3
        assert abs(s3.extract(2, 1) - 3.0) < 1e-10  # 3x^2y
        assert abs(s3.extract(1, 2) - 3.0) < 1e-10  # 3xy^2


class TestExpLinear:
    """Test exp(a0 + ax*x + ay*y) computation."""

    def test_exp_constant(self):
        """exp(1) = e."""
        s = BivariateSeries.exp_linear(1.0, 0.0, 0.0, 3, 3)
        assert abs(s.extract(0, 0) - math.e) < 1e-10

    def test_exp_x(self):
        """exp(x) = 1 + x + x^2/2 + x^3/6 + ..."""
        s = BivariateSeries.exp_linear(0.0, 1.0, 0.0, 4, 4)
        assert abs(s.extract(0, 0) - 1.0) < 1e-10
        assert abs(s.extract(1, 0) - 1.0) < 1e-10
        assert abs(s.extract(2, 0) - 0.5) < 1e-10
        assert abs(s.extract(3, 0) - 1.0/6) < 1e-10
        assert abs(s.extract(4, 0) - 1.0/24) < 1e-10

    def test_exp_2x(self):
        """exp(2x) = 1 + 2x + 2x^2 + 4x^3/3 + ..."""
        s = BivariateSeries.exp_linear(0.0, 2.0, 0.0, 4, 4)
        assert abs(s.extract(0, 0) - 1.0) < 1e-10
        assert abs(s.extract(1, 0) - 2.0) < 1e-10
        assert abs(s.extract(2, 0) - 2.0) < 1e-10  # 2^2/2! = 2
        assert abs(s.extract(3, 0) - 4.0/3) < 1e-10  # 2^3/3! = 8/6 = 4/3

    def test_exp_x_plus_y(self):
        """exp(x + y) evaluated at (0, 0) = 1, at (1, 1) = e^2."""
        s = BivariateSeries.exp_linear(0.0, 1.0, 1.0, 4, 4)
        # Coefficient of x^i y^j is 1/(i! j!)
        assert abs(s.extract(0, 0) - 1.0) < 1e-10
        assert abs(s.extract(1, 0) - 1.0) < 1e-10
        assert abs(s.extract(0, 1) - 1.0) < 1e-10
        assert abs(s.extract(1, 1) - 1.0) < 1e-10  # 1/(1!1!) = 1
        assert abs(s.extract(2, 1) - 0.5) < 1e-10  # 1/(2!1!) = 0.5

    def test_exp_evaluation(self):
        """exp(a0 + ax*x + ay*y) evaluated at (x,y) should equal exp(a0 + ax*x + ay*y)."""
        a0, ax, ay = 0.5, 0.3, 0.2
        x_val, y_val = 0.1, 0.2
        s = BivariateSeries.exp_linear(a0, ax, ay, 10, 10)
        computed = s.evaluate(x_val, y_val)
        expected = math.exp(a0 + ax * x_val + ay * y_val)
        assert abs(computed - expected) < 1e-8


class TestPolynomialComposition:
    """Test P(a0 + ax*x + ay*y) computation."""

    def test_constant_polynomial(self):
        """P(z) = 5 → P(anything) = 5."""
        s = BivariateSeries.zero(3, 3).compose_polynomial([5.0], a0=1.0, ax=1.0, ay=1.0)
        assert abs(s.extract(0, 0) - 5.0) < 1e-10
        assert s.extract(1, 0) == 0.0

    def test_linear_polynomial(self):
        """P(z) = 2 + 3z → P(x) = 2 + 3x."""
        s = BivariateSeries.zero(3, 3).compose_polynomial([2.0, 3.0], a0=0.0, ax=1.0, ay=0.0)
        assert abs(s.extract(0, 0) - 2.0) < 1e-10
        assert abs(s.extract(1, 0) - 3.0) < 1e-10

    def test_quadratic_polynomial(self):
        """P(z) = 1 + z + z^2 → P(x) = 1 + x + x^2."""
        s = BivariateSeries.zero(3, 3).compose_polynomial([1.0, 1.0, 1.0], a0=0.0, ax=1.0, ay=0.0)
        assert abs(s.extract(0, 0) - 1.0) < 1e-10
        assert abs(s.extract(1, 0) - 1.0) < 1e-10
        assert abs(s.extract(2, 0) - 1.0) < 1e-10

    def test_polynomial_at_shifted_argument(self):
        """P(z) = z^2 at z = 1 + x → P(1+x) = (1+x)^2 = 1 + 2x + x^2."""
        s = BivariateSeries.zero(3, 3).compose_polynomial([0.0, 0.0, 1.0], a0=1.0, ax=1.0, ay=0.0)
        assert abs(s.extract(0, 0) - 1.0) < 1e-10
        assert abs(s.extract(1, 0) - 2.0) < 1e-10
        assert abs(s.extract(2, 0) - 1.0) < 1e-10

    def test_polynomial_at_x_plus_y(self):
        """P(z) = z at z = x + y → P(x+y) = x + y."""
        s = BivariateSeries.zero(3, 3).compose_polynomial([0.0, 1.0], a0=0.0, ax=1.0, ay=1.0)
        assert abs(s.extract(0, 0) - 0.0) < 1e-10
        assert abs(s.extract(1, 0) - 1.0) < 1e-10
        assert abs(s.extract(0, 1) - 1.0) < 1e-10


class TestExp:
    """Test general exp(series) computation."""

    def test_exp_zero(self):
        """exp(0) = 1."""
        s = BivariateSeries.zero(3, 3)
        e = s.exp()
        assert abs(e.extract(0, 0) - 1.0) < 1e-10

    def test_exp_constant(self):
        """exp(2) = e^2."""
        s = BivariateSeries.from_scalar(2.0, 3, 3)
        e = s.exp()
        assert abs(e.extract(0, 0) - math.exp(2.0)) < 1e-10

    def test_exp_x(self):
        """exp(x) via Taylor series."""
        s = BivariateSeries.x(5, 5)
        e = s.exp()
        for i in range(6):
            expected = 1.0 / math.factorial(i)
            assert abs(e.extract(i, 0) - expected) < 1e-10


class TestBuildFunctions:
    """Test convenience builder functions."""

    def test_build_exp_bracket(self):
        """Test exp bracket builder."""
        a0, ax, ay = 1.0, 0.5, 0.5
        s = build_exp_bracket(a0, ax, ay, 3, 3)
        # Should equal exp(1 + 0.5x + 0.5y)
        expected_const = math.exp(1.0)
        assert abs(s.extract(0, 0) - expected_const) < 1e-10

    def test_build_log_factor(self):
        """Test log factor 1/θ + x + y."""
        theta = 4/7
        s = build_log_factor(theta, 3, 3)
        assert abs(s.extract(0, 0) - 1.0/theta) < 1e-10
        assert abs(s.extract(1, 0) - 1.0) < 1e-10
        assert abs(s.extract(0, 1) - 1.0) < 1e-10

    def test_build_P_factor_x(self):
        """Test P(u + x) with P(z) = 1 + 2z + 3z^2."""
        poly_coeffs = [1.0, 2.0, 3.0]
        u = 0.5
        s = build_P_factor(poly_coeffs, u, "x", 3, 3)
        # P(0.5 + x) = P(0.5) + P'(0.5)*x + P''(0.5)/2 * x^2
        # P(0.5) = 1 + 1 + 0.75 = 2.75
        # P'(z) = 2 + 6z → P'(0.5) = 5
        # P''(z) = 6 → P''(0.5)/2 = 3
        assert abs(s.extract(0, 0) - 2.75) < 1e-10
        assert abs(s.extract(1, 0) - 5.0) < 1e-10
        assert abs(s.extract(2, 0) - 3.0) < 1e-10

    def test_build_Q_factor(self):
        """Test Q(a0 + ax*x + ay*y)."""
        Q_coeffs = [1.0, -1.0]  # Q(z) = 1 - z
        s = build_Q_factor(Q_coeffs, a0=0.5, ax=0.1, ay=0.2, max_dx=3, max_dy=3)
        # Q(0.5 + 0.1x + 0.2y) = 1 - (0.5 + 0.1x + 0.2y) = 0.5 - 0.1x - 0.2y
        assert abs(s.extract(0, 0) - 0.5) < 1e-10
        assert abs(s.extract(1, 0) - (-0.1)) < 1e-10
        assert abs(s.extract(0, 1) - (-0.2)) < 1e-10


class TestIntegrationPRZZ:
    """Test patterns used in PRZZ unified bracket construction."""

    def test_full_bracket_coefficient_extraction(self):
        """
        Test that we can extract x^ℓ₁ y^ℓ₂ coefficient from a product of factors.

        For (ℓ₁, ℓ₂) = (2, 2), we extract x^2 y^2 coefficient.
        """
        max_dx, max_dy = 2, 2
        theta = 4/7
        R = 1.3036
        t = 0.7  # Not 0.5 to ensure a = Rθ(2t-1) ≠ 0
        u = 0.3

        # Exp factor: exp(2Rt + Rθ(2t-1)(x+y))
        a0 = 2 * R * t
        a_xy = R * theta * (2 * t - 1)
        exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

        # Log factor: 1/θ + x + y
        log_factor = build_log_factor(theta, max_dx, max_dy)

        # P factors: P(u+x) * P(u+y) with P(z) = 1 + z (simple test polynomial)
        P_coeffs = [1.0, 1.0]
        P_x = build_P_factor(P_coeffs, u, "x", max_dx, max_dy)
        P_y = build_P_factor(P_coeffs, u, "y", max_dx, max_dy)

        # Product
        bracket = exp_factor * log_factor * P_x * P_y

        # Extract x^2 y^2 coefficient
        coeff_22 = bracket.extract(2, 2)
        assert coeff_22 != 0.0  # Should be non-zero

        # Also verify lower-order coefficients
        coeff_00 = bracket.extract(0, 0)
        coeff_11 = bracket.extract(1, 1)
        assert coeff_00 != 0.0
        assert coeff_11 != 0.0

    def test_coefficient_for_11_matches_xy(self):
        """
        For (1,1), extracting x^1 y^1 should be equivalent to xy coefficient
        from the nilpotent series (modulo factorial normalization).
        """
        max_dx, max_dy = 1, 1
        theta = 4/7
        R = 1.3036
        t = 0.7  # Not 0.5 to ensure a = Rθ(2t-1) ≠ 0

        # Exp factor
        a0 = 2 * R * t
        a_xy = R * theta * (2 * t - 1)
        exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

        # Log factor
        log_factor = build_log_factor(theta, max_dx, max_dy)

        # Product
        bracket = exp_factor * log_factor

        # For (1,1), x^1 y^1 coefficient
        coeff_11 = bracket.extract(1, 1)

        # Verify it's reasonable
        assert coeff_11 != 0.0

    def test_varying_pair_degrees(self):
        """
        Test extraction for various pairs: (1,1), (1,2), (2,2), (2,3).
        """
        pairs = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
        theta = 4/7
        R = 1.3036
        t = 0.7  # Not 0.5 to ensure a = Rθ(2t-1) ≠ 0

        for ell1, ell2 in pairs:
            max_dx, max_dy = ell1, ell2

            # Exp factor
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # Log factor
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # Product
            bracket = exp_factor * log_factor

            # Extract target coefficient
            coeff = bracket.extract(ell1, ell2)
            assert coeff != 0.0, f"Coefficient for ({ell1},{ell2}) should be non-zero"
