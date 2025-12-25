"""
tests/test_i1_prefactor_product_rule.py
Unit test for I₁ algebraic prefactor product-rule identity.

This test verifies that the series engine correctly computes:
    ∂²ₓᵧ[(x + y + 1/θ) · G(x,y)]₀ = G_x(0) + G_y(0) + (1/θ)G_xy(0)

No PRZZ integrals - pure math test.
"""

import pytest
import numpy as np
import math

from src.series import TruncatedSeries
from src.term_dsl import AffineExpr, SeriesContext


class TestProductRuleIdentity:
    """Tests for the product-rule identity with algebraic prefactor."""

    @pytest.fixture
    def theta(self):
        """Standard θ value."""
        return 4.0 / 7.0

    @pytest.fixture
    def ctx(self):
        """SeriesContext with x and y variables."""
        return SeriesContext(var_names=("x", "y"))

    def test_prefactor_series_structure(self, theta, ctx):
        """Verify the prefactor (x + y + 1/θ) produces correct series."""
        # Create the algebraic prefactor
        prefactor = AffineExpr(
            a0=1.0 / theta,
            var_coeffs={"x": 1.0, "y": 1.0}
        )

        # Create dummy grid (not used for constants)
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        # Convert to series
        series = prefactor.to_series(U, T, ctx)

        # Should have: 1/θ + x + y
        # In TruncatedSeries format: coeffs[()] = 1/θ, coeffs[(0,)] = 1, coeffs[(1,)] = 1
        # where (0,) is x and (1,) is y

        # Extract coefficients
        const_term = series.extract(())  # constant term
        x_term = series.extract(("x",))    # coefficient of x (first var)
        y_term = series.extract(("y",))    # coefficient of y (second var)

        assert pytest.approx(float(const_term.flat[0]), abs=1e-10) == 1.0 / theta
        assert pytest.approx(float(x_term.flat[0]), abs=1e-10) == 1.0
        assert pytest.approx(float(y_term.flat[0]), abs=1e-10) == 1.0

    def test_product_rule_with_linear_G(self, theta, ctx):
        """
        Test: G(x,y) = a·x + b·y + c·x·y
        Then:
            G_x(0) = a
            G_y(0) = b
            G_xy(0) = c
        And:
            ∂²ₓᵧ[(x + y + 1/θ)·G]₀ = G_x(0) + G_y(0) + (1/θ)·G_xy(0)
                                    = a + b + c/θ
        """
        a, b, c = 2.0, 3.0, 5.0

        # Create G(x,y) = a·x + b·y + c·x·y as a TruncatedSeries
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        # Build G directly in series form
        # G = a·x + b·y + c·x·y
        G = ctx.scalar_series(np.zeros_like(U))
        G = G + ctx.variable_series("x") * a
        G = G + ctx.variable_series("y") * b
        G = G + ctx.variable_series("x") * ctx.variable_series("y") * c

        # Verify G structure
        assert pytest.approx(float(G.extract(("x",)).flat[0]), abs=1e-10) == a  # G has a·x
        assert pytest.approx(float(G.extract(("y",)).flat[0]), abs=1e-10) == b  # G has b·y
        assert pytest.approx(float(G.extract(("x", "y")).flat[0]), abs=1e-10) == c  # G has c·xy

        # Create prefactor (x + y + 1/θ)
        prefactor = AffineExpr(
            a0=1.0 / theta,
            var_coeffs={"x": 1.0, "y": 1.0}
        )
        P = prefactor.to_series(U, T, ctx)

        # Multiply: P · G
        PG = P * G

        # Extract ∂²ₓᵧ[PG]₀ = coefficient of x·y
        coeff_xy = PG.extract(("x", "y"))

        # Expected: a + b + c/θ
        expected = a + b + c / theta

        assert pytest.approx(float(coeff_xy.flat[0]), abs=1e-10) == expected, (
            f"Product rule failed: got {float(coeff_xy.flat[0])}, expected {expected}"
        )

    def test_product_rule_with_exponential_G(self, theta, ctx):
        """
        Test: G(x,y) = exp(a·x + b·y) truncated to first order
             = 1 + a·x + b·y + a·b·x·y + O(x²,y²)
        Then:
            G_x(0) = a
            G_y(0) = b
            G_xy(0) = a·b
        And:
            ∂²ₓᵧ[(x + y + 1/θ)·G]₀ = a + b + (a·b)/θ
        """
        a, b = 1.5, 2.5

        U = np.array([[0.5]])
        T = np.array([[0.5]])

        # Build truncated exp(a·x + b·y):
        # = 1 + a·x + b·y + a·b·x·y (to first order in each)
        G = ctx.scalar_series(np.ones_like(U))  # constant term = 1
        G = G + ctx.variable_series("x") * a
        G = G + ctx.variable_series("y") * b
        G = G + ctx.variable_series("x") * ctx.variable_series("y") * (a * b)

        # Verify G
        assert pytest.approx(float(G.extract(()).flat[0]), abs=1e-10) == 1.0
        assert pytest.approx(float(G.extract(("x",)).flat[0]), abs=1e-10) == a
        assert pytest.approx(float(G.extract(("y",)).flat[0]), abs=1e-10) == b
        assert pytest.approx(float(G.extract(("x", "y")).flat[0]), abs=1e-10) == a * b

        # Create prefactor
        prefactor = AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0, "y": 1.0})
        P = prefactor.to_series(U, T, ctx)

        # Multiply
        PG = P * G

        # Extract ∂²ₓᵧ[PG]₀
        coeff_xy = PG.extract(("x", "y"))

        # Expected: a + b + (a·b)/θ
        expected = a + b + (a * b) / theta

        assert pytest.approx(float(coeff_xy.flat[0]), abs=1e-10) == expected, (
            f"Product rule failed: got {float(coeff_xy.flat[0])}, expected {expected}"
        )

    def test_without_prefactor_just_G(self, theta, ctx):
        """
        Control test: Without prefactor, ∂²ₓᵧ[G]₀ = G_xy(0) = c

        This verifies that without the prefactor, we only get the mixed
        derivative term, NOT the G_x and G_y contributions.
        """
        a, b, c = 2.0, 3.0, 5.0

        U = np.array([[0.5]])
        T = np.array([[0.5]])

        # G = a·x + b·y + c·x·y
        G = ctx.scalar_series(np.zeros_like(U))
        G = G + ctx.variable_series("x") * a
        G = G + ctx.variable_series("y") * b
        G = G + ctx.variable_series("x") * ctx.variable_series("y") * c

        # Extract ∂²ₓᵧ[G]₀ directly
        coeff_xy = G.extract(("x", "y"))

        # Should just be c (no a or b contribution)
        expected = c

        assert pytest.approx(float(coeff_xy.flat[0]), abs=1e-10) == expected

    def test_product_rule_ratio(self, theta, ctx):
        """
        Verify the ratio of "with prefactor" to "without prefactor" shows
        the contribution from G_x and G_y.

        ratio = (a + b + c/θ) / (c/θ) = 1 + θ(a + b)/c
        """
        a, b, c = 2.0, 3.0, 5.0

        U = np.array([[0.5]])
        T = np.array([[0.5]])

        # G = a·x + b·y + c·x·y
        G = ctx.scalar_series(np.zeros_like(U))
        G = G + ctx.variable_series("x") * a
        G = G + ctx.variable_series("y") * b
        G = G + ctx.variable_series("x") * ctx.variable_series("y") * c

        # With prefactor
        prefactor = AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0, "y": 1.0})
        P = prefactor.to_series(U, T, ctx)
        PG = P * G
        with_prefactor = float(PG.extract(("x", "y")))

        # Without prefactor (just 1/θ)
        prefactor_const = AffineExpr(a0=1.0 / theta, var_coeffs={})
        P_const = prefactor_const.to_series(U, T, ctx)
        PG_const = P_const * G
        without_full_prefactor = float(PG_const.extract(("x", "y")))

        # Expected values
        expected_with = a + b + c / theta
        expected_without = c / theta

        # Ratio
        ratio = with_prefactor / without_full_prefactor
        expected_ratio = 1 + theta * (a + b) / c

        assert pytest.approx(ratio, rel=1e-10) == expected_ratio, (
            f"Ratio mismatch: got {ratio}, expected {expected_ratio}"
        )

        # Print for diagnostic
        print(f"\nWith prefactor (x+y+1/θ): {with_prefactor:.6f}")
        print(f"Without (just 1/θ):       {without_full_prefactor:.6f}")
        print(f"Ratio:                    {ratio:.6f}")
        print(f"Expected ratio:           {expected_ratio:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
