"""
tests/test_term_dsl.py
TDD tests for the Term DSL (Phase 0, Step 4).

Tests follow GPT guidance:
- Shape invariants are first-class testing targets
- Dtype preservation for complex scalars
- SeriesContext owns canonical var_names
- Missing mask extraction integrates to zero
- Composition module is the only Taylor engine
"""

import numpy as np
import pytest
from typing import Tuple


# =============================================================================
# Test Group A: SeriesContext
# =============================================================================

class TestSeriesContext:
    """SeriesContext owns canonical var_names for all series operations."""

    def test_context_creation(self):
        """Create context with var_names tuple."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))
        assert ctx.var_names == ("x1", "y1")

    def test_context_var_names_immutable(self):
        """var_names should be a tuple (immutable)."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))
        assert isinstance(ctx.var_names, tuple)

    def test_context_creates_zero_series(self):
        """Context can create a zero series with correct var_names."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))
        zero = ctx.zero_series()
        assert zero.var_names == ("x1", "y1")
        np.testing.assert_allclose(zero.extract(()), 0.0)

    def test_context_creates_scalar_series(self):
        """Context can create a scalar series with correct var_names."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))

        # Scalar value
        s = ctx.scalar_series(3.5)
        assert s.var_names == ("x1", "y1")
        np.testing.assert_allclose(s.extract(()), 3.5)

    def test_context_creates_scalar_series_with_grid(self):
        """Context can create a scalar series from grid array."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))

        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = ctx.scalar_series(grid)
        assert s.var_names == ("x1", "y1")
        np.testing.assert_allclose(s.extract(()), grid)

    def test_context_creates_variable_series(self):
        """Context can create a variable series."""
        from src.term_dsl import SeriesContext
        ctx = SeriesContext(var_names=("x1", "y1"))

        x1 = ctx.variable_series("x1")
        assert x1.var_names == ("x1", "y1")
        np.testing.assert_allclose(x1.extract(("x1",)), 1.0)
        np.testing.assert_allclose(x1.extract(("y1",)), 0.0)


# =============================================================================
# Test Group B: AffineExpr evaluation
# =============================================================================

class TestAffineExprEvaluation:
    """AffineExpr evaluates to grid-shaped arrays with dtype preservation."""

    def test_constant_a0_lifts_to_grid_shape(self):
        """Scalar a0 should lift to grid shape."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=2.5, var_coeffs={})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_a0(U, T)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 2.5)

    def test_callable_a0_evaluates_on_grid(self):
        """Callable a0 should evaluate on grid."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=lambda U, T: U + T, var_coeffs={})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_a0(U, T)
        np.testing.assert_allclose(result, U + T)

    def test_constant_coeff_lifts_to_grid_shape(self):
        """Scalar coefficient should lift to grid shape."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=0.0, var_coeffs={"x1": 3.0})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_coeff("x1", U, T)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 3.0)

    def test_callable_coeff_evaluates_on_grid(self):
        """Callable coefficient should evaluate on grid."""
        from src.term_dsl import AffineExpr

        theta = 4/7
        expr = AffineExpr(
            a0=0.0,
            var_coeffs={"x1": lambda U, T: theta * T}
        )
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_coeff("x1", U, T)
        np.testing.assert_allclose(result, theta * T)

    def test_missing_coeff_returns_zero_grid(self):
        """Missing coefficient should return zero with correct shape."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=1.0, var_coeffs={"x1": 2.0})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        # y1 is not in var_coeffs
        result = expr.evaluate_coeff("y1", U, T)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 0.0)

    def test_complex_scalar_preserves_dtype(self):
        """Complex scalar should preserve dtype (not cast to float)."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=1.0 + 2.0j, var_coeffs={})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_a0(U, T)
        assert result.dtype == np.complex128
        np.testing.assert_allclose(result, 1.0 + 2.0j)

    def test_complex_coeff_preserves_dtype(self):
        """Complex coefficient should preserve dtype."""
        from src.term_dsl import AffineExpr

        expr = AffineExpr(a0=0.0, var_coeffs={"x1": 1.0 + 1.0j})
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        result = expr.evaluate_coeff("x1", U, T)
        assert result.dtype == np.complex128
        np.testing.assert_allclose(result, 1.0 + 1.0j)


# =============================================================================
# Test Group C: AffineExpr to series conversion
# =============================================================================

class TestAffineExprToSeries:
    """AffineExpr converts to TruncatedSeries via SeriesContext."""

    def test_affine_to_u0_lin(self):
        """AffineExpr.to_u0_lin returns (u0, lin) for composition."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(
            a0=lambda U, T: T,
            var_coeffs={"x1": 2.0, "y1": lambda U, T: T - 1}
        )

        U = np.array([[0.3, 0.4]])
        T = np.array([[0.5, 0.6]])

        u0, lin = expr.to_u0_lin(U, T, ctx)

        np.testing.assert_allclose(u0, T)
        np.testing.assert_allclose(lin["x1"], 2.0)
        np.testing.assert_allclose(lin["y1"], T - 1)

    def test_affine_to_series_constant_only(self):
        """AffineExpr with only constant becomes scalar series."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(a0=5.0, var_coeffs={})

        U = np.array([[0.3, 0.4]])
        T = np.array([[0.5, 0.6]])

        series = expr.to_series(U, T, ctx)

        assert series.var_names == ("x1", "y1")
        np.testing.assert_allclose(series.extract(()), 5.0)
        np.testing.assert_allclose(series.extract(("x1",)), 0.0)

    def test_affine_to_series_with_variables(self):
        """AffineExpr with variables becomes linear series."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(a0=1.0, var_coeffs={"x1": 2.0, "y1": 3.0})

        U = np.array([[0.3]])
        T = np.array([[0.5]])

        series = expr.to_series(U, T, ctx)

        np.testing.assert_allclose(series.extract(()), 1.0)
        np.testing.assert_allclose(series.extract(("x1",)), 2.0)
        np.testing.assert_allclose(series.extract(("y1",)), 3.0)
        # Cross term is 0 for linear series
        np.testing.assert_allclose(series.extract(("x1", "y1")), 0.0)


# =============================================================================
# Test Group D: PolyFactor and ExpFactor structure
# =============================================================================

class TestFactorStructure:
    """PolyFactor and ExpFactor dataclass structure tests."""

    def test_poly_factor_default_power(self):
        """PolyFactor has default power=1."""
        from src.term_dsl import PolyFactor, AffineExpr

        expr = AffineExpr(a0=0.5, var_coeffs={})
        factor = PolyFactor(poly_name="Q", argument=expr)

        assert factor.poly_name == "Q"
        assert factor.power == 1

    def test_poly_factor_explicit_power(self):
        """PolyFactor can have explicit power."""
        from src.term_dsl import PolyFactor, AffineExpr

        expr = AffineExpr(a0=0.5, var_coeffs={})
        factor = PolyFactor(poly_name="Q", argument=expr, power=2)

        assert factor.power == 2

    def test_exp_factor_structure(self):
        """ExpFactor has scale and argument."""
        from src.term_dsl import ExpFactor, AffineExpr

        R = 1.3036
        expr = AffineExpr(a0=0.5, var_coeffs={"x1": 1.0})
        factor = ExpFactor(scale=2*R, argument=expr)

        assert factor.scale == 2 * R
        assert factor.argument == expr


# =============================================================================
# Test Group E: Term structure
# =============================================================================

class TestTermStructure:
    """Term dataclass structure and helper methods."""

    def test_term_total_vars(self):
        """Term.total_vars returns number of formal variables."""
        from src.term_dsl import Term, AffineExpr

        term = Term(
            name="test",
            pair=(1, 2),
            przz_reference=None,
            vars=("x1", "y1", "y2"),
            deriv_orders={"x1": 1, "y1": 1, "y2": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.total_vars() == 3

    def test_term_target_mask(self):
        """Term.target_mask returns all-1s bitmask for d=1."""
        from src.term_dsl import Term

        # 2 vars -> mask = 0b11 = 3
        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.target_mask() == 3  # 0b11

    def test_term_target_mask_3vars(self):
        """Term.target_mask for 3 variables."""
        from src.term_dsl import Term

        # 3 vars -> mask = 0b111 = 7
        term = Term(
            name="test",
            pair=(1, 2),
            przz_reference=None,
            vars=("x1", "y1", "y2"),
            deriv_orders={"x1": 1, "y1": 1, "y2": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.target_mask() == 7  # 0b111


# =============================================================================
# Test Group F: Factor evaluation (uses composition.py)
# =============================================================================

class TestFactorEvaluation:
    """Factor evaluation uses composition.py, not reimplemented Taylor loops."""

    def test_poly_factor_evaluates_via_composition(self):
        """PolyFactor evaluation calls compose_polynomial_on_affine."""
        from src.term_dsl import PolyFactor, AffineExpr, SeriesContext
        from src.polynomials import Polynomial

        ctx = SeriesContext(var_names=("x1", "y1"))

        # P(x) = 1 + 2x + 3x^2
        poly = Polynomial([1.0, 2.0, 3.0])

        # Argument: u + 2*x1 + 3*y1
        expr = AffineExpr(
            a0=lambda U, T: U,
            var_coeffs={"x1": 2.0, "y1": 3.0}
        )
        factor = PolyFactor(poly_name="test", argument=expr)

        U = np.array([[0.5]])
        T = np.array([[0.3]])

        # Evaluate factor
        series = factor.evaluate(poly, U, T, ctx)

        # Check results match compose_polynomial_on_affine
        from src.composition import compose_polynomial_on_affine
        u0, lin = expr.to_u0_lin(U, T, ctx)
        expected = compose_polynomial_on_affine(poly, u0, lin, ctx.var_names)

        np.testing.assert_allclose(series.extract(()), expected.extract(()))
        np.testing.assert_allclose(series.extract(("x1",)), expected.extract(("x1",)))
        np.testing.assert_allclose(series.extract(("y1",)), expected.extract(("y1",)))
        np.testing.assert_allclose(series.extract(("x1", "y1")), expected.extract(("x1", "y1")))

    def test_poly_factor_power_multiplies_series(self):
        """PolyFactor with power>1 multiplies series repeatedly."""
        from src.term_dsl import PolyFactor, AffineExpr, SeriesContext
        from src.polynomials import Polynomial

        ctx = SeriesContext(var_names=("x1",))

        # P(x) = 1 + x (linear)
        poly = Polynomial([1.0, 1.0])

        # Argument: u + x1
        expr = AffineExpr(a0=0.5, var_coeffs={"x1": 1.0})
        factor = PolyFactor(poly_name="test", argument=expr, power=2)

        U = np.array([[0.5]])
        T = np.array([[0.3]])

        # P(0.5 + x1)^2 = (1.5 + x1)^2 = 2.25 + 3*x1 + x1^2
        # But x1^2 = 0 (nilpotent), so result = 2.25 + 3*x1
        series = factor.evaluate(poly, U, T, ctx)

        np.testing.assert_allclose(series.extract(()), 2.25)
        np.testing.assert_allclose(series.extract(("x1",)), 3.0)

    def test_exp_factor_evaluates_via_series_exp(self):
        """ExpFactor evaluation builds affine series and calls .exp()."""
        from src.term_dsl import ExpFactor, AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        R = 1.0

        # Argument: t + x1 + y1
        expr = AffineExpr(
            a0=lambda U, T: T,
            var_coeffs={"x1": 1.0, "y1": 1.0}
        )
        factor = ExpFactor(scale=R, argument=expr)

        U = np.array([[0.5]])
        T = np.array([[0.3]])

        series = factor.evaluate(U, T, ctx)

        # exp(R*(t + x1 + y1)) at t=0.3
        # = exp(0.3) * exp(x1 + y1)
        # = exp(0.3) * (1 + x1 + y1 + x1*y1)
        exp_t = np.exp(0.3)
        np.testing.assert_allclose(series.extract(()), exp_t, rtol=1e-10)
        np.testing.assert_allclose(series.extract(("x1",)), exp_t, rtol=1e-10)
        np.testing.assert_allclose(series.extract(("y1",)), exp_t, rtol=1e-10)
        np.testing.assert_allclose(series.extract(("x1", "y1")), exp_t, rtol=1e-10)


# =============================================================================
# Test Group G: Missing mask and zero integration
# =============================================================================

class TestMissingMaskIntegration:
    """Missing mask extraction should integrate to zero cleanly."""

    def test_missing_mask_extracts_to_zero(self):
        """Extract of missing mask returns 0 with correct shape."""
        from src.term_dsl import SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1", "z1"))

        # Create series that only has x1 term
        series = ctx.zero_series()
        series = series + ctx.variable_series("x1") * 2.0

        # y1 and z1 terms should be zero
        np.testing.assert_allclose(series.extract(("y1",)), 0.0)
        np.testing.assert_allclose(series.extract(("z1",)), 0.0)

        # Cross terms should be zero
        np.testing.assert_allclose(series.extract(("x1", "y1")), 0.0)
        np.testing.assert_allclose(series.extract(("x1", "y1", "z1")), 0.0)

    def test_zero_coefficient_integrates_cleanly(self):
        """Zero coefficient array integrates to 0 without shape errors."""
        from src.term_dsl import SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))

        # Grid and weights
        U = np.linspace(0.1, 0.9, 5)
        W = np.ones(5) / 5  # Uniform weights

        # Series with only constant term
        series = ctx.scalar_series(np.ones(5) * 3.0)

        # x1*y1 coefficient should be zero but have shape (5,)
        coeff = series.extract(("x1", "y1"))

        # Integration should work and give 0
        integral = np.sum(W * coeff)
        np.testing.assert_allclose(integral, 0.0)


# =============================================================================
# Test Group H: Phase 0 d=1 enforcement
# =============================================================================

class TestD1Enforcement:
    """Phase 0 enforces d=1: all derivative orders must be 0 or 1."""

    def test_d1_order_1_accepted(self):
        """Derivative order 1 is valid for d=1."""
        from src.term_dsl import Term

        # Should not raise
        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )
        assert term.total_vars() == 2

    def test_d1_order_0_accepted(self):
        """Derivative order 0 is valid (no derivative on that variable)."""
        from src.term_dsl import Term

        # Should not raise - order 0 means no derivative
        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 0},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )
        assert term.deriv_orders["y1"] == 0

    def test_d1_order_2_rejected(self):
        """Derivative order > 1 is rejected in Phase 0."""
        from src.term_dsl import Term

        with pytest.raises(ValueError, match="order > 1 not supported"):
            Term(
                name="test",
                pair=(1, 1),
                przz_reference=None,
                vars=("x1", "y1"),
                deriv_orders={"x1": 2, "y1": 1},  # x1 has order 2 - invalid
                domain="[0,1]^2",
                numeric_prefactor=1.0,
                algebraic_prefactor=None,
                poly_prefactors=[],
                poly_factors=[],
                exp_factors=[]
            )

    def test_d1_negative_order_rejected(self):
        """Negative derivative orders are rejected."""
        from src.term_dsl import Term

        with pytest.raises(ValueError, match="non-negative"):
            Term(
                name="test",
                pair=(1, 1),
                przz_reference=None,
                vars=("x1", "y1"),
                deriv_orders={"x1": -1, "y1": 1},  # negative - invalid
                domain="[0,1]^2",
                numeric_prefactor=1.0,
                algebraic_prefactor=None,
                poly_prefactors=[],
                poly_factors=[],
                exp_factors=[]
            )

    def test_term_create_context(self):
        """Term.create_context() returns SeriesContext with term's vars."""
        from src.term_dsl import Term, SeriesContext

        term = Term(
            name="test",
            pair=(1, 2),
            przz_reference=None,
            vars=("x1", "y1", "y2"),
            deriv_orders={"x1": 1, "y1": 1, "y2": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        ctx = term.create_context()
        assert isinstance(ctx, SeriesContext)
        assert ctx.var_names == ("x1", "y1", "y2")


# =============================================================================
# Test Group I: Failure modes and robustness
# =============================================================================

class TestFailureModes:
    """Test that failure modes produce clear errors, not silent bugs."""

    def test_callable_returns_wrong_shape_raises(self):
        """Callable returning wrong shape should raise ValueError."""
        from src.term_dsl import AffineExpr, SeriesContext

        # Callable returns (n,) instead of (n,n)
        expr = AffineExpr(
            a0=lambda U, T: np.sum(U, axis=1),  # Returns (n,) not (n,n)
            var_coeffs={}
        )
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        with pytest.raises(ValueError, match="returned shape.*expected"):
            expr.evaluate_a0(U, T)

    def test_coeff_callable_returns_wrong_shape_raises(self):
        """Coefficient callable returning wrong shape should raise ValueError."""
        from src.term_dsl import AffineExpr, SeriesContext

        # Coefficient returns scalar array when grid expected
        expr = AffineExpr(
            a0=1.0,
            var_coeffs={"x1": lambda U, T: np.array([1.0, 2.0])}  # Wrong shape
        )
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        with pytest.raises(ValueError, match="returned shape.*expected"):
            expr.evaluate_coeff("x1", U, T)

    def test_affine_var_not_in_context_raises_on_to_u0_lin(self):
        """AffineExpr with var not in ctx.var_names should raise on to_u0_lin."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(
            a0=1.0,
            var_coeffs={"x1": 2.0, "z1": 3.0}  # z1 not in context
        )
        U = np.array([[0.3]])
        T = np.array([[0.5]])

        with pytest.raises(ValueError, match="z1.*not in.*var_names"):
            expr.to_u0_lin(U, T, ctx)

    def test_affine_var_not_in_context_raises_on_to_series(self):
        """AffineExpr with var not in ctx.var_names should raise on to_series."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(
            a0=1.0,
            var_coeffs={"bad_var": 2.0}  # bad_var not in context
        )
        U = np.array([[0.3]])
        T = np.array([[0.5]])

        with pytest.raises(ValueError, match="bad_var.*not in.*var_names"):
            expr.to_series(U, T, ctx)


# =============================================================================
# Test Group J: Zero-pruning behavior
# =============================================================================

class TestZeroPruning:
    """Test that zero coefficients are pruned from lin and series."""

    def test_zero_coeff_excluded_from_lin(self):
        """Zero coefficient should not appear in lin dict."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1", "z1"))
        expr = AffineExpr(
            a0=1.0,
            var_coeffs={
                "x1": 2.0,
                "y1": 0.0,  # Identically zero - should be pruned
                "z1": 3.0
            }
        )
        U = np.array([[0.3, 0.4]])
        T = np.array([[0.5, 0.6]])

        u0, lin = expr.to_u0_lin(U, T, ctx)

        # y1 should be excluded from lin
        assert "x1" in lin
        assert "y1" not in lin  # Pruned!
        assert "z1" in lin
        assert len(lin) == 2

    def test_zero_coeff_array_excluded_from_lin(self):
        """Zero coefficient array should not appear in lin dict."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(
            a0=lambda U, T: T,
            var_coeffs={
                "x1": lambda U, T: U,  # Non-zero
                "y1": lambda U, T: np.zeros_like(U)  # All zeros
            }
        )
        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.1, 0.2], [0.3, 0.4]])

        u0, lin = expr.to_u0_lin(U, T, ctx)

        assert "x1" in lin
        assert "y1" not in lin  # Pruned - all zeros
        assert len(lin) == 1

    def test_zero_coeff_not_added_to_series(self):
        """Zero coefficient should not add term to series."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1", "y1"))
        expr = AffineExpr(
            a0=5.0,
            var_coeffs={
                "x1": 2.0,
                "y1": 0.0  # Should be pruned
            }
        )
        U = np.array([[0.3]])
        T = np.array([[0.5]])

        series = expr.to_series(U, T, ctx)

        # Only constant and x1 terms should exist
        np.testing.assert_allclose(series.extract(()), 5.0)
        np.testing.assert_allclose(series.extract(("x1",)), 2.0)
        np.testing.assert_allclose(series.extract(("y1",)), 0.0)

        # Verify y1 mask not in coeffs dict (truly pruned)
        y1_mask = 1 << ctx.var_names.index("y1")
        assert y1_mask not in series.coeffs

    def test_precomputed_array_accepted(self):
        """Precomputed np.ndarray should be accepted as GridFunc."""
        from src.term_dsl import AffineExpr, SeriesContext

        ctx = SeriesContext(var_names=("x1",))
        U = np.array([[0.1, 0.2], [0.3, 0.4]])
        T = np.array([[0.5, 0.6], [0.7, 0.8]])

        # Pass precomputed array directly
        precomputed = U * 2 + T
        expr = AffineExpr(
            a0=precomputed,  # np.ndarray, not callable
            var_coeffs={"x1": 1.0}
        )

        result = expr.evaluate_a0(U, T)
        np.testing.assert_allclose(result, precomputed)


# =============================================================================
# Test Group K: Term vars/deriv_orders consistency
# =============================================================================

class TestTermVarsDerivOrdersConsistency:
    """Test that vars and deriv_orders must match exactly."""

    def test_deriv_orders_extra_var_rejected(self):
        """deriv_orders with var not in vars should raise."""
        from src.term_dsl import Term

        with pytest.raises(ValueError, match="not in vars"):
            Term(
                name="test",
                pair=(1, 1),
                przz_reference=None,
                vars=("x1", "y1"),
                deriv_orders={"x1": 1, "y1": 1, "z1": 1},  # z1 not in vars
                domain="[0,1]^2",
                numeric_prefactor=1.0,
                algebraic_prefactor=None,
                poly_prefactors=[],
                poly_factors=[],
                exp_factors=[]
            )

    def test_deriv_orders_missing_var_rejected(self):
        """vars with var not in deriv_orders should raise."""
        from src.term_dsl import Term

        with pytest.raises(ValueError, match="not in deriv_orders"):
            Term(
                name="test",
                pair=(1, 1),
                przz_reference=None,
                vars=("x1", "y1"),
                deriv_orders={"x1": 1},  # y1 missing
                domain="[0,1]^2",
                numeric_prefactor=1.0,
                algebraic_prefactor=None,
                poly_prefactors=[],
                poly_factors=[],
                exp_factors=[]
            )

    def test_vars_deriv_orders_match_accepted(self):
        """Matching vars and deriv_orders should be accepted."""
        from src.term_dsl import Term

        # Should not raise
        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )
        assert term.total_vars() == 2


# =============================================================================
# Test Group L: Term.deriv_tuple() helper
# =============================================================================

class TestTermDerivTuple:
    """Test deriv_tuple() returns correct tuple for series.extract()."""

    def test_deriv_tuple_all_order_1(self):
        """deriv_tuple returns all vars when all have order 1."""
        from src.term_dsl import Term

        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.deriv_tuple() == ("x1", "y1")

    def test_deriv_tuple_mixed_orders(self):
        """deriv_tuple returns only vars with order 1."""
        from src.term_dsl import Term

        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 1, "y1": 0},  # y1 has order 0
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.deriv_tuple() == ("x1",)

    def test_deriv_tuple_preserves_order(self):
        """deriv_tuple preserves canonical order from vars."""
        from src.term_dsl import Term

        term = Term(
            name="test",
            pair=(1, 2),
            przz_reference=None,
            vars=("x1", "y1", "y2"),
            deriv_orders={"x1": 1, "y1": 1, "y2": 1},
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        # Should be in same order as vars
        assert term.deriv_tuple() == ("x1", "y1", "y2")

    def test_deriv_tuple_empty_when_all_zero(self):
        """deriv_tuple returns empty tuple when all orders are 0."""
        from src.term_dsl import Term

        term = Term(
            name="test",
            pair=(1, 1),
            przz_reference=None,
            vars=("x1", "y1"),
            deriv_orders={"x1": 0, "y1": 0},  # No derivatives
            domain="[0,1]^2",
            numeric_prefactor=1.0,
            algebraic_prefactor=None,
            poly_prefactors=[],
            poly_factors=[],
            exp_factors=[]
        )

        assert term.deriv_tuple() == ()
