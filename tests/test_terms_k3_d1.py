"""
tests/test_terms_k3_d1.py
Tests for (1,1) I₁ term - the first and most critical term.

Test strategy:
1. Verify term structure is correct
2. Verify α and β arguments are DISTINCT
3. Symbolic sanity with toy polynomials at fixed (u,t)
4. Shape invariants for all coefficient arrays
5. Quadrature convergence at n=40,60,80

Only after these pass should I₂/I₃/I₄ be implemented.
"""

import numpy as np
import pytest
from typing import Tuple


# =============================================================================
# Test Group A: Term Structure
# =============================================================================

class TestI1TermStructure:
    """Verify I₁ term is constructed correctly."""

    def test_I1_11_vars_correct(self):
        """I₁ should have vars = ('x1', 'y1')."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert term.vars == ("x1", "y1")

    def test_I1_11_deriv_orders_correct(self):
        """I₁ should have deriv_orders = {'x1': 1, 'y1': 1}."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert term.deriv_orders == {"x1": 1, "y1": 1}

    def test_I1_11_deriv_tuple_correct(self):
        """deriv_tuple() should return ('x1', 'y1')."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert term.deriv_tuple() == ("x1", "y1")

    def test_I1_11_numeric_prefactor_is_1(self):
        """I₁ for (1,1) has pair sign +1 and no additional sign."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert term.numeric_prefactor == 1.0

    def test_I1_11_has_algebraic_prefactor(self):
        """I₁ should have algebraic prefactor (θS+1)/θ."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert term.algebraic_prefactor is not None

    def test_I1_11_has_poly_prefactor(self):
        """I₁ should have poly prefactor (1-u)²."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert len(term.poly_prefactors) == 1

    def test_I1_11_has_4_poly_factors(self):
        """I₁ should have 4 poly factors: P₁(x+u), P₁(y+u), Q(α), Q(β)."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert len(term.poly_factors) == 4

    def test_I1_11_has_2_exp_factors(self):
        """I₁ should have 2 exp factors: exp(R·α), exp(R·β)."""
        from src.terms_k3_d1 import make_I1_11

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)

        assert len(term.exp_factors) == 2


# =============================================================================
# Test Group B: α/β Arguments are DISTINCT
# =============================================================================

class TestAlphaBetaDistinct:
    """Verify Q_arg_alpha and Q_arg_beta are NOT equal."""

    def test_alpha_beta_constant_terms_equal(self):
        """Both α and β have constant term = t."""
        from src.terms_k3_d1 import make_Q_arg_alpha, make_Q_arg_beta

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        alpha = make_Q_arg_alpha(theta, x_vars, y_vars)
        beta = make_Q_arg_beta(theta, x_vars, y_vars)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        np.testing.assert_allclose(
            alpha.evaluate_a0(U, T),
            beta.evaluate_a0(U, T)
        )

    def test_alpha_beta_x_coeffs_differ(self):
        """α and β have DIFFERENT x1 coefficients."""
        from src.terms_k3_d1 import make_Q_arg_alpha, make_Q_arg_beta

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        alpha = make_Q_arg_alpha(theta, x_vars, y_vars)
        beta = make_Q_arg_beta(theta, x_vars, y_vars)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        alpha_x = alpha.evaluate_coeff("x1", U, T)
        beta_x = beta.evaluate_coeff("x1", U, T)

        # α: x1_coeff = θt
        # β: x1_coeff = θ(t-1)
        # These should NOT be equal
        assert not np.allclose(alpha_x, beta_x)

        # Verify specific values
        np.testing.assert_allclose(alpha_x, theta * T)
        np.testing.assert_allclose(beta_x, theta * (T - 1))

    def test_alpha_beta_y_coeffs_differ(self):
        """α and β have DIFFERENT y1 coefficients (swapped from x)."""
        from src.terms_k3_d1 import make_Q_arg_alpha, make_Q_arg_beta

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        alpha = make_Q_arg_alpha(theta, x_vars, y_vars)
        beta = make_Q_arg_beta(theta, x_vars, y_vars)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        alpha_y = alpha.evaluate_coeff("y1", U, T)
        beta_y = beta.evaluate_coeff("y1", U, T)

        # α: y1_coeff = θ(t-1)
        # β: y1_coeff = θt
        # These are swapped from x coefficients
        np.testing.assert_allclose(alpha_y, theta * (T - 1))
        np.testing.assert_allclose(beta_y, theta * T)

    def test_alpha_beta_swapped_symmetry(self):
        """α's x coeff = β's y coeff, and vice versa."""
        from src.terms_k3_d1 import make_Q_arg_alpha, make_Q_arg_beta

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        alpha = make_Q_arg_alpha(theta, x_vars, y_vars)
        beta = make_Q_arg_beta(theta, x_vars, y_vars)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # α's x coeff should equal β's y coeff
        np.testing.assert_allclose(
            alpha.evaluate_coeff("x1", U, T),
            beta.evaluate_coeff("y1", U, T)
        )

        # α's y coeff should equal β's x coeff
        np.testing.assert_allclose(
            alpha.evaluate_coeff("y1", U, T),
            beta.evaluate_coeff("x1", U, T)
        )


# =============================================================================
# Test Group C: Algebraic Prefactor
# =============================================================================

class TestAlgebraicPrefactor:
    """Verify algebraic prefactor (θS+1)/θ = 1/θ + x1 + y1."""

    def test_algebraic_prefactor_constant_term(self):
        """Constant term should be 1/θ."""
        from src.terms_k3_d1 import make_algebraic_prefactor_11

        theta = 4/7
        prefactor = make_algebraic_prefactor_11(theta)

        U = np.array([[0.3]])
        T = np.array([[0.5]])

        a0 = prefactor.evaluate_a0(U, T)
        np.testing.assert_allclose(a0, 1.0 / theta)

    def test_algebraic_prefactor_x_coeff(self):
        """x1 coefficient should be 1."""
        from src.terms_k3_d1 import make_algebraic_prefactor_11

        theta = 4/7
        prefactor = make_algebraic_prefactor_11(theta)

        U = np.array([[0.3]])
        T = np.array([[0.5]])

        x_coeff = prefactor.evaluate_coeff("x1", U, T)
        np.testing.assert_allclose(x_coeff, 1.0)

    def test_algebraic_prefactor_y_coeff(self):
        """y1 coefficient should be 1."""
        from src.terms_k3_d1 import make_algebraic_prefactor_11

        theta = 4/7
        prefactor = make_algebraic_prefactor_11(theta)

        U = np.array([[0.3]])
        T = np.array([[0.5]])

        y_coeff = prefactor.evaluate_coeff("y1", U, T)
        np.testing.assert_allclose(y_coeff, 1.0)


# =============================================================================
# Test Group D: Poly Prefactor
# =============================================================================

class TestPolyPrefactor:
    """Verify poly prefactor (1-u)²."""

    def test_poly_prefactor_at_u_0(self):
        """(1-0)² = 1."""
        from src.terms_k3_d1 import make_poly_prefactor_11

        prefactor = make_poly_prefactor_11()

        U = np.array([[0.0]])
        T = np.array([[0.5]])

        result = prefactor(U, T)
        np.testing.assert_allclose(result, 1.0)

    def test_poly_prefactor_at_u_1(self):
        """(1-1)² = 0."""
        from src.terms_k3_d1 import make_poly_prefactor_11

        prefactor = make_poly_prefactor_11()

        U = np.array([[1.0]])
        T = np.array([[0.5]])

        result = prefactor(U, T)
        np.testing.assert_allclose(result, 0.0)

    def test_poly_prefactor_at_u_half(self):
        """(1-0.5)² = 0.25."""
        from src.terms_k3_d1 import make_poly_prefactor_11

        prefactor = make_poly_prefactor_11()

        U = np.array([[0.5]])
        T = np.array([[0.5]])

        result = prefactor(U, T)
        np.testing.assert_allclose(result, 0.25)


# =============================================================================
# Test Group E: Symbolic Sanity with Toy Polynomials
# =============================================================================

class TestSymbolicSanity:
    """Test with simple polynomials to verify coefficient extraction."""

    def test_P_argument_structure(self):
        """P argument should be u + var."""
        from src.terms_k3_d1 import make_P_argument

        P_arg = make_P_argument("x1")

        U = np.array([[0.3]])
        T = np.array([[0.5]])

        # Constant term = u
        np.testing.assert_allclose(P_arg.evaluate_a0(U, T), U)

        # x1 coefficient = 1
        np.testing.assert_allclose(P_arg.evaluate_coeff("x1", U, T), 1.0)

    def test_factor_evaluation_with_toy_poly(self):
        """Evaluate PolyFactor with a simple polynomial."""
        from src.terms_k3_d1 import make_P_argument
        from src.term_dsl import PolyFactor, SeriesContext
        from src.polynomials import Polynomial

        ctx = SeriesContext(var_names=("x1", "y1"))

        # Simple polynomial: P(x) = 1 + x
        toy_poly = Polynomial([1.0, 1.0])

        # P argument: x1 + u
        P_arg = make_P_argument("x1")
        factor = PolyFactor("toy", P_arg)

        U = np.array([[0.5]])
        T = np.array([[0.3]])

        series = factor.evaluate(toy_poly, U, T, ctx)

        # P(u + x1) = 1 + (u + x1) = (1 + u) + x1
        # At u=0.5: constant = 1.5, x1_coeff = 1
        np.testing.assert_allclose(series.extract(()), 1.5)
        np.testing.assert_allclose(series.extract(("x1",)), 1.0)
        np.testing.assert_allclose(series.extract(("y1",)), 0.0)

    def test_exp_factor_evaluation(self):
        """Evaluate ExpFactor at a simple point."""
        from src.terms_k3_d1 import make_Q_arg_alpha
        from src.term_dsl import ExpFactor, SeriesContext

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        ctx = SeriesContext(var_names=("x1", "y1"))
        Q_arg = make_Q_arg_alpha(theta, x_vars, y_vars)
        factor = ExpFactor(scale=1.0, argument=Q_arg)  # scale=1 for simplicity

        U = np.array([[0.5]])
        T = np.array([[0.5]])

        series = factor.evaluate(U, T, ctx)

        # At t=0.5:
        # Arg_α = t + θt·x1 + θ(t-1)·y1 = 0.5 + θ·0.5·x1 + θ·(-0.5)·y1
        # exp(Arg_α) ≈ exp(0.5) * (1 + θ·0.5·x1 - θ·0.5·y1 + ...)

        # Just verify shape and that constant term is exp(t)
        np.testing.assert_allclose(series.extract(()), np.exp(0.5), rtol=1e-10)


# =============================================================================
# Test Group F: Shape Invariants
# =============================================================================

class TestShapeInvariants:
    """Verify all coefficient arrays have correct shape."""

    def test_all_affine_exprs_produce_correct_shapes(self):
        """All AffineExpr evaluations should return grid shape."""
        from src.terms_k3_d1 import (
            make_Q_arg_alpha, make_Q_arg_beta,
            make_algebraic_prefactor_11, make_P_argument
        )

        theta = 4/7
        x_vars = ("x1",)
        y_vars = ("y1",)

        exprs = [
            make_Q_arg_alpha(theta, x_vars, y_vars),
            make_Q_arg_beta(theta, x_vars, y_vars),
            make_algebraic_prefactor_11(theta),
            make_P_argument("x1"),
            make_P_argument("y1"),
        ]

        U = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        T = np.array([[0.7, 0.8, 0.9], [0.3, 0.4, 0.5]])

        for expr in exprs:
            a0 = expr.evaluate_a0(U, T)
            assert a0.shape == (2, 3), f"a0 shape mismatch: {a0.shape}"

            for var in expr.var_coeffs.keys():
                coeff = expr.evaluate_coeff(var, U, T)
                assert coeff.shape == (2, 3), f"coeff shape mismatch: {coeff.shape}"

    def test_extracted_coefficient_shape(self):
        """Extracted coefficient should match grid shape."""
        from src.terms_k3_d1 import make_I1_11
        from src.polynomials import Polynomial

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)
        ctx = term.create_context()

        # Need degree >= 2 for non-zero x1·y1 coefficient
        # The x1·y1 coeff comes from P''(u)·(coeff_x1)·(coeff_y1)
        toy_poly = Polynomial([1.0, 1.0, 1.0])  # 1 + x + x²

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Evaluate a Q factor - these have BOTH x1 and y1 in their arguments
        # Q_arg_alpha = t + θt·x1 + θ(t-1)·y1
        # So extracting ("x1", "y1") gives a non-zero grid-shaped result
        q_factor = term.poly_factors[2]  # Q(Arg_α)
        series = q_factor.evaluate(toy_poly, U, T, ctx)

        # Extract mixed derivative coefficient
        coeff = series.extract(term.deriv_tuple())
        assert coeff.shape == (2, 2), f"Expected (2, 2), got {coeff.shape}"

    def test_P_factor_single_var_extraction(self):
        """P factors only involve one variable, extract that one."""
        from src.terms_k3_d1 import make_I1_11
        from src.polynomials import Polynomial

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)
        ctx = term.create_context()

        toy_poly = Polynomial([1.0, 1.0])  # 1 + x

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # P₁(x1+u) only has x1 - extracting just x1 should work
        p_x_factor = term.poly_factors[0]  # P₁(x1+u)
        series = p_x_factor.evaluate(toy_poly, U, T, ctx)

        # Extract just x1 coefficient (not x1·y1)
        coeff_x1 = series.extract(("x1",))
        assert coeff_x1.shape == (2, 2), f"Expected (2, 2), got {coeff_x1.shape}"

        # The x1·y1 coefficient is zero (scalar) since P doesn't involve y1
        coeff_xy = series.extract(("x1", "y1"))
        assert np.asarray(coeff_xy).shape == () or np.allclose(coeff_xy, 0)


# =============================================================================
# Test Group G: Quadrature Convergence (placeholder for evaluate.py)
# =============================================================================

class TestQuadratureSetup:
    """Verify quadrature grid works with term evaluation."""

    def test_term_evaluates_on_quadrature_grid(self):
        """Term factors can be evaluated on quadrature grid."""
        from src.terms_k3_d1 import make_I1_11
        from src.quadrature import tensor_grid_2d
        from src.polynomials import Polynomial

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)
        ctx = term.create_context()

        # Use n=10 for quick test
        U, T, W = tensor_grid_2d(10)

        # Need degree >= 2 for non-zero x1·y1 coefficient
        toy_poly = Polynomial([1.0, 1.0, 1.0])  # 1 + x + x²

        # Evaluate a Q factor - has both x1 and y1 in its argument
        q_factor = term.poly_factors[2]  # Q(Arg_α)
        series = q_factor.evaluate(toy_poly, U, T, ctx)

        # Extract coefficient (x1·y1 for the mixed derivative)
        coeff = series.extract(term.deriv_tuple())

        # Should be (10, 10) grid
        assert coeff.shape == (10, 10), f"Expected (10, 10), got {coeff.shape}"

        # Integration should work
        integral = np.sum(W * coeff)
        assert np.isfinite(integral)

    def test_quadrature_convergence(self):
        """Verify integral converges to reference value as n increases.

        Uses a high-n reference (n=200) and checks that lower-n values
        are within tolerance. This is more robust than monotone convergence
        checks which can fail for some integrands.
        """
        from src.terms_k3_d1 import make_I1_11
        from src.quadrature import tensor_grid_2d
        from src.polynomials import Polynomial

        theta = 4/7
        R = 1.3036
        term = make_I1_11(theta, R)
        ctx = term.create_context()

        # Use degree-2 polynomial for non-trivial x1·y1 coefficient
        toy_poly = Polynomial([1.0, 1.0, 1.0])  # 1 + x + x²

        def compute_integral(n):
            U, T, W = tensor_grid_2d(n)
            q_factor = term.poly_factors[2]  # Q(Arg_α)
            series = q_factor.evaluate(toy_poly, U, T, ctx)
            coeff = series.extract(term.deriv_tuple())
            return np.sum(W * coeff)

        # Compute reference at high n
        ref = compute_integral(200)

        # Test values at n=40, 60, 80 are within tolerance of reference
        test_ns = [40, 60, 80]
        tol = 1e-10  # Gauss-Legendre converges fast for smooth integrands

        for n in test_ns:
            val = compute_integral(n)
            err = abs(val - ref)
            assert err < tol, (
                f"n={n} integral differs from reference by {err:.2e}, "
                f"exceeds tol={tol:.0e}. val={val:.12e}, ref={ref:.12e}"
            )
            assert np.isfinite(val), f"Integral at n={n} is not finite"


# =============================================================================
# Test Group H: I₂ Term Structure (Decoupled, no derivatives)
# =============================================================================

class TestI2TermStructure:
    """Verify I₂ term structure: no formal variables, Q(t)² with power, exp(2R·t)."""

    def test_I2_11_vars_empty(self):
        """I₂ has no formal variables."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert term.vars == ()

    def test_I2_11_deriv_orders_empty(self):
        """I₂ has no derivative orders."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert term.deriv_orders == {}

    def test_I2_11_deriv_tuple_empty(self):
        """I₂ deriv_tuple() returns empty tuple."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert term.deriv_tuple() == ()

    def test_I2_11_numeric_prefactor(self):
        """I₂ numeric_prefactor is 1/θ."""
        from src.terms_k3_d1 import make_I2_11
        theta = 4/7
        term = make_I2_11(theta, 1.3036)
        assert np.isclose(term.numeric_prefactor, 1.0 / theta)

    def test_I2_11_no_algebraic_prefactor(self):
        """I₂ has no algebraic prefactor."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert term.algebraic_prefactor is None

    def test_I2_11_no_poly_prefactors(self):
        """I₂ has no poly prefactors."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert term.poly_prefactors == []

    def test_I2_11_has_3_poly_factors(self):
        """I₂ has 3 poly factors: P₁(u), P₁(u), Q(t)²."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        assert len(term.poly_factors) == 3

    def test_I2_11_Q_factor_has_power_2(self):
        """I₂ Q factor has power=2."""
        from src.terms_k3_d1 import make_I2_11
        term = make_I2_11(4/7, 1.3036)
        q_factor = term.poly_factors[2]
        assert q_factor.poly_name == "Q"
        assert q_factor.power == 2

    def test_I2_11_exp_factor_scale_is_2R(self):
        """I₂ exp factor has scale=2R (not R)."""
        from src.terms_k3_d1 import make_I2_11
        R = 1.3036
        term = make_I2_11(4/7, R)
        assert len(term.exp_factors) == 1
        assert np.isclose(term.exp_factors[0].scale, 2 * R)


class TestI2Evaluation:
    """Verify I₂ evaluates correctly (no extraction needed)."""

    def test_I2_11_evaluates_finite(self):
        """I₂ poly factors evaluate to finite values."""
        from src.terms_k3_d1 import make_I2_11
        from src.polynomials import Polynomial

        term = make_I2_11(4/7, 1.3036)
        ctx = term.create_context()

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        toy_poly = Polynomial([1.0, 1.0, 1.0])

        # Evaluate all poly factors
        for factor in term.poly_factors:
            series = factor.evaluate(toy_poly, U, T, ctx)
            # For I₂, no formal variables, so extract() gives constant
            coeff = series.extract(())
            assert np.all(np.isfinite(coeff))

    def test_I2_11_exp_evaluates_finite(self):
        """I₂ exp factor evaluates to finite values."""
        from src.terms_k3_d1 import make_I2_11

        term = make_I2_11(4/7, 1.3036)
        ctx = term.create_context()

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Evaluate exp factor
        exp_factor = term.exp_factors[0]
        series = exp_factor.evaluate(U, T, ctx)
        coeff = series.extract(())
        assert np.all(np.isfinite(coeff))
        assert np.all(coeff > 0)  # exp is always positive


# =============================================================================
# Test Group I: I₃ Term Structure (Single x derivative)
# =============================================================================

class TestI3TermStructure:
    """Verify I₃ term structure: only x1 variable, negative prefactor."""

    def test_I3_11_vars_x1_only(self):
        """I₃ has only x1 as formal variable."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert term.vars == ("x1",)

    def test_I3_11_deriv_orders_x1(self):
        """I₃ derivative order is {x1: 1}."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert term.deriv_orders == {"x1": 1}

    def test_I3_11_deriv_tuple_x1(self):
        """I₃ deriv_tuple() returns ('x1',)."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert term.deriv_tuple() == ("x1",)

    def test_I3_11_numeric_prefactor_negative(self):
        """I₃ numeric_prefactor is -1/θ (PRZZ line 1562-1563)."""
        from src.terms_k3_d1 import make_I3_11
        theta = 4/7
        term = make_I3_11(theta, 1.3036)
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        assert abs(term.numeric_prefactor - (-1.0 / theta)) < 1e-10

    def test_I3_11_has_single_poly_prefactor(self):
        """I₃ has (1-u) poly prefactor (single power)."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert len(term.poly_prefactors) == 1

        # Verify it's (1-u)^1, not (1-u)^2
        U = np.array([0.0, 0.5, 1.0])
        T = np.array([0.5, 0.5, 0.5])
        prefactor = term.poly_prefactors[0]
        vals = prefactor(U, T)
        expected = 1 - U  # (1-u)^1
        assert np.allclose(vals, expected)

    def test_I3_11_has_4_poly_factors(self):
        """I₃ has 4 poly factors: P₁(x1+u), P₁(u), Q(α), Q(β)."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert len(term.poly_factors) == 4

    def test_I3_11_has_2_exp_factors(self):
        """I₃ has 2 exp factors."""
        from src.terms_k3_d1 import make_I3_11
        term = make_I3_11(4/7, 1.3036)
        assert len(term.exp_factors) == 2


class TestI3AlphaBetaDistinct:
    """Verify I₃ α and β arguments are distinct (even with y=0)."""

    def test_I3_alpha_beta_x_coeffs_differ(self):
        """I₃ α and β have different x1 coefficients."""
        from src.terms_k3_d1 import make_Q_arg_alpha_x_only, make_Q_arg_beta_x_only

        theta = 4/7
        alpha = make_Q_arg_alpha_x_only(theta)
        beta = make_Q_arg_beta_x_only(theta)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        alpha_x = alpha.evaluate_coeff("x1", U, T)  # θt
        beta_x = beta.evaluate_coeff("x1", U, T)    # θ(t-1)

        # These should NOT be equal
        assert not np.allclose(alpha_x, beta_x)

    def test_I3_alpha_beta_constant_equal(self):
        """I₃ α and β have the same constant term (t)."""
        from src.terms_k3_d1 import make_Q_arg_alpha_x_only, make_Q_arg_beta_x_only

        theta = 4/7
        alpha = make_Q_arg_alpha_x_only(theta)
        beta = make_Q_arg_beta_x_only(theta)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Constant terms should be equal (both = t)
        assert np.allclose(alpha.evaluate_a0(U, T), beta.evaluate_a0(U, T))
        assert np.allclose(alpha.evaluate_a0(U, T), T)


class TestI3Evaluation:
    """Verify I₃ evaluates correctly."""

    def test_I3_11_x1_coeff_extraction(self):
        """I₃ Q factor x1 coefficient extraction works."""
        from src.terms_k3_d1 import make_I3_11
        from src.polynomials import Polynomial

        term = make_I3_11(4/7, 1.3036)
        ctx = term.create_context()

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Need degree >= 1 for non-zero x1 coefficient
        toy_poly = Polynomial([1.0, 1.0, 1.0])

        # Evaluate a Q factor
        q_factor = term.poly_factors[2]
        series = q_factor.evaluate(toy_poly, U, T, ctx)

        # Extract x1 coefficient
        coeff = series.extract(term.deriv_tuple())
        assert coeff.shape == (2, 2)
        assert np.all(np.isfinite(coeff))


# =============================================================================
# Test Group J: I₄ Term Structure (Single y derivative)
# =============================================================================

class TestI4TermStructure:
    """Verify I₄ term structure: only y1 variable, negative prefactor."""

    def test_I4_11_vars_y1_only(self):
        """I₄ has only y1 as formal variable."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert term.vars == ("y1",)

    def test_I4_11_deriv_orders_y1(self):
        """I₄ derivative order is {y1: 1}."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert term.deriv_orders == {"y1": 1}

    def test_I4_11_deriv_tuple_y1(self):
        """I₄ deriv_tuple() returns ('y1',)."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert term.deriv_tuple() == ("y1",)

    def test_I4_11_numeric_prefactor_negative(self):
        """I₄ numeric_prefactor is -1/θ (PRZZ line 1568-1569)."""
        from src.terms_k3_d1 import make_I4_11
        theta = 4/7
        term = make_I4_11(theta, 1.3036)
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        assert abs(term.numeric_prefactor - (-1.0 / theta)) < 1e-10

    def test_I4_11_has_single_poly_prefactor(self):
        """I₄ has (1-u) poly prefactor (single power)."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert len(term.poly_prefactors) == 1

        # Verify it's (1-u)^1
        U = np.array([0.0, 0.5, 1.0])
        T = np.array([0.5, 0.5, 0.5])
        prefactor = term.poly_prefactors[0]
        vals = prefactor(U, T)
        expected = 1 - U
        assert np.allclose(vals, expected)

    def test_I4_11_has_4_poly_factors(self):
        """I₄ has 4 poly factors: P₁(u), P₁(y1+u), Q(α), Q(β)."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert len(term.poly_factors) == 4

    def test_I4_11_has_2_exp_factors(self):
        """I₄ has 2 exp factors."""
        from src.terms_k3_d1 import make_I4_11
        term = make_I4_11(4/7, 1.3036)
        assert len(term.exp_factors) == 2


class TestI4AlphaBetaDistinct:
    """Verify I₄ α and β arguments are distinct (even with x=0)."""

    def test_I4_alpha_beta_y_coeffs_differ(self):
        """I₄ α and β have different y1 coefficients."""
        from src.terms_k3_d1 import make_Q_arg_alpha_y_only, make_Q_arg_beta_y_only

        theta = 4/7
        alpha = make_Q_arg_alpha_y_only(theta)
        beta = make_Q_arg_beta_y_only(theta)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        alpha_y = alpha.evaluate_coeff("y1", U, T)  # θ(t-1)
        beta_y = beta.evaluate_coeff("y1", U, T)    # θt

        # These should NOT be equal
        assert not np.allclose(alpha_y, beta_y)

    def test_I4_alpha_beta_constant_equal(self):
        """I₄ α and β have the same constant term (t)."""
        from src.terms_k3_d1 import make_Q_arg_alpha_y_only, make_Q_arg_beta_y_only

        theta = 4/7
        alpha = make_Q_arg_alpha_y_only(theta)
        beta = make_Q_arg_beta_y_only(theta)

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Constant terms should be equal (both = t)
        assert np.allclose(alpha.evaluate_a0(U, T), beta.evaluate_a0(U, T))
        assert np.allclose(alpha.evaluate_a0(U, T), T)


class TestI4Evaluation:
    """Verify I₄ evaluates correctly."""

    def test_I4_11_y1_coeff_extraction(self):
        """I₄ Q factor y1 coefficient extraction works."""
        from src.terms_k3_d1 import make_I4_11
        from src.polynomials import Polynomial

        term = make_I4_11(4/7, 1.3036)
        ctx = term.create_context()

        U = np.array([[0.3, 0.4], [0.5, 0.6]])
        T = np.array([[0.2, 0.25], [0.3, 0.35]])

        # Need degree >= 1 for non-zero y1 coefficient
        toy_poly = Polynomial([1.0, 1.0, 1.0])

        # Evaluate a Q factor
        q_factor = term.poly_factors[2]
        series = q_factor.evaluate(toy_poly, U, T, ctx)

        # Extract y1 coefficient
        coeff = series.extract(term.deriv_tuple())
        assert coeff.shape == (2, 2)
        assert np.all(np.isfinite(coeff))


# =============================================================================
# Test Group K: make_all_terms_11 convenience function
# =============================================================================

class TestMakeAllTerms11:
    """Verify make_all_terms_11 returns all 4 terms."""

    def test_returns_4_terms(self):
        """make_all_terms_11 returns exactly 4 terms."""
        from src.terms_k3_d1 import make_all_terms_11
        terms = make_all_terms_11(4/7, 1.3036)
        assert len(terms) == 4

    def test_terms_have_correct_names(self):
        """Terms have names I1_11, I2_11, I3_11, I4_11."""
        from src.terms_k3_d1 import make_all_terms_11
        terms = make_all_terms_11(4/7, 1.3036)
        names = [t.name for t in terms]
        assert names == ["I1_11", "I2_11", "I3_11", "I4_11"]

    def test_all_terms_build_without_error(self):
        """All terms build successfully with PRZZ parameters."""
        from src.terms_k3_d1 import make_all_terms_11
        theta = 4/7
        R = 1.3036
        terms = make_all_terms_11(theta, R)
        for term in terms:
            assert term.pair == (1, 1)
            ctx = term.create_context()  # Should not raise


# =============================================================================
# Test Group L: P Factor Structure (CRITICAL - locks in Interpretation B)
# =============================================================================

class TestPFactorStructure:
    """
    CRITICAL: These tests lock in "Interpretation B" (summed P arguments)
    from TECHNICAL_ANALYSIS.md Section 10.1.

    The number of P factors must be exactly 2 for all I₁ terms (one per side).
    This prevents flip-flopping between interpretations.
    """

    def test_I1_11_has_2_P_factors(self):
        """I₁ for (1,1) should have exactly 2 P factors."""
        from src.terms_k3_d1 import make_I1_11
        term = make_I1_11(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_22_has_2_P_factors(self):
        """I₁ for (2,2) should have exactly 2 P factors (summed args)."""
        from src.terms_k3_d1 import make_I1_22
        term = make_I1_22(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_33_has_2_P_factors(self):
        """I₁ for (3,3) should have exactly 2 P factors (summed args)."""
        from src.terms_k3_d1 import make_I1_33
        term = make_I1_33(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_12_has_2_P_factors(self):
        """I₁ for (1,2) should have exactly 2 P factors."""
        from src.terms_k3_d1 import make_I1_12
        term = make_I1_12(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_13_has_2_P_factors(self):
        """I₁ for (1,3) should have exactly 2 P factors."""
        from src.terms_k3_d1 import make_I1_13
        term = make_I1_13(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_23_has_2_P_factors(self):
        """I₁ for (2,3) should have exactly 2 P factors."""
        from src.terms_k3_d1 import make_I1_23
        term = make_I1_23(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name.startswith("P")]
        assert len(p_factors) == 2, f"Expected 2 P factors, got {len(p_factors)}"

    def test_I1_22_P_left_has_both_x_vars(self):
        """I₁ for (2,2): left P factor should have x1 AND x2 in argument."""
        from src.terms_k3_d1 import make_I1_22
        term = make_I1_22(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name == "P2"]
        p_left = p_factors[0]

        # Both x1 and x2 should be in the argument's var_coeffs
        arg_vars = set(p_left.argument.var_coeffs.keys())
        assert "x1" in arg_vars and "x2" in arg_vars, \
            f"P_left argument should contain x1 and x2, got {arg_vars}"

    def test_I1_22_P_right_has_both_y_vars(self):
        """I₁ for (2,2): right P factor should have y1 AND y2 in argument."""
        from src.terms_k3_d1 import make_I1_22
        term = make_I1_22(4/7, 1.3036)
        p_factors = [f for f in term.poly_factors if f.poly_name == "P2"]
        p_right = p_factors[1]

        # Both y1 and y2 should be in the argument's var_coeffs
        arg_vars = set(p_right.argument.var_coeffs.keys())
        assert "y1" in arg_vars and "y2" in arg_vars, \
            f"P_right argument should contain y1 and y2, got {arg_vars}"


class TestLinearPLitmusTest:
    """
    CRITICAL: Linear polynomial litmus test.

    If P(u) = 1 + u (linear), then:
    - Summed args P(u + x1 + x2): coeff of x1*x2 = P''(u) = 0
    - Separate factors P(u+x1)*P(u+x2): coeff of x1*x2 = (P'(u))² = 1

    This test ENFORCES Interpretation B by verifying zero x1*x2 coefficient.
    """

    def test_linear_P_gives_zero_x1x2_coefficient(self):
        """With linear P, summed-arg P factor should give 0 for x1*x2 coeff."""
        from src.polynomials import Polynomial
        from src.composition import compose_polynomial_on_affine
        import numpy as np

        # Linear polynomial P(u) = 1 + u
        P = Polynomial([1.0, 1.0])

        # Test at u = 0.5
        u0 = np.array([[0.5]])
        var_names = ("x1", "x2")

        # Build P(u + x1 + x2) with summed argument
        lin = {"x1": np.array([[1.0]]), "x2": np.array([[1.0]])}
        series = compose_polynomial_on_affine(P, u0, lin, var_names)

        # Extract x1*x2 coefficient
        coeff = series.extract(("x1", "x2"))
        coeff_val = float(np.asarray(coeff).flat[0]) if np.asarray(coeff).size > 0 else 0.0

        # MUST be 0 for summed args (P''(u) = 0 for linear P)
        assert abs(coeff_val) < 1e-14, \
            f"Summed args: x1*x2 coeff should be 0, got {coeff_val}. " \
            "This means we're NOT using Interpretation B (summed args)!"
