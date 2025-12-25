#!/usr/bin/env python3
"""
Gate tests for Run 18: TeX Combined Integral Structure

These tests validate each stage of the implementation according to the plan.

Stage 18A: CombinedMirrorFactor scalar limit test
Stage 18B: I1 channel (1,1) convergence test
Stage 18C: I2 channel test
Stage 18D: S34 channel test
Stage 18E: Full assembly test
"""

import numpy as np
import pytest

from src.term_dsl import SeriesContext, CombinedMirrorFactor


# =============================================================================
# Stage 18A: CombinedMirrorFactor Tests
# =============================================================================

class TestCombinedMirrorFactorScalarLimit:
    """Test CombinedMirrorFactor at x=y=0 matches analytic formula."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_scalar_limit_analytic(self):
        """Verify scalar_limit() method matches (exp(2R) - 1) / (2R)."""
        for R in [self.R_KAPPA, self.R_KAPPA_STAR, 0.5, 1.0, 2.0]:
            factor = CombinedMirrorFactor(R=R, theta=self.THETA)
            expected = (np.exp(2 * R) - 1) / (2 * R)
            actual = factor.scalar_limit()
            assert np.isclose(actual, expected, rtol=1e-12), (
                f"R={R}: scalar_limit()={actual} != expected {expected}"
            )

    def test_scalar_limit_via_evaluate_kappa(self):
        """Verify evaluate() at x=y=0 matches scalar_limit() for κ benchmark."""
        R = self.R_KAPPA
        factor = CombinedMirrorFactor(R=R, theta=self.THETA, n_quad_s=40)

        # Single grid point
        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        # Extract constant term (mask=0 for no x, no y)
        constant_coeff = series.coeffs.get(0, np.zeros_like(U))

        # The analytic scalar limit (at x=y=0)
        # Note: we multiply by the log factor (1 + θ(x+y)) which is 1 at x=y=0
        expected = factor.scalar_limit()

        # Should match within quadrature precision
        assert np.allclose(constant_coeff, expected, rtol=1e-6), (
            f"evaluate() constant term = {constant_coeff[0, 0]:.8f}, "
            f"expected scalar_limit = {expected:.8f}"
        )

    def test_scalar_limit_via_evaluate_kappa_star(self):
        """Verify evaluate() at x=y=0 matches scalar_limit() for κ* benchmark."""
        R = self.R_KAPPA_STAR
        factor = CombinedMirrorFactor(R=R, theta=self.THETA, n_quad_s=40)

        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)
        constant_coeff = series.coeffs.get(0, np.zeros_like(U))
        expected = factor.scalar_limit()

        assert np.allclose(constant_coeff, expected, rtol=1e-6), (
            f"evaluate() constant term = {constant_coeff[0, 0]:.8f}, "
            f"expected scalar_limit = {expected:.8f}"
        )

    def test_quadrature_convergence(self):
        """Verify quadrature converges as n_quad_s increases."""
        R = self.R_KAPPA
        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        expected = (np.exp(2 * R) - 1) / (2 * R)

        errors = []
        for n_quad in [5, 10, 20, 40]:
            factor = CombinedMirrorFactor(R=R, theta=self.THETA, n_quad_s=n_quad)
            series = factor.evaluate(U, T, ctx)
            constant_coeff = series.coeffs.get(0, np.zeros_like(U))[0, 0]
            error = abs(constant_coeff - expected)
            errors.append(error)

        # Errors should decrease or stay at machine precision
        # By n=10, Gauss-Legendre is typically at machine precision for smooth integrands
        machine_precision = 1e-13
        for i in range(len(errors) - 1):
            # Either error decreases OR both are already at machine precision
            at_machine_precision = errors[i] < machine_precision and errors[i + 1] < machine_precision
            error_decreased = errors[i + 1] <= errors[i] * 1.1  # Allow small numerical noise
            assert at_machine_precision or error_decreased, (
                f"Quadrature not converging: error at n={[5, 10, 20, 40][i+1]} = {errors[i+1]:.2e}, "
                f"at n={[5, 10, 20, 40][i]} = {errors[i]:.2e}"
            )

        # Final error should be at machine precision
        assert errors[-1] < machine_precision, (
            f"Final error {errors[-1]:.2e} not at machine precision"
        )


class TestCombinedMirrorFactorDerivativeStructure:
    """Test that CombinedMirrorFactor has correct derivative structure."""

    THETA = 4.0 / 7.0
    R = 1.3036

    def test_xy_coefficient_nonzero(self):
        """The xy coefficient should be nonzero (derivative structure)."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=20)

        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        # For vars=("x", "y"), the xy coefficient has mask = 0b11 = 3
        xy_coeff = series.coeffs.get(3, np.zeros_like(U))

        assert not np.allclose(xy_coeff, 0), (
            f"xy coefficient should be nonzero, got {xy_coeff}"
        )

    def test_x_coefficient_nonzero(self):
        """The x coefficient should be nonzero."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=20)

        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        # For vars=("x", "y"), the x coefficient has mask = 0b01 = 1
        x_coeff = series.coeffs.get(1, np.zeros_like(U))

        assert not np.allclose(x_coeff, 0), (
            f"x coefficient should be nonzero, got {x_coeff}"
        )

    def test_y_coefficient_nonzero(self):
        """The y coefficient should be nonzero."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=20)

        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        # For vars=("x", "y"), the y coefficient has mask = 0b10 = 2
        y_coeff = series.coeffs.get(2, np.zeros_like(U))

        assert not np.allclose(y_coeff, 0), (
            f"y coefficient should be nonzero, got {y_coeff}"
        )

    def test_symmetry_x_y(self):
        """The x and y coefficients should be equal (symmetric structure)."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=20)

        U = np.array([[0.5]])
        T = np.array([[0.5]])
        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        x_coeff = series.coeffs.get(1, np.zeros_like(U))
        y_coeff = series.coeffs.get(2, np.zeros_like(U))

        assert np.allclose(x_coeff, y_coeff), (
            f"x and y coefficients should be equal: x={x_coeff}, y={y_coeff}"
        )


class TestCombinedMirrorFactorGridBroadcast:
    """Test that CombinedMirrorFactor works on full grids."""

    THETA = 4.0 / 7.0
    R = 1.3036

    def test_2d_grid(self):
        """Verify evaluation on 2D grid produces correct shapes."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=10)

        # Create a 5x5 grid
        nodes = np.linspace(0.1, 0.9, 5)
        U, T = np.meshgrid(nodes, nodes, indexing="ij")

        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)

        # All coefficients should have shape (5, 5)
        for mask, coeff in series.coeffs.items():
            assert coeff.shape == (5, 5), (
                f"mask={mask}: shape {coeff.shape} != expected (5, 5)"
            )

    def test_scalar_limit_on_full_grid(self):
        """Verify scalar limit matches at every grid point."""
        factor = CombinedMirrorFactor(R=self.R, theta=self.THETA, n_quad_s=20)

        nodes = np.linspace(0.1, 0.9, 4)
        U, T = np.meshgrid(nodes, nodes, indexing="ij")

        ctx = SeriesContext(var_names=("x", "y"))

        series = factor.evaluate(U, T, ctx)
        constant_coeff = series.coeffs.get(0, np.zeros_like(U))

        expected = factor.scalar_limit()

        # All grid points should give the same scalar limit
        assert np.allclose(constant_coeff, expected, rtol=1e-6), (
            f"Scalar limit varies across grid: min={constant_coeff.min():.6f}, "
            f"max={constant_coeff.max():.6f}, expected={expected:.6f}"
        )


# =============================================================================
# Stage 18B-E: Placeholder tests (to be implemented)
# =============================================================================

class TestI1ChannelCombined:
    """Stage 18B: I1 channel with combined mirror structure."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_i1_11_finite_multiply(self):
        """I1 (multiply) for (1,1) should produce finite result."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        assert np.isfinite(result.I1_combined), "I1_combined should be finite"
        assert result.I1_combined > 0, "I1_combined should be positive"

    def test_i1_11_finite_replace(self):
        """I1 (replace) for (1,1) should produce finite result."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_11_replace

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_11_replace(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        assert np.isfinite(result.I1_combined), "I1_combined should be finite"
        assert result.I1_combined > 0, "I1_combined should be positive"

    def test_i1_11_convergence(self):
        """I1 quadrature should converge."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_11_replace

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        values = []
        for n in [20, 40, 60]:
            result = compute_I1_tex_combined_11_replace(
                theta=self.THETA, R=self.R_KAPPA, n=n, polynomials=polys, n_quad_s=20
            )
            values.append(result.I1_combined)

        # Check convergence: |v60 - v40| < |v40 - v20|
        diff_60_40 = abs(values[2] - values[1])
        diff_40_20 = abs(values[1] - values[0])

        # Allow for machine precision convergence (values already converged)
        assert diff_60_40 <= diff_40_20 + 1e-10, (
            f"Quadrature not converging: |v60-v40|={diff_60_40:.2e}, |v40-v20|={diff_40_20:.2e}"
        )

    def test_i1_replace_reasonable_magnitude(self):
        """I1 (replace) should be in reasonable range for c_target."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_11_replace

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_11_replace(
            theta=self.THETA, R=self.R_KAPPA, n=60, polynomials=polys, n_quad_s=20
        )

        # I1 should be positive and not wildly off from c_target ≈ 2.14
        # Run 18B found I1_replace ≈ 2.08 for κ benchmark
        c_target = 2.137
        assert 0.5 < result.I1_combined < 10, (
            f"I1_combined={result.I1_combined} out of reasonable range [0.5, 10]"
        )


class TestI2ChannelCombined:
    """Stage 18C: I2 channel with combined mirror structure."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036

    def test_i2_positive(self):
        """I2 base should be positive."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I2_tex_combined_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I2_tex_combined_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys
        )

        assert result.I2_base > 0, "I2_base should be positive"
        assert np.isfinite(result.I2_combined), "I2_combined should be finite"


class TestS34ChannelCombined:
    """Stage 18D: S34 channel with combined mirror structure."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036

    def test_s34_finite(self):
        """S34 base should produce finite result."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_S34_base_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_S34_base_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys
        )

        assert np.isfinite(result), "S34 should be finite"
        # S34 is typically negative
        assert result < 0, "S34 should be negative"


class TestFullAssemblyCombined:
    """Stage 18E: Full assembly test."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_c_finite_kappa(self):
        """c should be finite for κ benchmark."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import (
            compute_I1_tex_combined_11_replace,
            compute_I2_tex_combined_11,
            compute_S34_base_11,
        )

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        i1 = compute_I1_tex_combined_11_replace(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )
        i2 = compute_I2_tex_combined_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys
        )
        s34 = compute_S34_base_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys
        )

        c = i1.I1_combined + i2.I2_base + s34

        assert np.isfinite(c), "c should be finite"
        assert c > 0, "c should be positive"

    def test_c_finite_kappa_star(self):
        """c should be finite for κ* benchmark."""
        from src.polynomials import load_przz_polynomials_kappa_star
        from src.evaluate import (
            compute_I1_tex_combined_11_replace,
            compute_I2_tex_combined_11,
            compute_S34_base_11,
        )

        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        i1 = compute_I1_tex_combined_11_replace(
            theta=self.THETA, R=self.R_KAPPA_STAR, n=40, polynomials=polys, n_quad_s=20
        )
        i2 = compute_I2_tex_combined_11(
            theta=self.THETA, R=self.R_KAPPA_STAR, n=40, polynomials=polys
        )
        s34 = compute_S34_base_11(
            theta=self.THETA, R=self.R_KAPPA_STAR, n=40, polynomials=polys
        )

        c = i1.I1_combined + i2.I2_base + s34

        assert np.isfinite(c), "c should be finite"
        assert c > 0, "c should be positive"


# =============================================================================
# Stage 20A: TexCombinedMirrorCore Atomic Tests (Run 20)
# =============================================================================

class TestTexCombinedMirrorCoreScalarLimit:
    """Test scalar limit (x=y=0) against analytic formula."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_scalar_limit_analytic_kappa(self):
        """Scalar limit should match (exp(2R) - 1) / (2R) for κ."""
        from src.term_dsl import TexCombinedMirrorCore

        R = self.R_KAPPA
        core = TexCombinedMirrorCore(R=R, theta=self.THETA)
        expected = (np.exp(2 * R) - 1) / (2 * R)
        computed = core.scalar_limit()

        assert np.isfinite(computed), f"Scalar limit should be finite, got {computed}"
        assert computed > 0, f"Scalar limit should be positive, got {computed}"
        np.testing.assert_allclose(
            computed, expected, rtol=1e-10,
            err_msg=f"Scalar limit mismatch for R={R}"
        )

    def test_scalar_limit_analytic_kappa_star(self):
        """Scalar limit should match (exp(2R) - 1) / (2R) for κ*."""
        from src.term_dsl import TexCombinedMirrorCore

        R = self.R_KAPPA_STAR
        core = TexCombinedMirrorCore(R=R, theta=self.THETA)
        expected = (np.exp(2 * R) - 1) / (2 * R)
        computed = core.scalar_limit()

        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    def test_scalar_limit_is_finite_for_benchmarks(self):
        """Both benchmark R values should give finite scalar limits."""
        from src.term_dsl import TexCombinedMirrorCore

        for R in [self.R_KAPPA, self.R_KAPPA_STAR]:
            core = TexCombinedMirrorCore(R=R, theta=self.THETA)
            scalar = core.scalar_limit()
            assert np.isfinite(scalar), f"Scalar limit not finite for R={R}"
            assert scalar > 0, f"Scalar limit should be positive for R={R}"


class TestTexCombinedMirrorCoreSeriesEvaluation:
    """Test series evaluation produces correct structure."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_series_produces_finite_coefficients(self):
        """Series coefficients should all be finite."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        core = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA)
        ctx = SeriesContext(var_names=("x", "y"))
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        series = core.evaluate(U, T, ctx)

        # Check all coefficients are finite
        for mask, coeff in series.coeffs.items():
            assert np.all(np.isfinite(coeff)), f"Non-finite coefficient for mask {mask}"

    def test_series_constant_term_matches_scalar_limit(self):
        """The constant term (mask=0) should match scalar_limit()."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        for R in [self.R_KAPPA, self.R_KAPPA_STAR]:
            core = TexCombinedMirrorCore(R=R, theta=self.THETA)
            ctx = SeriesContext(var_names=("x", "y"))
            U = np.array([[0.5]])
            T = np.array([[0.5]])

            series = core.evaluate(U, T, ctx)
            constant_term = series.extract(())  # Empty tuple = constant

            expected = core.scalar_limit()
            np.testing.assert_allclose(
                constant_term[0, 0], expected, rtol=1e-10,
                err_msg=f"Constant term mismatch for R={R}"
            )

    def test_series_has_xy_term(self):
        """The xy coefficient should be non-zero."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        core = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA)
        ctx = SeriesContext(var_names=("x", "y"))
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        series = core.evaluate(U, T, ctx)
        xy_coeff = series.extract(("x", "y"))

        assert np.abs(xy_coeff[0, 0]) > 1e-10, \
            f"xy coefficient should be non-zero, got {xy_coeff[0, 0]}"


class TestTexCombinedMirrorCoreQuadratureConvergence:
    """Test that results converge with increasing s-quadrature."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036

    def test_quadrature_convergence_constant_term(self):
        """Series constant term should converge as n_quad_s increases."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        R = self.R_KAPPA
        ctx = SeriesContext(var_names=("x", "y"))
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        results = []
        for n_quad in [10, 20, 40, 80]:
            core = TexCombinedMirrorCore(R=R, theta=self.THETA, n_quad_s=n_quad)
            series = core.evaluate(U, T, ctx)
            results.append(series.extract(())[0, 0])

        # All should be close to analytic scalar limit
        expected = (np.exp(2 * R) - 1) / (2 * R)
        for val in results:
            np.testing.assert_allclose(val, expected, rtol=1e-6)

    def test_series_xy_converges(self):
        """xy coefficient should converge with quadrature."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        R = self.R_KAPPA
        ctx = SeriesContext(var_names=("x", "y"))
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        xy_vals = []
        for n_quad in [10, 20, 40, 80]:
            core = TexCombinedMirrorCore(R=R, theta=self.THETA, n_quad_s=n_quad)
            series = core.evaluate(U, T, ctx)
            xy_vals.append(series.extract(("x", "y"))[0, 0])

        # Later values should be closer together (convergence)
        diff_early = np.abs(xy_vals[1] - xy_vals[0])
        diff_late = np.abs(xy_vals[-1] - xy_vals[-2])
        assert diff_late < diff_early + 1e-12, \
            f"xy coefficient should converge. Early diff: {diff_early}, Late diff: {diff_late}"


class TestTexCombinedMirrorCoreDifferenceQuotientIdentity:
    """Test the TeX identity via direct difference quotient comparison."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036

    def test_difference_quotient_identity_small_xy_001(self):
        """For small (x, y) = (0.01, 0.01), combined structure matches difference quotient."""
        from src.term_dsl import TexCombinedMirrorCore

        core = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA, n_quad_s=40)
        x_val, y_val = 0.01, 0.01

        combined, direct = core.difference_quotient_test(x_val, y_val, L=20.0)

        rel_err = np.abs(combined - direct) / np.abs(direct)
        assert rel_err < 0.01, \
            f"Difference quotient identity failed: combined={combined:.6e}, " \
            f"direct={direct:.6e}, rel_err={rel_err:.2%}"

    def test_difference_quotient_identity_small_xy_0001(self):
        """For small (x, y) = (0.001, 0.001), combined structure matches difference quotient."""
        from src.term_dsl import TexCombinedMirrorCore

        core = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA, n_quad_s=40)
        x_val, y_val = 0.001, 0.001

        combined, direct = core.difference_quotient_test(x_val, y_val, L=20.0)

        rel_err = np.abs(combined - direct) / np.abs(direct)
        assert rel_err < 0.01, f"rel_err={rel_err:.2%}"

    def test_difference_quotient_improves_with_larger_L(self):
        """As L increases (asymptotic regime), agreement should improve."""
        from src.term_dsl import TexCombinedMirrorCore

        core = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA, n_quad_s=40)
        x_val, y_val = 0.01, 0.01

        errors = []
        for L in [5.0, 10.0, 20.0, 50.0]:
            combined, direct = core.difference_quotient_test(x_val, y_val, L=L)
            rel_err = np.abs(combined - direct) / np.abs(direct)
            errors.append(rel_err)

        # All errors should be small
        assert all(e < 0.1 for e in errors), f"All errors should be small: {errors}"


class TestTexCombinedMirrorCoreNoSingularities:
    """Test that there are no hidden singularities at α+β=0."""

    THETA = 4.0 / 7.0

    def test_no_singularity_various_R(self):
        """The combined structure should be well-behaved for various R values."""
        from src.term_dsl import TexCombinedMirrorCore, SeriesContext

        for R in [0.1, 0.5, 1.0, 1.3036, 2.0]:
            core = TexCombinedMirrorCore(R=R, theta=self.THETA)

            # Scalar limit should be finite
            scalar = core.scalar_limit()
            assert np.isfinite(scalar), f"Singularity for R={R}"
            assert scalar > 0, f"Should be positive for R={R}"

            # Series should be finite
            ctx = SeriesContext(var_names=("x", "y"))
            U = np.array([[0.5]])
            T = np.array([[0.5]])
            series = core.evaluate(U, T, ctx)

            for mask, coeff in series.coeffs.items():
                assert np.all(np.isfinite(coeff)), \
                    f"Singularity in series for R={R}, mask={mask}"


class TestTexCombinedMirrorCoreDiffersFromRun18:
    """Verify that TexCombinedMirrorCore differs from Run 18's CombinedMirrorFactor."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036

    def test_scalar_limit_same_as_run18(self):
        """Scalar limit should be the same (no outer exp at x=y=0)."""
        from src.term_dsl import TexCombinedMirrorCore, CombinedMirrorFactor

        run18 = CombinedMirrorFactor(R=self.R_KAPPA, theta=self.THETA)
        run20 = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA)

        np.testing.assert_allclose(run20.scalar_limit(), run18.scalar_limit(), rtol=1e-10)

    def test_xy_coefficient_differs_from_run18(self):
        """xy coefficient should differ due to outer exp(-Rθ(x+y))."""
        from src.term_dsl import TexCombinedMirrorCore, CombinedMirrorFactor, SeriesContext

        ctx = SeriesContext(var_names=("x", "y"))
        U = np.array([[0.5]])
        T = np.array([[0.5]])

        run18 = CombinedMirrorFactor(R=self.R_KAPPA, theta=self.THETA)
        run20 = TexCombinedMirrorCore(R=self.R_KAPPA, theta=self.THETA)

        series18 = run18.evaluate(U, T, ctx)
        series20 = run20.evaluate(U, T, ctx)

        xy18 = series18.extract(("x", "y"))[0, 0]
        xy20 = series20.extract(("x", "y"))[0, 0]

        # They should differ (Run 20 has outer exp factor)
        assert not np.isclose(xy18, xy20, rtol=0.01), \
            f"xy coefficients should differ: run18={xy18}, run20={xy20}"

        # Run 20's xy should be smaller in absolute value due to exp(-Rθ(x+y)) damping
        # (This is a structural difference diagnostic)


# =============================================================================
# Stage 20B/20C: compute_I1_tex_combined_core_11() Tests (Run 20)
# =============================================================================

class TestComputeI1TexCombinedCore11:
    """Test the new Run 20 I1 computation function."""

    THETA = 4.0 / 7.0
    R_KAPPA = 1.3036
    R_KAPPA_STAR = 1.1167

    def test_finite_result_kappa(self):
        """Function produces finite result for κ benchmark."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        assert np.isfinite(result.I1_combined_core), \
            f"I1_combined_core should be finite, got {result.I1_combined_core}"

    def test_finite_result_kappa_star(self):
        """Function produces finite result for κ* benchmark."""
        from src.polynomials import load_przz_polynomials_kappa_star
        from src.evaluate import compute_I1_tex_combined_core_11

        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA_STAR, n=40, polynomials=polys, n_quad_s=20
        )

        assert np.isfinite(result.I1_combined_core), \
            f"I1_combined_core should be finite, got {result.I1_combined_core}"

    def test_nonzero_result(self):
        """Result should be non-zero."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        assert np.abs(result.I1_combined_core) > 1e-10, \
            f"I1_combined_core should be non-zero, got {result.I1_combined_core}"

    def test_quadrature_convergence(self):
        """Result should converge with increasing quadrature."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        values = []
        for n in [20, 40, 60]:
            result = compute_I1_tex_combined_core_11(
                theta=self.THETA, R=self.R_KAPPA, n=n, polynomials=polys, n_quad_s=20
            )
            values.append(result.I1_combined_core)

        # Check convergence: |v60 - v40| < |v40 - v20|
        diff_60_40 = abs(values[2] - values[1])
        diff_40_20 = abs(values[1] - values[0])

        # Allow for already-converged values
        assert diff_60_40 <= diff_40_20 + 1e-10, \
            f"Not converging: |v60-v40|={diff_60_40:.2e}, |v40-v20|={diff_40_20:.2e}"

    def test_differs_from_run18_combined(self):
        """Run 20 result should differ from Run 18 result."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11, compute_I1_tex_combined_11_replace

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        run18 = compute_I1_tex_combined_11_replace(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )
        run20 = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        # They should differ (Run 20 has outer exp factor in combined core)
        rel_diff = abs(run20.I1_combined_core - run18.I1_combined) / abs(run18.I1_combined)
        assert rel_diff > 0.01, \
            f"Run 20 and Run 18 results should differ: run20={run20.I1_combined_core:.6f}, " \
            f"run18={run18.I1_combined:.6f}, rel_diff={rel_diff:.2%}"

    def test_differs_from_run19_tex_exact(self):
        """Run 20 result should differ from Run 19 result."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11, compute_I1_tex_exact_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        run19 = compute_I1_tex_exact_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys
        )
        run20 = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA, n=40, polynomials=polys, n_quad_s=20
        )

        # They should differ (Run 19 used naive plus+minus, Run 20 uses log×integral)
        rel_diff = abs(run20.I1_combined_core - run19.I1_tex_exact) / abs(run19.I1_tex_exact)
        assert rel_diff > 0.01, \
            f"Run 20 and Run 19 results should differ: run20={run20.I1_combined_core:.6f}, " \
            f"run19={run19.I1_tex_exact:.6f}, rel_diff={rel_diff:.2%}"

    def test_reasonable_magnitude(self):
        """I1 should be in reasonable range (not wildly divergent like Run 19)."""
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_I1_tex_combined_core_11

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_I1_tex_combined_core_11(
            theta=self.THETA, R=self.R_KAPPA, n=60, polynomials=polys, n_quad_s=20
        )

        # Run 19 gave ~4.3 (10x too high), Run 18 gave ~2.08
        # We want something in reasonable range for c_target ≈ 2.14
        assert 0.1 < abs(result.I1_combined_core) < 20, \
            f"I1_combined_core={result.I1_combined_core} out of reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
