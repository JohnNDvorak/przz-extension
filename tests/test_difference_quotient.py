"""
tests/test_difference_quotient.py
Phase 21: Gate Tests for Difference Quotient Implementation

PURPOSE:
========
Verify that the difference quotient module correctly implements the
PRZZ identity (TeX Lines 1502-1511) and can be used to achieve D = 0.

GATE TESTS:
===========
1. Scalar identity verification: (exp(2R)-1)/(2R) == ∫exp(2Rt)dt
2. Series coefficient extraction works correctly
3. Q factor integration works correctly
4. Cross-benchmark consistency

USAGE:
======
    pytest tests/test_difference_quotient.py -v
"""

import pytest
import numpy as np

from src.difference_quotient import (
    # Scalar functions
    scalar_difference_quotient_lhs,
    scalar_difference_quotient_rhs,
    verify_scalar_difference_quotient,
    przz_scalar_limit,
    przz_scalar_limit_via_t_integral,
    verify_przz_scalar_limit,
    # Eigenvalue functions
    get_direct_eigenvalues,
    get_mirror_eigenvalues,
    get_unified_bracket_eigenvalues,
    # Series construction
    build_bracket_exp_series,
    build_log_factor_series,
    build_q_factor_series,
    # Main class
    DifferenceQuotientBracket,
    BracketEvaluationResult,
    # Convenience functions
    create_difference_quotient_evaluator,
    run_scalar_gate_test,
)


class TestScalarDifferenceQuotient:
    """Test the scalar difference quotient identity."""

    def test_lhs_basic(self):
        """LHS should compute (1 - z^{-s}) / s correctly."""
        z = 2.0
        s = 1.0
        result = scalar_difference_quotient_lhs(z, s)
        expected = (1 - 2**(-1)) / 1
        assert abs(result - expected) < 1e-14

    def test_rhs_basic(self):
        """RHS should compute log(z) * integral correctly."""
        z = 2.0
        s = 1.0
        result = scalar_difference_quotient_rhs(z, s, n_quad=50)
        # RHS = log(z) * ∫₀¹ z^{-ts} dt
        # Should equal LHS = (1 - z^{-s})/s = (1 - 0.5)/1 = 0.5
        expected = 0.5
        assert abs(result - expected) < 1e-10

    def test_identity_holds(self):
        """The difference quotient identity should hold."""
        z = 3.0
        s = 2.0
        lhs, rhs, rel_error = verify_scalar_difference_quotient(z, s, n_quad=50)
        assert rel_error < 1e-10

    @pytest.mark.parametrize("z,s", [
        (2.0, 0.5),
        (3.0, 1.5),
        (10.0, 0.1),
        (1.5, 3.0),
    ])
    def test_identity_multiple_values(self, z, s):
        """Identity should hold for various z, s values."""
        lhs, rhs, rel_error = verify_scalar_difference_quotient(z, s, n_quad=60)
        assert rel_error < 1e-9, f"z={z}, s={s}: rel_error={rel_error}"

    def test_lhs_raises_on_zero_s(self):
        """LHS should raise on s=0."""
        with pytest.raises(ValueError):
            scalar_difference_quotient_lhs(2.0, 0.0)


class TestPRZZScalarLimit:
    """Test PRZZ-specific scalar limit computations."""

    def test_przz_scalar_limit_kappa(self):
        """Scalar limit for κ benchmark should match analytic."""
        R = 1.3036
        analytic = przz_scalar_limit(R)
        expected = (np.exp(2 * R) - 1) / (2 * R)
        assert abs(analytic - expected) < 1e-14

    def test_przz_scalar_limit_kappa_star(self):
        """Scalar limit for κ* benchmark should match analytic."""
        R = 1.1167
        analytic = przz_scalar_limit(R)
        expected = (np.exp(2 * R) - 1) / (2 * R)
        assert abs(analytic - expected) < 1e-14

    def test_t_integral_matches_analytic_kappa(self):
        """t-integral should match analytic for κ."""
        R = 1.3036
        analytic, quadrature, rel_error = verify_przz_scalar_limit(R, n_quad=50)
        assert rel_error < 1e-10

    def test_t_integral_matches_analytic_kappa_star(self):
        """t-integral should match analytic for κ*."""
        R = 1.1167
        analytic, quadrature, rel_error = verify_przz_scalar_limit(R, n_quad=50)
        assert rel_error < 1e-10

    def test_scalar_limit_increases_with_R(self):
        """Scalar limit should increase with R."""
        R1, R2 = 1.0, 1.5
        limit1 = przz_scalar_limit(R1)
        limit2 = przz_scalar_limit(R2)
        assert limit2 > limit1


class TestEigenvalues:
    """Test eigenvalue computation functions."""

    def test_direct_eigenvalues_at_t_zero(self):
        """At t=0, eigenvalues should have specific structure."""
        t = 0.0
        theta = 4.0 / 7.0
        (u0_a, x_a, y_a), (u0_b, x_b, y_b) = get_direct_eigenvalues(t, theta)

        assert abs(u0_a - 0.0) < 1e-14
        assert abs(x_a - theta * (0 - 1)) < 1e-14  # θ(t-1) = -θ
        assert abs(y_a - 0.0) < 1e-14  # θt = 0

    def test_direct_eigenvalues_at_t_one(self):
        """At t=1, eigenvalues should have specific structure."""
        t = 1.0
        theta = 4.0 / 7.0
        (u0_a, x_a, y_a), (u0_b, x_b, y_b) = get_direct_eigenvalues(t, theta)

        assert abs(u0_a - 1.0) < 1e-14
        assert abs(x_a - 0.0) < 1e-14  # θ(t-1) = 0
        assert abs(y_a - theta) < 1e-14  # θt = θ

    def test_eigenvalues_symmetric_at_t_half(self):
        """At t=0.5, eigenvalues should show symmetry."""
        t = 0.5
        theta = 4.0 / 7.0
        (u0_a, x_a, y_a), (u0_b, x_b, y_b) = get_direct_eigenvalues(t, theta)

        # u0 values should be equal
        assert abs(u0_a - u0_b) < 1e-14

        # Linear coefficients should be swapped
        assert abs(x_a - y_b) < 1e-14
        assert abs(y_a - x_b) < 1e-14

    def test_mirror_eigenvalues(self):
        """Mirror eigenvalues should be θ for both."""
        theta = 4.0 / 7.0
        y_coeff, x_coeff = get_mirror_eigenvalues(theta)
        assert abs(y_coeff - theta) < 1e-14
        assert abs(x_coeff - theta) < 1e-14

    def test_unified_equals_direct(self):
        """Unified eigenvalues should equal direct eigenvalues."""
        t = 0.3
        theta = 4.0 / 7.0
        direct = get_direct_eigenvalues(t, theta)
        unified = get_unified_bracket_eigenvalues(t, theta)
        assert direct == unified


class TestSeriesConstruction:
    """Test series construction functions."""

    def test_exp_series_at_t_zero(self):
        """At t=0, exp series should be exp(0) = 1."""
        t = 0.0
        theta = 4.0 / 7.0
        R = 1.3036
        series = build_bracket_exp_series(t, theta, R)

        # Constant term should be exp(0) = 1
        constant = series.coeffs.get(0, 0.0)
        assert abs(float(constant) - 1.0) < 1e-14

    def test_exp_series_at_t_one(self):
        """At t=1, exp series constant should be exp(2R)."""
        t = 1.0
        theta = 4.0 / 7.0
        R = 1.3036
        series = build_bracket_exp_series(t, theta, R)

        constant = series.coeffs.get(0, 0.0)
        expected = np.exp(2 * R)
        assert abs(float(constant) - expected) < 1e-10

    def test_log_factor_structure(self):
        """Log factor should be 1 + θ(x+y)."""
        theta = 4.0 / 7.0
        series = build_log_factor_series(theta)

        # Should have constant term 1
        assert abs(float(series.coeffs.get(0, 0)) - 1.0) < 1e-14

        # Should have x coefficient θ
        x_mask = 1 << 0  # x is first variable
        assert abs(float(series.coeffs.get(x_mask, 0)) - theta) < 1e-14

        # Should have y coefficient θ
        y_mask = 1 << 1  # y is second variable
        assert abs(float(series.coeffs.get(y_mask, 0)) - theta) < 1e-14

    def test_exp_series_has_linear_terms(self):
        """Exp series should have x and y linear terms (at t != 0.5)."""
        # Note: At t=0.5, lin_coeff = Rθ(2t-1) = 0, so use t=0.7 instead
        t = 0.7
        theta = 4.0 / 7.0
        R = 1.3036
        series = build_bracket_exp_series(t, theta, R)

        # Check x coefficient exists
        x_mask = 1 << 0
        x_coeff = series.coeffs.get(x_mask, 0.0)
        assert abs(float(x_coeff)) > 1e-10, "Should have x linear term"

        # Check y coefficient exists
        y_mask = 1 << 1
        y_coeff = series.coeffs.get(y_mask, 0.0)
        assert abs(float(y_coeff)) > 1e-10, "Should have y linear term"

        # x and y coefficients should be equal (symmetric in x,y)
        assert abs(float(x_coeff) - float(y_coeff)) < 1e-12

    def test_exp_series_zero_linear_at_t_half(self):
        """At t=0.5, linear coefficients should be zero (Rθ(2t-1)=0)."""
        t = 0.5
        theta = 4.0 / 7.0
        R = 1.3036
        series = build_bracket_exp_series(t, theta, R)

        # At t=0.5, the linear coefficient Rθ(2*0.5-1) = 0
        x_mask = 1 << 0
        x_coeff = series.coeffs.get(x_mask, 0.0)
        assert abs(float(x_coeff)) < 1e-14, "Should be zero at t=0.5"


class TestDifferenceQuotientBracket:
    """Test the main DifferenceQuotientBracket class."""

    def test_creation_kappa(self):
        """Should create bracket for κ benchmark."""
        bracket = DifferenceQuotientBracket(theta=4/7, R=1.3036)
        assert bracket.theta == 4/7
        assert bracket.R == 1.3036

    def test_creation_kappa_star(self):
        """Should create bracket for κ* benchmark."""
        bracket = DifferenceQuotientBracket(theta=4/7, R=1.1167)
        assert bracket.R == 1.1167

    def test_verify_scalar_identity_kappa(self):
        """Scalar identity should pass for κ."""
        bracket = DifferenceQuotientBracket(R=1.3036)
        passed, analytic, quadrature, rel_error = bracket.verify_scalar_identity()
        assert passed
        assert rel_error < 1e-10

    def test_verify_scalar_identity_kappa_star(self):
        """Scalar identity should pass for κ*."""
        bracket = DifferenceQuotientBracket(R=1.1167)
        passed, analytic, quadrature, rel_error = bracket.verify_scalar_identity()
        assert passed
        assert rel_error < 1e-10

    def test_evaluate_scalar_integral_kappa(self):
        """Scalar integral should match analytic for κ."""
        bracket = DifferenceQuotientBracket(R=1.3036)
        quadrature = bracket.evaluate_scalar_integral()
        analytic = przz_scalar_limit(1.3036)
        assert abs(quadrature - analytic) / abs(analytic) < 1e-10

    def test_evaluate_xy_coefficient_nonzero(self):
        """xy coefficient integral should be non-zero."""
        bracket = DifferenceQuotientBracket(R=1.3036)
        xy_coeff = bracket.evaluate_xy_coefficient_integral()
        assert abs(xy_coeff) > 0.1  # Should be substantial

    def test_compute_bracket_result_returns_dataclass(self):
        """compute_bracket_result should return BracketEvaluationResult."""
        bracket = DifferenceQuotientBracket(R=1.3036)
        result = bracket.compute_bracket_result()
        assert isinstance(result, BracketEvaluationResult)

    def test_result_has_all_fields(self):
        """Result should have all expected fields."""
        bracket = DifferenceQuotientBracket(R=1.3036)
        result = bracket.compute_bracket_result()

        assert hasattr(result, 'scalar_limit')
        assert hasattr(result, 't_values')
        assert hasattr(result, 'integrand_values')
        assert hasattr(result, 'integrated_value')
        assert hasattr(result, 'analytic_expectation')
        assert hasattr(result, 'relative_error')


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_evaluator_kappa(self):
        """Should create evaluator for κ."""
        bracket = create_difference_quotient_evaluator("kappa")
        assert bracket.R == 1.3036

    def test_create_evaluator_kappa_star(self):
        """Should create evaluator for κ*."""
        bracket = create_difference_quotient_evaluator("kappa_star")
        assert bracket.R == 1.1167

    def test_create_evaluator_invalid(self):
        """Should raise for invalid benchmark."""
        with pytest.raises(ValueError):
            create_difference_quotient_evaluator("invalid")

    def test_run_scalar_gate_test_kappa(self):
        """Scalar gate test should pass for κ."""
        passed = run_scalar_gate_test(R=1.3036)
        assert passed

    def test_run_scalar_gate_test_kappa_star(self):
        """Scalar gate test should pass for κ*."""
        passed = run_scalar_gate_test(R=1.1167)
        assert passed


class TestQuadratureConvergence:
    """Test that quadrature achieves high accuracy."""

    def test_scalar_limit_high_accuracy(self):
        """Scalar limit should achieve machine precision accuracy."""
        R = 1.3036
        analytic = przz_scalar_limit(R)

        # With n=40, should be at machine precision
        quadrature = przz_scalar_limit_via_t_integral(R, n_quad=40)
        rel_error = abs(quadrature - analytic) / abs(analytic)

        # Should be at machine precision
        assert rel_error < 1e-12, f"Should achieve high accuracy: rel_error={rel_error}"

    def test_xy_coefficient_stable(self):
        """xy coefficient should be stable across quadrature settings."""
        n_values = [20, 40, 80]
        results = []

        for n in n_values:
            bracket = DifferenceQuotientBracket(R=1.3036, n_quad_t=n)
            xy_coeff = bracket.evaluate_xy_coefficient_integral()
            results.append(xy_coeff)

        # All values should be close to each other (stable)
        avg = sum(results) / len(results)
        for r in results:
            rel_diff = abs(r - avg) / abs(avg)
            assert rel_diff < 1e-10, f"Should be stable: {results}"


class TestCrossBenchmarkConsistency:
    """Test consistency between κ and κ* benchmarks."""

    def test_both_benchmarks_pass_scalar_identity(self):
        """Both benchmarks should pass scalar identity."""
        for benchmark in ["kappa", "kappa_star"]:
            bracket = create_difference_quotient_evaluator(benchmark)
            passed, _, _, rel_error = bracket.verify_scalar_identity()
            assert passed, f"{benchmark} failed: rel_error={rel_error}"

    def test_xy_coefficient_positive_both(self):
        """xy coefficient should be positive for both benchmarks."""
        for benchmark in ["kappa", "kappa_star"]:
            bracket = create_difference_quotient_evaluator(benchmark)
            xy_coeff = bracket.evaluate_xy_coefficient_integral()
            assert xy_coeff > 0, f"{benchmark}: xy_coeff = {xy_coeff}"

    def test_scalar_limit_larger_for_kappa(self):
        """κ has larger R, so should have larger scalar limit."""
        kappa = create_difference_quotient_evaluator("kappa")
        kappa_star = create_difference_quotient_evaluator("kappa_star")

        limit_kappa = kappa.evaluate_scalar_integral()
        limit_kappa_star = kappa_star.evaluate_scalar_integral()

        assert limit_kappa > limit_kappa_star
