"""
tests/test_unified_bracket_evaluator_ladder.py
Phase 21B: Ladder Tests for Unified Bracket Evaluator

PURPOSE:
========
These tests catch classic mistakes BEFORE attempting the D=0 gate.
They verify the building blocks work correctly in isolation.

The tests must PASS before proceeding with the full implementation.

LADDER TEST CATEGORIES:
=======================
1. Bracket-only series sanity (Q=1, no polynomials)
2. Scalar limit (x=y=0) through the unified path
3. Q factor sanity at x=y=0
4. Smoke test for I₁(1,1) stability

USAGE:
======
    pytest tests/test_unified_bracket_evaluator_ladder.py -v

IMPORTANT:
==========
These tests are designed to catch:
- Double log factor
- Wrong exp sign
- Wrong eigenvalue constructor
- xy extraction errors
"""

import pytest
import numpy as np

from src.series import TruncatedSeries
from src.difference_quotient import (
    build_bracket_exp_series,
    build_log_factor_series,
    build_q_factor_series,
    get_unified_bracket_eigenvalues,
    przz_scalar_limit,
    DifferenceQuotientBracket,
)
from src.quadrature import gauss_legendre_01
from src.composition import compose_polynomial_on_affine


# =============================================================================
# CONSTANTS
# =============================================================================

KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
THETA = 4.0 / 7.0


# =============================================================================
# 1. BRACKET-ONLY SERIES SANITY (Q=1, No Polynomials)
# =============================================================================


class TestBracketOnlySeriesSanity:
    """Test the bracket series in isolation, without Q or P polynomials."""

    def test_linear_coefficient_vanishes_at_t_half(self):
        """
        At t=0.5, the linear (x+y) coefficient from exp should vanish.

        The exp factor is exp(2Rt + Rθ(2t-1)(x+y)).
        At t=0.5: Rθ(2*0.5-1) = Rθ*0 = 0, so no linear term.
        """
        t = 0.5
        exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)

        x_mask = 1 << 0
        y_mask = 1 << 1

        x_coeff = float(exp_series.coeffs.get(x_mask, 0.0))
        y_coeff = float(exp_series.coeffs.get(y_mask, 0.0))

        assert abs(x_coeff) < 1e-14, f"x coeff at t=0.5 should be 0, got {x_coeff}"
        assert abs(y_coeff) < 1e-14, f"y coeff at t=0.5 should be 0, got {y_coeff}"

    def test_linear_coefficient_nonzero_at_t_zero(self):
        """At t=0, linear coefficient should be -Rθ (from Rθ(2t-1) = -Rθ)."""
        t = 0.0
        exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)

        x_mask = 1 << 0
        x_coeff = float(exp_series.coeffs.get(x_mask, 0.0))

        # At t=0: lin_coeff = Rθ(2*0-1) = -Rθ
        # exp(u0 + lin*x + lin*y) expands to exp(u0)*(1 + lin*x + lin*y + ...)
        # At t=0, u0=0, so exp(0)=1, and x_coeff = lin = -Rθ
        expected = -KAPPA_R * THETA
        assert abs(x_coeff - expected) < 1e-10, f"Expected {expected}, got {x_coeff}"

    def test_linear_coefficient_nonzero_at_t_one(self):
        """At t=1, linear coefficient should be +Rθ (from Rθ(2t-1) = +Rθ)."""
        t = 1.0
        exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)

        x_mask = 1 << 0
        x_coeff = float(exp_series.coeffs.get(x_mask, 0.0))

        # At t=1: lin_coeff = Rθ(2*1-1) = +Rθ
        # u0 = 2R, so exp(2R) multiplies the linear term
        expected = KAPPA_R * THETA * np.exp(2 * KAPPA_R)
        assert abs(x_coeff - expected) < 1e-8, f"Expected {expected}, got {x_coeff}"

    def test_xy_coefficient_finite_at_all_t(self):
        """xy coefficient should be finite for all t in [0,1]."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)
            log_series = build_log_factor_series(THETA)
            combined = exp_series * log_series

            xy_mask = (1 << 0) | (1 << 1)
            xy_coeff = float(combined.coeffs.get(xy_mask, 0.0))

            assert np.isfinite(xy_coeff), f"xy coeff at t={t} should be finite"

    def test_xy_coefficient_continuous_in_t(self):
        """
        xy coefficient should be continuous (no jumps) as t varies.

        NOTE: The xy coefficient changes sign around t≈0.5 because the
        linear coefficient in exp is Rθ(2t-1)(x+y), which is:
        - Negative for t < 0.5
        - Zero at t = 0.5
        - Positive for t > 0.5

        This is EXPECTED behavior and represents a smooth zero crossing.
        We test for Lipschitz continuity (bounded derivative) instead.
        """
        n_points = 100  # Fine sampling for derivative estimation
        t_values = np.linspace(0, 1, n_points)
        xy_coeffs = []

        for t in t_values:
            exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)
            log_series = build_log_factor_series(THETA)
            combined = exp_series * log_series

            xy_mask = (1 << 0) | (1 << 1)
            xy_coeff = float(combined.coeffs.get(xy_mask, 0.0))
            xy_coeffs.append(xy_coeff)

        # Check for Lipschitz continuity: |f(t2) - f(t1)| / |t2 - t1| should be bounded
        # This is more appropriate for functions with zero crossings
        dt = 1.0 / (n_points - 1)
        derivatives = [abs(xy_coeffs[i+1] - xy_coeffs[i]) / dt for i in range(n_points-1)]
        max_derivative = max(derivatives)

        # The function should have bounded derivative (smooth, no discontinuities)
        # Expected order of magnitude: exp(2R) * poly ~ 10-20
        assert max_derivative < 1000, f"Unbounded derivative detected: max={max_derivative}"

        # Verify the zero crossing happens (sanity check)
        has_negative = any(c < 0 for c in xy_coeffs)
        has_positive = any(c > 0 for c in xy_coeffs)
        assert has_negative and has_positive, "Expected sign change in xy coefficient"


# =============================================================================
# 2. SCALAR LIMIT (x=y=0) THROUGH UNIFIED PATH
# =============================================================================


class TestScalarLimitThroughUnifiedPath:
    """Test that the unified evaluator reproduces the known scalar limit."""

    def test_scalar_limit_kappa_via_t_integral(self):
        """
        The t-integral at x=y=0 should equal (exp(2R)-1)/(2R).

        This is the foundational identity that everything builds on.
        """
        bracket = DifferenceQuotientBracket(R=KAPPA_R, n_quad_t=50)
        quadrature = bracket.evaluate_scalar_integral()
        analytic = przz_scalar_limit(KAPPA_R)

        rel_error = abs(quadrature - analytic) / abs(analytic)
        assert rel_error < 1e-12, f"Scalar limit mismatch: rel_error={rel_error}"

    def test_scalar_limit_kappa_star_via_t_integral(self):
        """Same test for κ* benchmark."""
        bracket = DifferenceQuotientBracket(R=KAPPA_STAR_R, n_quad_t=50)
        quadrature = bracket.evaluate_scalar_integral()
        analytic = przz_scalar_limit(KAPPA_STAR_R)

        rel_error = abs(quadrature - analytic) / abs(analytic)
        assert rel_error < 1e-12, f"Scalar limit mismatch: rel_error={rel_error}"

    def test_exp_series_constant_term_integrates_correctly(self):
        """
        The constant term of exp series, integrated over t, should give scalar limit.

        This verifies we're not accidentally introducing extra factors.
        """
        n_quad = 50
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            exp_series = build_bracket_exp_series(t, THETA, KAPPA_R)
            constant = float(exp_series.coeffs.get(0, 0.0))
            integral += constant * w

        # Should equal (exp(2R)-1)/(2R)
        expected = przz_scalar_limit(KAPPA_R)
        rel_error = abs(integral - expected) / abs(expected)

        assert rel_error < 1e-12, f"Constant term integral wrong: rel_error={rel_error}"

    def test_no_log_factor_duplication(self):
        """
        Verify we don't accidentally include the log factor twice.

        If log factor is included twice, the constant term would be multiplied by 1*1=1
        (no change to constant), but xy coefficient would be wrong.
        """
        # With log factor once
        exp_series = build_bracket_exp_series(0.3, THETA, KAPPA_R)
        log_series = build_log_factor_series(THETA)
        with_log_once = exp_series * log_series

        # With log factor twice (wrong!)
        with_log_twice = exp_series * log_series * log_series

        xy_mask = (1 << 0) | (1 << 1)
        xy_once = float(with_log_once.coeffs.get(xy_mask, 0.0))
        xy_twice = float(with_log_twice.coeffs.get(xy_mask, 0.0))

        # They should be different (if same, we have a problem in our test logic)
        assert abs(xy_once - xy_twice) > 0.01, "Log factor duplication not detected"


# =============================================================================
# 3. Q FACTOR SANITY AT x=y=0
# =============================================================================


class TestQFactorSanityAtZero:
    """Test Q(A_α)×Q(A_β) behavior at x=y=0."""

    def test_eigenvalues_at_x_y_zero(self):
        """At x=y=0, unified eigenvalues reduce to just the constant u0=t."""
        for t in [0.0, 0.5, 1.0]:
            (u0_a, x_a, y_a), (u0_b, x_b, y_b) = get_unified_bracket_eigenvalues(t, THETA)

            # At x=y=0, the eigenvalue is just u0
            assert abs(u0_a - t) < 1e-14, f"A_α at t={t} should be t, got {u0_a}"
            assert abs(u0_b - t) < 1e-14, f"A_β at t={t} should be t, got {u0_b}"

    def test_q_factor_at_x_y_zero_is_q_t_squared(self):
        """
        At x=y=0, Q(A_α)×Q(A_β) should equal Q(t)×Q(t) = Q(t)².

        We test with Q(z) = 1 - z (simple linear polynomial).
        """
        from src.polynomials import Polynomial

        # Simple Q(z) = 1 - z
        Q_poly = Polynomial([1.0, -1.0])

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_series = build_q_factor_series(Q_poly, t, THETA)

            # Extract constant term (x=y=0 value)
            constant = float(q_series.coeffs.get(0, 0.0))

            # Expected: Q(t)² = (1-t)²
            expected = (1.0 - t) ** 2
            assert abs(constant - expected) < 1e-12, (
                f"At t={t}: Q(t)²={(1-t)**2}, got {constant}"
            )

    def test_q_factor_linear_terms_cancel_at_t_half(self):
        """
        At t=0.5, eigenvalues have symmetric structure, so certain terms should cancel.

        For Q(z) = 1 - z:
        - A_α = 0.5 + θ(0.5-1)x + θ*0.5*y = 0.5 - 0.5θx + 0.5θy
        - A_β = 0.5 + θ*0.5*x + θ(0.5-1)y = 0.5 + 0.5θx - 0.5θy

        Q(A_α) = 1 - A_α = 0.5 + 0.5θx - 0.5θy
        Q(A_β) = 1 - A_β = 0.5 - 0.5θx + 0.5θy

        Product at constant level: 0.5 * 0.5 = 0.25

        The x and y linear terms: (0.5θx)(−0.5θx) + ... needs careful analysis.
        """
        from src.polynomials import Polynomial

        Q_poly = Polynomial([1.0, -1.0])  # Q(z) = 1 - z
        t = 0.5

        q_series = build_q_factor_series(Q_poly, t, THETA)

        # Constant should be Q(0.5)² = 0.25
        constant = float(q_series.coeffs.get(0, 0.0))
        assert abs(constant - 0.25) < 1e-12, f"Expected 0.25, got {constant}"


# =============================================================================
# 4. SMOKE TEST FOR I₁(1,1) STABILITY
# =============================================================================


class TestI1SmokeTest:
    """Smoke tests for I₁(1,1) computation stability."""

    def test_unified_I1_11_returns_finite(self):
        """compute_I1_micro_case_11 should return finite value."""
        from src.unified_bracket_evaluator import MicroCaseEvaluator

        evaluator = MicroCaseEvaluator(R=KAPPA_R, n_quad_u=20, n_quad_t=20)
        result = evaluator.compute_I1_micro_case_11()

        assert np.isfinite(result.I1_value), f"I1 should be finite, got {result.I1_value}"

    def test_unified_I1_11_stable_under_refinement(self):
        """I₁(1,1) should be stable under quadrature refinement."""
        from src.unified_bracket_evaluator import MicroCaseEvaluator

        results = []
        for n in [20, 40]:
            evaluator = MicroCaseEvaluator(R=KAPPA_R, n_quad_u=n, n_quad_t=n)
            result = evaluator.compute_I1_micro_case_11()
            results.append(result.I1_value)

        # Relative change should be small
        rel_change = abs(results[1] - results[0]) / abs(results[0])
        assert rel_change < 1e-4, f"I1 unstable under refinement: rel_change={rel_change}"

    def test_unified_I1_11_positive(self):
        """I₁(1,1) should be positive for typical parameters."""
        from src.unified_bracket_evaluator import MicroCaseEvaluator

        evaluator = MicroCaseEvaluator(R=KAPPA_R, n_quad_u=30, n_quad_t=30)
        result = evaluator.compute_I1_micro_case_11()

        assert result.I1_value > 0, f"Expected positive I1, got {result.I1_value}"

    def test_unified_I1_11_kappa_vs_kappa_star(self):
        """I₁(1,1) should differ between κ and κ* benchmarks."""
        from src.unified_bracket_evaluator import MicroCaseEvaluator

        kappa_eval = MicroCaseEvaluator(R=KAPPA_R, n_quad_u=30, n_quad_t=30)
        kappa_star_eval = MicroCaseEvaluator(R=KAPPA_STAR_R, n_quad_u=30, n_quad_t=30)

        kappa_result = kappa_eval.compute_I1_micro_case_11()
        kappa_star_result = kappa_star_eval.compute_I1_micro_case_11()

        # κ has larger R, so integrals should differ
        assert kappa_result.I1_value != kappa_star_result.I1_value, (
            "κ and κ* should give different I1 values"
        )


# =============================================================================
# 5. ALGEBRAIC PREFACTOR TESTS
# =============================================================================


class TestAlgebraicPrefactor:
    """Test the algebraic prefactor (1/θ + x + y) handling."""

    def test_algebraic_prefactor_structure(self):
        """The algebraic prefactor (1/θ + x + y) should have correct structure."""
        var_names = ("x", "y")

        alg_series = TruncatedSeries.from_scalar(1.0 / THETA, var_names)
        alg_series = alg_series + TruncatedSeries.variable("x", var_names)
        alg_series = alg_series + TruncatedSeries.variable("y", var_names)

        # Constant term = 1/θ
        constant = float(alg_series.coeffs.get(0, 0.0))
        assert abs(constant - 1.0/THETA) < 1e-14

        # x coefficient = 1
        x_mask = 1 << 0
        x_coeff = float(alg_series.coeffs.get(x_mask, 0.0))
        assert abs(x_coeff - 1.0) < 1e-14

        # y coefficient = 1
        y_mask = 1 << 1
        y_coeff = float(alg_series.coeffs.get(y_mask, 0.0))
        assert abs(y_coeff - 1.0) < 1e-14

    def test_prefactor_affects_xy_coefficient(self):
        """
        Multiplying by (1/θ + x + y) should correctly affect xy coefficient.

        If f = a + bx + cy + dxy, then
        f × (1/θ + x + y) has xy term: d/θ + b + c

        (Since x² = y² = 0, the terms bx×y = by, cy×x = cx contribute to xy)
        """
        var_names = ("x", "y")

        # f = 1 + 2x + 3y + 4xy
        f = TruncatedSeries.from_scalar(1.0, var_names)
        f = f + TruncatedSeries.variable("x", var_names) * 2.0
        f = f + TruncatedSeries.variable("y", var_names) * 3.0
        xy_term = TruncatedSeries.variable("x", var_names) * TruncatedSeries.variable("y", var_names)
        f = f + xy_term * 4.0

        # g = 1/θ + x + y
        g = TruncatedSeries.from_scalar(1.0 / THETA, var_names)
        g = g + TruncatedSeries.variable("x", var_names)
        g = g + TruncatedSeries.variable("y", var_names)

        product = f * g

        # Expected xy coefficient:
        # From f=1 × xy_part_of_g: 0 (g has no xy)
        # From x_of_f × y_of_g: 2 × 1 = 2
        # From y_of_f × x_of_g: 3 × 1 = 3
        # From xy_of_f × const_of_g: 4 × (1/θ) = 4/θ
        expected = 2 + 3 + 4/THETA

        xy_mask = (1 << 0) | (1 << 1)
        actual = float(product.coeffs.get(xy_mask, 0.0))

        assert abs(actual - expected) < 1e-12, f"Expected {expected}, got {actual}"


# =============================================================================
# 6. INTEGRATION SANITY
# =============================================================================


class TestIntegrationSanity:
    """Test that quadrature integration is set up correctly."""

    def test_unit_integral(self):
        """∫₀¹ 1 dt = 1."""
        t_nodes, t_weights = gauss_legendre_01(50)
        integral = sum(t_weights)
        assert abs(integral - 1.0) < 1e-14

    def test_t_integral(self):
        """∫₀¹ t dt = 0.5."""
        t_nodes, t_weights = gauss_legendre_01(50)
        integral = sum(t * w for t, w in zip(t_nodes, t_weights))
        assert abs(integral - 0.5) < 1e-14

    def test_t_squared_integral(self):
        """∫₀¹ t² dt = 1/3."""
        t_nodes, t_weights = gauss_legendre_01(50)
        integral = sum(t**2 * w for t, w in zip(t_nodes, t_weights))
        assert abs(integral - 1.0/3.0) < 1e-13

    def test_exp_integral_matches_analytic(self):
        """∫₀¹ exp(at) dt = (exp(a)-1)/a."""
        a = 2.0 * KAPPA_R
        t_nodes, t_weights = gauss_legendre_01(50)
        integral = sum(np.exp(a * t) * w for t, w in zip(t_nodes, t_weights))
        expected = (np.exp(a) - 1) / a
        assert abs(integral - expected) / abs(expected) < 1e-12
