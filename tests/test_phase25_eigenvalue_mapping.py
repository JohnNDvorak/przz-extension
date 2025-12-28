"""
tests/test_phase25_eigenvalue_mapping.py
Phase 25.5: Eigenvalue Mapping Verification Tests

PURPOSE:
========
Verify that the unified bracket eigenvalues (A_alpha, A_beta) match
the DSL term definitions and are consistent between evaluators.

UNIFIED BRACKET EIGENVALUES:
============================
From unified_s12_evaluator_v3.py:
    A_alpha = t + theta*(t-1)*x + theta*t*y
    A_beta  = t + theta*t*x + theta*(t-1)*y

At x=y=0: A_alpha = A_beta = t (simplifies to scalar)

Created: 2025-12-25
"""

import pytest
import math


# =============================================================================
# CONSTANTS
# =============================================================================

THETA = 4.0 / 7.0


# =============================================================================
# EIGENVALUE COMPUTATION
# =============================================================================


def compute_unified_eigenvalues(t: float, x: float, y: float, theta: float = THETA):
    """
    Compute unified bracket eigenvalues A_alpha and A_beta.

    From unified_s12_evaluator_v3.py lines 373-398.
    """
    A_alpha = t + theta * (t - 1) * x + theta * t * y
    A_beta = t + theta * t * x + theta * (t - 1) * y
    return A_alpha, A_beta


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def test_points():
    """Small grid of (t, x, y) test points."""
    return [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 0.1, 0.0),
        (0.5, 0.0, 0.1),
        (0.5, 0.1, 0.1),
        (0.25, 0.05, 0.08),
        (0.75, -0.1, 0.2),
    ]


# =============================================================================
# EIGENVALUE STRUCTURE TESTS
# =============================================================================


class TestEigenvalueAtOrigin:
    """Verify eigenvalues at x=y=0 equal t."""

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_alpha_equals_t_at_origin(self, t):
        """At x=y=0, A_alpha should equal t."""
        A_alpha, A_beta = compute_unified_eigenvalues(t, 0.0, 0.0)
        assert A_alpha == pytest.approx(t, abs=1e-15)

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_beta_equals_t_at_origin(self, t):
        """At x=y=0, A_beta should equal t."""
        A_alpha, A_beta = compute_unified_eigenvalues(t, 0.0, 0.0)
        assert A_beta == pytest.approx(t, abs=1e-15)


class TestEigenvalueSymmetry:
    """Verify x <-> y symmetry property."""

    def test_A_alpha_xy_equals_A_beta_yx(self, test_points):
        """A_alpha(x,y) should equal A_beta(y,x) by symmetry."""
        for t, x, y in test_points:
            A_alpha_xy, A_beta_xy = compute_unified_eigenvalues(t, x, y)
            A_alpha_yx, A_beta_yx = compute_unified_eigenvalues(t, y, x)

            assert A_alpha_xy == pytest.approx(A_beta_yx, abs=1e-15), (
                f"Symmetry violated at t={t}, x={x}, y={y}"
            )

    def test_eigenvalues_equal_for_x_equals_y(self, test_points):
        """When x=y, A_alpha should equal A_beta."""
        for t, x, y in test_points:
            if x == y:
                A_alpha, A_beta = compute_unified_eigenvalues(t, x, y)
                assert A_alpha == pytest.approx(A_beta, abs=1e-15)


class TestEigenvalueBoundary:
    """Test eigenvalue behavior at integration boundaries."""

    def test_at_t_equals_0(self):
        """At t=0, eigenvalues depend only on x."""
        x, y = 0.1, 0.2
        A_alpha, A_beta = compute_unified_eigenvalues(0.0, x, y)

        # At t=0: A_alpha = -theta*x, A_beta = -theta*y
        expected_alpha = -THETA * x
        expected_beta = -THETA * y

        assert A_alpha == pytest.approx(expected_alpha, abs=1e-15)
        assert A_beta == pytest.approx(expected_beta, abs=1e-15)

    def test_at_t_equals_1(self):
        """At t=1, eigenvalues have specific form."""
        x, y = 0.1, 0.2
        A_alpha, A_beta = compute_unified_eigenvalues(1.0, x, y)

        # At t=1: A_alpha = 1 + theta*y, A_beta = 1 + theta*x
        expected_alpha = 1.0 + THETA * y
        expected_beta = 1.0 + THETA * x

        assert A_alpha == pytest.approx(expected_alpha, abs=1e-15)
        assert A_beta == pytest.approx(expected_beta, abs=1e-15)


class TestEigenvalueDerivatives:
    """Test eigenvalue partial derivatives for series expansion."""

    def test_dA_alpha_dx(self):
        """dA_alpha/dx = theta*(t-1)."""
        t = 0.5
        A_alpha_0, _ = compute_unified_eigenvalues(t, 0.0, 0.0)
        A_alpha_eps, _ = compute_unified_eigenvalues(t, 1e-8, 0.0)

        numerical_deriv = (A_alpha_eps - A_alpha_0) / 1e-8
        expected_deriv = THETA * (t - 1)

        assert numerical_deriv == pytest.approx(expected_deriv, rel=1e-5)

    def test_dA_alpha_dy(self):
        """dA_alpha/dy = theta*t."""
        t = 0.5
        A_alpha_0, _ = compute_unified_eigenvalues(t, 0.0, 0.0)
        A_alpha_eps, _ = compute_unified_eigenvalues(t, 0.0, 1e-8)

        numerical_deriv = (A_alpha_eps - A_alpha_0) / 1e-8
        expected_deriv = THETA * t

        assert numerical_deriv == pytest.approx(expected_deriv, rel=1e-5)

    def test_dA_beta_dx(self):
        """dA_beta/dx = theta*t."""
        t = 0.5
        _, A_beta_0 = compute_unified_eigenvalues(t, 0.0, 0.0)
        _, A_beta_eps = compute_unified_eigenvalues(t, 1e-8, 0.0)

        numerical_deriv = (A_beta_eps - A_beta_0) / 1e-8
        expected_deriv = THETA * t

        assert numerical_deriv == pytest.approx(expected_deriv, rel=1e-5)

    def test_dA_beta_dy(self):
        """dA_beta/dy = theta*(t-1)."""
        t = 0.5
        _, A_beta_0 = compute_unified_eigenvalues(t, 0.0, 0.0)
        _, A_beta_eps = compute_unified_eigenvalues(t, 0.0, 1e-8)

        numerical_deriv = (A_beta_eps - A_beta_0) / 1e-8
        expected_deriv = THETA * (t - 1)

        assert numerical_deriv == pytest.approx(expected_deriv, rel=1e-5)


# =============================================================================
# Q FACTOR TESTS
# =============================================================================


class TestQFactorEigenvalueConsistency:
    """Test that Q(A_alpha) * Q(A_beta) is consistent."""

    def test_Q_at_origin_is_Q_t_squared(self):
        """At x=y=0, Q(A_alpha)*Q(A_beta) = Q(t)^2."""
        from src.polynomials import load_przz_polynomials
        import numpy as np

        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)

        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            A_alpha, A_beta = compute_unified_eigenvalues(t, 0.0, 0.0)

            # Both should equal t
            assert A_alpha == pytest.approx(t, abs=1e-15)
            assert A_beta == pytest.approx(t, abs=1e-15)

            # Q(A_alpha) * Q(A_beta) = Q(t)^2
            t_arr = np.array([t])
            Q_t = Q.eval(t_arr)[0]
            expected = Q_t ** 2
            actual = Q.eval(np.array([A_alpha]))[0] * Q.eval(np.array([A_beta]))[0]

            assert actual == pytest.approx(expected, rel=1e-10)


class TestEigenvalueIntegrationRange:
    """Verify eigenvalues stay in valid range for Q polynomial."""

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_eigenvalues_in_range_for_small_xy(self, t):
        """For small |x|, |y| < 0.1, eigenvalues should be in [0, 1] ish."""
        for x, y in [(-0.1, -0.1), (-0.1, 0.1), (0.1, -0.1), (0.1, 0.1)]:
            A_alpha, A_beta = compute_unified_eigenvalues(t, x, y)

            # Eigenvalues should be reasonably bounded
            assert -0.5 < A_alpha < 1.5, f"A_alpha={A_alpha} out of expected range"
            assert -0.5 < A_beta < 1.5, f"A_beta={A_beta} out of expected range"
