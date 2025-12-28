"""
tests/test_q_affine_expansion.py
Phase 46A: Tests for Q-Derivative Analytic Kernel Module

Tests the nilpotent Taylor expansion and Q-moment computation.

KEY GATES:
==========
1. Q=1 gate: With Q(t)=1, Q'=Q''=0, so all derivative contributions must be 0
2. Formula vs series: Closed-form must match brute-force series computation
3. Quadrature stability: Results stable under refinement

Created: 2025-12-27 (Phase 46A)
"""
import pytest
import numpy as np
from src.polynomials import Polynomial
from src.unified_s12.q_affine_expansion import (
    q_affine_series_at_xy,
    q_product_xy_coeff,
    q_product_xy_coeff_post_identity,
    q_product_xy_coeff_post_identity_vectorized,
    compute_q_moments_under_weight,
    compute_frozen_vs_full_xy_ratio,
    verify_formula_vs_series,
    NilpotentSeriesCoeffs,
)


class TestNilpotentSeriesBasics:
    """Test basic nilpotent series expansion."""

    def test_constant_polynomial_has_zero_derivatives(self):
        """Q(t) = 1 should have Q' = Q'' = 0."""
        Q_one = Polynomial(np.array([1.0]))  # Q(t) = 1
        t = 0.5
        a, b = 0.3, 0.7

        coeffs = q_affine_series_at_xy(Q_one, t, a, b)

        assert coeffs.c0 == pytest.approx(1.0, abs=1e-14)
        assert coeffs.cx == pytest.approx(0.0, abs=1e-14)
        assert coeffs.cy == pytest.approx(0.0, abs=1e-14)
        assert coeffs.cxy == pytest.approx(0.0, abs=1e-14)

    def test_linear_polynomial_has_zero_second_derivative(self):
        """Q(t) = t should have Q'' = 0."""
        Q_linear = Polynomial(np.array([0.0, 1.0]))  # Q(t) = t
        t = 0.5
        a, b = 0.3, 0.7

        coeffs = q_affine_series_at_xy(Q_linear, t, a, b)

        assert coeffs.c0 == pytest.approx(0.5, abs=1e-14)  # Q(0.5) = 0.5
        assert coeffs.cx == pytest.approx(a, abs=1e-14)    # Q'(t)*a = 1*a
        assert coeffs.cy == pytest.approx(b, abs=1e-14)    # Q'(t)*b = 1*b
        assert coeffs.cxy == pytest.approx(0.0, abs=1e-14) # Q''(t)*a*b = 0

    def test_quadratic_polynomial_has_constant_second_derivative(self):
        """Q(t) = t^2 should have Q'' = 2."""
        Q_quad = Polynomial(np.array([0.0, 0.0, 1.0]))  # Q(t) = t^2
        t = 0.5
        a, b = 0.3, 0.7

        coeffs = q_affine_series_at_xy(Q_quad, t, a, b)

        assert coeffs.c0 == pytest.approx(0.25, abs=1e-14)  # Q(0.5) = 0.25
        assert coeffs.cx == pytest.approx(2 * t * a, abs=1e-14)  # Q'(t) = 2t
        assert coeffs.cy == pytest.approx(2 * t * b, abs=1e-14)
        assert coeffs.cxy == pytest.approx(2 * a * b, abs=1e-14)  # Q'' = 2


class TestQProductXYCoeff:
    """Test the [xy] coefficient extraction from product."""

    def test_product_of_constants_has_zero_xy(self):
        """Product of two constant series has [xy] = 0."""
        c1 = NilpotentSeriesCoeffs(c0=2.0, cx=0.0, cy=0.0, cxy=0.0)
        c2 = NilpotentSeriesCoeffs(c0=3.0, cx=0.0, cy=0.0, cxy=0.0)

        xy_coeff = q_product_xy_coeff(c1, c2)

        assert xy_coeff == pytest.approx(0.0, abs=1e-14)

    def test_product_xy_formula(self):
        """Test the [xy] extraction formula directly."""
        # Known coefficients
        c1 = NilpotentSeriesCoeffs(c0=1.0, cx=2.0, cy=3.0, cxy=0.5)
        c2 = NilpotentSeriesCoeffs(c0=2.0, cx=1.0, cy=0.5, cxy=0.3)

        # [xy](Q1*Q2) = c0_1*cxy_2 + cxy_1*c0_2 + cx_1*cy_2 + cy_1*cx_2
        expected = 1.0*0.3 + 0.5*2.0 + 2.0*0.5 + 3.0*1.0
        # = 0.3 + 1.0 + 1.0 + 3.0 = 5.3

        xy_coeff = q_product_xy_coeff(c1, c2)

        assert xy_coeff == pytest.approx(5.3, abs=1e-14)


class TestQ1Gate:
    """Critical Q=1 gate: all derivative contributions must vanish."""

    def test_q1_xy_coeff_is_zero_post_identity(self):
        """With Q(t)=1, [xy] Q(A_alpha)Q(A_beta) = 0."""
        Q_one = Polynomial(np.array([1.0]))
        theta = 4/7

        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            xy_coeff = q_product_xy_coeff_post_identity(Q_one, t, theta)
            assert xy_coeff == pytest.approx(0.0, abs=1e-14), f"Failed at t={t}"

    def test_q1_xy_coeff_is_zero_vectorized(self):
        """Vectorized version also gives zero for Q=1."""
        Q_one = Polynomial(np.array([1.0]))
        theta = 4/7
        t_arr = np.linspace(0.01, 0.99, 50)

        xy_coeffs = q_product_xy_coeff_post_identity_vectorized(Q_one, t_arr, theta)

        assert np.allclose(xy_coeffs, 0.0, atol=1e-14)

    def test_q1_moments_are_zero(self):
        """With Q=1, all derivative-based moments should be zero."""
        Q_one = Polynomial(np.array([1.0]))
        theta = 4/7
        R = 1.3036

        result = compute_q_moments_under_weight(Q_one, theta, R, n_quad=60)

        # Q(t)^2 = 1, so this should be exp(2R) integral (not zero)
        assert result.Q_squared_moment > 0

        # But Q'' = 0 and Q' = 0, so these must be zero
        assert result.Q_Q_double_prime_moment == pytest.approx(0.0, abs=1e-14)
        assert result.Q_prime_squared_moment == pytest.approx(0.0, abs=1e-14)
        assert result.t_times_t_minus_1_Q_Qpp == pytest.approx(0.0, abs=1e-14)
        assert result.t2_plus_tm1_2_Qp2 == pytest.approx(0.0, abs=1e-14)
        assert result.xy_coeff_moment == pytest.approx(0.0, abs=1e-14)


class TestFormulaVsSeries:
    """Verify closed-form formula matches brute-force series computation."""

    def test_formula_matches_series_quadratic(self):
        """Test with Q(t) = t^2."""
        Q = Polynomial(np.array([0.0, 0.0, 1.0]))
        theta = 4/7

        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = verify_formula_vs_series(Q, t, theta)
            assert result["match"], f"Mismatch at t={t}: diff={result['absolute_diff']}"

    def test_formula_matches_series_cubic(self):
        """Test with Q(t) = t^3."""
        Q = Polynomial(np.array([0.0, 0.0, 0.0, 1.0]))
        theta = 4/7

        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = verify_formula_vs_series(Q, t, theta)
            assert result["match"], f"Mismatch at t={t}: diff={result['absolute_diff']}"

    def test_formula_matches_series_przz_q(self):
        """Test with actual PRZZ Q polynomial."""
        # PRZZ Q polynomial coefficients
        Q_coeffs = np.array([
            0.083377,
            0.833797,
            -1.875080,
            8.166908,
            -20.416973,
            29.166631,
            -22.916671,
            9.166667,
            -1.458339,
        ])
        Q = Polynomial(Q_coeffs)
        theta = 4/7

        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = verify_formula_vs_series(Q, t, theta)
            assert result["match"], f"Mismatch at t={t}: diff={result['absolute_diff']}"


class TestPostIdentityFormula:
    """Test the post-identity [xy] formula structure."""

    def test_formula_is_theta_squared_scaled(self):
        """The formula should scale as theta^2."""
        Q = Polynomial(np.array([0.0, 0.0, 1.0]))  # Q = t^2
        t = 0.5

        xy_1 = q_product_xy_coeff_post_identity(Q, t, theta=1.0)
        xy_half = q_product_xy_coeff_post_identity(Q, t, theta=0.5)

        # Should scale as theta^2
        assert xy_half == pytest.approx(xy_1 * 0.25, rel=1e-10)

    def test_geometric_coefficients_at_t_half(self):
        """At t=0.5, verify geometric coefficients."""
        # At t=0.5:
        # geom_QQpp = 2*t*(t-1) = 2*0.5*(-0.5) = -0.5
        # geom_Qp2 = t^2 + (t-1)^2 = 0.25 + 0.25 = 0.5

        Q = Polynomial(np.array([0.0, 0.0, 1.0]))  # Q = t^2
        t = 0.5
        theta = 1.0  # Use theta=1 to see raw structure

        # Q(0.5) = 0.25, Q'(0.5) = 1.0, Q''(0.5) = 2.0
        # [xy] = theta^2 * (-0.5 * 0.25 * 2.0 + 0.5 * 1.0^2)
        #      = 1.0 * (-0.25 + 0.5) = 0.25
        expected = 0.25

        xy_coeff = q_product_xy_coeff_post_identity(Q, t, theta)

        assert xy_coeff == pytest.approx(expected, abs=1e-14)

    def test_symmetry_at_t_half(self):
        """At t=0.5, eigenvalue coefficients are symmetric."""
        t = 0.5
        theta = 4/7

        # a_alpha = theta*(t-1) = theta*(-0.5)
        # b_alpha = theta*t = theta*0.5
        # a_beta = theta*t = theta*0.5
        # b_beta = theta*(t-1) = theta*(-0.5)

        # So a_alpha = -b_alpha = -a_beta = b_beta (up to sign)
        a_alpha = theta * (t - 1)
        b_alpha = theta * t
        a_beta = theta * t
        b_beta = theta * (t - 1)

        assert a_alpha == pytest.approx(-b_alpha, abs=1e-14)
        assert a_beta == pytest.approx(-b_beta, abs=1e-14)


class TestQMomentsComputation:
    """Test Q-moment computation under integration."""

    def test_uniform_weight_q1_integral(self):
        """With Q=1 and uniform weight, Q^2 integral = 1."""
        Q_one = Polynomial(np.array([1.0]))

        result = compute_q_moments_under_weight(
            Q_one, theta=4/7, R=1.0, n_quad=60, weight_type="uniform"
        )

        # integral of 1 over [0,1] = 1
        assert result.Q_squared_moment == pytest.approx(1.0, abs=1e-10)

    def test_exp_2R_weight_integral(self):
        """Test exp(2Rt) weight integral."""
        Q_one = Polynomial(np.array([1.0]))
        R = 1.0

        result = compute_q_moments_under_weight(
            Q_one, theta=4/7, R=R, n_quad=60, weight_type="exp_2R"
        )

        # integral of exp(2Rt) from 0 to 1 = (exp(2R) - 1) / (2R)
        expected = (np.exp(2*R) - 1) / (2*R)
        assert result.Q_squared_moment == pytest.approx(expected, rel=1e-8)

    def test_quadrature_stability(self):
        """Results should be stable under quadrature refinement."""
        Q_coeffs = np.array([0.083377, 0.833797, -1.875080, 8.166908])
        Q = Polynomial(Q_coeffs)
        theta = 4/7
        R = 1.3036

        result_60 = compute_q_moments_under_weight(Q, theta, R, n_quad=60)
        result_80 = compute_q_moments_under_weight(Q, theta, R, n_quad=80)
        result_100 = compute_q_moments_under_weight(Q, theta, R, n_quad=100)

        # All should agree to at least 1e-6
        assert result_60.xy_coeff_moment == pytest.approx(result_80.xy_coeff_moment, rel=1e-6)
        assert result_80.xy_coeff_moment == pytest.approx(result_100.xy_coeff_moment, rel=1e-6)


class TestFrozenVsFullComparison:
    """Test the frozen Q^2 vs full Q(A_alpha)Q(A_beta) comparison."""

    def test_q1_has_zero_derivative_ratio(self):
        """With Q=1, derivative contribution is zero."""
        Q_one = Polynomial(np.array([1.0]))

        result = compute_frozen_vs_full_xy_ratio(Q_one, theta=4/7, R=1.3036)

        assert result["xy_deriv_plus"] == pytest.approx(0.0, abs=1e-14)
        assert result["xy_deriv_minus"] == pytest.approx(0.0, abs=1e-14)

    def test_przz_q_has_nonzero_derivative_contribution(self):
        """PRZZ Q polynomial should have significant derivative contribution."""
        Q_coeffs = np.array([
            0.083377, 0.833797, -1.875080, 8.166908,
            -20.416973, 29.166631, -22.916671, 9.166667, -1.458339,
        ])
        Q = Polynomial(Q_coeffs)

        result = compute_frozen_vs_full_xy_ratio(Q, theta=4/7, R=1.3036)

        # Should have nonzero derivative contribution
        assert abs(result["xy_deriv_plus"]) > 1e-6
        assert abs(result["xy_deriv_minus"]) > 1e-6

        # The ratio tells us the relative size of Q-derivative effect
        # This should be nonzero but smaller than 1
        assert abs(result["ratio_plus"]) < 1.0
        assert abs(result["ratio_minus"]) < 1.0


class TestPRZZQPolynomial:
    """Test with actual PRZZ Q polynomial parameters."""

    @pytest.fixture
    def przz_Q(self):
        """PRZZ Q polynomial."""
        return Polynomial(np.array([
            0.083377,
            0.833797,
            -1.875080,
            8.166908,
            -20.416973,
            29.166631,
            -22.916671,
            9.166667,
            -1.458339,
        ]))

    def test_q_derivative_moments_przz(self, przz_Q):
        """Compute Q-derivative moments for PRZZ Q."""
        result = compute_q_moments_under_weight(
            przz_Q, theta=4/7, R=1.3036, n_quad=80
        )

        # Just verify these are finite and reasonable
        assert np.isfinite(result.Q_squared_moment)
        assert np.isfinite(result.xy_coeff_moment)

        # Log for analysis
        print(f"\nPRZZ Q moments (R=1.3036, exp(2Rt) weight):")
        print(f"  Q^2 moment: {result.Q_squared_moment:.6f}")
        print(f"  Q*Q'' moment: {result.Q_Q_double_prime_moment:.6f}")
        print(f"  (Q')^2 moment: {result.Q_prime_squared_moment:.6f}")
        print(f"  [xy] moment: {result.xy_coeff_moment:.6f}")
        print(f"  ratio [xy]/Q^2: {result.xy_coeff_moment / result.Q_squared_moment:.6f}")

    def test_verify_formula_exhaustively(self, przz_Q):
        """Verify formula matches series at many t points."""
        theta = 4/7

        for t in np.linspace(0.01, 0.99, 20):
            result = verify_formula_vs_series(przz_Q, t, theta)
            assert result["match"], f"Formula mismatch at t={t}"
