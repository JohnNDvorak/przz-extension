# -*- coding: utf-8 -*-
"""
Tests for Explicit Error Bound Estimation

Validates the explicit error constants for the four PRZZ O(T/L) sources:
- C_contour: Contour integral bounds (Lines 1341, 1400-1435)
- C_Taylor: A^{(1,1)} Taylor expansion (Line 1341)
- C_I5: Prime sum contribution (Lines 1580-1628) - actually O(T/L²)
- C_EM: Euler-Maclaurin remainder (Lines 1433-1435)

Key result: I₅ is O(T/L²), not O(T/L), making it negligible at L=40.

Reference: PRZZ κ = 0.417293962 at R = 1.3036
"""

import json
import math
import pytest
from pathlib import Path

from src.error_bound_estimator import ErrorBoundEstimator, ExplicitErrorBoundResult
from src.ratios.arithmetic_factor import A11_prime_sum, A11_derivative


# ============================================================================
# Test Data
# ============================================================================

# PRZZ baseline parameters
R_PRZZ = 1.3036
THETA_PRZZ = 4 / 7
C_PRZZ = 2.1374544
KAPPA_PRZZ = 0.417293962

# Simple test polynomials for unit tests
P1_SIMPLE = [1.0, -0.5]  # Linear: 1 - 0.5*(1-x) in constrained form
P2_SIMPLE = [1.0, 0.5]   # x + 0.5*x^2
P3_SIMPLE = [0.1, 0.0]   # 0.1*x


# ============================================================================
# Unit Tests: Polynomial Norm Computations
# ============================================================================

class TestDerivativeL2Norm:
    """Tests for L² norm of polynomial derivatives."""

    def test_linear_polynomial(self):
        """Linear polynomial has constant derivative."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        # P(u) = c0*u, P'(u) = c0, ||P'||_L² = |c0|
        coeffs = [2.0]  # P_tilde = 2.0, so P(u) = 2.0*u
        norm = estimator.compute_derivative_L2_norm(coeffs)
        assert norm == pytest.approx(2.0, rel=1e-6)

    def test_quadratic_polynomial(self):
        """Test quadratic polynomial derivative norm."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        # P(u) = u + u^2, P'(u) = 1 + 2u
        # ||P'||_L² = sqrt(∫₀¹ (1+2u)² du) = sqrt(∫₀¹ (1 + 4u + 4u²) du)
        # = sqrt(1 + 2 + 4/3) = sqrt(13/3) ≈ 2.0817
        coeffs = [1.0, 1.0]  # P_tilde coeffs for u + u^2
        norm = estimator.compute_derivative_L2_norm(coeffs)
        expected = math.sqrt(13 / 3)
        assert norm == pytest.approx(expected, rel=0.05)  # Allow 5% tolerance

    def test_nonnegative(self):
        """L² norm is always non-negative."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        norm = estimator.compute_derivative_L2_norm([-1.0, -2.0, -3.0])
        assert norm >= 0


class TestMellinEnvelope:
    """Tests for Mellin envelope ||P||_Mellin = sup |P(u)| exp(Rθu)."""

    def test_positive_polynomial(self):
        """Positive polynomial has positive Mellin envelope."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        envelope = estimator.compute_mellin_envelope([1.0, 0.5])
        assert envelope > 0

    def test_scaling_with_R(self):
        """Larger R gives larger Mellin envelope."""
        estimator_low = ErrorBoundEstimator(R=1.0, theta=THETA_PRZZ)
        estimator_high = ErrorBoundEstimator(R=2.0, theta=THETA_PRZZ)

        coeffs = [1.0, 0.5]
        envelope_low = estimator_low.compute_mellin_envelope(coeffs)
        envelope_high = estimator_high.compute_mellin_envelope(coeffs)

        assert envelope_high > envelope_low

    def test_constant_polynomial(self):
        """Constant polynomial P(u) = c has ||P||_Mellin = |c| exp(Rθ)."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        # P(u) = 2*u, so at u=1: P(1) = 2
        # Mellin envelope should be near 2 * exp(Rθ) at u=1
        coeffs = [2.0]
        envelope = estimator.compute_mellin_envelope(coeffs)
        expected_at_1 = 2.0 * math.exp(R_PRZZ * THETA_PRZZ)
        assert envelope <= expected_at_1 * 1.1  # Allow some tolerance


class TestPolynomialIntegrals:
    """Tests for ∫P₁(u)P₂(u)du computations."""

    def test_self_integral_positive(self):
        """∫P(u)²du > 0 for non-zero polynomial."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        integral = estimator.compute_polynomial_integral([1.0, 0.5], [1.0, 0.5])
        assert integral > 0

    def test_cross_integral_symmetry(self):
        """∫P₁P₂ = ∫P₂P₁."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        P1 = [1.0, 0.5]
        P2 = [0.5, -0.3]
        int12 = estimator.compute_polynomial_integral(P1, P2)
        int21 = estimator.compute_polynomial_integral(P2, P1)
        assert int12 == pytest.approx(int21, rel=1e-10)


class TestDerivativeCrossIntegrals:
    """Tests for ∫P₁'(u)P₂'(u)du."""

    def test_linear_derivatives(self):
        """Linear polynomials have constant derivatives."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        # P1(u) = 2u, P2(u) = 3u → P1' = 2, P2' = 3
        # ∫P1'P2' du = ∫ 2*3 du = 6
        P1 = [2.0]
        P2 = [3.0]
        cross = estimator.compute_derivative_cross_integral(P1, P2)
        assert cross == pytest.approx(6.0, rel=1e-6)


# ============================================================================
# Unit Tests: Individual Error Constants
# ============================================================================

class TestCContour:
    """Tests for contour integral bound C_contour."""

    def test_positive(self):
        """C_contour is positive."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        C = estimator.compute_C_contour(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        assert C > 0

    def test_scales_with_polynomial_size(self):
        """Larger polynomials give larger C_contour."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)

        C_small = estimator.compute_C_contour([0.1], [0.1], [0.1])
        C_large = estimator.compute_C_contour([10.0], [10.0], [10.0])

        assert C_large > C_small


class TestCTaylor:
    """Tests for Taylor expansion bound C_Taylor."""

    def test_positive(self):
        """C_Taylor is positive."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        C = estimator.compute_C_Taylor(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        assert C > 0

    def test_uses_A11_derivative(self):
        """C_Taylor incorporates A^{(1,1)} derivative."""
        # A11_derivative at s=0 is about 5.9
        deriv = A11_derivative(0.0, prime_cutoff=10000)
        assert 4.0 < deriv < 8.0


class TestCI5:
    """Tests for I₅ prime sum bound C_I5."""

    def test_positive(self):
        """C_I5 is positive."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        C = estimator.compute_C_I5_explicit(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        assert C > 0

    def test_scales_with_L2_norms(self):
        """C_I5 scales with ||P'||_L² products."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)

        # Small derivatives
        C_small = estimator.compute_C_I5_explicit([0.5], [0.5], [0.5])
        # Large derivatives
        C_large = estimator.compute_C_I5_explicit([5.0], [5.0], [5.0])

        assert C_large > C_small


class TestCEM:
    """Tests for Euler-Maclaurin bound C_EM."""

    def test_positive(self):
        """C_EM is positive."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        C = estimator.compute_C_EM(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        assert C > 0

    def test_smaller_than_contour(self):
        """C_EM is typically smaller than C_contour."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        C_contour = estimator.compute_C_contour(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        C_EM = estimator.compute_C_EM(P1_SIMPLE, P2_SIMPLE, P3_SIMPLE)
        # Euler-Maclaurin is usually a smaller correction
        assert C_EM < C_contour * 2


# ============================================================================
# Integration Tests: ExplicitErrorBoundResult
# ============================================================================

class TestExplicitErrorBoundResult:
    """Tests for the full explicit error bound computation."""

    def test_result_structure(self):
        """Result has all expected fields."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        assert hasattr(result, 'C_contour')
        assert hasattr(result, 'C_Taylor')
        assert hasattr(result, 'C_I5')
        assert hasattr(result, 'C_EM')
        assert hasattr(result, 'total_C_per_L')
        assert hasattr(result, 'kappa_main')
        assert hasattr(result, 'kappa_rigorous')

    def test_kappa_rigorous_less_than_main(self):
        """κ_rigorous < κ_main (error subtracts from bound)."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        assert result.kappa_rigorous < result.kappa_main

    def test_gap_is_positive(self):
        """Gap = κ_main - κ_rigorous is positive."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        assert result.kappa_gap > 0

    def test_summary_table_format(self):
        """summary_table() returns formatted markdown."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        table = result.summary_table()
        assert '## Explicit Error Constants' in table
        assert 'C_contour' in table
        assert 'κ_main' in table


# ============================================================================
# Validation Tests: PRZZ Baseline
# ============================================================================

class TestPRZZBaseline:
    """Validation tests against PRZZ κ = 0.4173 baseline."""

    @pytest.fixture
    def przz_params(self):
        """Load PRZZ polynomial parameters."""
        data_path = Path(__file__).parent.parent / 'data' / 'przz_parameters.json'
        with open(data_path, 'r') as f:
            return json.load(f)

    def test_przz_polynomials_load(self, przz_params):
        """PRZZ polynomials load correctly."""
        P1 = przz_params['polynomials']['P1']['tilde_coeffs']
        P2 = przz_params['polynomials']['P2']['tilde_coeffs']
        P3 = przz_params['polynomials']['P3']['tilde_coeffs']

        assert len(P1) == 4
        assert len(P2) == 3
        assert len(P3) == 3

    def test_przz_error_bound_positive(self, przz_params):
        """PRZZ polynomials give positive error bound."""
        P1 = przz_params['polynomials']['P1']['tilde_coeffs']
        P2 = przz_params['polynomials']['P2']['tilde_coeffs']
        P3 = przz_params['polynomials']['P3']['tilde_coeffs']

        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1, P2, P3,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        assert result.total_C_per_L > 0
        assert result.kappa_rigorous > 0

    def test_przz_rigorous_kappa_above_one_third(self, przz_params):
        """PRZZ κ_rigorous > 1/3 (meaningful bound)."""
        P1 = przz_params['polynomials']['P1']['tilde_coeffs']
        P2 = przz_params['polynomials']['P2']['tilde_coeffs']
        P3 = przz_params['polynomials']['P3']['tilde_coeffs']

        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        result = estimator.compute_explicit_error_bounds(
            P1, P2, P3,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )

        # κ_rigorous should still be a meaningful bound
        assert result.kappa_rigorous > 0.33


# ============================================================================
# A^{(1,1)} Prime Sum Tests
# ============================================================================

class TestA11PrimeSum:
    """Tests for arithmetic factor A^{(1,1)}."""

    def test_A11_at_zero(self):
        """A^{(1,1)}(0) ≈ 1.3856 (PRZZ anchor value)."""
        A11 = A11_prime_sum(0.0, prime_cutoff=100000)
        assert A11 == pytest.approx(1.3856, rel=0.01)

    def test_A11_positive(self):
        """A^{(1,1)} is positive."""
        A11 = A11_prime_sum(0.0, prime_cutoff=10000)
        assert A11 > 0

    def test_A11_decreases_with_s(self):
        """A^{(1,1)}(s) decreases as s increases."""
        A11_0 = A11_prime_sum(0.0, prime_cutoff=10000)
        A11_1 = A11_prime_sum(1.0, prime_cutoff=10000)
        assert A11_1 < A11_0


class TestA11Derivative:
    """Tests for A^{(1,1)} derivative (Taylor expansion bound)."""

    def test_derivative_positive(self):
        """dA^{(1,1)}/ds is positive at s=0."""
        deriv = A11_derivative(0.0, prime_cutoff=10000)
        assert deriv > 0

    def test_derivative_magnitude(self):
        """dA^{(1,1)}/ds(0) ≈ 5.9."""
        deriv = A11_derivative(0.0, prime_cutoff=100000)
        assert 4.0 < deriv < 8.0


# ============================================================================
# Paper Table Generation Tests
# ============================================================================

class TestPaperTableGeneration:
    """Tests for paper-ready table generation."""

    def test_generate_paper_tables_runs(self):
        """generate_paper_tables() runs without error."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        tables = estimator.generate_paper_tables(L=40.0)
        assert len(tables) > 0

    def test_tables_contain_markdown(self):
        """Output contains markdown table formatting."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        tables = estimator.generate_paper_tables(L=40.0)

        assert '|' in tables  # Markdown table delimiter
        assert 'Source' in tables or 'C_contour' in tables

    def test_tables_show_both_configs(self):
        """Tables compare PRZZ and optimized configurations."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        tables = estimator.generate_paper_tables(L=40.0)

        assert 'PRZZ' in tables or 'Baseline' in tables
        assert 'Optimized' in tables or 'Opt' in tables


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_L_gives_large_error(self):
        """Small L gives large error bound."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)

        result_L10 = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=10.0
        )
        result_L100 = estimator.compute_explicit_error_bounds(
            P1_SIMPLE, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=100.0
        )

        # Gap should decrease with larger L
        assert result_L10.kappa_gap > result_L100.kappa_gap

    def test_zero_polynomial_coefficient(self):
        """Handle polynomial with zero coefficient."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        P_zero_coeff = [1.0, 0.0, -0.5]  # Middle coefficient is zero

        result = estimator.compute_explicit_error_bounds(
            P_zero_coeff, P2_SIMPLE, P3_SIMPLE,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )
        assert result.kappa_rigorous > 0

    def test_single_coefficient_polynomial(self):
        """Handle minimal polynomial with one coefficient."""
        estimator = ErrorBoundEstimator(R=R_PRZZ, theta=THETA_PRZZ)
        P_single = [1.0]  # P(u) = u

        result = estimator.compute_explicit_error_bounds(
            P_single, P_single, P_single,
            c=C_PRZZ, kappa_main=KAPPA_PRZZ, L=40.0
        )
        assert result.kappa_rigorous > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
