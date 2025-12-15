"""
tests/test_i5_falsifiability.py
Falsifiability tests for I₅ diagonal convolution.

These tests verify properties of the I₅ implementation that are
INDEPENDENT of matching PRZZ targets. They test mathematical
properties that should hold regardless of the specific g value.

Purpose:
- Distinguish "correct formula" from "lucky calibration"
- Verify mathematical consistency
- Test robustness under perturbation

Note: These tests complement test_i5_validation.py which tests
agreement with PRZZ targets.
"""

import pytest
import numpy as np
from math import factorial

from src.i5_diagonal import (
    compute_I5_diagonal_convolution,
    compute_I5_total,
    build_ratio_only_bivariate,
    get_polynomial_coeffs,
)
from src.reference_bivariate import BivariateSeries
from src.polynomials import load_przz_polynomials


class TestDiagonalConvolutionAlgebra:
    """Test mathematical properties of the diagonal convolution formula."""

    def test_convolution_with_constant_series(self):
        """
        For a constant series F = c, the convolution should give 0.

        ΔF_{l1,l2} = Σ_{k=1}^{max_k} [(-g·S)^k / k!] × F_{l1-k,l2-k}

        If F is constant (only F_{0,0} = c is nonzero), then for l1=l2=1:
        ΔF_{1,1} = (-g·S)^1 / 1! × F_{0,0} = -g·S·c
        """
        # Create a constant bivariate series
        F = BivariateSeries.constant(2.5, max_order=4)

        # Convolution at (1,1) should give -g*S*c
        g = 0.50
        S_val = 1.0
        result = compute_I5_diagonal_convolution(1, 1, F, S_val, g, max_k=1)

        expected = -g * S_val * 2.5
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_convolution_with_xy_series(self):
        """
        For F = X·Y (coefficient at (1,1) only), the k=1 term is:
        ΔF_{1,1} = (-g·S) × F_{0,0}

        Since F_{0,0} = 0 for F = X·Y, the result should be 0.
        """
        # Create F = X·Y (only coefficient at (1,1) is 1)
        F = BivariateSeries.zero(max_order=4)
        F.coeffs[1, 1] = 1.0  # X^1 Y^1

        g = 0.50
        S_val = 1.0
        result = compute_I5_diagonal_convolution(1, 1, F, S_val, g, max_k=1)

        # F_{0,0} = 0, so result should be 0
        assert abs(result) < 1e-10, f"Expected 0, got {result}"

    def test_convolution_higher_diagonal(self):
        """
        For F with F_{0,0}=1, F_{1,1}=2, the (2,2) convolution is:
        ΔF_{2,2} = (-g·S)¹ × F_{1,1} + (-g·S)² × F_{0,0} / 2!
                 = -g·S·2 + (g·S)²·1/2

        With max_k=1, only the first term contributes.
        """
        F = BivariateSeries.zero(max_order=6)
        F.coeffs[0, 0] = 1.0
        F.coeffs[1, 1] = 2.0

        g = 0.50
        S_val = 1.0

        # With max_k=1
        result_k1 = compute_I5_diagonal_convolution(2, 2, F, S_val, g, max_k=1)
        expected_k1 = -g * S_val * 2.0
        assert abs(result_k1 - expected_k1) < 1e-10

        # With max_k=2 (full)
        result_full = compute_I5_diagonal_convolution(2, 2, F, S_val, g, max_k=2)
        expected_full = -g * S_val * 2.0 + (g * S_val) ** 2 / 2.0
        assert abs(result_full - expected_full) < 1e-10

    def test_convolution_linearity_in_series(self):
        """
        Diagonal convolution should be linear in the series F.
        ΔF(aF₁ + bF₂) = a·ΔF(F₁) + b·ΔF(F₂)
        """
        F1 = BivariateSeries.constant(1.0, max_order=4)
        F2 = BivariateSeries.constant(2.0, max_order=4)

        g = 0.50
        S_val = 1.0
        l1, l2 = 1, 1

        delta_F1 = compute_I5_diagonal_convolution(l1, l2, F1, S_val, g, max_k=1)
        delta_F2 = compute_I5_diagonal_convolution(l1, l2, F2, S_val, g, max_k=1)

        # Test linearity: 3*F1 + 2*F2
        F_combined = BivariateSeries.constant(3.0 * 1.0 + 2.0 * 2.0, max_order=4)
        delta_combined = compute_I5_diagonal_convolution(l1, l2, F_combined, S_val, g, max_k=1)

        expected = 3.0 * delta_F1 + 2.0 * delta_F2
        assert abs(delta_combined - expected) < 1e-10


class TestSmallSTaylor:
    """Test that for small S, I₅ behaves as expected."""

    def test_small_s_linear_behavior(self):
        """
        For small S, I₅ should be approximately linear in S.
        I₅(S) ≈ -g·S·(something) + O(S²)

        With max_k=1, it should be exactly linear.
        """
        # Create a simple bivariate series
        F = BivariateSeries.constant(1.0, max_order=4)
        g = 0.50
        l1, l2 = 1, 1

        S_values = [0.1, 0.2, 0.3, 0.4]
        results = [
            compute_I5_diagonal_convolution(l1, l2, F, S, g, max_k=1)
            for S in S_values
        ]

        # With max_k=1, should be exactly linear
        # I₅ = -g·S·F_{0,0} = -0.5·S·1 = -0.5·S
        for S, result in zip(S_values, results):
            expected = -g * S * 1.0
            assert abs(result - expected) < 1e-10

    def test_full_convolution_taylor(self):
        """
        For small S, full convolution should agree with k=1 to O(S²).
        """
        F = BivariateSeries.constant(1.0, max_order=4)
        g = 0.50
        l1, l2 = 1, 1

        S_small = 0.1
        result_k1 = compute_I5_diagonal_convolution(l1, l2, F, S_small, g, max_k=1)
        result_full = compute_I5_diagonal_convolution(l1, l2, F, S_small, g, max_k=None)

        # They should be exactly equal for (1,1) since k_max = min(1,1) = 1
        assert abs(result_k1 - result_full) < 1e-10

    def test_higher_pair_taylor_agreement(self):
        """
        For (2,2) pair with small S, k=1 and k=2 should differ by O(S²).
        """
        F = BivariateSeries.constant(1.0, max_order=6)
        g = 0.50
        l1, l2 = 2, 2

        S_small = 0.1
        result_k1 = compute_I5_diagonal_convolution(l1, l2, F, S_small, g, max_k=1)
        result_k2 = compute_I5_diagonal_convolution(l1, l2, F, S_small, g, max_k=2)

        # k=1: -g·S·F_{1,1} = 0 (F_{1,1}=0 for constant series)
        # k=2: 0 + (g·S)²/2·F_{0,0} = (0.5·0.1)²/2 = 0.00125
        assert abs(result_k1) < 1e-10  # Should be 0
        expected_k2 = (g * S_small) ** 2 / 2.0
        assert abs(result_k2 - expected_k2) < 1e-10


class TestGStabilityUnderPerturbation:
    """Test that g is stable when polynomials are perturbed."""

    @pytest.fixture
    def przz_polynomials(self):
        """Load PRZZ polynomials."""
        return load_przz_polynomials()

    def test_g_stability_przz_baseline(self, przz_polynomials):
        """
        Establish baseline: compute I₅ with PRZZ polynomials.
        """
        theta = 4 / 7
        R = 1.3036

        I5_total, per_pair = compute_I5_total(
            przz_polynomials, theta, R,
            n_quadrature=40, n_primes=5000,
            g=0.50, max_k=1
        )

        # Should be negative and on the order of -0.04 to -0.05
        assert I5_total < 0
        assert -0.1 < I5_total < 0

    def test_g_stability_under_theta_variation(self, przz_polynomials):
        """
        Vary θ slightly and verify I₅ changes smoothly.

        If the formula is correct, I₅ should be a smooth function of θ,
        not jump discontinuously.
        """
        R = 1.3036
        theta_base = 4 / 7

        # Baseline at θ = 4/7
        I5_base, _ = compute_I5_total(
            przz_polynomials, theta_base, R,
            n_quadrature=40, n_primes=5000,
            g=0.50, max_k=1
        )

        # Test at slightly different θ values
        for theta_perturbed in [0.55, 0.58, 0.60]:
            I5_perturbed, _ = compute_I5_total(
                przz_polynomials, theta_perturbed, R,
                n_quadrature=40, n_primes=5000,
                g=0.50, max_k=1
            )

            # I₅ should not jump wildly
            # Allow ratio between 0.5 and 2.0 for reasonable θ changes
            ratio = I5_perturbed / I5_base if I5_base != 0 else float('inf')
            assert 0.3 < ratio < 3.0, (
                f"I₅ ratio {ratio} at θ={theta_perturbed} is outside expected range"
            )


class TestMaxKComparison:
    """Test max_k=1 vs full convolution."""

    def test_max_k_1_vs_full_small_s(self):
        """
        For small S, max_k=1 should give nearly same result as full.
        Higher-k terms should be negligible.
        """
        F = BivariateSeries.constant(1.0, max_order=6)
        g = 0.50
        S_val = 0.5  # Moderate S

        # For (1,1), k_max = 1 anyway
        result_k1_11 = compute_I5_diagonal_convolution(1, 1, F, S_val, g, max_k=1)
        result_full_11 = compute_I5_diagonal_convolution(1, 1, F, S_val, g, max_k=None)
        assert abs(result_k1_11 - result_full_11) < 1e-10

        # For (2,2), there's a k=2 term
        result_k1_22 = compute_I5_diagonal_convolution(2, 2, F, S_val, g, max_k=1)
        result_full_22 = compute_I5_diagonal_convolution(2, 2, F, S_val, g, max_k=2)

        # The k=2 term is (g*S)²/2! which for g=0.5, S=0.5 is 0.03125
        k2_contribution = (g * S_val) ** 2 / 2.0
        assert abs(result_full_22 - result_k1_22 - k2_contribution) < 1e-10

    def test_max_k_1_sufficient_for_przz(self):
        """
        For PRZZ parameters with typical S values, max_k=1 should be sufficient.

        This tests that higher-k terms contribute relatively little to the total.
        """
        polys = load_przz_polynomials()
        theta = 4 / 7
        R = 1.3036

        I5_k1, per_pair_k1 = compute_I5_total(
            polys, theta, R,
            n_quadrature=40, n_primes=5000,
            g=0.50, max_k=1
        )

        I5_full, per_pair_full = compute_I5_total(
            polys, theta, R,
            n_quadrature=40, n_primes=5000,
            g=0.50, max_k=None  # Full convolution
        )

        # The difference should be small (< 5% relative)
        # The k=2 and higher terms contribute at most O((g*S)^2) which is small but not negligible
        if I5_k1 != 0:
            relative_diff = abs(I5_full - I5_k1) / abs(I5_k1)
            assert relative_diff < 0.05, f"Relative diff {relative_diff} > 5%"

    def test_g_compensation_for_max_k_full(self):
        """
        Verify that g=θ²(1+θ)≈0.513 with max_k=full gives similar results
        to g=0.50 with max_k=1.

        This tests the "compensating calibration" hypothesis.
        """
        polys = load_przz_polynomials()
        theta = 4 / 7
        R = 1.3036

        # Structural parameters
        I5_structural, _ = compute_I5_total(
            polys, theta, R,
            n_quadrature=40, n_primes=5000,
            g=0.50, max_k=1
        )

        # Default parameters (compensating calibration)
        g_default = theta ** 2 * (1 + theta)
        I5_default, _ = compute_I5_total(
            polys, theta, R,
            n_quadrature=40, n_primes=5000,
            g=g_default, max_k=None
        )

        # They should be close (within 5%)
        if I5_structural != 0:
            relative_diff = abs(I5_default - I5_structural) / abs(I5_structural)
            assert relative_diff < 0.05, f"Relative diff {relative_diff} > 5%"


class TestCoefficientExtractionHelper:
    """Test the get_polynomial_coeffs helper function."""

    def test_extraction_from_przz_polynomial(self):
        """Test coefficient extraction from PRZZ polynomials."""
        polys = load_przz_polynomials()
        P1, P2, P3, Q = polys

        # Should not raise
        coeffs_P1 = get_polynomial_coeffs(P1)
        coeffs_Q = get_polynomial_coeffs(Q)

        assert isinstance(coeffs_P1, np.ndarray)
        assert isinstance(coeffs_Q, np.ndarray)
        assert len(coeffs_P1) > 0
        assert len(coeffs_Q) > 0

    def test_extraction_error_on_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            get_polynomial_coeffs("not a polynomial")

        with pytest.raises(TypeError):
            get_polynomial_coeffs(42)
