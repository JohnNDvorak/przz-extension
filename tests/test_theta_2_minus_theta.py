"""
tests/test_theta_2_minus_theta.py
Tests for the breakthrough θ(2-θ) formula

The formula:
  g_I1 = 1.0
  g_I2 = 1 + θ(2-θ)/(2K(2K+1))

This achieves ~0.02% accuracy on both benchmarks without calibration.

Created: 2025-12-27 (Phase 46)
"""
import pytest
import numpy as np
from src.evaluator.correction_policy import (
    CorrectionMode,
    get_g_correction,
    is_derived_mode,
)


class TestTheta2MinusThetaFormula:
    """Test the θ(2-θ) formula itself."""

    def test_g_I2_value_for_standard_parameters(self):
        """Verify g_I2 = 1 + θ(2-θ)/(2K(2K+1)) for θ=4/7, K=3."""
        theta = 4/7
        K = 3

        # Expected: 1 + (4/7)(10/7)/42 = 1 + 40/2058 = 1.01943635...
        expected = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))

        result = get_g_correction(
            R=1.3036,
            theta=theta,
            K=K,
            f_I1=0.5,  # Arbitrary for g_I2 extraction
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        assert result.g_I2 == pytest.approx(expected, abs=1e-10)
        assert result.g_I2 == pytest.approx(1.019436346, rel=1e-8)

    def test_g_I1_is_exactly_one(self):
        """g_I1 should be exactly 1.0 in θ(2-θ) mode."""
        result = get_g_correction(
            R=1.3036,
            theta=4/7,
            K=3,
            f_I1=0.5,
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        assert result.g_I1 == pytest.approx(1.0, abs=1e-15)

    def test_g_I2_matches_calibrated_closely(self):
        """g_I2 should be within 0.002% of calibrated value."""
        G_I2_CALIBRATED = 1.01945154

        result = get_g_correction(
            R=1.3036,
            theta=4/7,
            K=3,
            f_I1=0.5,
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        gap_pct = abs(result.g_I2 / G_I2_CALIBRATED - 1) * 100
        assert gap_pct < 0.002, f"Gap {gap_pct}% > 0.002%"

    def test_mode_is_classified_as_derived(self):
        """THETA_2_MINUS_THETA should be classified as derived (not anchored)."""
        assert is_derived_mode(CorrectionMode.THETA_2_MINUS_THETA)


class TestTheta2MinusThetaOnBenchmarks:
    """Test θ(2-θ) formula on the PRZZ benchmarks."""

    def test_g_total_kappa_benchmark(self):
        """Test g_total for kappa benchmark (R=1.3036)."""
        result = get_g_correction(
            R=1.3036,
            theta=4/7,
            K=3,
            f_I1=0.233,  # Typical f_I1 for kappa
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        # g_total = f_I1 * 1.0 + (1-f_I1) * g_I2
        # = 0.233 * 1.0 + 0.767 * 1.01943635
        # ≈ 1.0149
        assert result.g == pytest.approx(1.0149, rel=0.001)

    def test_g_total_kappa_star_benchmark(self):
        """Test g_total for kappa* benchmark (R=1.1167)."""
        result = get_g_correction(
            R=1.1167,
            theta=4/7,
            K=3,
            f_I1=0.326,  # Typical f_I1 for kappa*
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        # g_total = f_I1 * 1.0 + (1-f_I1) * g_I2
        # ≈ 1.0131
        assert result.g == pytest.approx(1.0131, rel=0.001)

    def test_comparison_to_calibrated_within_0_1_percent(self):
        """θ(2-θ) should match calibrated g_total within 0.1%."""
        for R, f_I1 in [(1.3036, 0.233), (1.1167, 0.326)]:
            derived = get_g_correction(
                R=R,
                theta=4/7,
                K=3,
                f_I1=f_I1,
                mode=CorrectionMode.THETA_2_MINUS_THETA,
            )

            calibrated = get_g_correction(
                R=R,
                theta=4/7,
                K=3,
                f_I1=f_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
                allow_target_anchoring=True,
            )

            gap_pct = abs(derived.g / calibrated.g - 1) * 100
            assert gap_pct < 0.1, f"Gap {gap_pct}% > 0.1% at R={R}"


class TestTheta2MinusThetaRequiresF_I1:
    """Test that θ(2-θ) mode requires f_I1 parameter."""

    def test_raises_without_f_I1(self):
        """Should raise ValueError if f_I1 is not provided."""
        with pytest.raises(ValueError, match="f_I1 parameter"):
            get_g_correction(
                R=1.3036,
                theta=4/7,
                K=3,
                mode=CorrectionMode.THETA_2_MINUS_THETA,
            )


class TestTheta2MinusThetaFormulaMathematics:
    """Test the mathematical properties of the θ(2-θ) formula."""

    def test_theta_2_minus_theta_is_quadratic_in_theta(self):
        """θ(2-θ) = 2θ - θ² should be quadratic."""
        for theta in [0.1, 0.3, 0.5, 4/7, 0.8]:
            K = 3
            result = get_g_correction(
                R=1.0,
                theta=theta,
                K=K,
                f_I1=0.5,
                mode=CorrectionMode.THETA_2_MINUS_THETA,
            )

            expected_g_I2 = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))
            assert result.g_I2 == pytest.approx(expected_g_I2, abs=1e-12)

    def test_theta_0_gives_g_I2_equals_1(self):
        """When θ=0, g_I2 should be 1.0."""
        result = get_g_correction(
            R=1.0,
            theta=0.0,
            K=3,
            f_I1=0.5,
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        assert result.g_I2 == pytest.approx(1.0, abs=1e-15)

    def test_theta_1_gives_g_I2_equals_1_plus_1_over_denominator(self):
        """When θ=1, g_I2 should be 1 + 1/(2K(2K+1))."""
        K = 3
        result = get_g_correction(
            R=1.0,
            theta=1.0,
            K=K,
            f_I1=0.5,
            mode=CorrectionMode.THETA_2_MINUS_THETA,
        )

        # θ(2-θ) = 1*(2-1) = 1
        expected = 1 + 1 / (2 * K * (2 * K + 1))
        assert result.g_I2 == pytest.approx(expected, abs=1e-12)

    def test_maximum_at_theta_equals_1(self):
        """θ(2-θ) has maximum value of 1 at θ=1."""
        K = 3
        theta_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        g_I2_values = []
        for theta in theta_values:
            result = get_g_correction(
                R=1.0,
                theta=theta,
                K=K,
                f_I1=0.5,
                mode=CorrectionMode.THETA_2_MINUS_THETA,
            )
            g_I2_values.append(result.g_I2)

        # Maximum should be at θ=1
        assert max(g_I2_values) == g_I2_values[-1]
