"""
tests/test_correction_policy.py
Phase 45.1: Tests for Correction Policy Infrastructure

These tests ensure:
1. Default mode is DERIVED_BASELINE_ONLY (not anchored)
2. Baseline mode does NOT reference κ/κ* targets
3. Anchored mode DOES use calibrated constants, and logs them
4. Explicit opt-in is required for anchored mode

Created: 2025-12-27 (Phase 45.1)
"""

import pytest
import math
import logging

from src.evaluator.correction_policy import (
    CorrectionMode,
    CorrectionResult,
    compute_g_baseline,
    compute_base,
    compute_g_anchored,
    get_g_correction,
    get_mirror_multiplier,
    is_derived_mode,
    is_anchored_mode,
    get_default_mode,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


class TestCorrectionModeEnum:
    """Test the CorrectionMode enum."""

    def test_enum_values(self):
        """Verify enum has expected values."""
        assert CorrectionMode.DERIVED_BASELINE_ONLY.value == "derived_baseline_only"
        assert CorrectionMode.FIRST_PRINCIPLES_I1_I2.value == "first_principles_i1_i2"
        assert CorrectionMode.ANCHORED_TWO_BENCHMARKS.value == "anchored_two_benchmarks"
        # Legacy alias should have same value
        assert CorrectionMode.COMPONENT_RENORM_ANCHORED.value == "anchored_two_benchmarks"

    def test_default_is_derived(self):
        """The default mode MUST be DERIVED_BASELINE_ONLY."""
        assert get_default_mode() == CorrectionMode.DERIVED_BASELINE_ONLY

    def test_is_derived_mode(self):
        """Test is_derived_mode helper."""
        assert is_derived_mode(CorrectionMode.DERIVED_BASELINE_ONLY) is True
        assert is_derived_mode(CorrectionMode.COMPONENT_RENORM_ANCHORED) is False

    def test_is_anchored_mode(self):
        """Test is_anchored_mode helper."""
        assert is_anchored_mode(CorrectionMode.DERIVED_BASELINE_ONLY) is False
        assert is_anchored_mode(CorrectionMode.COMPONENT_RENORM_ANCHORED) is True


class TestDerivedBaseline:
    """Test the derived baseline formulas (first-principles, no anchoring)."""

    def test_g_baseline_formula(self):
        """
        g_baseline = 1 + θ/(2K(2K+1))

        For K=3, θ=4/7:
        g = 1 + (4/7)/(6×7) = 1 + (4/7)/42 = 1 + 4/294 ≈ 1.01360544
        """
        theta = 4 / 7
        K = 3
        g = compute_g_baseline(theta, K)

        expected = 1 + theta / (2 * K * (2 * K + 1))
        assert g == pytest.approx(expected, rel=1e-10)
        assert g == pytest.approx(1.01360544217687, rel=1e-8)

    def test_base_formula(self):
        """
        base = exp(R) + (2K-1)

        For R=1.3036, K=3:
        base = exp(1.3036) + 5 ≈ 3.6825 + 5 = 8.6825
        """
        R = 1.3036
        K = 3
        base = compute_base(R, K)

        expected = math.exp(R) + (2 * K - 1)
        assert base == pytest.approx(expected, rel=1e-10)
        assert base == pytest.approx(8.6825299, rel=1e-5)

    def test_derived_mode_no_anchored_constants(self):
        """
        CRITICAL: Derived mode must NOT use G_I1_CALIBRATED or G_I2_CALIBRATED.

        This test verifies the result is independent of the calibrated constants.
        """
        R = 1.3036
        theta = 4 / 7
        K = 3

        result = get_g_correction(R, theta, K, mode=CorrectionMode.DERIVED_BASELINE_ONLY)

        # The g value should equal g_baseline exactly
        g_baseline = compute_g_baseline(theta, K)
        assert result.g == pytest.approx(g_baseline, rel=1e-10)

        # The result should NOT have g_I1 or g_I2 set
        assert result.g_I1 is None
        assert result.g_I2 is None
        assert result.f_I1 is None


class TestAnchoredMode:
    """Test the anchored mode (uses calibrated constants)."""

    def test_anchored_requires_f_I1(self):
        """Anchored mode MUST require f_I1 parameter."""
        R = 1.3036
        theta = 4 / 7
        K = 3

        with pytest.raises(ValueError, match="requires f_I1"):
            get_g_correction(
                R, theta, K,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
                allow_target_anchoring=True  # Allow anchoring, but still missing f_I1
            )

    def test_anchored_uses_calibrated_constants(self):
        """Anchored mode must use G_I1_CALIBRATED and G_I2_CALIBRATED."""
        R = 1.3036
        theta = 4 / 7
        K = 3
        f_I1 = 0.2329  # κ benchmark value

        result = get_g_correction(
            R, theta, K, f_I1=f_I1,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            allow_target_anchoring=True
        )

        # Verify the weighted formula
        expected_g = f_I1 * G_I1_CALIBRATED + (1 - f_I1) * G_I2_CALIBRATED
        assert result.g == pytest.approx(expected_g, rel=1e-10)

        # Verify the calibrated constants are recorded
        assert result.g_I1 == G_I1_CALIBRATED
        assert result.g_I2 == G_I2_CALIBRATED
        assert result.f_I1 == f_I1

    def test_anchored_logs_constants(self, caplog):
        """Anchored mode must LOG the calibrated constants used."""
        R = 1.3036
        theta = 4 / 7
        K = 3
        f_I1 = 0.2329

        with caplog.at_level(logging.WARNING):  # Changed to WARNING level
            get_g_correction(
                R, theta, K, f_I1=f_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
                allow_target_anchoring=True
            )

        # Check that constants were logged
        assert "ANCHORED_TWO_BENCHMARKS" in caplog.text
        assert "calibrated" in caplog.text
        assert str(G_I1_CALIBRATED)[:6] in caplog.text  # At least first 6 digits


class TestCalibratedConstants:
    """Verify the calibrated constants have expected values."""

    def test_g_I1_value(self):
        """g_I1 should be very close to 1.0 (I1 self-corrects)."""
        assert G_I1_CALIBRATED == pytest.approx(1.00091428, rel=1e-6)
        assert G_I1_CALIBRATED > 1.0  # Slightly above 1
        assert G_I1_CALIBRATED < 1.001  # But very close

    def test_g_I2_value(self):
        """g_I2 should be larger than g_baseline (I2 needs extra correction)."""
        theta = 4 / 7
        K = 3
        g_baseline = compute_g_baseline(theta, K)

        assert G_I2_CALIBRATED == pytest.approx(1.01945154, rel=1e-6)
        assert G_I2_CALIBRATED > g_baseline  # Larger than baseline

    def test_weighted_formula_kappa(self):
        """
        For κ benchmark (f_I1 ≈ 0.2329), the weighted formula should give
        g_total ≈ 1.0151 (which is what Phase 45 achieved).
        """
        f_I1 = 0.2329
        g_total = compute_g_anchored(f_I1)

        assert g_total == pytest.approx(1.0151, rel=1e-3)

    def test_weighted_formula_kappa_star(self):
        """
        For κ* benchmark (f_I1 ≈ 0.3263), the weighted formula should give
        g_total ≈ 1.0134.
        """
        f_I1 = 0.3263
        g_total = compute_g_anchored(f_I1)

        assert g_total == pytest.approx(1.0134, rel=1e-3)


class TestMirrorMultiplier:
    """Test the full mirror multiplier m = g × base."""

    def test_multiplier_derived(self):
        """Test mirror multiplier in derived mode."""
        R = 1.3036
        theta = 4 / 7
        K = 3

        m = get_mirror_multiplier(R, theta, K, mode=CorrectionMode.DERIVED_BASELINE_ONLY)

        g_baseline = compute_g_baseline(theta, K)
        base = compute_base(R, K)
        expected_m = g_baseline * base

        assert m == pytest.approx(expected_m, rel=1e-10)
        assert m == pytest.approx(8.8007, rel=1e-3)

    def test_multiplier_anchored(self):
        """Test mirror multiplier in anchored mode."""
        R = 1.3036
        theta = 4 / 7
        K = 3
        f_I1 = 0.2329

        m = get_mirror_multiplier(
            R, theta, K, f_I1=f_I1,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            allow_target_anchoring=True
        )

        g_anchored = compute_g_anchored(f_I1)
        base = compute_base(R, K)
        expected_m = g_anchored * base

        assert m == pytest.approx(expected_m, rel=1e-10)


class TestDefaultBehavior:
    """Test that default behavior is first-principles (not anchored)."""

    def test_get_g_correction_default(self):
        """get_g_correction default should be DERIVED_BASELINE_ONLY."""
        R = 1.3036
        theta = 4 / 7
        K = 3

        # No mode specified - should use default
        result = get_g_correction(R, theta, K)

        assert result.mode == CorrectionMode.DERIVED_BASELINE_ONLY
        assert result.g_I1 is None
        assert result.g_I2 is None

    def test_get_mirror_multiplier_default(self):
        """get_mirror_multiplier default should be DERIVED_BASELINE_ONLY."""
        R = 1.3036
        theta = 4 / 7
        K = 3

        # Compare default to explicit derived mode
        m_default = get_mirror_multiplier(R, theta, K)
        m_derived = get_mirror_multiplier(R, theta, K, mode=CorrectionMode.DERIVED_BASELINE_ONLY)

        assert m_default == m_derived


class TestCorrectionResultDataclass:
    """Test the CorrectionResult dataclass."""

    def test_result_fields_derived(self):
        """Verify all fields are set correctly in derived mode."""
        R = 1.3036
        theta = 4 / 7
        K = 3

        result = get_g_correction(R, theta, K, mode=CorrectionMode.DERIVED_BASELINE_ONLY)

        assert isinstance(result, CorrectionResult)
        assert result.g > 0
        assert result.base > 0
        assert result.m == pytest.approx(result.g * result.base, rel=1e-10)
        assert result.mode == CorrectionMode.DERIVED_BASELINE_ONLY
        assert result.g == result.g_baseline  # In derived mode, g equals baseline

    def test_result_fields_anchored(self):
        """Verify all fields are set correctly in anchored mode."""
        R = 1.3036
        theta = 4 / 7
        K = 3
        f_I1 = 0.2329

        result = get_g_correction(
            R, theta, K, f_I1=f_I1,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            allow_target_anchoring=True
        )

        assert isinstance(result, CorrectionResult)
        assert result.g > 0
        assert result.base > 0
        assert result.m == pytest.approx(result.g * result.base, rel=1e-10)
        assert result.mode == CorrectionMode.ANCHORED_TWO_BENCHMARKS
        assert result.g_I1 is not None
        assert result.g_I2 is not None
        assert result.f_I1 == f_I1
        assert result.g != result.g_baseline  # In anchored mode, g differs from baseline
