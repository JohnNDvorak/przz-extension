"""
tests/test_correction_policy_lock.py
Phase 46.0: Tests for correction policy anchoring guard

These tests verify that:
1. ANCHORED_TWO_BENCHMARKS mode is BLOCKED by default
2. Explicit allow_target_anchoring=True is required to use anchored mode
3. Derived modes (DERIVED_BASELINE_ONLY, FIRST_PRINCIPLES_I1_I2) work without guard
4. The guard cannot be accidentally bypassed

Created: 2025-12-27 (Phase 46.0)
"""
import pytest
from src.evaluator.correction_policy import (
    CorrectionMode,
    get_g_correction,
    get_mirror_multiplier,
    is_derived_mode,
    is_anchored_mode,
    get_default_mode,
    get_all_derived_modes,
    get_all_anchored_modes,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


# Test parameters
THETA = 4 / 7
K = 3
R = 1.3036
F_I1 = 0.233  # Typical I1 fraction


class TestAnchoringGuard:
    """Tests for the anchoring guard (Phase 46.0 core requirement)."""

    def test_anchored_mode_blocked_by_default(self):
        """ANCHORED_TWO_BENCHMARKS must fail without allow_target_anchoring=True."""
        with pytest.raises(ValueError) as exc_info:
            get_g_correction(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            )

        assert "allow_target_anchoring=True" in str(exc_info.value)
        assert "NOT first-principles" in str(exc_info.value)

    def test_anchored_mode_blocked_with_explicit_false(self):
        """ANCHORED_TWO_BENCHMARKS fails even with allow_target_anchoring=False."""
        with pytest.raises(ValueError) as exc_info:
            get_g_correction(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
                allow_target_anchoring=False,
            )

        assert "allow_target_anchoring=True" in str(exc_info.value)

    def test_anchored_mode_works_with_explicit_true(self):
        """ANCHORED_TWO_BENCHMARKS succeeds with allow_target_anchoring=True."""
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            allow_target_anchoring=True,
        )

        assert result.mode == CorrectionMode.ANCHORED_TWO_BENCHMARKS
        assert result.g_I1 == G_I1_CALIBRATED
        assert result.g_I2 == G_I2_CALIBRATED

    def test_legacy_alias_also_blocked(self):
        """COMPONENT_RENORM_ANCHORED (legacy) is also blocked."""
        with pytest.raises(ValueError) as exc_info:
            get_g_correction(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode.COMPONENT_RENORM_ANCHORED,
            )

        assert "allow_target_anchoring=True" in str(exc_info.value)

    def test_legacy_alias_works_with_opt_in(self):
        """COMPONENT_RENORM_ANCHORED works with opt-in and normalizes to new name."""
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.COMPONENT_RENORM_ANCHORED,
            allow_target_anchoring=True,
        )

        # Should normalize to the new canonical name
        assert result.mode == CorrectionMode.ANCHORED_TWO_BENCHMARKS

    def test_get_mirror_multiplier_respects_guard(self):
        """Convenience function also respects the guard."""
        with pytest.raises(ValueError):
            get_mirror_multiplier(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            )


class TestDerivedModesNoGuard:
    """Tests that derived modes work without the guard."""

    def test_derived_baseline_works_by_default(self):
        """DERIVED_BASELINE_ONLY works without any special flags."""
        result = get_g_correction(R=R, theta=THETA, K=K)

        assert result.mode == CorrectionMode.DERIVED_BASELINE_ONLY
        assert result.g == result.g_baseline

    def test_first_principles_i1_i2_works_without_guard(self):
        """FIRST_PRINCIPLES_I1_I2 works without allow_target_anchoring."""
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
        )

        assert result.mode == CorrectionMode.FIRST_PRINCIPLES_I1_I2
        assert result.g_I1 == 1.0  # Derived value
        assert result.g_I2 == result.g_baseline  # Derived value

    def test_first_principles_i1_i2_requires_f_I1(self):
        """FIRST_PRINCIPLES_I1_I2 requires f_I1 parameter."""
        with pytest.raises(ValueError) as exc_info:
            get_g_correction(
                R=R, theta=THETA, K=K,
                mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
            )

        assert "f_I1" in str(exc_info.value)


class TestModeClassification:
    """Tests for mode classification helpers."""

    def test_derived_mode_classification(self):
        """is_derived_mode correctly identifies derived modes."""
        assert is_derived_mode(CorrectionMode.DERIVED_BASELINE_ONLY)
        assert is_derived_mode(CorrectionMode.FIRST_PRINCIPLES_I1_I2)
        assert not is_derived_mode(CorrectionMode.ANCHORED_TWO_BENCHMARKS)

    def test_anchored_mode_classification(self):
        """is_anchored_mode correctly identifies anchored modes."""
        assert is_anchored_mode(CorrectionMode.ANCHORED_TWO_BENCHMARKS)
        assert is_anchored_mode(CorrectionMode.COMPONENT_RENORM_ANCHORED)  # Legacy
        assert not is_anchored_mode(CorrectionMode.DERIVED_BASELINE_ONLY)
        assert not is_anchored_mode(CorrectionMode.FIRST_PRINCIPLES_I1_I2)

    def test_default_mode_is_derived(self):
        """Default mode must be a derived mode."""
        default = get_default_mode()
        assert is_derived_mode(default)
        assert default == CorrectionMode.DERIVED_BASELINE_ONLY

    def test_all_derived_modes_list(self):
        """get_all_derived_modes returns all derived modes."""
        modes = get_all_derived_modes()
        assert CorrectionMode.DERIVED_BASELINE_ONLY in modes
        assert CorrectionMode.FIRST_PRINCIPLES_I1_I2 in modes
        assert len(modes) == 2

    def test_all_anchored_modes_list(self):
        """get_all_anchored_modes returns all anchored modes."""
        modes = get_all_anchored_modes()
        assert CorrectionMode.ANCHORED_TWO_BENCHMARKS in modes
        assert len(modes) == 1


class TestFirstPrinciplesValues:
    """Tests that FIRST_PRINCIPLES_I1_I2 uses truly derived values."""

    def test_g_I1_is_exactly_one(self):
        """g_I1 in first-principles mode is exactly 1.0."""
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
        )

        assert result.g_I1 == 1.0

    def test_g_I2_is_g_baseline(self):
        """g_I2 in first-principles mode equals g_baseline."""
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
        )

        assert result.g_I2 == result.g_baseline

    def test_first_principles_differs_from_calibrated(self):
        """First-principles values differ from calibrated values."""
        fp_result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
        )

        anchored_result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=F_I1,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            allow_target_anchoring=True,
        )

        # g_I1: first-principles is 1.0, calibrated is 1.00091
        assert fp_result.g_I1 != anchored_result.g_I1
        assert abs(fp_result.g_I1 - 1.0) < 1e-10
        assert abs(anchored_result.g_I1 - G_I1_CALIBRATED) < 1e-10

        # g_I2: first-principles is g_baseline, calibrated is 1.01945
        assert fp_result.g_I2 != anchored_result.g_I2

    def test_weighted_g_formula(self):
        """Total g follows the weighted formula g = f_I1*g_I1 + (1-f_I1)*g_I2."""
        f_I1_test = 0.3
        result = get_g_correction(
            R=R, theta=THETA, K=K, f_I1=f_I1_test,
            mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2,
        )

        expected_g = f_I1_test * result.g_I1 + (1 - f_I1_test) * result.g_I2
        assert abs(result.g - expected_g) < 1e-10


class TestGuardCannotBeBypassed:
    """Tests that ensure the guard cannot be accidentally bypassed."""

    def test_cannot_set_mode_value_directly(self):
        """Cannot bypass guard by using raw enum value."""
        # Even if someone tries to pass the raw value, it should still be blocked
        with pytest.raises(ValueError):
            get_g_correction(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode("anchored_two_benchmarks"),
            )

    def test_guard_in_error_message_is_informative(self):
        """Error message explains why anchoring is blocked."""
        with pytest.raises(ValueError) as exc_info:
            get_g_correction(
                R=R, theta=THETA, K=K, f_I1=F_I1,
                mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            )

        error_msg = str(exc_info.value)
        # Should explain what went wrong
        assert "ANCHORED_TWO_BENCHMARKS" in error_msg
        # Should explain how to fix it
        assert "allow_target_anchoring=True" in error_msg
        # Should explain WHY this guard exists
        assert "first-principles" in error_msg.lower()
