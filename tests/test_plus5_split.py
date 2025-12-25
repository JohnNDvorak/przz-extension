"""
Tests for +5 signature split analysis (Phase 19.1.1).

These tests verify the main-only vs with-error mode separation,
ensuring we can distinguish B/A contributions from main terms
versus J₁,₅ error terms.
"""

import pytest
import numpy as np

from src.ratios.plus5_harness import (
    compute_plus5_signature_split,
    run_plus5_split_report,
    check_main_term_sufficiency,
    Plus5SplitResult,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestPlus5SplitComputation:
    """Test basic split computation functionality."""

    def test_compute_split_kappa(self):
        """Split computation works for kappa benchmark."""
        result = compute_plus5_signature_split("kappa")

        assert isinstance(result, Plus5SplitResult)
        assert result.benchmark == "kappa"
        assert np.isfinite(result.B_over_A)
        assert np.isfinite(result.B_over_A_main_only)

    def test_compute_split_kappa_star(self):
        """Split computation works for kappa_star benchmark."""
        result = compute_plus5_signature_split("kappa_star")

        assert isinstance(result, Plus5SplitResult)
        assert result.benchmark == "kappa_star"
        assert np.isfinite(result.B_over_A)
        assert np.isfinite(result.B_over_A_main_only)

    def test_full_and_with_error_are_equal(self):
        """B_over_A and B_over_A_with_error should be identical."""
        result = compute_plus5_signature_split("kappa")

        # These should be the same value (just different labels)
        assert abs(result.B_over_A - result.B_over_A_with_error) < 1e-10

    def test_j15_contribution_is_difference(self):
        """J15 contribution ratio equals full minus main-only."""
        result = compute_plus5_signature_split("kappa")

        expected_contribution = result.B_over_A - result.B_over_A_main_only
        assert abs(result.j15_contribution_ratio - expected_contribution) < 1e-10


class TestJ15Separation:
    """Test that J15 is properly separated between modes."""

    def test_main_only_excludes_j15(self):
        """Main-only mode should have smaller values (J15 excluded)."""
        result = compute_plus5_signature_split("kappa")

        # With J15 excluded, A and B should be smaller
        assert result.A_main_only < result.A
        assert result.B_main_only < result.B

    def test_j15_contribution_positive(self):
        """J15 contribution should be positive (adds to A and B)."""
        result = compute_plus5_signature_split("kappa")

        assert result.j15_contribution_A > 0
        assert result.j15_contribution_B > 0

    def test_j15_contribution_to_ratio_positive(self):
        """J15 increases the B/A ratio toward 5."""
        result = compute_plus5_signature_split("kappa")

        # J15 should add to the ratio
        assert result.j15_contribution_ratio > 0

    def test_main_only_below_full(self):
        """Main-only B/A should be below full B/A."""
        for bench in ["kappa", "kappa_star"]:
            result = compute_plus5_signature_split(bench)
            assert result.B_over_A_main_only < result.B_over_A


class TestJ15RequiredFlag:
    """Test the j15_required_for_target flag."""

    def test_flag_set_when_main_far_full_close(self):
        """Flag should be True when main-only is far but full is close."""
        result = compute_plus5_signature_split("kappa")

        # Based on current implementation:
        # - main-only gap > 5% AND full gap < 5% => j15_required = True
        if abs(result.gap_percent_main_only) > 5.0 and abs(result.gap_percent) < 5.0:
            assert result.j15_required_for_target is True

    def test_both_benchmarks_show_j15_required(self):
        """Both benchmarks currently show J15 is required."""
        # This is a documentation test - current state shows reliance on J15
        for bench in ["kappa", "kappa_star"]:
            result = compute_plus5_signature_split(bench)

            # Current findings: main-only has ~14-16% gap
            assert abs(result.gap_percent_main_only) > 10.0


class TestGapCalculations:
    """Test gap percentage calculations."""

    def test_gap_percent_formula(self):
        """Gap percent should be (B/A - 5) / 5 * 100."""
        result = compute_plus5_signature_split("kappa")

        expected_gap = (result.B_over_A - 5.0) / 5.0 * 100
        assert abs(result.gap_percent - expected_gap) < 1e-6

    def test_gap_percent_main_only_formula(self):
        """Main-only gap percent should follow same formula."""
        result = compute_plus5_signature_split("kappa")

        expected_gap = (result.B_over_A_main_only - 5.0) / 5.0 * 100
        assert abs(result.gap_percent_main_only - expected_gap) < 1e-6

    def test_full_gap_smaller_than_main_gap(self):
        """Full gap should be closer to 0 than main-only gap."""
        for bench in ["kappa", "kappa_star"]:
            result = compute_plus5_signature_split(bench)

            # Full B/A is closer to 5 than main-only
            assert abs(result.gap_percent) < abs(result.gap_percent_main_only)


class TestCheckMainTermSufficiency:
    """Test the main-term sufficiency gate function."""

    def test_check_returns_bool(self):
        """check_main_term_sufficiency returns a boolean."""
        result = check_main_term_sufficiency("kappa")
        assert isinstance(result, bool)

    def test_check_with_loose_tolerance(self):
        """With very loose tolerance, main-term might pass."""
        # Very loose tolerance (50%) should pass
        result = check_main_term_sufficiency("kappa", tolerance_percent=50.0)
        # Main-only is ~4.28, gap is ~14%, should pass at 50%
        assert result is True

    def test_check_with_strict_tolerance(self):
        """With strict tolerance, main-term should fail."""
        # Strict tolerance (5%) should fail
        result = check_main_term_sufficiency("kappa", tolerance_percent=5.0)
        # Main-only gap is ~14%, should fail at 5%
        assert result is False


class TestReportGeneration:
    """Test report generation functionality."""

    def test_run_report_returns_dict(self):
        """run_plus5_split_report returns dict of results."""
        results = run_plus5_split_report(verbose=False)

        assert isinstance(results, dict)
        assert "kappa" in results
        assert "kappa_star" in results

    def test_report_results_are_valid(self):
        """Report results are valid Plus5SplitResult objects."""
        results = run_plus5_split_report(verbose=False)

        for result in results.values():
            assert isinstance(result, Plus5SplitResult)
            assert np.isfinite(result.B_over_A)
            assert np.isfinite(result.B_over_A_main_only)


class TestResultDataclass:
    """Test Plus5SplitResult dataclass functionality."""

    def test_to_dict_serializable(self):
        """to_dict produces JSON-serializable output."""
        import json

        result = compute_plus5_signature_split("kappa")
        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        recovered = json.loads(json_str)

        assert recovered["benchmark"] == result.benchmark
        assert abs(recovered["B_over_A"] - result.B_over_A) < 1e-10

    def test_result_frozen(self):
        """Plus5SplitResult is immutable (frozen)."""
        result = compute_plus5_signature_split("kappa")

        with pytest.raises(AttributeError):
            result.B_over_A = 99.0


class TestLaurentModeConsistency:
    """Test behavior across Laurent modes."""

    def test_mode_in_result(self):
        """Laurent mode is recorded in result."""
        result = compute_plus5_signature_split(
            "kappa", laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )
        assert result.mode == "actual_logderiv"

    def test_different_modes_give_different_results(self):
        """Different Laurent modes should give different B/A values."""
        result_actual = compute_plus5_signature_split(
            "kappa", laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )
        result_raw = compute_plus5_signature_split(
            "kappa", laurent_mode=LaurentMode.RAW_LOGDERIV
        )

        # Results should differ
        assert abs(result_actual.B_over_A - result_raw.B_over_A) > 0.01


class TestDeltaMetrics:
    """Test delta metrics in split result."""

    def test_delta_present(self):
        """Delta metrics are present in result."""
        result = compute_plus5_signature_split("kappa")

        assert hasattr(result, 'delta')
        assert hasattr(result, 'delta_main_only')
        assert np.isfinite(result.delta)
        assert np.isfinite(result.delta_main_only)

    def test_delta_values_finite(self):
        """Delta values are finite (can be negative or positive)."""
        result = compute_plus5_signature_split("kappa")

        # delta = D/A where D = I12+ + I34+ (contamination from +R branch)
        # D can be negative if I12+ and I34+ have opposite signs
        # Just verify they're finite
        assert np.isfinite(result.delta)
        assert np.isfinite(result.delta_main_only)

        # Main-only delta should have larger magnitude (A is smaller)
        # delta_main_only = D_main / A_main where A_main < A
        assert abs(result.delta_main_only) > abs(result.delta)


class TestCriticalFinding:
    """Document the critical Phase 19.1 finding."""

    def test_j15_is_required_for_both_benchmarks(self):
        """
        CRITICAL FINDING: J₁,₅ is required to achieve B/A ≈ 5.

        Per TRUTH_SPEC Lines 1621-1628, J₁,₅ involves A^{(1,1)} which
        is explicitly an error term. If we need J₁,₅ to match the
        target, we're computing the wrong main-term object.

        Current status:
        - kappa main-only B/A: ~4.28 (14% gap)
        - kappa_star main-only B/A: ~4.22 (16% gap)

        This suggests the derivation needs revision.
        """
        for bench in ["kappa", "kappa_star"]:
            result = compute_plus5_signature_split(bench)

            # Document current findings
            assert result.j15_required_for_target is True
            assert abs(result.gap_percent_main_only) > 10.0
            assert result.j15_contribution_ratio > 0.5

    def test_j15_contribution_consistent(self):
        """J15 contribution should be similar magnitude across benchmarks."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # J15 contribution to B/A should be in same ballpark
        ratio = kappa.j15_contribution_ratio / kappa_star.j15_contribution_ratio
        assert 0.8 < ratio < 1.2
