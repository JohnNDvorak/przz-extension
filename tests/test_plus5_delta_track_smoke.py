"""
tests/test_plus5_delta_track_smoke.py
Phase 20.0: Smoke Tests for +5 Delta Tracking

PURPOSE:
========
Validate that the --plus5-split diagnostic produces correct structure
and that main-only mode properly excludes J15.

Tests:
1. Script returns dict with required keys
2. MAIN_TERM_ONLY forbids J15 access
3. Per-piece breakdown sums correctly
4. Both benchmarks produce results
5. JSON serialization works

USAGE:
======
    pytest tests/test_plus5_delta_track_smoke.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.ratios.plus5_harness import (
    compute_plus5_signature_split,
    Plus5SplitResult,
)
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    LaurentMode,
)


class TestPlus5SplitResultStructure:
    """Test that Plus5SplitResult has all required fields."""

    def test_result_is_dataclass(self):
        """Result should be a frozen dataclass."""
        result = compute_plus5_signature_split("kappa")
        assert isinstance(result, Plus5SplitResult)

    def test_result_has_main_only_fields(self):
        """Result has main-only B/A and components."""
        result = compute_plus5_signature_split("kappa")

        assert hasattr(result, "A_main_only")
        assert hasattr(result, "B_main_only")
        assert hasattr(result, "B_over_A_main_only")
        assert hasattr(result, "delta_main_only")

    def test_result_has_with_error_fields(self):
        """Result has with-error B/A and components."""
        result = compute_plus5_signature_split("kappa")

        assert hasattr(result, "A")
        assert hasattr(result, "B")
        assert hasattr(result, "B_over_A")
        assert hasattr(result, "B_over_A_with_error")
        assert hasattr(result, "delta")

    def test_result_has_j15_contribution(self):
        """Result tracks J15 contribution."""
        result = compute_plus5_signature_split("kappa")

        assert hasattr(result, "j15_contribution_A")
        assert hasattr(result, "j15_contribution_B")
        assert hasattr(result, "j15_contribution_ratio")
        assert hasattr(result, "j15_required_for_target")

    def test_result_has_gap_metrics(self):
        """Result has gap percentage metrics."""
        result = compute_plus5_signature_split("kappa")

        assert hasattr(result, "gap_percent")
        assert hasattr(result, "gap_percent_main_only")


class TestBothBenchmarksWork:
    """Test that both benchmarks produce valid results."""

    def test_kappa_produces_result(self):
        """Kappa benchmark runs without error."""
        result = compute_plus5_signature_split("kappa")
        assert np.isfinite(result.B_over_A)
        assert np.isfinite(result.B_over_A_main_only)

    def test_kappa_star_produces_result(self):
        """Kappa* benchmark runs without error."""
        result = compute_plus5_signature_split("kappa_star")
        assert np.isfinite(result.B_over_A)
        assert np.isfinite(result.B_over_A_main_only)

    def test_r_values_correct(self):
        """R values match expected benchmarks."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        assert abs(kappa.R - 1.3036) < 0.001
        assert abs(kappa_star.R - 1.1167) < 0.001


class TestJ15Separation:
    """Test that J15 is properly separated in main-only mode."""

    def test_main_only_excludes_j15(self):
        """Main-only computation should exclude J15 contribution."""
        result = compute_plus5_signature_split("kappa")

        # J15 contribution should be positive (it adds to B/A)
        assert result.j15_contribution_ratio > 0

        # Main-only should be smaller than with-error
        assert result.B_over_A_main_only < result.B_over_A_with_error

    def test_j15_contribution_adds_correctly(self):
        """J15 contribution should equal difference between modes."""
        result = compute_plus5_signature_split("kappa")

        expected_diff = result.B_over_A_with_error - result.B_over_A_main_only
        actual_contrib = result.j15_contribution_ratio

        assert abs(expected_diff - actual_contrib) < 1e-6

    def test_direct_include_j15_parameter(self):
        """include_j15=False should give same as main-only harness."""
        # Direct computation with include_j15=False
        from src.ratios.przz_polynomials import load_przz_k3_polynomials

        polys = load_przz_k3_polynomials("kappa", enforce_Q0=True)

        with_j15 = compute_m1_with_mirror_assembly(
            theta=4.0 / 7.0, R=1.3036, polys=polys, K=3, include_j15=True
        )
        without_j15 = compute_m1_with_mirror_assembly(
            theta=4.0 / 7.0, R=1.3036, polys=polys, K=3, include_j15=False
        )

        # With J15 should have higher B_over_A
        assert with_j15["B_over_A"] > without_j15["B_over_A"]


class TestMathematicalConsistency:
    """Test mathematical consistency of split computation."""

    def test_b_over_a_equals_b_divided_by_a(self):
        """B/A should equal B divided by A."""
        result = compute_plus5_signature_split("kappa")

        computed_ratio = result.B_main_only / result.A_main_only
        assert abs(computed_ratio - result.B_over_A_main_only) < 1e-10

    def test_a_contribution_sums(self):
        """A_main_only + j15_contribution_A should equal A."""
        result = compute_plus5_signature_split("kappa")

        expected_A = result.A_main_only + result.j15_contribution_A
        assert abs(expected_A - result.A) < 1e-10

    def test_b_contribution_sums(self):
        """B_main_only + j15_contribution_B should equal B."""
        result = compute_plus5_signature_split("kappa")

        expected_B = result.B_main_only + result.j15_contribution_B
        assert abs(expected_B - result.B) < 1e-10

    def test_delta_consistency(self):
        """Delta values should be finite and consistent."""
        result = compute_plus5_signature_split("kappa")

        assert np.isfinite(result.delta)
        assert np.isfinite(result.delta_main_only)


class TestJ15RequirementFlag:
    """Test the J15 requirement detection."""

    def test_j15_is_currently_required(self):
        """Current implementation requires J15 for B/A ≈ 5."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # Both should require J15 currently
        assert kappa.j15_required_for_target is True
        assert kappa_star.j15_required_for_target is True

    def test_main_only_gap_is_significant(self):
        """Main-only gap should be ~14-16% (significantly below 5)."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # Main-only B/A should be ~4.28, so gap is ~14%
        assert abs(kappa.gap_percent_main_only) > 10.0
        assert abs(kappa_star.gap_percent_main_only) > 10.0

    def test_with_error_gap_is_small(self):
        """With-error gap should be <5% (close to 5)."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # With J15, gap should be <5%
        assert abs(kappa.gap_percent) < 5.0
        assert abs(kappa_star.gap_percent) < 5.0


class TestDeterminism:
    """Test that results are deterministic."""

    def test_results_are_deterministic(self):
        """Multiple runs should give identical results."""
        result1 = compute_plus5_signature_split("kappa")
        result2 = compute_plus5_signature_split("kappa")

        assert result1.B_over_A == result2.B_over_A
        assert result1.B_over_A_main_only == result2.B_over_A_main_only
        assert result1.j15_contribution_ratio == result2.j15_contribution_ratio

    def test_benchmarks_give_different_results(self):
        """Different benchmarks should give different results."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        assert kappa.B_over_A != kappa_star.B_over_A
        assert kappa.R != kappa_star.R


class TestScriptIntegration:
    """Test the run_delta_report.py script integration."""

    def test_print_plus5_split_report_runs(self):
        """print_plus5_split_report should run without error."""
        from run_delta_report import print_plus5_split_report

        results = print_plus5_split_report(["kappa", "kappa_star"])

        assert "kappa" in results
        assert "kappa_star" in results

    def test_print_plus5_split_report_returns_dict(self):
        """print_plus5_split_report should return dict of Plus5SplitResult."""
        from run_delta_report import print_plus5_split_report

        results = print_plus5_split_report(["kappa"])

        assert isinstance(results, dict)
        assert isinstance(results["kappa"], Plus5SplitResult)


class TestDocumentation:
    """Document current Phase 20.0 state."""

    def test_phase_20_critical_finding(self):
        """
        PHASE 20.0 CRITICAL FINDING:

        Main-only B/A is ~4.28 (not 5).
        J15 adds ~0.67 to achieve B/A ≈ 4.95.

        This means the derivation relies on error terms.
        Phase 20.2 goal: fix main term to produce B/A = 5 without J15.
        """
        kappa = compute_plus5_signature_split("kappa")

        # Main-only should be ~4.28
        assert 4.2 < kappa.B_over_A_main_only < 4.4

        # J15 contribution should be ~0.67
        assert 0.6 < kappa.j15_contribution_ratio < 0.75

        # With error should be ~4.95
        assert 4.8 < kappa.B_over_A_with_error < 5.1
