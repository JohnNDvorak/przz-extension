"""
Tests for K-sweep universality analysis (Phase 19.5).

These tests verify that B/A = 2K-1 holds across K values.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from k_sweep import (
    compute_k_result,
    run_k_sweep,
    run_j15_comparison_sweep,
    KSweepResult,
    KSweepReport,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestComputeKResult:
    """Test single K computation."""

    def test_compute_k3_kappa(self):
        """K=3 computation works for kappa."""
        result = compute_k_result("kappa", K=3)

        assert isinstance(result, KSweepResult)
        assert result.K == 3
        assert result.target_B_over_A == 5
        assert np.isfinite(result.B_over_A)

    def test_compute_k4_kappa(self):
        """K=4 computation works for kappa (using K=3 polys)."""
        result = compute_k_result("kappa", K=4)

        assert result.K == 4
        assert result.target_B_over_A == 7
        assert np.isfinite(result.B_over_A)

    def test_compute_k5_kappa_star(self):
        """K=5 computation works for kappa_star."""
        result = compute_k_result("kappa_star", K=5)

        assert result.K == 5
        assert result.target_B_over_A == 9
        assert np.isfinite(result.B_over_A)

    def test_target_is_2K_minus_1(self):
        """Target B/A is always 2K-1."""
        for K in [3, 4, 5, 6]:
            result = compute_k_result("kappa", K=K)
            assert result.target_B_over_A == 2 * K - 1


class TestKSweepReport:
    """Test K-sweep report generation."""

    def test_run_sweep_returns_report(self):
        """run_k_sweep returns a KSweepReport."""
        report = run_k_sweep(K_values=[3, 4], verbose=False)

        assert isinstance(report, KSweepReport)
        assert "kappa" in report.results
        assert "kappa_star" in report.results

    def test_report_has_all_k_values(self):
        """Report includes all requested K values."""
        K_values = [3, 4, 5]
        report = run_k_sweep(K_values=K_values, verbose=False)

        for bench in ["kappa", "kappa_star"]:
            for K in K_values:
                assert K in report.results[bench]

    def test_report_tracks_universality(self):
        """Report tracks whether universality holds."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        assert isinstance(report.universality_holds, bool)
        assert isinstance(report.max_gap_percent, float)
        assert report.gap_trend in ["shrinking", "constant", "growing", "unknown"]


class TestGapBehavior:
    """Test gap behavior across K values."""

    def test_gap_shrinks_percentage_wise(self):
        """Gap percentage should shrink with K (constant absolute gap)."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        for bench in ["kappa", "kappa_star"]:
            gaps = [report.results[bench][K].gap_percent for K in [3, 4, 5]]

            # Gap percentages should decrease (less negative)
            assert gaps[0] < gaps[1] < gaps[2] < 0

    def test_absolute_gap_constant(self):
        """Absolute gap (B/A - target) should be nearly constant."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        for bench in ["kappa", "kappa_star"]:
            gaps = [report.results[bench][K].gap for K in [3, 4, 5]]

            # All gaps should be within 0.01 of each other
            assert max(gaps) - min(gaps) < 0.01

    def test_b_over_a_increases_by_2(self):
        """B/A should increase by ~2 for each K increment."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        for bench in ["kappa", "kappa_star"]:
            b_over_a = [report.results[bench][K].B_over_A for K in [3, 4, 5]]

            # Differences should be ~2
            diff1 = b_over_a[1] - b_over_a[0]
            diff2 = b_over_a[2] - b_over_a[1]

            assert abs(diff1 - 2.0) < 0.01
            assert abs(diff2 - 2.0) < 0.01


class TestUniversality:
    """Test universality criteria."""

    def test_universality_holds_for_default_sweep(self):
        """Universality should hold for K=3,4,5."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        # All gaps should be < 10%
        assert report.universality_holds is True

    def test_max_gap_is_at_k3(self):
        """Maximum gap percent should be at K=3 (smallest K)."""
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        for bench in ["kappa", "kappa_star"]:
            gaps = {K: abs(report.results[bench][K].gap_percent) for K in [3, 4, 5]}
            max_K = max(gaps, key=gaps.get)
            assert max_K == 3


class TestJ15Comparison:
    """Test J15 contribution comparison."""

    def test_j15_comparison_runs(self):
        """J15 comparison sweep runs without error."""
        results = run_j15_comparison_sweep(
            K_values=[3, 4], verbose=False
        )

        assert "kappa" in results
        assert "kappa_star" in results

    def test_j15_contribution_constant_across_k(self):
        """J15 contribution should be constant across K values."""
        results = run_j15_comparison_sweep(
            K_values=[3, 4, 5], verbose=False
        )

        for bench in ["kappa", "kappa_star"]:
            contributions = [
                results[bench][K]["j15_contribution"]
                for K in [3, 4, 5]
            ]

            # All contributions should be within 0.001 of each other
            assert max(contributions) - min(contributions) < 0.001

    def test_j15_makes_gap_smaller(self):
        """Including J15 should reduce the gap magnitude."""
        results = run_j15_comparison_sweep(
            K_values=[3], verbose=False
        )

        for bench in ["kappa", "kappa_star"]:
            with_gap = abs(results[bench][3]["with_gap_pct"])
            without_gap = abs(results[bench][3]["without_gap_pct"])

            # Gap should be smaller with J15
            assert with_gap < without_gap


class TestLaurentModes:
    """Test behavior across Laurent modes."""

    def test_actual_mode_gives_small_gaps(self):
        """ACTUAL_LOGDERIV mode should give small gaps."""
        report = run_k_sweep(
            K_values=[3],
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
            verbose=False,
        )

        for bench in ["kappa", "kappa_star"]:
            gap = abs(report.results[bench][3].gap_percent)
            assert gap < 5.0  # Less than 5% gap

    def test_different_modes_give_different_results(self):
        """Different Laurent modes should give different B/A values."""
        actual = run_k_sweep(
            K_values=[3],
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
            verbose=False,
        )
        raw = run_k_sweep(
            K_values=[3],
            laurent_mode=LaurentMode.RAW_LOGDERIV,
            verbose=False,
        )

        for bench in ["kappa", "kappa_star"]:
            actual_ba = actual.results[bench][3].B_over_A
            raw_ba = raw.results[bench][3].B_over_A

            # Results should differ
            assert abs(actual_ba - raw_ba) > 0.1


class TestWarnings:
    """Test warning generation."""

    def test_k4_generates_proxy_warning(self):
        """K=4 should generate a proxy polynomial warning."""
        report = run_k_sweep(K_values=[4], verbose=False)

        # Should have warnings about proxy polynomials
        assert len(report.warnings) > 0
        assert any("proxy" in w.lower() for w in report.warnings)

    def test_k3_no_proxy_warning(self):
        """K=3 should not generate proxy warnings."""
        report = run_k_sweep(K_values=[3], verbose=False)

        # Should have no warnings (or not proxy warnings)
        proxy_warnings = [w for w in report.warnings if "proxy" in w.lower()]
        assert len(proxy_warnings) == 0


class TestResultDataclass:
    """Test KSweepResult dataclass."""

    def test_result_fields_present(self):
        """KSweepResult has all required fields."""
        result = compute_k_result("kappa", K=3)

        assert hasattr(result, 'benchmark')
        assert hasattr(result, 'K')
        assert hasattr(result, 'target_B_over_A')
        assert hasattr(result, 'B_over_A')
        assert hasattr(result, 'gap')
        assert hasattr(result, 'gap_percent')

    def test_gap_calculation_correct(self):
        """Gap is correctly calculated as B/A - target."""
        result = compute_k_result("kappa", K=3)

        expected_gap = result.B_over_A - result.target_B_over_A
        assert abs(result.gap - expected_gap) < 1e-10

    def test_gap_percent_calculation_correct(self):
        """Gap percent is correctly calculated."""
        result = compute_k_result("kappa", K=3)

        expected_pct = result.gap / result.target_B_over_A * 100
        assert abs(result.gap_percent - expected_pct) < 1e-6


class TestDocumentation:
    """Document key findings from K-sweep."""

    def test_k_sweep_findings(self):
        """
        PHASE 19.5 FINDINGS:

        1. B/A increases by exactly 2 for each K increment
        2. Absolute gap is constant (~-0.05 for kappa, ~-0.13 for kappa_star)
        3. Gap percentage SHRINKS with K (good extrapolation)
        4. J15 contribution is constant across K (~0.67)
        5. Universality holds: formula B/A = 2K-1 is structurally correct
        """
        report = run_k_sweep(K_values=[3, 4, 5], verbose=False)

        # Document current state
        assert report.gap_trend == "shrinking"
        assert report.universality_holds is True
        assert report.max_gap_percent < 5.0  # Current: ~2.66%
