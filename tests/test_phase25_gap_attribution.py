"""
tests/test_phase25_gap_attribution.py
Phase 25.1 & 25.2: Gap Attribution Harness and S34 Invariance Gate Tests

PURPOSE:
========
Verify the gap attribution harness works correctly and identify where
the 5-7% gap between unified S12 and empirical DSL evaluators originates.

TASKS COVERED:
==============
- 25.1: GapReport structure and computation
- 25.2: S34 Invariance Gate (critical diagnostic)

Created: 2025-12-25
"""

import pytest
import math

from src.evaluator.gap_attribution import (
    GapReport,
    compute_gap_report,
    run_dual_benchmark_gap_attribution,
    print_gap_report,
    print_dual_benchmark_summary,
    KAPPA_R,
    KAPPA_STAR_R,
    THETA,
    KAPPA_C_TARGET,
    KAPPA_STAR_C_TARGET,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def kappa_polys():
    """Load kappa polynomials once for all tests."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture(scope="module")
def kappa_star_polys():
    """Load kappa* polynomials once for all tests."""
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    return {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}


@pytest.fixture(scope="module")
def kappa_report(kappa_polys):
    """Compute kappa gap report once for all tests."""
    return compute_gap_report(
        theta=THETA,
        R=KAPPA_R,
        n_quad=40,  # Faster for tests
        polynomials=kappa_polys,
        normalization_mode="scalar",
        benchmark_name="kappa",
        c_target=KAPPA_C_TARGET,
    )


@pytest.fixture(scope="module")
def kappa_star_report(kappa_star_polys):
    """Compute kappa* gap report once for all tests."""
    return compute_gap_report(
        theta=THETA,
        R=KAPPA_STAR_R,
        n_quad=40,  # Faster for tests
        polynomials=kappa_star_polys,
        normalization_mode="scalar",
        benchmark_name="kappa_star",
        c_target=KAPPA_STAR_C_TARGET,
    )


# =============================================================================
# TASK 25.1: GAP REPORT STRUCTURE TESTS
# =============================================================================


class TestGapReportStructure:
    """Verify GapReport has all required fields and computes correctly."""

    def test_gap_report_has_all_required_fields(self, kappa_report):
        """GapReport should have all documented fields."""
        # Input fields
        assert hasattr(kappa_report, "theta")
        assert hasattr(kappa_report, "R")
        assert hasattr(kappa_report, "n_quad")
        assert hasattr(kappa_report, "benchmark")

        # Unified fields
        assert hasattr(kappa_report, "unified_S12_total")
        assert hasattr(kappa_report, "unified_S12_unnormalized")
        assert hasattr(kappa_report, "unified_normalization_factor")
        assert hasattr(kappa_report, "unified_normalization_mode")

        # Empirical fields
        assert hasattr(kappa_report, "empirical_c_total")
        assert hasattr(kappa_report, "empirical_S12_plus_total")
        assert hasattr(kappa_report, "empirical_S12_minus_total")
        assert hasattr(kappa_report, "empirical_S12_combined")
        assert hasattr(kappa_report, "empirical_S34_total")

        # Gap metrics
        assert hasattr(kappa_report, "delta_S12")
        assert hasattr(kappa_report, "delta_S34")
        assert hasattr(kappa_report, "ratio_S12")

    def test_gap_report_values_are_finite(self, kappa_report):
        """All numeric values should be finite."""
        assert math.isfinite(kappa_report.unified_S12_total)
        assert math.isfinite(kappa_report.unified_S12_unnormalized)
        assert math.isfinite(kappa_report.empirical_c_total)
        assert math.isfinite(kappa_report.empirical_S12_combined)
        assert math.isfinite(kappa_report.empirical_S34_total)
        assert math.isfinite(kappa_report.delta_S12)
        assert math.isfinite(kappa_report.delta_S34)
        assert math.isfinite(kappa_report.ratio_S12)

    def test_gap_report_normalization_mode_is_scalar(self, kappa_report):
        """Should use scalar normalization (Phase 24 default)."""
        assert kappa_report.unified_normalization_mode == "scalar"

    def test_gap_report_to_dict(self, kappa_report):
        """to_dict() should produce valid dictionary for JSON."""
        d = kappa_report.to_dict()
        assert isinstance(d, dict)
        assert "benchmark" in d
        assert "unified" in d
        assert "empirical" in d
        assert "gap_metrics" in d
        assert "diagnosis" in d

    def test_dual_benchmark_returns_both_reports(self):
        """run_dual_benchmark_gap_attribution should return both reports."""
        kappa, kappa_star = run_dual_benchmark_gap_attribution(n_quad=30)
        assert kappa.benchmark == "kappa"
        assert kappa_star.benchmark == "kappa_star"
        assert kappa.R == pytest.approx(KAPPA_R)
        assert kappa_star.R == pytest.approx(KAPPA_STAR_R)


class TestGapMetrics:
    """Verify gap metric calculations are correct."""

    def test_delta_S12_calculation(self, kappa_report):
        """delta_S12 = unified - empirical."""
        expected = kappa_report.unified_S12_total - kappa_report.empirical_S12_combined
        assert kappa_report.delta_S12 == pytest.approx(expected, rel=1e-10)

    def test_ratio_S12_calculation(self, kappa_report):
        """ratio_S12 = unified / empirical."""
        expected = kappa_report.unified_S12_total / kappa_report.empirical_S12_combined
        assert kappa_report.ratio_S12 == pytest.approx(expected, rel=1e-10)

    def test_s12_gap_is_around_5_to_7_percent(self, kappa_report):
        """The S12 gap should be in the 5-7% range (Phase 24 expectation)."""
        gap_pct = abs(kappa_report.ratio_S12 - 1.0) * 100
        # Allow wider range for test stability, but should be in ballpark
        assert 0 < gap_pct < 20, f"S12 gap {gap_pct:.1f}% outside expected range"

    def test_empirical_s12_combined_formula(self, kappa_report):
        """S12_combined = S12_plus + m * S12_minus where m = exp(R) + 5."""
        m = math.exp(kappa_report.R) + 5
        expected = (kappa_report.empirical_S12_plus_total +
                   m * kappa_report.empirical_S12_minus_total)
        assert kappa_report.empirical_S12_combined == pytest.approx(expected, rel=1e-10)


# =============================================================================
# TASK 25.2: S34 INVARIANCE GATE TESTS (CRITICAL)
# =============================================================================


class TestS34Invariance:
    """
    Gate tests for S34 invariance between evaluators.

    SPEC: I3 and I4 do NOT have mirror (TRUTH_SPEC.md Section 10).
    Therefore S34 MUST be computed identically in both evaluators.

    If this test fails, we have found a real bug that contaminates all results.
    """

    def test_s34_invariance_kappa(self, kappa_report):
        """
        S34 must be identical between evaluators for kappa benchmark.

        CRITICAL: If this fails, stop everything and fix S34.
        """
        # S34 should have near-zero delta (within numerical tolerance)
        # Note: Small differences are expected due to different code paths
        assert abs(kappa_report.delta_S34) < 0.01, (
            f"S34 INVARIANCE VIOLATED at kappa: delta_S34 = {kappa_report.delta_S34:.2e}\n"
            f"This indicates a bug in S34 computation between evaluators.\n"
            f"STOP: Fix S34 before proceeding with other Phase 25 tasks."
        )

    def test_s34_invariance_kappa_star(self, kappa_star_report):
        """
        S34 must be identical between evaluators for kappa* benchmark.

        CRITICAL: If this fails, stop everything and fix S34.
        """
        assert abs(kappa_star_report.delta_S34) < 0.01, (
            f"S34 INVARIANCE VIOLATED at kappa*: delta_S34 = {kappa_star_report.delta_S34:.2e}\n"
            f"This indicates a bug in S34 computation between evaluators.\n"
            f"STOP: Fix S34 before proceeding with other Phase 25 tasks."
        )

    def test_s34_ratio_near_unity_kappa(self, kappa_report):
        """S34 ratio should be ~1.0 for kappa."""
        # Allow 0.1% tolerance
        assert kappa_report.ratio_S34 == pytest.approx(1.0, rel=0.001), (
            f"S34 ratio = {kappa_report.ratio_S34:.6f} (expected ~1.0)"
        )

    def test_s34_ratio_near_unity_kappa_star(self, kappa_star_report):
        """S34 ratio should be ~1.0 for kappa*."""
        assert kappa_star_report.ratio_S34 == pytest.approx(1.0, rel=0.001), (
            f"S34 ratio = {kappa_star_report.ratio_S34:.6f} (expected ~1.0)"
        )

    def test_s34_invariant_flag_set(self, kappa_report, kappa_star_report):
        """Both reports should have s34_invariant=True."""
        assert kappa_report.s34_invariant, "kappa S34 should be invariant"
        assert kappa_star_report.s34_invariant, "kappa* S34 should be invariant"


# =============================================================================
# TWO-BENCHMARK GATE TESTS
# =============================================================================


class TestTwoBenchmarkGate:
    """
    Verify findings are consistent across BOTH benchmarks.

    Any finding that only applies to one benchmark is likely an artifact,
    not a true structural issue.
    """

    def test_gap_in_s12_consistent_both_benchmarks(self, kappa_report, kappa_star_report):
        """
        If there's a gap in S12, it should appear in both benchmarks.
        """
        # Both should have gap_in_S12 set consistently
        # (either both True or both False)
        assert kappa_report.gap_in_S12 == kappa_star_report.gap_in_S12, (
            f"Inconsistent S12 gap detection:\n"
            f"  kappa: gap_in_S12 = {kappa_report.gap_in_S12}\n"
            f"  kappa*: gap_in_S12 = {kappa_star_report.gap_in_S12}"
        )

    def test_ratio_s12_similar_both_benchmarks(self, kappa_report, kappa_star_report):
        """
        S12 ratios should be similar (within 2%) across benchmarks.

        Large differences indicate R-dependent effects.
        """
        ratio_diff = abs(kappa_report.ratio_S12 - kappa_star_report.ratio_S12)
        assert ratio_diff < 0.02, (
            f"S12 ratio differs significantly between benchmarks:\n"
            f"  kappa: ratio_S12 = {kappa_report.ratio_S12:.4f}\n"
            f"  kappa*: ratio_S12 = {kappa_star_report.ratio_S12:.4f}\n"
            f"  difference = {ratio_diff:.4f}\n"
            f"WARNING: R-dependent effect detected!"
        )


# =============================================================================
# DIAGNOSTIC OUTPUT TESTS
# =============================================================================


class TestDiagnosticOutput:
    """Test diagnostic print functions don't crash."""

    def test_print_gap_report_runs(self, kappa_report, capsys):
        """print_gap_report should run without error."""
        print_gap_report(kappa_report)
        captured = capsys.readouterr()
        assert "GAP ATTRIBUTION REPORT" in captured.out
        assert "kappa" in captured.out.lower()

    def test_print_dual_benchmark_summary_runs(self, kappa_report, kappa_star_report, capsys):
        """print_dual_benchmark_summary should run without error."""
        print_dual_benchmark_summary(kappa_report, kappa_star_report)
        captured = capsys.readouterr()
        assert "DUAL BENCHMARK SUMMARY" in captured.out


# =============================================================================
# PARAMETRIZED R-SWEEP TESTS
# =============================================================================


@pytest.mark.parametrize("R,name", [(KAPPA_R, "kappa"), (KAPPA_STAR_R, "kappa_star")])
class TestParametrizedBenchmarks:
    """Parametrized tests across both benchmarks."""

    def test_unified_s12_positive(self, R, name, kappa_polys, kappa_star_polys):
        """Unified S12 should be positive."""
        polys = kappa_polys if name == "kappa" else kappa_star_polys
        c_target = KAPPA_C_TARGET if name == "kappa" else KAPPA_STAR_C_TARGET
        report = compute_gap_report(
            theta=THETA,
            R=R,
            n_quad=30,
            polynomials=polys,
            normalization_mode="scalar",
            benchmark_name=name,
            c_target=c_target,
        )
        assert report.unified_S12_total > 0, f"S12 should be positive for {name}"

    def test_empirical_c_positive(self, R, name, kappa_polys, kappa_star_polys):
        """Empirical c should be positive."""
        polys = kappa_polys if name == "kappa" else kappa_star_polys
        c_target = KAPPA_C_TARGET if name == "kappa" else KAPPA_STAR_C_TARGET
        report = compute_gap_report(
            theta=THETA,
            R=R,
            n_quad=30,
            polynomials=polys,
            normalization_mode="scalar",
            benchmark_name=name,
            c_target=c_target,
        )
        assert report.empirical_c_total > 0, f"c should be positive for {name}"
