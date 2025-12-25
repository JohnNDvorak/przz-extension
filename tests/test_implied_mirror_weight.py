"""
tests/test_implied_mirror_weight.py
Tests for the implied mirror weight diagnostic (Phase 18.1).
"""

import pytest
import math

from src.diagnostics.implied_mirror_weight import (
    compute_implied_m1,
    compute_implied_m1_with_breakdown,
    run_implied_m1_comparison,
    ImpliedM1Result,
    BENCHMARKS,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestComputeImpliedM1:
    """Tests for the basic implied m1 computation."""

    def test_simple_case(self):
        """Test implied m1 computation with known values."""
        # c = I12+ + m1 * I12- + I34+
        # 10 = 2 + m1 * 3 + 1
        # m1 = (10 - 2 - 1) / 3 = 7/3
        m1 = compute_implied_m1(
            c_target=10.0,
            I12_plus=2.0,
            I12_minus=3.0,
            I34_plus=1.0,
            R=1.0,
            K=3,
        )
        assert abs(m1 - 7.0 / 3.0) < 1e-10

    def test_residual_is_zero(self):
        """Verify that computed c matches target exactly."""
        c_target = 5.0
        I12_plus = 1.0
        I12_minus = 2.0
        I34_plus = 0.5

        m1 = compute_implied_m1(
            c_target=c_target,
            I12_plus=I12_plus,
            I12_minus=I12_minus,
            I34_plus=I34_plus,
            R=1.0,
            K=3,
        )

        c_computed = I12_plus + m1 * I12_minus + I34_plus
        assert abs(c_computed - c_target) < 1e-10

    def test_raises_on_zero_denominator(self):
        """Test that zero I12_minus raises ValueError."""
        with pytest.raises(ValueError, match="I12_minus is too small"):
            compute_implied_m1(
                c_target=5.0,
                I12_plus=1.0,
                I12_minus=0.0,
                I34_plus=0.5,
                R=1.0,
                K=3,
            )


class TestComputeImpliedM1WithBreakdown:
    """Tests for the full implied m1 diagnostic."""

    def test_kappa_benchmark_runs(self):
        """Test that kappa benchmark computes without error."""
        result = compute_implied_m1_with_breakdown("kappa")
        assert isinstance(result, ImpliedM1Result)
        assert result.benchmark == "kappa"
        assert result.R == 1.3036

    def test_kappa_star_benchmark_runs(self):
        """Test that kappa* benchmark computes without error."""
        result = compute_implied_m1_with_breakdown("kappa_star")
        assert isinstance(result, ImpliedM1Result)
        assert result.benchmark == "kappa_star"
        assert result.R == 1.1167

    def test_residual_near_zero(self):
        """Verify c_computed matches c_target within tolerance."""
        for benchmark in ["kappa", "kappa_star"]:
            result = compute_implied_m1_with_breakdown(benchmark)
            assert abs(result.residual) < 1e-10, (
                f"Residual too large for {benchmark}: {result.residual}"
            )

    def test_implied_m1_is_positive(self):
        """Verify implied m1 is positive for both benchmarks."""
        for benchmark in ["kappa", "kappa_star"]:
            result = compute_implied_m1_with_breakdown(benchmark)
            assert result.m1_implied > 0, (
                f"Implied m1 should be positive for {benchmark}"
            )

    def test_j1x_implied_differs_from_empirical(self):
        """
        Verify J1x implied m1 differs significantly from empirical.

        This is expected because J1x uses Case B-only, not full Case C.
        The ratio should be ~0.15-0.25 (J1x needs much less m1 weight).
        """
        for benchmark in ["kappa", "kappa_star"]:
            result = compute_implied_m1_with_breakdown(benchmark)
            # J1x implied is much smaller than empirical
            assert result.ratio_to_empirical < 0.5, (
                f"J1x implied m1 should be << empirical for {benchmark}"
            )
            assert result.ratio_to_empirical > 0.1, (
                f"J1x implied m1 should still be positive for {benchmark}"
            )

    def test_channel_values_are_reasonable(self):
        """Verify channel values are in expected ranges."""
        for benchmark in ["kappa", "kappa_star"]:
            result = compute_implied_m1_with_breakdown(benchmark)
            # I12_plus should be positive
            assert result.I12_plus > 0
            # I12_minus should be positive (for mirror assembly)
            assert result.I12_minus > 0
            # I34_plus is typically negative (J13/J14 structure)
            # Don't assert sign - just check it's not huge
            assert abs(result.I34_plus) < 10

    def test_unknown_benchmark_raises(self):
        """Test that unknown benchmark raises ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            compute_implied_m1_with_breakdown("unknown")


class TestRunImpliedM1Comparison:
    """Tests for the comparison across benchmarks."""

    def test_returns_both_benchmarks(self):
        """Test that comparison returns results for both benchmarks."""
        results = run_implied_m1_comparison(verbose=False)
        assert "kappa" in results
        assert "kappa_star" in results

    def test_both_residuals_near_zero(self):
        """Test that both benchmarks have near-zero residuals."""
        results = run_implied_m1_comparison(verbose=False)
        for name, result in results.items():
            assert abs(result.residual) < 1e-10, (
                f"Residual too large for {name}: {result.residual}"
            )

    def test_ratio_difference_is_documented(self):
        """
        Test that ratio difference between benchmarks is understood.

        The kappa vs kappa* ratio difference indicates R-dependence
        in the J1x channel structure.
        """
        results = run_implied_m1_comparison(verbose=False)
        ratio_kappa = results["kappa"].ratio_to_empirical
        ratio_kappa_star = results["kappa_star"].ratio_to_empirical
        ratio_diff = abs(ratio_kappa - ratio_kappa_star)

        # Should be non-zero but not huge
        assert ratio_diff < 0.2, "Ratio difference too large"
        # The finding is that they differ - this is expected
        # due to R-dependent effects in J1x channels


class TestLaurentModeEffect:
    """Test the effect of Laurent mode on implied m1."""

    def test_actual_vs_raw_mode(self):
        """Test that ACTUAL and RAW modes give different results."""
        result_actual = compute_implied_m1_with_breakdown(
            "kappa",
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
        )
        result_raw = compute_implied_m1_with_breakdown(
            "kappa",
            laurent_mode=LaurentMode.RAW_LOGDERIV,
        )

        # Channel values should differ between modes
        assert result_actual.I12_plus != result_raw.I12_plus or \
               result_actual.I34_plus != result_raw.I34_plus, (
            "ACTUAL and RAW modes should give different channel values"
        )
