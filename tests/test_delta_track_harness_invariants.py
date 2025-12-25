"""
tests/test_delta_track_harness_invariants.py
Phase 14I Task I3: Invariants and Regression Tests for Delta Harness

PURPOSE:
========
Ensure delta computations satisfy algebraic invariants and prevent
regressions from previous phases.

INVARIANTS TESTED:
==================
1. Algebraic identities:
   - delta == D/A (definition)
   - B_over_A == (2K-1) + delta for K=3

2. Phase 14E regression prevention:
   - I12_plus != I12_minus (the abs(R) bug made these equal)

3. Phase 14F stability:
   - delta < 0.5 for both benchmarks in default mode

4. Attribution consistency:
   - delta_s12 + delta_s34 == delta (by definition)
"""

import pytest
from src.ratios.delta_track_harness import (
    compute_delta_metrics,
    compute_delta_metrics_extended,
    DeltaMetrics,
    DeltaMetricsExtended,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestAlgebraicIdentities:
    """Test that delta satisfies its defining algebraic relations."""

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    @pytest.mark.parametrize("mode", [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED])
    def test_delta_equals_D_over_A(self, benchmark, mode):
        """delta = D / A by definition."""
        metrics = compute_delta_metrics_extended(benchmark, mode)

        expected_delta = metrics.D / metrics.A
        actual_delta = metrics.delta

        assert abs(expected_delta - actual_delta) < 1e-12, (
            f"delta should equal D/A for {benchmark} {mode.value}: "
            f"D={metrics.D:.10f}, A={metrics.A:.10f}, "
            f"D/A={expected_delta:.10f}, delta={actual_delta:.10f}"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    @pytest.mark.parametrize("mode", [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED])
    def test_B_over_A_equals_2K_minus_1_plus_delta(self, benchmark, mode):
        """B/A = (2K-1) + delta for K=3 pieces."""
        K = 3  # Fixed for our implementation
        metrics = compute_delta_metrics_extended(benchmark, mode)

        expected_B_over_A = (2 * K - 1) + metrics.delta
        actual_B_over_A = metrics.B_over_A

        assert abs(expected_B_over_A - actual_B_over_A) < 1e-12, (
            f"B/A should equal 5 + delta for K=3 in {benchmark} {mode.value}: "
            f"expected={expected_B_over_A:.10f}, actual={actual_B_over_A:.10f}, "
            f"delta={metrics.delta:.10f}"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_attribution_sums_to_delta(self, benchmark):
        """delta_s12 + delta_s34 should equal delta."""
        metrics = compute_delta_metrics_extended(benchmark, LaurentMode.RAW_LOGDERIV)

        attribution_sum = metrics.delta_s12 + metrics.delta_s34
        delta = metrics.delta

        # Allow small tolerance for floating point
        assert abs(attribution_sum - delta) < 1e-10, (
            f"delta_s12 + delta_s34 should equal delta for {benchmark}: "
            f"delta_s12={metrics.delta_s12:.10f}, delta_s34={metrics.delta_s34:.10f}, "
            f"sum={attribution_sum:.10f}, delta={delta:.10f}"
        )


class TestPhase14ERegressionPrevention:
    """Prevent the Phase 14E abs(R) bug from reoccurring.

    The bug caused I12_plus and I12_minus to be computed at the same
    absolute value of R, making them incorrectly equal.
    """

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    @pytest.mark.parametrize("mode", [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED])
    def test_I12_plus_differs_from_I12_minus(self, benchmark, mode):
        """I12(+R) must differ from I12(-R) - regression test for abs(R) bug."""
        metrics = compute_delta_metrics_extended(benchmark, mode)

        # Compute relative difference
        avg = (abs(metrics.I12_plus) + abs(metrics.I12_minus)) / 2
        if avg < 1e-14:
            pytest.skip("Both I12 values near zero, cannot test relative difference")

        rel_diff = abs(metrics.I12_plus - metrics.I12_minus) / avg

        # Should differ by more than 1% (actually they differ by ~40-70%)
        assert rel_diff > 0.01, (
            f"I12_plus and I12_minus should differ for {benchmark} {mode.value}: "
            f"I12_plus={metrics.I12_plus:.6f}, I12_minus={metrics.I12_minus:.6f}, "
            f"rel_diff={rel_diff*100:.2f}%"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_I12_signs_differ_in_raw_mode(self, benchmark):
        """In raw_logderiv mode, I12_plus and I12_minus should have same sign but differ."""
        metrics = compute_delta_metrics_extended(benchmark, LaurentMode.RAW_LOGDERIV)

        # Both should be positive for raw_logderiv mode
        assert metrics.I12_plus > 0, f"I12_plus should be positive for {benchmark}"
        assert metrics.I12_minus > 0, f"I12_minus should be positive for {benchmark}"

        # And they should differ significantly
        ratio = metrics.I12_plus / metrics.I12_minus
        assert 0.3 < ratio < 0.9, (
            f"I12 ratio should be in (0.3, 0.9) for {benchmark}: "
            f"ratio={ratio:.4f}, I12_plus={metrics.I12_plus:.6f}, I12_minus={metrics.I12_minus:.6f}"
        )


class TestPhase14FStability:
    """Preserve the stability discovered in Phase 14F.

    Phase 14F showed that delta < 0.5 for both benchmarks in default mode.
    """

    @pytest.mark.parametrize("benchmark,max_delta", [
        ("kappa", 0.35),       # Phase 14G measured 0.253
        ("kappa_star", 0.15),  # Phase 14G measured 0.079
    ])
    def test_delta_below_threshold_raw_mode(self, benchmark, max_delta):
        """delta should stay below threshold in raw_logderiv mode."""
        metrics = compute_delta_metrics(benchmark, LaurentMode.RAW_LOGDERIV)

        assert metrics.delta < max_delta, (
            f"delta exceeds threshold for {benchmark}: "
            f"delta={metrics.delta:.4f}, max_delta={max_delta}"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_delta_below_half(self, benchmark):
        """delta should be less than 0.5 for any benchmark in default mode."""
        metrics = compute_delta_metrics(benchmark, LaurentMode.RAW_LOGDERIV)

        assert metrics.delta < 0.5, (
            f"delta should be < 0.5 for {benchmark}: delta={metrics.delta:.4f}"
        )

    def test_kappa_star_has_smaller_delta_than_kappa(self):
        """κ* should have smaller delta than κ (Phase 14F finding)."""
        metrics_k = compute_delta_metrics("kappa", LaurentMode.RAW_LOGDERIV)
        metrics_ks = compute_delta_metrics("kappa_star", LaurentMode.RAW_LOGDERIV)

        assert metrics_ks.delta < metrics_k.delta, (
            f"κ* should have smaller delta: "
            f"κ delta={metrics_k.delta:.4f}, κ* delta={metrics_ks.delta:.4f}"
        )


class TestPoleCancelledModeWorse:
    """Document that pole_cancelled mode makes results worse.

    Phase 14G discovery: pole_cancelled INCREASES delta, not decreases.
    Phase 14H semantic tests proved RAW_LOGDERIV is correct.
    """

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_pole_cancelled_has_larger_delta(self, benchmark):
        """pole_cancelled mode should have larger delta than raw_logderiv."""
        raw = compute_delta_metrics(benchmark, LaurentMode.RAW_LOGDERIV)
        pole = compute_delta_metrics(benchmark, LaurentMode.POLE_CANCELLED)

        assert pole.delta > raw.delta, (
            f"pole_cancelled should have larger delta for {benchmark}: "
            f"raw={raw.delta:.4f}, pole_cancelled={pole.delta:.4f}"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_pole_cancelled_increase_significant(self, benchmark):
        """pole_cancelled increases delta by more than 10%."""
        raw = compute_delta_metrics(benchmark, LaurentMode.RAW_LOGDERIV)
        pole = compute_delta_metrics(benchmark, LaurentMode.POLE_CANCELLED)

        increase = (pole.delta - raw.delta) / raw.delta

        # Phase 14G showed increases of 30% (κ) and 200% (κ*)
        assert increase > 0.10, (
            f"pole_cancelled increase should be >10% for {benchmark}: "
            f"increase={increase*100:.1f}%"
        )


class TestExtendedMetricsConsistency:
    """Test consistency between basic and extended metrics."""

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    @pytest.mark.parametrize("mode", [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED])
    def test_extended_matches_basic(self, benchmark, mode):
        """Extended metrics should have same core values as basic metrics."""
        basic = compute_delta_metrics(benchmark, mode)
        extended = compute_delta_metrics_extended(benchmark, mode)

        assert basic.benchmark == extended.benchmark
        assert basic.laurent_mode == extended.laurent_mode
        assert abs(basic.R - extended.R) < 1e-10
        assert abs(basic.A - extended.A) < 1e-10
        assert abs(basic.D - extended.D) < 1e-10
        assert abs(basic.delta - extended.delta) < 1e-10
        assert abs(basic.B_over_A - extended.B_over_A) < 1e-10

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_A_equals_I12_minus(self, benchmark):
        """A should equal I12_minus (mirror coefficient)."""
        metrics = compute_delta_metrics_extended(benchmark, LaurentMode.RAW_LOGDERIV)

        assert abs(metrics.A - metrics.I12_minus) < 1e-12, (
            f"A should equal I12_minus for {benchmark}: "
            f"A={metrics.A:.10f}, I12_minus={metrics.I12_minus:.10f}"
        )

    @pytest.mark.parametrize("benchmark", ["kappa", "kappa_star"])
    def test_D_equals_I12_plus_plus_I34_plus(self, benchmark):
        """D should equal I12_plus + I34_plus."""
        metrics = compute_delta_metrics_extended(benchmark, LaurentMode.RAW_LOGDERIV)

        expected_D = metrics.I12_plus + metrics.I34_plus
        actual_D = metrics.D

        assert abs(expected_D - actual_D) < 1e-12, (
            f"D should equal I12_plus + I34_plus for {benchmark}: "
            f"I12_plus={metrics.I12_plus:.10f}, I34_plus={metrics.I34_plus:.10f}, "
            f"sum={expected_D:.10f}, D={actual_D:.10f}"
        )


class TestBenchmarkRValues:
    """Verify correct R values are used for each benchmark."""

    def test_kappa_uses_correct_R(self):
        """κ benchmark should use R = 1.3036."""
        metrics = compute_delta_metrics("kappa", LaurentMode.RAW_LOGDERIV)
        assert abs(metrics.R - 1.3036) < 0.0001, f"κ R should be 1.3036, got {metrics.R}"

    def test_kappa_star_uses_correct_R(self):
        """κ* benchmark should use R = 1.1167."""
        metrics = compute_delta_metrics("kappa_star", LaurentMode.RAW_LOGDERIV)
        assert abs(metrics.R - 1.1167) < 0.0001, f"κ* R should be 1.1167, got {metrics.R}"

    def test_different_benchmarks_have_different_R(self):
        """The two benchmarks should have different R values."""
        metrics_k = compute_delta_metrics("kappa", LaurentMode.RAW_LOGDERIV)
        metrics_ks = compute_delta_metrics("kappa_star", LaurentMode.RAW_LOGDERIV)

        assert abs(metrics_k.R - metrics_ks.R) > 0.1, (
            f"Benchmarks should have different R: κ R={metrics_k.R}, κ* R={metrics_ks.R}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
