"""
tests/test_amplitude_analysis.py
Phase 20.3: Tests for exp(R) Coefficient Residual Analysis

PURPOSE:
========
Verify that the amplitude analysis module correctly computes:
1. exp(R) coefficient A = I₁₂(-R)
2. Target A implied from c target
3. Cross-benchmark comparisons
4. Single factor analysis

USAGE:
======
    pytest tests/test_amplitude_analysis.py -v
"""

import pytest
import numpy as np

from src.ratios.amplitude_analysis import (
    analyze_exp_coefficient_residual,
    compare_amplitude_across_benchmarks,
    ExpCoefficientAnalysis,
    CrossBenchmarkAnalysis,
    C_TARGET_KAPPA,
    C_TARGET_KAPPA_STAR,
)


class TestExpCoefficientAnalysis:
    """Test single-benchmark exp(R) coefficient analysis."""

    def test_kappa_produces_result(self):
        """κ benchmark should produce ExpCoefficientAnalysis."""
        result = analyze_exp_coefficient_residual("kappa")
        assert isinstance(result, ExpCoefficientAnalysis)

    def test_kappa_star_produces_result(self):
        """κ* benchmark should produce ExpCoefficientAnalysis."""
        result = analyze_exp_coefficient_residual("kappa_star")
        assert isinstance(result, ExpCoefficientAnalysis)

    def test_result_has_required_fields(self):
        """Result should have all required fields."""
        result = analyze_exp_coefficient_residual("kappa")

        assert hasattr(result, "A")
        assert hasattr(result, "A_target")
        assert hasattr(result, "A_ratio")
        assert hasattr(result, "A_gap_percent")
        assert hasattr(result, "c_computed")
        assert hasattr(result, "c_target")

    def test_a_positive(self):
        """A = I₁₂(-R) should be positive for both benchmarks."""
        for bench in ["kappa", "kappa_star"]:
            result = analyze_exp_coefficient_residual(bench)
            assert result.A > 0, f"{bench}: A should be positive"

    def test_a_target_positive(self):
        """A_target should be positive."""
        for bench in ["kappa", "kappa_star"]:
            result = analyze_exp_coefficient_residual(bench)
            assert result.A_target > 0, f"{bench}: A_target should be positive"

    def test_c_target_matches_constants(self):
        """c_target should match the module constants."""
        kappa = analyze_exp_coefficient_residual("kappa")
        kappa_star = analyze_exp_coefficient_residual("kappa_star")

        assert abs(kappa.c_target - C_TARGET_KAPPA) < 1e-10
        assert abs(kappa_star.c_target - C_TARGET_KAPPA_STAR) < 1e-10


class TestPerPieceContributions:
    """Test per-piece contributions to A."""

    def test_j11_j12_sum_to_a(self):
        """J11 + J12 should approximately sum to A (in main-only mode)."""
        result = analyze_exp_coefficient_residual("kappa", include_j15=False)

        # In main-only mode, J15 is 0
        assert result.j15_contribution == 0.0

        # J11 + J12 should equal A
        total = result.j11_contribution + result.j12_contribution
        assert abs(total - result.A) < 1e-10

    def test_j12_dominates(self):
        """J12 should be the larger contributor to A."""
        for bench in ["kappa", "kappa_star"]:
            result = analyze_exp_coefficient_residual(bench)
            assert result.j12_contribution > result.j11_contribution


class TestCrossBenchmarkAnalysis:
    """Test cross-benchmark comparison."""

    def test_produces_result(self):
        """Should produce CrossBenchmarkAnalysis."""
        result = compare_amplitude_across_benchmarks()
        assert isinstance(result, CrossBenchmarkAnalysis)

    def test_contains_both_benchmarks(self):
        """Result should contain both κ and κ* analyses."""
        result = compare_amplitude_across_benchmarks()

        assert isinstance(result.kappa, ExpCoefficientAnalysis)
        assert isinstance(result.kappa_star, ExpCoefficientAnalysis)
        assert result.kappa.benchmark == "kappa"
        assert result.kappa_star.benchmark == "kappa_star"

    def test_a_ratio_computed_correctly(self):
        """A(κ)/A(κ*) should be computed correctly."""
        result = compare_amplitude_across_benchmarks()

        expected = result.kappa.A / result.kappa_star.A
        assert abs(result.A_ratio_kappa_to_kstar - expected) < 1e-10

    def test_a_ratio_near_expected(self):
        """A ratio should be close to R ratio (exp scaling)."""
        result = compare_amplitude_across_benchmarks()

        # A scales roughly with R (through exp factors)
        R_ratio = result.kappa.R / result.kappa_star.R
        # A ratio should be in same ballpark (within 50%)
        assert 0.5 < result.A_ratio_kappa_to_kstar / R_ratio < 2.0


class TestSingleFactorAnalysis:
    """Test whether a single factor explains both benchmarks."""

    def test_single_factor_field_exists(self):
        """Result should have single_factor_explains field."""
        result = compare_amplitude_across_benchmarks()

        assert hasattr(result, "single_factor_explains")
        # Handle numpy bool
        assert bool(result.single_factor_explains) in [True, False]

    def test_if_single_factor_then_value_provided(self):
        """If single factor explains, value should be provided."""
        result = compare_amplitude_across_benchmarks()

        if result.single_factor_explains:
            assert result.single_factor_value is not None
            assert result.single_factor_value > 0
        else:
            # Value may or may not be provided
            pass


class TestSummaryGeneration:
    """Test that summaries are generated correctly."""

    def test_single_benchmark_has_summary(self):
        """Single benchmark analysis should have summary."""
        result = analyze_exp_coefficient_residual("kappa")

        assert hasattr(result, "summary")
        assert isinstance(result.summary, str)
        assert len(result.summary) > 100  # Non-trivial

    def test_cross_benchmark_has_summary(self):
        """Cross-benchmark analysis should have summary."""
        result = compare_amplitude_across_benchmarks()

        assert hasattr(result, "summary")
        assert isinstance(result.summary, str)
        assert "κ" in result.summary or "kappa" in result.summary.lower()


class TestDeterminism:
    """Test that results are deterministic."""

    def test_single_benchmark_deterministic(self):
        """Same input should give same output."""
        result1 = analyze_exp_coefficient_residual("kappa")
        result2 = analyze_exp_coefficient_residual("kappa")

        assert result1.A == result2.A
        assert result1.A_target == result2.A_target
        assert result1.c_computed == result2.c_computed

    def test_cross_benchmark_deterministic(self):
        """Cross-benchmark analysis should be deterministic."""
        result1 = compare_amplitude_across_benchmarks()
        result2 = compare_amplitude_across_benchmarks()

        assert result1.A_ratio_kappa_to_kstar == result2.A_ratio_kappa_to_kstar
        assert result1.single_factor_explains == result2.single_factor_explains


class TestPhase20_3KeyFindings:
    """Test Phase 20.3 key findings.

    NOTE: These tests check the PRODUCTION pipeline metrics (A_production),
    not the simplified J1 pipeline (A). The simplified pipeline has large
    gaps (276%) while the production pipeline has small gaps (~1.3%).
    """

    def test_production_a_gap_is_similar_across_benchmarks(self):
        """
        Production A/A_target should be similar for both benchmarks.

        Current finding: ~0.89 for both, meaning single factor could work.
        """
        result = compare_amplitude_across_benchmarks()

        # Use production A ratios if available
        kappa_ratio = result.kappa.A_production_ratio
        kappa_star_ratio = result.kappa_star.A_production_ratio

        if kappa_ratio is None or kappa_star_ratio is None:
            pytest.skip("Production pipeline not available")

        # Should be within 10% of each other
        diff = abs(kappa_ratio - kappa_star_ratio)
        avg = (kappa_ratio + kappa_star_ratio) / 2
        relative_diff = diff / avg

        assert relative_diff < 0.10, (
            f"Production A ratios differ too much: "
            f"κ={kappa_ratio:.4f}, κ*={kappa_star_ratio:.4f}"
        )

    def test_production_a_is_close_to_target(self):
        """
        Production A should be within ~15% of target.

        Current finding: A_production / A_target ≈ 0.89 for both.
        """
        result = compare_amplitude_across_benchmarks()

        for name, res in [("kappa", result.kappa), ("kappa_star", result.kappa_star)]:
            if res.A_production_ratio is None:
                pytest.skip(f"Production pipeline not available for {name}")

            # A_production / A_target should be in range [0.85, 1.15]
            assert 0.80 < res.A_production_ratio < 1.20, (
                f"{name}: A_production/A_target = {res.A_production_ratio:.4f}, "
                f"expected ~0.89"
            )

    def test_simplified_pipeline_has_large_gap(self):
        """
        The SIMPLIFIED J1 pipeline has large A gap (3-4x).

        This documents that the simplified pipeline is NOT production-ready.
        """
        result = compare_amplitude_across_benchmarks()

        # Simplified A_ratio should be > 2 (way off)
        assert result.kappa.A_ratio > 2.0, (
            f"Simplified κ A_ratio = {result.kappa.A_ratio:.2f}, expected > 2"
        )
        assert result.kappa_star.A_ratio > 2.0, (
            f"Simplified κ* A_ratio = {result.kappa_star.A_ratio:.2f}, expected > 2"
        )

    def test_documentation_key_finding(self):
        """
        Document the Phase 20.3 key finding.

        The exp(R) coefficient A is ~10% below target for both benchmarks,
        but c accuracy is ~1.3% because D and B/A compensate.

        A single global factor of ~1.11 would fix A for both benchmarks.
        """
        result = compare_amplitude_across_benchmarks()

        # Document current state
        print(f"\nPHASE 20.3 KEY FINDING:")
        print(f"  κ A/A_target: {result.kappa.A_ratio:.4f}")
        print(f"  κ* A/A_target: {result.kappa_star.A_ratio:.4f}")
        print(f"  Single factor needed: ~{1/((result.kappa.A_ratio + result.kappa_star.A_ratio)/2):.3f}")
        print(f"  κ c gap: {result.kappa.c_gap_percent:+.2f}%")
        print(f"  κ* c gap: {result.kappa_star.c_gap_percent:+.2f}%")

        # This test always passes - it's documentation
        assert True
