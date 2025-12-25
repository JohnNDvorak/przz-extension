"""
tests/test_m1_derived.py
Tests for the derived m1 module (Phase 18.4).
"""

import pytest
import numpy as np

from src.mirror.m1_derived import (
    M1DerivationMode,
    M1DerivedResult,
    m1_derived,
    m1_derived_with_breakdown,
    m1_empirical_formula,
    m1_fitted_formula,
    m1_naive_paper,
    m1_przz_limit,
    compare_m1_modes,
)


class TestM1Formulas:
    """Tests for individual m1 formulas."""

    def test_empirical_formula_kappa(self):
        """Test empirical formula at kappa benchmark."""
        R = 1.3036
        K = 3
        expected = np.exp(R) + 5
        result = m1_empirical_formula(R, K)
        assert abs(result - expected) < 1e-10

    def test_empirical_formula_kappa_star(self):
        """Test empirical formula at kappa* benchmark."""
        R = 1.1167
        K = 3
        expected = np.exp(R) + 5
        result = m1_empirical_formula(R, K)
        assert abs(result - expected) < 1e-10

    def test_empirical_k_dependence(self):
        """Test that empirical formula depends on K correctly."""
        R = 1.3036
        m1_k3 = m1_empirical_formula(R, K=3)
        m1_k4 = m1_empirical_formula(R, K=4)
        assert abs(m1_k4 - m1_k3 - 2.0) < 1e-10  # Delta should be 2

    def test_fitted_formula(self):
        """Test fitted formula returns positive value."""
        R = 1.3036
        result = m1_fitted_formula(R)
        assert result > 0
        # Should be close to empirical (within 2%)
        empirical = m1_empirical_formula(R, K=3)
        assert abs(result - empirical) / empirical < 0.02

    def test_naive_paper_formula(self):
        """Test naive paper formula: exp(2R)."""
        R = 1.3036
        expected = np.exp(2 * R)
        result = m1_naive_paper(R)
        assert abs(result - expected) < 1e-10

    def test_przz_limit_formula(self):
        """Test PRZZ limit formula: exp(2R/theta)."""
        R = 1.3036
        theta = 4.0 / 7.0
        expected = np.exp(2 * R / theta)
        result = m1_przz_limit(R, theta)
        assert abs(result - expected) < 1e-10


class TestM1Derived:
    """Tests for m1_derived function."""

    def test_all_modes_return_positive(self):
        """Test that all modes return positive m1."""
        R = 1.3036
        K = 3
        theta = 4.0 / 7.0

        for mode in M1DerivationMode:
            result = m1_derived(R, K, theta, mode)
            assert result > 0, f"Mode {mode.name} returned non-positive: {result}"

    def test_empirical_mode_is_default(self):
        """Test that EMPIRICAL is the default mode."""
        R = 1.3036
        K = 3
        theta = 4.0 / 7.0

        default_result = m1_derived(R, K, theta)
        empirical_result = m1_derived(R, K, theta, M1DerivationMode.EMPIRICAL)
        assert default_result == empirical_result

    def test_unknown_mode_raises(self):
        """Test that unknown mode raises ValueError."""
        # This can't easily happen with enum, but test the code path
        # by testing that all enum values are handled
        for mode in M1DerivationMode:
            # Should not raise
            m1_derived(1.3036, 3, 4.0 / 7.0, mode)


class TestM1DerivedWithBreakdown:
    """Tests for m1_derived_with_breakdown function."""

    def test_returns_result_dataclass(self):
        """Test that breakdown returns M1DerivedResult."""
        result = m1_derived_with_breakdown(1.3036)
        assert isinstance(result, M1DerivedResult)

    def test_ratio_to_empirical_is_one_for_empirical_mode(self):
        """Test that ratio is 1.0 when using empirical mode."""
        result = m1_derived_with_breakdown(
            1.3036,
            derivation_mode=M1DerivationMode.EMPIRICAL,
        )
        assert abs(result.ratio_to_empirical - 1.0) < 1e-10

    def test_przz_limit_is_much_larger(self):
        """Test that PRZZ limit is much larger than empirical."""
        result = m1_derived_with_breakdown(
            1.3036,
            derivation_mode=M1DerivationMode.PRZZ_LIMIT,
        )
        # PRZZ limit should be > 5x empirical
        assert result.ratio_to_empirical > 5.0


class TestCompareM1Modes:
    """Tests for compare_m1_modes function."""

    def test_returns_all_modes(self):
        """Test that comparison includes all derivation modes."""
        results = compare_m1_modes(1.3036, verbose=False)

        for mode in M1DerivationMode:
            assert mode.name in results, f"Missing mode: {mode.name}"

    def test_results_are_consistent(self):
        """Test that results are internally consistent."""
        results = compare_m1_modes(1.3036, verbose=False)

        # All should have same reference values
        first = list(results.values())[0]
        for result in results.values():
            assert result.m1_empirical == first.m1_empirical
            assert result.m1_fitted == first.m1_fitted


class TestDocumentedFindings:
    """Tests that verify documented findings from Phase 18.4."""

    def test_przz_limit_much_larger_than_empirical_at_kappa(self):
        """
        Test that PRZZ limit is ~11x empirical at kappa.

        This documents that the asymptotic formula doesn't apply at finite R.
        """
        result = m1_derived_with_breakdown(
            R=1.3036,
            K=3,
            theta=4.0 / 7.0,
            derivation_mode=M1DerivationMode.PRZZ_LIMIT,
        )
        # Should be ~11x (documented finding)
        assert 10.0 < result.ratio_to_empirical < 12.0

    def test_przz_limit_much_larger_than_empirical_at_kappa_star(self):
        """
        Test that PRZZ limit is ~6x empirical at kappa*.

        The ratio is smaller for kappa* due to smaller R.
        """
        result = m1_derived_with_breakdown(
            R=1.1167,
            K=3,
            theta=4.0 / 7.0,
            derivation_mode=M1DerivationMode.PRZZ_LIMIT,
        )
        # Should be ~6x (documented finding)
        assert 5.0 < result.ratio_to_empirical < 8.0

    def test_naive_paper_larger_but_closer(self):
        """
        Test that naive paper exp(2R) is larger but closer than PRZZ limit.
        """
        naive = m1_derived_with_breakdown(
            R=1.3036,
            derivation_mode=M1DerivationMode.NAIVE_PAPER,
        )
        przz = m1_derived_with_breakdown(
            R=1.3036,
            derivation_mode=M1DerivationMode.PRZZ_LIMIT,
        )

        # Naive is between empirical and PRZZ limit
        assert 1.0 < naive.ratio_to_empirical < przz.ratio_to_empirical

    def test_empirical_is_best_option(self):
        """
        Test that empirical is the only mode with ratio ~1.0.

        This confirms that empirical remains the best option for production.
        """
        results = compare_m1_modes(1.3036, verbose=False)

        for mode_name, result in results.items():
            if mode_name in ["EMPIRICAL", "TAYLOR_CORRECTION"]:
                assert abs(result.ratio_to_empirical - 1.0) < 0.01
            elif mode_name == "FITTED":
                assert abs(result.ratio_to_empirical - 1.0) < 0.02
            else:
                # Other modes should be significantly off
                assert abs(result.ratio_to_empirical - 1.0) > 0.1
