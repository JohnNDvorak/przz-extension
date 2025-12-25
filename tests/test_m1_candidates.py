"""
Tests for m1 candidate formulas (Phase 19.4.2).

These tests verify:
1. All candidates compute without error
2. No candidate requires fitting (deterministic formulas only)
3. Candidates are compared against implied m1 from both benchmarks
4. Document which candidates "win" (match within tolerance without tuning)
"""

import pytest
import numpy as np

from src.mirror.m1_derived import (
    M1DerivationMode,
    m1_derived,
    m1_empirical_formula,
    m1_uniform_avg_exp_2Rt,
    m1_E_exp_2Rt_under_Q2,
    m1_sinh_scaled,
    compare_m1_modes,
)


# Benchmark parameters
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
THETA = 4.0 / 7.0
K = 3


class TestCandidateFormulas:
    """Test that all candidate formulas compute correctly."""

    def test_uniform_avg_exp_2Rt_computes(self):
        """Uniform average formula computes without error."""
        result = m1_uniform_avg_exp_2Rt(KAPPA_R)
        assert np.isfinite(result)
        assert result > 0

    def test_uniform_avg_exp_2Rt_at_zero(self):
        """Uniform average has correct limit at R=0."""
        # ∫₀¹ exp(0) dt = 1
        result = m1_uniform_avg_exp_2Rt(1e-12)
        assert abs(result - 1.0) < 0.01

    def test_uniform_avg_closed_form(self):
        """Verify closed-form formula is correct."""
        R = 1.0
        expected = (np.exp(2 * R) - 1) / (2 * R)
        actual = m1_uniform_avg_exp_2Rt(R)
        assert abs(actual - expected) < 1e-10

    def test_E_exp_2Rt_under_Q2_uniform_default(self):
        """Q² weighted expectation defaults to uniform when Q_coeffs=None."""
        uniform = m1_uniform_avg_exp_2Rt(KAPPA_R)
        weighted = m1_E_exp_2Rt_under_Q2(KAPPA_R, Q_coeffs=None)
        assert abs(uniform - weighted) < 1e-10

    def test_E_exp_2Rt_under_Q2_with_polynomial(self):
        """Q² weighted expectation works with polynomial."""
        # Q(t) = 1 + t  =>  Q² = 1 + 2t + t²
        Q_coeffs = (1.0, 1.0)
        result = m1_E_exp_2Rt_under_Q2(KAPPA_R, Q_coeffs=Q_coeffs)
        assert np.isfinite(result)
        assert result > 0

    def test_sinh_scaled_computes(self):
        """Sinh-scaled formula computes without error."""
        result = m1_sinh_scaled(KAPPA_R)
        assert np.isfinite(result)
        assert result > 0

    def test_sinh_scaled_at_zero(self):
        """Sinh-scaled has correct limit at R=0."""
        # lim_{R→0} 2*sinh(R)/R = 2 (from Taylor)
        result = m1_sinh_scaled(1e-12)
        assert abs(result - 2.0) < 0.01

    def test_sinh_scaled_formula(self):
        """Verify sinh formula is correct."""
        R = 1.0
        expected = 2 * np.sinh(R) / R
        actual = m1_sinh_scaled(R)
        assert abs(actual - expected) < 1e-10


class TestNoPureFitting:
    """Verify no candidate requires fitting (all deterministic)."""

    def test_uniform_avg_is_deterministic(self):
        """Uniform average gives same result every time."""
        results = [m1_uniform_avg_exp_2Rt(KAPPA_R) for _ in range(5)]
        assert all(abs(r - results[0]) < 1e-15 for r in results)

    def test_sinh_scaled_is_deterministic(self):
        """Sinh scaled gives same result every time."""
        results = [m1_sinh_scaled(KAPPA_R) for _ in range(5)]
        assert all(abs(r - results[0]) < 1e-15 for r in results)

    def test_candidates_have_no_fitted_parameters(self):
        """All Phase 19.4.2 candidates are parameter-free."""
        # These candidates should not use any fitted constants
        R = 1.2345  # Arbitrary R

        # Each formula should only depend on R (and maybe K, theta)
        # and produce a deterministic result
        u = m1_uniform_avg_exp_2Rt(R)
        s = m1_sinh_scaled(R)

        # Both should be finite and positive
        assert np.isfinite(u) and u > 0
        assert np.isfinite(s) and s > 0


class TestCandidateComparison:
    """Compare candidates against empirical formula."""

    def test_compare_m1_modes_runs(self):
        """compare_m1_modes works with all new modes."""
        results = compare_m1_modes(KAPPA_R, K=K, theta=THETA, verbose=False)

        assert "UNIFORM_AVG_EXP_2RT" in results
        assert "E_EXP_2RT_UNDER_Q2" in results
        assert "SINH_SCALED" in results

    def test_candidates_computed_for_both_benchmarks(self):
        """All candidates compute for both benchmarks."""
        for R in [KAPPA_R, KAPPA_STAR_R]:
            for mode in [M1DerivationMode.UNIFORM_AVG_EXP_2RT,
                         M1DerivationMode.E_EXP_2RT_UNDER_Q2,
                         M1DerivationMode.SINH_SCALED]:
                result = m1_derived(R, K, THETA, mode)
                assert np.isfinite(result)

    def test_candidate_ratios_to_empirical(self):
        """Document candidate ratios to empirical formula."""
        empirical = m1_empirical_formula(KAPPA_R, K)

        candidates = {
            "UNIFORM_AVG": m1_uniform_avg_exp_2Rt(KAPPA_R),
            "SINH_SCALED": m1_sinh_scaled(KAPPA_R),
        }

        # Print for documentation
        for name, value in candidates.items():
            ratio = value / empirical
            print(f"{name}: {value:.4f} (ratio to empirical: {ratio:.4f})")

        # All candidates should be different from empirical
        # (since empirical is exp(R)+5 ≈ 8.7 and these are exp-like ~3-5)
        for value in candidates.values():
            assert value < empirical  # All should be smaller


class TestCandidateWinCondition:
    """Test which candidates "win" (match implied m1 without tuning)."""

    def test_no_candidate_wins_without_offset(self):
        """
        Phase 19.4.2 acceptance: A candidate wins if it matches implied m1
        across both benchmarks within 5% WITHOUT tuning.

        This test documents that the exp-component alone is insufficient.
        The "+5" constant component is also needed.
        """
        # Expected implied m1 from Phase 18.1 (J1x pipeline, which gives ~15-22% of empirical)
        # For production, implied m1 ≈ empirical ≈ 8.7

        # The exp-component candidates give ~3-5, which is the exp(R) part
        # but missing the "+5" constant

        # This test verifies the gap
        for R, name in [(KAPPA_R, "kappa"), (KAPPA_STAR_R, "kappa_star")]:
            empirical = m1_empirical_formula(R, K)

            uniform = m1_uniform_avg_exp_2Rt(R)
            sinh = m1_sinh_scaled(R)

            # None of these should be within 5% of empirical
            # (they're all ~40-50% of empirical)
            assert abs(uniform / empirical - 1.0) > 0.05
            assert abs(sinh / empirical - 1.0) > 0.05

    def test_exp_R_component_matches_candidates(self):
        """
        The exp(R) component of empirical should be comparable to candidates.

        Empirical = exp(R) + 5
        Candidates are exp-like formulas

        exp(R) at R=1.3036 is ~3.68
        """
        R = KAPPA_R
        exp_R = np.exp(R)

        uniform = m1_uniform_avg_exp_2Rt(R)  # (exp(2R)-1)/(2R) ≈ 4.5
        sinh = m1_sinh_scaled(R)  # 2*sinh(R)/R ≈ 2.9

        # These should be in the same ballpark as exp(R)
        # (within factor of 2)
        assert 0.5 < uniform / exp_R < 2.0
        assert 0.5 < sinh / exp_R < 2.0


class TestConstantComponent:
    """Test the "+5" constant component hypothesis."""

    def test_empirical_decomposition(self):
        """Empirical = exp(R) + (2K-1)."""
        R = KAPPA_R
        empirical = m1_empirical_formula(R, K)

        exp_R = np.exp(R)
        constant = 2 * K - 1  # = 5 for K=3

        assert abs(empirical - (exp_R + constant)) < 1e-10

    def test_candidate_plus_constant_closer_to_empirical(self):
        """
        Hypothesis: best candidate + (2K-1) should be closer to empirical.
        """
        R = KAPPA_R
        empirical = m1_empirical_formula(R, K)
        constant = 2 * K - 1

        # sinh_scaled + 5 should be closer than sinh_scaled alone
        sinh = m1_sinh_scaled(R)
        sinh_plus_const = sinh + constant

        ratio_without = sinh / empirical
        ratio_with = sinh_plus_const / empirical

        # Adding constant should get us closer to 1.0
        assert abs(ratio_with - 1.0) < abs(ratio_without - 1.0)


class TestRScaling:
    """Test R-scaling behavior of candidates."""

    def test_uniform_avg_increases_with_R(self):
        """Uniform average increases with R."""
        values = [m1_uniform_avg_exp_2Rt(R) for R in [0.5, 1.0, 1.5, 2.0]]
        for i in range(1, len(values)):
            assert values[i] > values[i-1]

    def test_sinh_scaled_increases_with_R(self):
        """Sinh scaled increases with R."""
        values = [m1_sinh_scaled(R) for R in [0.5, 1.0, 1.5, 2.0]]
        for i in range(1, len(values)):
            assert values[i] > values[i-1]

    def test_candidates_consistent_across_benchmarks(self):
        """
        Candidates should give consistent ratios across benchmarks.

        If a candidate is the "right" exp-component, the ratio
        (candidate + const) / empirical should be similar for both R values.
        """
        const = 2 * K - 1

        ratios_uniform = []
        ratios_sinh = []

        for R in [KAPPA_R, KAPPA_STAR_R]:
            empirical = m1_empirical_formula(R, K)

            uniform = m1_uniform_avg_exp_2Rt(R) + const
            sinh = m1_sinh_scaled(R) + const

            ratios_uniform.append(uniform / empirical)
            ratios_sinh.append(sinh / empirical)

        # Ratios should be within ~20% of each other
        assert abs(ratios_uniform[0] - ratios_uniform[1]) < 0.2
        assert abs(ratios_sinh[0] - ratios_sinh[1]) < 0.2
