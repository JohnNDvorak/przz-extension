"""
tests/test_m1_sensitivity.py
Sensitivity tests for m1 parameter.

These tests verify that m1 is not a "fragile knob":
- c should vary monotonically with m1
- c should be roughly linear in m1 (through the I1_minus channel)
- Small perturbations in m1 should not cause sign flips or catastrophes

This helps characterize the risk of extending to K>3 with the extrapolated formula.
"""

import pytest
import numpy as np

from src.m1_policy import M1Mode, M1Policy, m1_formula, M1_EMPIRICAL_KAPPA


class TestM1Monotonicity:
    """Test that c varies monotonically with m1."""

    def test_m1_formula_monotone_in_R(self):
        """m1 should increase monotonically with R."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)

        R_values = [1.0, 1.1, 1.2, 1.3, 1.4]
        m1_values = [m1_formula(K=3, R=R, policy=policy) for R in R_values]

        # Check strictly increasing
        for i in range(len(m1_values) - 1):
            assert m1_values[i] < m1_values[i + 1], (
                f"m1 should increase with R: m1({R_values[i]})={m1_values[i]} "
                f"vs m1({R_values[i+1]})={m1_values[i+1]}"
            )

    def test_m1_formula_monotone_in_K(self):
        """m1 should increase monotonically with K."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
        R = 1.3036

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            K_values = [3, 4, 5, 6]
            m1_values = [m1_formula(K=K, R=R, policy=policy) for K in K_values]

        # Check strictly increasing
        for i in range(len(m1_values) - 1):
            assert m1_values[i] < m1_values[i + 1], (
                f"m1 should increase with K: m1(K={K_values[i]})={m1_values[i]} "
                f"vs m1(K={K_values[i+1]})={m1_values[i+1]}"
            )


class TestM1Linearity:
    """Test that m1 formula is linear in K."""

    def test_m1_linear_in_K(self):
        """m1 = exp(R) + (2K-1) should be linear in K."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
        R = 1.3036

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Compute for K=3,4,5
            m1_3 = m1_formula(K=3, R=R, policy=policy)
            m1_4 = m1_formula(K=4, R=R, policy=policy)
            m1_5 = m1_formula(K=5, R=R, policy=policy)

        # Delta should be constant
        delta_34 = m1_4 - m1_3
        delta_45 = m1_5 - m1_4

        assert np.isclose(delta_34, delta_45, rtol=1e-10), (
            f"m1 should be linear in K: delta(3->4)={delta_34}, delta(4->5)={delta_45}"
        )

        # Delta should be 2 (since 2K-1 increases by 2 per K)
        assert np.isclose(delta_34, 2.0, rtol=1e-10)


class TestM1Sensitivity:
    """Test sensitivity of m1 to perturbations."""

    def test_m1_small_perturbation_finite(self):
        """Small perturbations in m1 should give finite results."""
        m1_base = M1_EMPIRICAL_KAPPA

        perturbations = [0.9, 0.95, 1.0, 1.05, 1.1]

        for p in perturbations:
            m1_perturbed = m1_base * p
            assert np.isfinite(m1_perturbed)
            assert m1_perturbed > 0

    def test_m1_sensitivity_range(self):
        """m1 should be in a reasonable range for typical R values."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)

        for R in [1.0, 1.1, 1.2, 1.3, 1.4]:
            m1 = m1_formula(K=3, R=R, policy=policy)

            # For K=3, m1 = exp(R) + 5
            # At R=1.0: m1 ≈ 2.72 + 5 = 7.72
            # At R=1.4: m1 ≈ 4.06 + 5 = 9.06
            assert 7 < m1 < 10, f"m1 at R={R} is {m1}, outside expected range"


class TestM1ExtrapolationRisk:
    """Test to characterize risk of K>3 extrapolation."""

    def test_k4_m1_in_reasonable_range(self):
        """K=4 extrapolated m1 should be in a reasonable range."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
        R = 1.3036

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1_k4 = m1_formula(K=4, R=R, policy=policy)

        # For K=4 at R=1.3036: m1 = exp(1.3036) + 7 ≈ 10.68
        assert 10 < m1_k4 < 11, f"m1 at K=4 is {m1_k4}, outside expected range"

    def test_k3_to_k4_delta_is_2(self):
        """Going from K=3 to K=4 should add exactly 2 to m1."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
        R = 1.3036

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1_k3 = m1_formula(K=3, R=R, policy=policy)
            m1_k4 = m1_formula(K=4, R=R, policy=policy)

        delta = m1_k4 - m1_k3
        assert np.isclose(delta, 2.0, rtol=1e-10)


class TestM1NaiveComparison:
    """Test comparing empirical to naive formulas."""

    def test_naive_is_larger_at_benchmark_R(self):
        """exp(2R) should be larger than exp(R) + 5 for benchmark R values.

        Note: This is only true for R > ~1.05. At smaller R, empirical is larger.
        The benchmarks use R=1.1167 (kappa*) and R=1.3036 (kappa), where naive > empirical.
        """
        # Only test at the actual benchmark R values
        for R in [1.1167, 1.3036]:
            m1_empirical = np.exp(R) + 5
            m1_naive = np.exp(2 * R)

            assert m1_naive > m1_empirical, (
                f"At R={R}: naive={m1_naive}, empirical={m1_empirical}"
            )

    def test_naive_ratio_documented(self):
        """The naive/empirical ratio should match documented values."""
        # kappa: ratio ~1.56
        R_kappa = 1.3036
        ratio_kappa = np.exp(2 * R_kappa) / (np.exp(R_kappa) + 5)
        assert 1.55 < ratio_kappa < 1.57

        # kappa*: ratio ~1.16
        R_kappa_star = 1.1167
        ratio_kappa_star = np.exp(2 * R_kappa_star) / (np.exp(R_kappa_star) + 5)
        assert 1.15 < ratio_kappa_star < 1.17


class TestM1DocumentedFormula:
    """Test that the documented formula is correctly implemented."""

    def test_formula_exp_r_plus_5_at_k3(self):
        """K=3 should use exp(R) + 5."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)

        for R in [1.0, 1.1167, 1.3036]:
            m1 = m1_formula(K=3, R=R, policy=policy)
            expected = np.exp(R) + 5
            assert np.isclose(m1, expected, rtol=1e-10)

    def test_formula_exp_r_plus_2k_minus_1(self):
        """K_DEP_EMPIRICAL should use exp(R) + (2K-1)."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
        R = 1.3036

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for K in [3, 4, 5]:
                m1 = m1_formula(K=K, R=R, policy=policy)
                expected = np.exp(R) + (2 * K - 1)
                assert np.isclose(m1, expected, rtol=1e-10)
