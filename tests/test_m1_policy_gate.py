"""
tests/test_m1_policy_gate.py
Gate tests for M1Policy - HARD SAFETY LOCK for K>3 Extension.

These tests verify that:
1. K=3 works correctly with K3_EMPIRICAL and K_DEP_EMPIRICAL
2. K>3 RAISES unless explicitly enabled
3. The empirical values match expected benchmarks
4. Override mode works correctly

See: src/m1_policy.py for implementation details.
"""

import pytest
import numpy as np
import warnings

from src.m1_policy import (
    M1Mode,
    M1Policy,
    M1ExtrapolationError,
    m1_formula,
    get_m1_reference_values,
    M1_EMPIRICAL_KAPPA,
    M1_EMPIRICAL_KAPPA_STAR,
)


class TestM1ModeEnum:
    """Test M1Mode enum definition."""

    def test_all_modes_defined(self):
        """All expected modes should be defined."""
        assert hasattr(M1Mode, "K3_EMPIRICAL")
        assert hasattr(M1Mode, "K_DEP_EMPIRICAL")
        assert hasattr(M1Mode, "PAPER_NAIVE")
        assert hasattr(M1Mode, "OVERRIDE")

    def test_modes_are_distinct(self):
        """Each mode should have a unique value."""
        modes = [M1Mode.K3_EMPIRICAL, M1Mode.K_DEP_EMPIRICAL,
                 M1Mode.PAPER_NAIVE, M1Mode.OVERRIDE]
        assert len(modes) == len(set(modes))


class TestM1PolicyDataclass:
    """Test M1Policy dataclass."""

    def test_default_no_extrapolation(self):
        """Default should be allow_extrapolation=False."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        assert policy.allow_extrapolation is False

    def test_default_no_override_value(self):
        """Default should be override_value=None."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        assert policy.override_value is None

    def test_frozen(self):
        """M1Policy should be frozen (immutable)."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        with pytest.raises(Exception):  # FrozenInstanceError
            policy.mode = M1Mode.OVERRIDE


class TestK3Empirical:
    """Test M1Mode.K3_EMPIRICAL mode."""

    def test_k3_works(self):
        """K=3 should work with K3_EMPIRICAL."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)
        assert np.isfinite(m1)
        assert np.isclose(m1, np.exp(1.3036) + 5, rtol=1e-10)

    def test_k4_raises(self):
        """K=4 should RAISE with K3_EMPIRICAL (not extrapolatable)."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        with pytest.raises(ValueError) as exc_info:
            m1_formula(K=4, R=1.3036, policy=policy)

        assert "only valid for K=3" in str(exc_info.value)

    def test_k2_raises(self):
        """K=2 should also RAISE (formula is for K=3 only)."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        with pytest.raises(ValueError):
            m1_formula(K=2, R=1.3036, policy=policy)

    def test_allow_extrapolation_ignored(self):
        """allow_extrapolation should not help K3_EMPIRICAL."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL, allow_extrapolation=True)
        # K!=3 should still raise
        with pytest.raises(ValueError):
            m1_formula(K=4, R=1.3036, policy=policy)


class TestKDepEmpirical:
    """Test M1Mode.K_DEP_EMPIRICAL mode."""

    def test_k3_works(self):
        """K=3 should work without opt-in."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)
        # Should match K3_EMPIRICAL result
        assert np.isclose(m1, np.exp(1.3036) + 5, rtol=1e-10)

    def test_k4_raises_without_optin(self):
        """K=4 should RAISE without allow_extrapolation=True."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)
        with pytest.raises(M1ExtrapolationError) as exc_info:
            m1_formula(K=4, R=1.3036, policy=policy)

        error_msg = str(exc_info.value)
        assert "allow_extrapolation=True" in error_msg
        assert "UNVALIDATED" in error_msg

    def test_k4_works_with_optin(self):
        """K=4 should work with allow_extrapolation=True."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)

        # Should work but emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m1 = m1_formula(K=4, R=1.3036, policy=policy)

            # Should have warning
            assert len(w) == 1
            assert "EXTRAPOLATED" in str(w[0].message)
            assert "UNVALIDATED" in str(w[0].message)

        # Result should be exp(R) + 7
        assert np.isclose(m1, np.exp(1.3036) + 7, rtol=1e-10)

    def test_k5_works_with_optin(self):
        """K=5 should work with allow_extrapolation=True."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m1 = m1_formula(K=5, R=1.3036, policy=policy)

        # Result should be exp(R) + 9
        assert np.isclose(m1, np.exp(1.3036) + 9, rtol=1e-10)

    def test_k3_no_warning(self):
        """K=3 should not emit warning even with allow_extrapolation=True."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m1_formula(K=3, R=1.3036, policy=policy)

            # Should have NO warning for K=3
            assert len(w) == 0


class TestPaperNaive:
    """Test M1Mode.PAPER_NAIVE mode."""

    def test_k3_works(self):
        """K=3 should work without opt-in."""
        policy = M1Policy(mode=M1Mode.PAPER_NAIVE)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)
        # Result should be exp(2R)
        assert np.isclose(m1, np.exp(2 * 1.3036), rtol=1e-10)

    def test_naive_is_larger_than_empirical(self):
        """PAPER_NAIVE should give larger values than empirical."""
        policy_naive = M1Policy(mode=M1Mode.PAPER_NAIVE)
        policy_emp = M1Policy(mode=M1Mode.K3_EMPIRICAL)

        m1_naive = m1_formula(K=3, R=1.3036, policy=policy_naive)
        m1_emp = m1_formula(K=3, R=1.3036, policy=policy_emp)

        # Naive should be ~1.56x empirical
        ratio = m1_naive / m1_emp
        assert 1.5 < ratio < 1.6

    def test_k4_raises_without_optin(self):
        """K=4 should RAISE without allow_extrapolation=True."""
        policy = M1Policy(mode=M1Mode.PAPER_NAIVE)
        with pytest.raises(M1ExtrapolationError):
            m1_formula(K=4, R=1.3036, policy=policy)


class TestOverride:
    """Test M1Mode.OVERRIDE mode."""

    def test_override_works(self):
        """Override should use the specified value."""
        policy = M1Policy(mode=M1Mode.OVERRIDE, override_value=42.0)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)
        assert m1 == 42.0

    def test_override_requires_value(self):
        """Override without value should RAISE."""
        policy = M1Policy(mode=M1Mode.OVERRIDE)
        with pytest.raises(ValueError) as exc_info:
            m1_formula(K=3, R=1.3036, policy=policy)

        assert "override_value" in str(exc_info.value)

    def test_override_any_k(self):
        """Override should work for any K (no validation)."""
        policy = M1Policy(mode=M1Mode.OVERRIDE, override_value=10.0)

        m1_k3 = m1_formula(K=3, R=1.3036, policy=policy)
        m1_k4 = m1_formula(K=4, R=1.3036, policy=policy)
        m1_k10 = m1_formula(K=10, R=1.3036, policy=policy)

        assert m1_k3 == 10.0
        assert m1_k4 == 10.0
        assert m1_k10 == 10.0


class TestBenchmarkValidation:
    """Test that m1 matches expected benchmark values."""

    def test_kappa_benchmark(self):
        """K3_EMPIRICAL at R=1.3036 should match kappa benchmark."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)

        expected = M1_EMPIRICAL_KAPPA
        assert np.isclose(m1, expected, rtol=1e-10)
        assert np.isclose(m1, 8.683, rtol=0.001)  # Approximate value

    def test_kappa_star_benchmark(self):
        """K3_EMPIRICAL at R=1.1167 should match kappa* benchmark."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.1167, policy=policy)

        expected = M1_EMPIRICAL_KAPPA_STAR
        assert np.isclose(m1, expected, rtol=1e-10)
        assert np.isclose(m1, 8.055, rtol=0.001)  # Approximate value

    def test_k_dep_matches_k3_at_k3(self):
        """K_DEP_EMPIRICAL should match K3_EMPIRICAL at K=3."""
        policy_k3 = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        policy_dep = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)

        for R in [1.3036, 1.1167]:
            m1_k3 = m1_formula(K=3, R=R, policy=policy_k3)
            m1_dep = m1_formula(K=3, R=R, policy=policy_dep)
            assert np.isclose(m1_k3, m1_dep, rtol=1e-10)


class TestReferenceValues:
    """Test reference value functions."""

    def test_get_m1_reference_values(self):
        """get_m1_reference_values should return expected structure."""
        refs = get_m1_reference_values()

        assert "empirical" in refs
        assert "naive" in refs
        assert "naive_to_empirical_ratio" in refs

        assert np.isclose(refs["empirical"]["kappa"], M1_EMPIRICAL_KAPPA)
        assert np.isclose(refs["empirical"]["kappa_star"], M1_EMPIRICAL_KAPPA_STAR)

    def test_naive_to_empirical_ratio(self):
        """Naive/empirical ratio should be documented correctly."""
        refs = get_m1_reference_values()

        # kappa: naive is ~1.56x empirical
        assert 1.5 < refs["naive_to_empirical_ratio"]["kappa"] < 1.6

        # kappa*: naive is ~1.16x empirical
        assert 1.1 < refs["naive_to_empirical_ratio"]["kappa_star"] < 1.2


class TestM1ExtrapolationError:
    """Test M1ExtrapolationError exception."""

    def test_is_value_error_subclass(self):
        """M1ExtrapolationError should be a ValueError subclass."""
        assert issubclass(M1ExtrapolationError, ValueError)

    def test_can_be_caught(self):
        """M1ExtrapolationError should be catchable."""
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)

        try:
            m1_formula(K=4, R=1.3036, policy=policy)
            assert False, "Should have raised"
        except M1ExtrapolationError as e:
            assert "K=4" in str(e)
