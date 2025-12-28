"""
tests/test_mirror_formula_locked.py
Phase 36: Locked Formula Verification Tests

These tests verify that the derived mirror multiplier formula is:
1. K-generalized correctly (θ/42 for K=3, θ/72 for K=4, etc.)
2. The production default
3. Consistent across K values

FORMULA (Phase 36 locked):
    m(K, R) = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]

Created: 2025-12-26 (Phase 36)
"""
import math
import pytest
import warnings

from src.evaluator.decomposition import compute_mirror_multiplier


class TestMirrorFormulaKDependence:
    """Test that K-dependence is wired correctly."""

    THETA = 4 / 7
    R = 1.3036  # PRZZ κ benchmark

    def test_k3_correction_is_theta_over_42(self):
        """K=3 should use θ/(2×3×7) = θ/42."""
        K = 3
        denom = 2 * K * (2 * K + 1)  # 2×3×7 = 42
        expected_correction = 1 + self.THETA / denom

        m, desc = compute_mirror_multiplier(self.R, K, formula="derived")

        # Extract the correction factor from m
        base = math.exp(self.R) + (2 * K - 1)
        actual_correction = m / base

        assert abs(actual_correction - expected_correction) < 1e-10, (
            f"K=3 correction should be {expected_correction:.8f}, got {actual_correction:.8f}"
        )
        assert abs(denom - 42) < 1e-10, f"K=3 denominator should be 42, got {denom}"

    def test_k4_correction_is_theta_over_72(self):
        """K=4 should use θ/(2×4×9) = θ/72."""
        K = 4
        denom = 2 * K * (2 * K + 1)  # 2×4×9 = 72
        expected_correction = 1 + self.THETA / denom

        m, desc = compute_mirror_multiplier(self.R, K, formula="derived")

        base = math.exp(self.R) + (2 * K - 1)
        actual_correction = m / base

        assert abs(actual_correction - expected_correction) < 1e-10, (
            f"K=4 correction should be {expected_correction:.8f}, got {actual_correction:.8f}"
        )
        assert abs(denom - 72) < 1e-10, f"K=4 denominator should be 72, got {denom}"

    def test_k5_correction_is_theta_over_110(self):
        """K=5 should use θ/(2×5×11) = θ/110."""
        K = 5
        denom = 2 * K * (2 * K + 1)  # 2×5×11 = 110
        expected_correction = 1 + self.THETA / denom

        m, desc = compute_mirror_multiplier(self.R, K, formula="derived")

        base = math.exp(self.R) + (2 * K - 1)
        actual_correction = m / base

        assert abs(actual_correction - expected_correction) < 1e-10, (
            f"K=5 correction should be {expected_correction:.8f}, got {actual_correction:.8f}"
        )
        assert abs(denom - 110) < 1e-10, f"K=5 denominator should be 110, got {denom}"

    def test_correction_shrinks_with_k(self):
        """The Beta correction should shrink as K increases."""
        corrections = []
        for K in [3, 4, 5, 6]:
            m, _ = compute_mirror_multiplier(self.R, K, formula="derived")
            base = math.exp(self.R) + (2 * K - 1)
            corrections.append((K, m / base))

        # Verify monotonically decreasing
        for i in range(len(corrections) - 1):
            K_curr, corr_curr = corrections[i]
            K_next, corr_next = corrections[i + 1]
            assert corr_curr > corr_next, (
                f"Correction should shrink: K={K_curr} has {corr_curr:.6f}, "
                f"K={K_next} has {corr_next:.6f}"
            )

    def test_base_term_is_2k_minus_1(self):
        """The base term should be exp(R) + (2K-1)."""
        for K in [3, 4, 5]:
            expected_base = math.exp(self.R) + (2 * K - 1)

            # Compute with empirical (which is just the base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_emp, _ = compute_mirror_multiplier(self.R, K, formula="empirical")

            assert abs(m_emp - expected_base) < 1e-10, (
                f"K={K}: empirical m should be exp(R)+{2*K-1}={expected_base:.6f}, "
                f"got {m_emp:.6f}"
            )


class TestMirrorFormulaDefault:
    """Test that derived formula is the production default."""

    def test_derived_is_default(self):
        """compute_mirror_multiplier should default to 'derived'."""
        R, K = 1.3036, 3

        # Call without specifying formula
        m_default, desc_default = compute_mirror_multiplier(R, K)

        # Call with explicit 'derived'
        m_derived, desc_derived = compute_mirror_multiplier(R, K, formula="derived")

        assert abs(m_default - m_derived) < 1e-15, (
            "Default formula should be 'derived'"
        )
        assert "[1 + θ/(2K(2K+1))]" in desc_default, (
            "Default description should indicate derived formula"
        )


class TestMirrorFormulaWarnings:
    """Test that non-production formulas emit warnings."""

    def test_empirical_warns(self):
        """Using 'empirical' should warn."""
        with pytest.warns(UserWarning, match="Use 'derived' for production"):
            compute_mirror_multiplier(1.3036, 3, formula="empirical")

    def test_derived_full_warns(self):
        """Using 'derived_full' should warn about deprecation."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            compute_mirror_multiplier(1.3036, 3, formula="derived_full")

    def test_derived_does_not_warn(self):
        """Using 'derived' should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            compute_mirror_multiplier(1.3036, 3, formula="derived")


class TestMirrorFormulaGeneral:
    """General formula tests."""

    def test_formula_structure_k_general(self):
        """Test the general formula structure for arbitrary K."""
        R = 1.2
        theta = 4 / 7

        for K in range(2, 8):
            m, desc = compute_mirror_multiplier(R, K, formula="derived", theta=theta)

            # Verify formula: m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
            denom = 2 * K * (2 * K + 1)
            expected_correction = 1 + theta / denom
            expected_base = math.exp(R) + (2 * K - 1)
            expected_m = expected_correction * expected_base

            assert abs(m - expected_m) < 1e-10, (
                f"K={K}: m should be {expected_m:.8f}, got {m:.8f}"
            )

    def test_theta_parameter_is_used(self):
        """Test that the theta parameter is actually used."""
        R, K = 1.3036, 3

        m_default, _ = compute_mirror_multiplier(R, K, formula="derived")
        m_custom, _ = compute_mirror_multiplier(R, K, formula="derived", theta=0.5)

        # They should differ since theta is different
        assert m_default != m_custom, "Different theta should give different m"

        # Verify the custom theta is applied correctly
        expected_correction = 1 + 0.5 / 42
        expected_m = expected_correction * (math.exp(R) + 5)
        assert abs(m_custom - expected_m) < 1e-10
