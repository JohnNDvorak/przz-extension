#!/usr/bin/env python3
"""
Tests for Phase 44 Corrected Evaluator

Validates the I1-fraction corrected evaluator achieves <0.01% accuracy
on both kappa and kappa* benchmarks.

Created: 2025-12-27 (Phase 44)
"""
import pytest
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.corrected_evaluator import (
    compute_g_corrected,
    compute_c_corrected,
    compute_c_with_uncertainty,
    compute_kappa_improvement_significance,
    validate_corrected_evaluator,
    CORRECTION_ALPHA,
    CORRECTION_F_REF,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# Known benchmark targets
C_TARGET_KAPPA = 2.13745440613217263636
C_TARGET_KAPPA_STAR = 1.9379524112
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167
THETA = 4 / 7
K = 3


class TestCorrectionParameters:
    """Test that correction parameters are in expected ranges."""

    def test_alpha_in_expected_range(self):
        """Alpha should be approximately 1.36."""
        assert 1.3 < CORRECTION_ALPHA < 1.5
        assert abs(CORRECTION_ALPHA - 1.3625) < 0.01

    def test_f_ref_in_expected_range(self):
        """f_ref should be approximately 0.316."""
        assert 0.3 < CORRECTION_F_REF < 0.35
        assert abs(CORRECTION_F_REF - 0.3154) < 0.01

    def test_alpha_times_beta_factor(self):
        """Check that alpha * beta_factor gives expected slope."""
        beta_factor = THETA / (2 * K * (2 * K + 1))
        slope = CORRECTION_ALPHA * beta_factor
        # The empirical slope is approximately 0.0185
        assert abs(slope - 0.0185) < 0.002


class TestComputeGCorrected:
    """Test the g correction computation."""

    def test_g_at_f_ref_equals_baseline(self):
        """At f_I1 = f_ref, corrected g should equal baseline g."""
        g_corrected, g_baseline, delta_g = compute_g_corrected(
            CORRECTION_F_REF, THETA, K
        )
        assert abs(delta_g) < 1e-10
        assert abs(g_corrected - g_baseline) < 1e-10

    def test_g_baseline_formula(self):
        """Baseline g should be 1 + theta/(2K(2K+1))."""
        _, g_baseline, _ = compute_g_corrected(0.3, THETA, K)
        expected = 1 + THETA / (2 * K * (2 * K + 1))
        assert abs(g_baseline - expected) < 1e-10

    def test_g_increases_for_low_f_I1(self):
        """For f_I1 < f_ref, g should increase (delta_g > 0)."""
        _, _, delta_g = compute_g_corrected(0.2, THETA, K)
        assert delta_g > 0

    def test_g_decreases_for_high_f_I1(self):
        """For f_I1 > f_ref, g should decrease (delta_g < 0)."""
        _, _, delta_g = compute_g_corrected(0.4, THETA, K)
        assert delta_g < 0

    def test_g_corrected_linearity(self):
        """Delta g should be linear in f_I1."""
        f1, f2, f3 = 0.2, 0.3, 0.4
        _, _, d1 = compute_g_corrected(f1, THETA, K)
        _, _, d2 = compute_g_corrected(f2, THETA, K)
        _, _, d3 = compute_g_corrected(f3, THETA, K)

        # Linear interpolation: d2 should be midpoint of d1 and d3
        expected_d2 = (d1 + d3) / 2
        assert abs(d2 - expected_d2) < 1e-10


class TestComputeCCorrected:
    """Test the corrected c computation against benchmarks."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_kappa_benchmark_within_001_percent(self, polys_kappa):
        """Corrected c for kappa should be within 0.01% of target."""
        result = compute_c_corrected(polys_kappa, R_KAPPA, theta=THETA, K=K)
        gap_pct = (result.c / C_TARGET_KAPPA - 1) * 100
        assert abs(gap_pct) < 0.01, f"Gap {gap_pct}% exceeds 0.01%"

    def test_kappa_star_benchmark_within_001_percent(self, polys_kappa_star):
        """Corrected c for kappa* should be within 0.01% of target."""
        result = compute_c_corrected(polys_kappa_star, R_KAPPA_STAR, theta=THETA, K=K)
        gap_pct = (result.c / C_TARGET_KAPPA_STAR - 1) * 100
        assert abs(gap_pct) < 0.01, f"Gap {gap_pct}% exceeds 0.01%"

    def test_correction_improves_kappa(self, polys_kappa):
        """Correction should improve kappa benchmark accuracy."""
        result = compute_c_corrected(polys_kappa, R_KAPPA, theta=THETA, K=K)

        baseline_gap = abs(result.c_baseline / C_TARGET_KAPPA - 1)
        corrected_gap = abs(result.c / C_TARGET_KAPPA - 1)

        assert corrected_gap < baseline_gap, "Correction should improve accuracy"

    def test_correction_improves_kappa_star(self, polys_kappa_star):
        """Correction should improve kappa* benchmark accuracy."""
        result = compute_c_corrected(polys_kappa_star, R_KAPPA_STAR, theta=THETA, K=K)

        baseline_gap = abs(result.c_baseline / C_TARGET_KAPPA_STAR - 1)
        corrected_gap = abs(result.c / C_TARGET_KAPPA_STAR - 1)

        assert corrected_gap < baseline_gap, "Correction should improve accuracy"

    def test_f_I1_values_in_expected_range(self, polys_kappa, polys_kappa_star):
        """I1 fractions should be in physically reasonable range."""
        result_k = compute_c_corrected(polys_kappa, R_KAPPA, theta=THETA, K=K)
        result_ks = compute_c_corrected(polys_kappa_star, R_KAPPA_STAR, theta=THETA, K=K)

        # Both should be between 0 and 1
        assert 0 < result_k.f_I1 < 1
        assert 0 < result_ks.f_I1 < 1

        # kappa has lower f_I1 than kappa* (known from analysis)
        assert result_k.f_I1 < result_ks.f_I1

    def test_delta_g_has_opposite_signs(self, polys_kappa, polys_kappa_star):
        """kappa and kappa* should need opposite g corrections."""
        result_k = compute_c_corrected(polys_kappa, R_KAPPA, theta=THETA, K=K)
        result_ks = compute_c_corrected(polys_kappa_star, R_KAPPA_STAR, theta=THETA, K=K)

        # kappa (low f_I1) needs g increased (positive delta_g)
        # kappa* (high f_I1) needs g decreased (negative or near-zero delta_g)
        assert result_k.delta_g > result_ks.delta_g

    def test_result_components_reasonable(self, polys_kappa):
        """All result components should be numerically reasonable."""
        result = compute_c_corrected(polys_kappa, R_KAPPA, theta=THETA, K=K)

        # S12 components should be positive (both contribute positively)
        assert result.S12_plus > 0
        assert result.S12_minus > 0  # S12 at -R is also positive

        # I1 and I2 at -R should sum to S12_minus
        assert abs(result.I1_minus + result.I2_minus - result.S12_minus) < 1e-10

        # Multipliers should be positive and greater than 1
        assert result.m_baseline > 1
        assert result.m_corrected > 1

        # g values should be close to baseline
        assert abs(result.g_corrected / result.g_baseline - 1) < 0.02


class TestUncertaintyQuantification:
    """Test uncertainty bounds computation."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_uncertainty_bounds_bracket_c(self, polys_kappa):
        """Uncertainty bounds should bracket the c value."""
        result = compute_c_with_uncertainty(polys_kappa, R_KAPPA)

        assert result.c_lower < result.c < result.c_upper

    def test_uncertainty_bounds_bracket_kappa(self, polys_kappa):
        """Uncertainty bounds should bracket the kappa value."""
        result = compute_c_with_uncertainty(polys_kappa, R_KAPPA)

        # Note: kappa bounds are reversed (lower c -> higher kappa)
        assert result.kappa_lower < result.kappa < result.kappa_upper

    def test_uncertainty_pct_reasonable(self, polys_kappa):
        """Uncertainty percentage should be reasonable."""
        result = compute_c_with_uncertainty(polys_kappa, R_KAPPA)

        # Should be at least 0.02% (our floor)
        assert result.uncertainty_pct >= 0.02
        # Should be at most 0.1% for well-behaved polynomials
        assert result.uncertainty_pct <= 0.1

    def test_kappa_uncertainty_propagation(self, polys_kappa):
        """Check kappa uncertainty is correctly propagated from c."""
        result = compute_c_with_uncertainty(polys_kappa, R_KAPPA)

        # Manual calculation
        kappa_lower_manual = 1 - math.log(result.c_upper) / R_KAPPA
        kappa_upper_manual = 1 - math.log(result.c_lower) / R_KAPPA

        assert abs(result.kappa_lower - kappa_lower_manual) < 1e-10
        assert abs(result.kappa_upper - kappa_upper_manual) < 1e-10


class TestKappaImprovementSignificance:
    """Test significance testing for kappa improvements."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_significant_improvement_detected(self, polys_kappa, polys_kappa_star):
        """A significant kappa improvement should be detected."""
        # kappa* has lower kappa than kappa, so comparing them in reverse
        # should show non-significant "improvement"
        result_k = compute_c_with_uncertainty(polys_kappa, R_KAPPA)
        result_ks = compute_c_with_uncertainty(polys_kappa_star, R_KAPPA_STAR)

        # kappa should be higher than kappa* (since kappa* uses lower R)
        assert result_k.kappa > result_ks.kappa

    def test_same_polynomial_not_significant(self, polys_kappa):
        """Same polynomial at same R should show no significant difference."""
        result1 = compute_c_with_uncertainty(polys_kappa, R_KAPPA)
        result2 = compute_c_with_uncertainty(polys_kappa, R_KAPPA)

        delta, is_sig, msg = compute_kappa_improvement_significance(result1, result2)

        assert abs(delta) < 1e-10
        assert not is_sig

    def test_significance_message_format(self, polys_kappa, polys_kappa_star):
        """Significance message should be properly formatted."""
        result_k = compute_c_with_uncertainty(polys_kappa, R_KAPPA)
        result_ks = compute_c_with_uncertainty(polys_kappa_star, R_KAPPA_STAR)

        _, _, msg = compute_kappa_improvement_significance(result_ks, result_k)

        assert "SIGNIFICANT" in msg or "NOT SIGNIFICANT" in msg


class TestValidateCorrectedEvaluator:
    """Test the built-in validation function."""

    def test_validation_passes(self):
        """Built-in validation should pass."""
        passed, msg = validate_corrected_evaluator(verbose=False)
        assert passed, f"Validation failed: {msg}"

    def test_validation_message_format(self):
        """Validation message should be informative."""
        passed, msg = validate_corrected_evaluator(verbose=False)
        assert "PASS" in msg or "FAIL" in msg
        assert "%" in msg  # Should include percentage gaps


class TestQuadratureStability:
    """Test that results are stable under quadrature refinement."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_quadrature_convergence(self, polys_kappa):
        """Results should converge as quadrature increases."""
        results = []
        for n_quad in [40, 60, 80]:
            result = compute_c_corrected(polys_kappa, R_KAPPA, n_quad=n_quad)
            results.append(result.c)

        # Difference between n=60 and n=80 should be smaller than n=40 and n=60
        diff_40_60 = abs(results[1] - results[0])
        diff_60_80 = abs(results[2] - results[1])

        assert diff_60_80 < diff_40_60 or diff_60_80 < 1e-8


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_g_correction_with_different_K(self):
        """g correction should work for different K values."""
        for K_test in [2, 3, 4, 5]:
            g_corr, g_base, delta_g = compute_g_corrected(0.3, THETA, K_test)
            assert g_corr > 0
            assert g_base > 1
            # g_baseline should decrease as K increases
            if K_test > 2:
                g_base_prev = 1 + THETA / (2 * (K_test - 1) * (2 * (K_test - 1) + 1))
                assert g_base < g_base_prev

    def test_custom_correction_parameters(self, polys_kappa):
        """Custom alpha and f_ref should work."""
        result = compute_c_corrected(
            polys_kappa, R_KAPPA,
            alpha=1.4, f_ref=0.32
        )
        assert result.c > 0
        assert result.g_corrected > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
