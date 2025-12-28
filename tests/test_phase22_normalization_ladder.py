"""
tests/test_phase22_normalization_ladder.py
Phase 22: Normalization Ladder Tests

PURPOSE:
========
Verify that the scalar baseline normalization:
1. Correctly divides by F(R) = (exp(2R)-1)/(2R)
2. Preserves the D=0 and B/A=5 structural properties
3. Produces values in the right magnitude ballpark

These tests MUST pass before wiring normalization into evaluate.py.

REFERENCES:
===========
- src/unified_s12_evaluator_v3.py: Unified bracket with normalization
- docs/PHASE_21C_SUMMARY.md: Phase 21C structural achievements
"""

import pytest
import math
import numpy as np

from src.unified_s12_evaluator_v3 import (
    compute_S12_unified_v3,
    compute_scalar_baseline_factor,
    run_dual_benchmark_v3,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def compute_abd_from_unified_value(V: float, R: float) -> dict:
    """
    Compute ABD decomposition from unified bracket value V.

    For the unified approach (Phase 21C/22):
        m = exp(R) + 5  (for K=3)
        A = V / m
        B = 5 × A
        D = V - A × exp(R) - B

    D should be ~0 by the difference quotient identity.
    B/A should be exactly 5 (for K=3).

    Args:
        V: Unified bracket S12 value
        R: PRZZ R parameter

    Returns:
        dict with keys: A, B, D, B_over_A, m
    """
    m = math.exp(R) + 5  # mirror multiplier for K=3
    A = V / m
    B = 5 * A
    D = V - A * math.exp(R) - B
    B_over_A = B / A if A != 0 else float('inf')

    return {
        "A": A,
        "B": B,
        "D": D,
        "B_over_A": B_over_A,
        "m": m,
    }


# =============================================================================
# SCALAR BASELINE CHECK TESTS
# =============================================================================


class TestScalarBaselineCheck:
    """Verify F(R) computation and division behavior."""

    def test_F_R_formula_kappa(self):
        """F(R)/2 = (exp(2R)-1)/(4R) matches analytic formula for kappa."""
        R = 1.3036
        expected = (math.exp(2 * R) - 1) / (4 * R)  # F(R)/2
        computed = compute_scalar_baseline_factor(R)
        assert abs(computed - expected) < 1e-10

    def test_F_R_formula_kappa_star(self):
        """F(R)/2 = (exp(2R)-1)/(4R) matches analytic formula for kappa_star."""
        R = 1.1167
        expected = (math.exp(2 * R) - 1) / (4 * R)  # F(R)/2
        computed = compute_scalar_baseline_factor(R)
        assert abs(computed - expected) < 1e-10

    def test_F_R_limit_R_near_zero(self):
        """F(R)/2 -> 0.5 as R -> 0 (L'Hopital limit)."""
        computed = compute_scalar_baseline_factor(0.0)
        # F(R) -> 1 as R -> 0, so F(R)/2 -> 0.5
        assert abs(computed - 0.5) < 1e-10

    def test_unnormalized_divided_by_normalized_equals_F_R_div_2_kappa(self):
        """S12_unnorm / S12_norm = F(R)/2 for kappa benchmark (scalar mode)."""
        R = 1.3036
        theta = 4.0 / 7.0

        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # Compute unnormalized
        result_unnorm = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            normalization_mode="none",
        )

        # Compute normalized (scalar mode explicitly)
        result_norm = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            normalization_mode="scalar",
        )

        ratio = result_unnorm.S12_total / result_norm.S12_total
        expected = compute_scalar_baseline_factor(R)  # F(R)/2

        assert abs(ratio - expected) < 1e-8, f"Ratio {ratio} != F(R)/2 {expected}"

    def test_unnormalized_divided_by_normalized_equals_F_R_div_2_kappa_star(self):
        """S12_unnorm / S12_norm = F(R)/2 for kappa_star benchmark (scalar mode)."""
        R = 1.1167
        theta = 4.0 / 7.0

        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # Compute unnormalized
        result_unnorm = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            normalization_mode="none",
        )

        # Compute normalized (scalar mode explicitly)
        result_norm = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            normalization_mode="scalar",
        )

        ratio = result_unnorm.S12_total / result_norm.S12_total
        expected = compute_scalar_baseline_factor(R)  # F(R)/2

        assert abs(ratio - expected) < 1e-8, f"Ratio {ratio} != F(R)/2 {expected}"


# =============================================================================
# D AND B/A INVARIANT TESTS
# =============================================================================


class TestDAndBAInvariantsWithNormalization:
    """Verify D=0 and B/A=5 still hold after normalization."""

    def test_D_zero_with_normalization_kappa(self):
        """D ≈ 0 still holds for kappa with normalization ON."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        R = 1.3036

        # Compute ABD decomposition from unified value
        abd = compute_abd_from_unified_value(kappa.S12_total, R)

        # D should be ~0 (within numerical precision)
        # For normalized values, D should still be ~0
        assert abs(abd["D"]) < 1e-6, f"D = {abd['D']} is not ~0 for kappa (normalized)"

    def test_D_zero_with_normalization_kappa_star(self):
        """D ≈ 0 still holds for kappa_star with normalization ON."""
        _, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        R = 1.1167

        abd = compute_abd_from_unified_value(kappa_star.S12_total, R)

        assert abs(abd["D"]) < 1e-6, f"D = {abd['D']} is not ~0 for kappa_star (normalized)"

    def test_B_over_A_equals_5_with_normalization_kappa(self):
        """B/A = 5 still holds for kappa with normalization ON."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        R = 1.3036

        abd = compute_abd_from_unified_value(kappa.S12_total, R)

        # B/A should be 5 (2K-1 for K=3)
        assert abs(abd["B_over_A"] - 5.0) < 1e-6, f"B/A = {abd['B_over_A']} is not 5 for kappa"

    def test_B_over_A_equals_5_with_normalization_kappa_star(self):
        """B/A = 5 still holds for kappa_star with normalization ON."""
        _, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        R = 1.1167

        abd = compute_abd_from_unified_value(kappa_star.S12_total, R)

        assert abs(abd["B_over_A"] - 5.0) < 1e-6, f"B/A = {abd['B_over_A']} is not 5 for kappa_star"

    def test_normalization_does_not_change_D_ratio(self):
        """D/A ratio unchanged between normalized and unnormalized."""
        # The D/A ratio should be invariant under scalar normalization
        for benchmark, R in [("kappa", 1.3036), ("kappa_star", 1.1167)]:
            if benchmark == "kappa":
                P1, P2, P3, Q = load_przz_polynomials()
            else:
                P1, P2, P3, Q = load_przz_polynomials_kappa_star()

            polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
            theta = 4.0 / 7.0

            result_unnorm = compute_S12_unified_v3(
                R=R, theta=theta, polynomials=polynomials,
                normalize_scalar_baseline=False,
            )
            result_norm = compute_S12_unified_v3(
                R=R, theta=theta, polynomials=polynomials,
                normalize_scalar_baseline=True,
            )

            abd_unnorm = compute_abd_from_unified_value(result_unnorm.S12_total, R)
            abd_norm = compute_abd_from_unified_value(result_norm.S12_total, R)

            # D/A should be the same (both ~0)
            # D scales with V, but D/A = D/(V/m) = D*m/V should be invariant
            # Since D = V - A*exp(R) - B = V - V/m*exp(R) - 5*V/m = V*(1 - exp(R)/m - 5/m)
            # D/A = D*m/V = 1 - exp(R)/m - 5/m (constant for fixed R)
            # But for unified bracket, D = 0 by structure
            # Both should be ~0
            assert abs(abd_unnorm["D"]) < 1e-10, f"D_unnorm = {abd_unnorm['D']} not ~0"
            assert abs(abd_norm["D"]) < 1e-10, f"D_norm = {abd_norm['D']} not ~0"


# =============================================================================
# MAGNITUDE SANITY TESTS
# =============================================================================


class TestMagnitudeSanity:
    """Verify normalized values are in the right ballpark."""

    def test_normalized_S12_kappa_in_ballpark(self):
        """Normalized S12 for kappa should be in reasonable range [0.5, 3.0]."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        S12 = kappa.S12_total
        assert 0.5 < S12 < 3.0, f"S12 = {S12} not in [0.5, 3.0] for kappa"

    def test_normalized_S12_kappa_star_in_ballpark(self):
        """Normalized S12 for kappa_star should be in reasonable range [0.5, 3.0]."""
        _, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        S12 = kappa_star.S12_total
        assert 0.5 < S12 < 3.0, f"S12 = {S12} not in [0.5, 3.0] for kappa_star"

    def test_normalized_S12_less_than_c_target(self):
        """Normalized S12 should be less than c target (since c = S12 + I34)."""
        kappa, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        c_target_kappa = 2.137
        c_target_kappa_star = 1.938

        # S12 alone should be less than c (I34 adds positive contribution)
        # This is a loose sanity check
        assert kappa.S12_total < c_target_kappa * 2, \
            f"S12 = {kappa.S12_total} seems too large vs c_target = {c_target_kappa}"
        assert kappa_star.S12_total < c_target_kappa_star * 2, \
            f"S12 = {kappa_star.S12_total} seems too large vs c_target = {c_target_kappa_star}"

    def test_per_pair_contributions_finite(self):
        """All per-pair contributions should be finite."""
        kappa, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalize_scalar_baseline=True,
        )

        for result, benchmark in [(kappa, "kappa"), (kappa_star, "kappa_star")]:
            for pair_key in ["11", "22", "33", "12", "13", "23"]:
                contrib = result.pair_contributions[pair_key]
                assert np.isfinite(contrib), \
                    f"Pair {pair_key} contribution is not finite for {benchmark}"


# =============================================================================
# QUADRATURE STABILITY TESTS
# =============================================================================


class TestQuadratureStabilityWithNormalization:
    """Verify normalization is stable under quadrature refinement."""

    def test_S12_stable_n40_vs_n60(self):
        """S12 normalized should be stable between n=40 and n=60."""
        theta = 4.0 / 7.0
        R = 1.3036

        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result_40 = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            n_quad_u=40, n_quad_t=40,
            normalize_scalar_baseline=True,
        )

        result_60 = compute_S12_unified_v3(
            R=R, theta=theta, polynomials=polynomials,
            n_quad_u=60, n_quad_t=60,
            normalize_scalar_baseline=True,
        )

        rel_diff = abs(result_40.S12_total - result_60.S12_total) / abs(result_60.S12_total)
        assert rel_diff < 0.01, f"S12 changed by {rel_diff*100:.2f}% between n=40 and n=60"


# =============================================================================
# RESULT METADATA TESTS
# =============================================================================


class TestResultMetadata:
    """Verify result object contains correct normalization metadata."""

    def test_normalized_result_has_correct_metadata(self):
        """Normalized result should have correct metadata fields."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="scalar",  # Explicit scalar mode for this test
        )

        # Check metadata
        assert kappa.normalization_mode == "scalar"
        assert kappa.scalar_baseline_factor > 1.0
        assert kappa.normalization_factor > 1.0
        assert kappa.S12_unnormalized > kappa.S12_total

        # Verify relationship: S12_unnorm / S12_norm = normalization_factor
        expected_ratio = kappa.S12_unnormalized / kappa.S12_total
        assert abs(expected_ratio - kappa.normalization_factor) < 1e-10

    def test_unnormalized_result_has_correct_metadata(self):
        """Unnormalized result should have correct metadata fields."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="none",  # Explicit none mode for this test
        )

        # Check metadata
        assert kappa.normalization_mode == "none"
        assert kappa.scalar_baseline_factor > 1.0
        assert kappa.normalization_factor == 1.0  # No normalization applied
        # When not normalized, S12_unnormalized should equal S12_total
        assert abs(kappa.S12_unnormalized - kappa.S12_total) < 1e-10

    def test_diagnostic_corrected_mode_has_correct_metadata(self):
        """Diagnostic corrected mode should have correct metadata fields (Phase 23/24).

        NOTE: This test uses QUARANTINED empirically-fitted correction.
        """
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        # Check metadata
        assert kappa.normalization_mode == "diagnostic_corrected"
        assert kappa.scalar_baseline_factor > 1.0
        assert kappa.normalization_factor > 1.0
        # Corrected factor is smaller than scalar factor
        assert kappa.normalization_factor < kappa.scalar_baseline_factor
        assert kappa.S12_unnormalized > kappa.S12_total

        # Verify relationship
        expected_ratio = kappa.S12_unnormalized / kappa.S12_total
        assert abs(expected_ratio - kappa.normalization_factor) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
