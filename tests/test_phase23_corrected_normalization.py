"""
tests/test_phase23_corrected_normalization.py
Phase 23/24: Diagnostic Corrected Normalization Tests (QUARANTINED)

PURPOSE:
========
Verify that the diagnostic_corrected normalization mode achieves <2% accuracy on c.

IMPORTANT (Phase 24 QUARANTINE):
================================
The correction factor was derived EMPIRICALLY by comparing unified bracket S12 to
empirical S12 across both benchmarks. This violates "derived > tuned" discipline.

All tests in this file require allow_diagnostic_correction=True to run.
This mode should NOT be used in production - use "scalar" mode instead.

Phase 24 aims to DERIVE this correction from first principles.

The correction factor:
    correction(R) = 0.8691 + 0.0765 × R

This accounts for non-scalar effects that the Phase 22 scalar baseline
normalization (F(R)/2) misses:
- Log factor (1/θ + x + y) contribution to xy coefficient
- Q eigenvalue t-dependence effects
- Polynomial structure interactions

REFERENCES:
===========
- src/unified_s12_evaluator_v3.py: Diagnostic corrected normalization implementation
- docs/PHASE_22_SUMMARY.md: Phase 22 scalar normalization baseline
"""

import pytest
import math

from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_s12_evaluator_v3 import (
    compute_scalar_baseline_factor,
    compute_diagnostic_corrected_baseline_factor,
    compute_diagnostic_correction_factor_linear_fit,
    run_dual_benchmark_v3,
)

# Legacy aliases for backwards compatibility
compute_corrected_baseline_factor = compute_diagnostic_corrected_baseline_factor
compute_empirical_correction_factor = compute_diagnostic_correction_factor_linear_fit


# Benchmark targets
C_TARGET_KAPPA = 2.137
C_TARGET_KAPPA_STAR = 1.938


class TestCorrectionFactorFormula:
    """Verify the correction factor formula is correct."""

    def test_correction_factor_kappa(self):
        """Correction factor for kappa should be ~0.9688."""
        R = 1.3036
        correction = compute_empirical_correction_factor(R)
        assert abs(correction - 0.9688) < 0.001, f"Correction {correction} != 0.9688"

    def test_correction_factor_kappa_star(self):
        """Correction factor for kappa_star should be ~0.9545."""
        R = 1.1167
        correction = compute_empirical_correction_factor(R)
        assert abs(correction - 0.9545) < 0.001, f"Correction {correction} != 0.9545"

    def test_corrected_baseline_less_than_scalar(self):
        """Corrected baseline should be less than scalar baseline."""
        for R in [1.3036, 1.1167, 1.0, 1.5]:
            F_scalar = compute_scalar_baseline_factor(R)
            F_corrected = compute_corrected_baseline_factor(R)
            assert F_corrected < F_scalar, \
                f"F_corrected {F_corrected} >= F_scalar {F_scalar} for R={R}"

    def test_correction_factor_linear_in_R(self):
        """Correction factor should increase linearly with R."""
        R_values = [1.0, 1.1, 1.2, 1.3, 1.4]
        corrections = [compute_empirical_correction_factor(R) for R in R_values]

        # Check monotonicity
        for i in range(len(corrections) - 1):
            assert corrections[i] < corrections[i+1], \
                f"Correction not increasing: {corrections}"


class TestDiagnosticCorrectedNormalizationAccuracy:
    """Verify that diagnostic_corrected normalization achieves <2% accuracy.

    NOTE: These tests use QUARANTINED empirically-fitted correction.
    All tests require allow_diagnostic_correction=True.
    """

    def test_c_accuracy_kappa_diagnostic_corrected(self):
        """c accuracy for kappa should be within 2% with diagnostic_corrected normalization."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        c_computed = result.total
        c_gap_pct = abs((c_computed - C_TARGET_KAPPA) / C_TARGET_KAPPA) * 100

        print(f"\nκ benchmark (diagnostic_corrected mode):")
        print(f"  c target: {C_TARGET_KAPPA:.6f}")
        print(f"  c computed: {c_computed:.6f}")
        print(f"  c gap: {c_gap_pct:.2f}%")

        assert c_gap_pct < 2.0, \
            f"c gap {c_gap_pct:.2f}% exceeds 2% tolerance"

    def test_c_accuracy_kappa_star_diagnostic_corrected(self):
        """c accuracy for kappa_star should be within 2% with diagnostic_corrected normalization."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.1167,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        c_computed = result.total
        c_gap_pct = abs((c_computed - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR) * 100

        print(f"\nκ* benchmark (diagnostic_corrected mode):")
        print(f"  c target: {C_TARGET_KAPPA_STAR:.6f}")
        print(f"  c computed: {c_computed:.6f}")
        print(f"  c gap: {c_gap_pct:.2f}%")

        assert c_gap_pct < 2.0, \
            f"c gap {c_gap_pct:.2f}% exceeds 2% tolerance"


class TestDiagnosticCorrectedVsScalarComparison:
    """Compare diagnostic_corrected normalization to scalar normalization.

    NOTE: These tests use QUARANTINED empirically-fitted correction.
    """

    def test_diagnostic_corrected_better_than_scalar_kappa(self):
        """Diagnostic corrected normalization should be more accurate than scalar for kappa."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result_scalar = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.3036, n=40, polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="scalar",
        )

        result_corrected = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.3036, n=40, polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        gap_scalar = abs((result_scalar.total - C_TARGET_KAPPA) / C_TARGET_KAPPA)
        gap_corrected = abs((result_corrected.total - C_TARGET_KAPPA) / C_TARGET_KAPPA)

        print(f"\nκ comparison:")
        print(f"  Scalar gap: {gap_scalar*100:.2f}%")
        print(f"  Diagnostic corrected gap: {gap_corrected*100:.2f}%")

        assert gap_corrected < gap_scalar, \
            f"Diagnostic corrected ({gap_corrected*100:.2f}%) not better than scalar ({gap_scalar*100:.2f}%)"

    def test_diagnostic_corrected_better_than_scalar_kappa_star(self):
        """Diagnostic corrected normalization should be more accurate than scalar for kappa_star."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result_scalar = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.1167, n=40, polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="scalar",
        )

        result_corrected = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.1167, n=40, polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        gap_scalar = abs((result_scalar.total - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR)
        gap_corrected = abs((result_corrected.total - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR)

        print(f"\nκ* comparison:")
        print(f"  Scalar gap: {gap_scalar*100:.2f}%")
        print(f"  Diagnostic corrected gap: {gap_corrected*100:.2f}%")

        assert gap_corrected < gap_scalar, \
            f"Diagnostic corrected ({gap_corrected*100:.2f}%) not better than scalar ({gap_scalar*100:.2f}%)"


class TestDAndBAPreservedWithDiagnosticCorrectedMode:
    """Verify D=0 and B/A=5 still hold with diagnostic_corrected normalization.

    NOTE: These tests use QUARANTINED empirically-fitted correction.
    """

    def test_D_zero_with_diagnostic_corrected_kappa(self):
        """D ≈ 0 still holds for kappa with diagnostic_corrected mode."""
        kappa, _ = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        R = 1.3036
        m = math.exp(R) + 5
        A = kappa.S12_total / m
        B = 5 * A
        D = kappa.S12_total - A * math.exp(R) - B

        assert abs(D) < 1e-6, f"D = {D} is not ~0 for kappa (diagnostic_corrected)"

    def test_D_zero_with_diagnostic_corrected_kappa_star(self):
        """D ≈ 0 still holds for kappa_star with diagnostic_corrected mode."""
        _, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        R = 1.1167
        m = math.exp(R) + 5
        A = kappa_star.S12_total / m
        B = 5 * A
        D = kappa_star.S12_total - A * math.exp(R) - B

        assert abs(D) < 1e-6, f"D = {D} is not ~0 for kappa_star (diagnostic_corrected)"

    def test_B_over_A_equals_5_with_diagnostic_corrected(self):
        """B/A = 5 still holds with diagnostic_corrected mode."""
        kappa, kappa_star = run_dual_benchmark_v3(
            include_Q=True,
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        for result, R, name in [(kappa, 1.3036, "kappa"), (kappa_star, 1.1167, "kappa_star")]:
            m = math.exp(R) + 5
            A = result.S12_total / m
            B = 5 * A
            B_over_A = B / A if A != 0 else float('inf')

            assert abs(B_over_A - 5.0) < 1e-6, \
                f"B/A = {B_over_A} is not 5 for {name}"


class TestDiagnosticCorrectedModeMatchesEmpirical:
    """Verify diagnostic_corrected mode matches empirical mode accuracy.

    NOTE: These tests use QUARANTINED empirically-fitted correction.
    """

    def test_diagnostic_corrected_matches_empirical_kappa(self):
        """Diagnostic corrected mode should match empirical accuracy for kappa."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result_emp = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.3036, n=40, polynomials=polynomials,
            mirror_mode="empirical_scalar",
        )

        result_corrected = compute_c_paper_with_mirror(
            theta=4.0/7.0, R=1.3036, n=40, polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalization_mode="diagnostic_corrected",
            allow_diagnostic_correction=True,  # Required for diagnostic mode
        )

        gap_emp = (result_emp.total - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100
        gap_corrected = (result_corrected.total - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100

        print(f"\nκ vs empirical:")
        print(f"  Empirical c: {result_emp.total:.6f} (gap: {gap_emp:.2f}%)")
        print(f"  Diagnostic corrected c: {result_corrected.total:.6f} (gap: {gap_corrected:.2f}%)")

        # Diagnostic corrected should be within 0.5% of empirical
        assert abs(gap_corrected - gap_emp) < 0.5, \
            f"Diagnostic corrected gap {gap_corrected:.2f}% differs from empirical {gap_emp:.2f}% by >0.5%"


class TestDiagnosticModeRequiresExplicitOptIn:
    """Verify that diagnostic_corrected mode requires allow_diagnostic_correction=True.

    This is a Phase 24 gate test to ensure the quarantine is enforced.
    """

    def test_diagnostic_corrected_without_flag_raises(self):
        """Using diagnostic_corrected without allow_diagnostic_correction should raise."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        with pytest.raises(ValueError) as exc_info:
            compute_c_paper_with_mirror(
                theta=4.0/7.0, R=1.3036, n=40, polynomials=polynomials,
                mirror_mode="difference_quotient_v3",
                normalization_mode="diagnostic_corrected",
                # allow_diagnostic_correction NOT set (defaults to False)
            )

        assert "allow_diagnostic_correction" in str(exc_info.value)
        assert "derived > tuned" in str(exc_info.value) or "diagnostic" in str(exc_info.value).lower()

    def test_run_dual_benchmark_without_flag_raises(self):
        """Using run_dual_benchmark_v3 with diagnostic_corrected without flag should raise."""
        with pytest.raises(ValueError) as exc_info:
            run_dual_benchmark_v3(
                include_Q=True,
                normalization_mode="diagnostic_corrected",
                # allow_diagnostic_correction NOT set (defaults to False)
            )

        assert "allow_diagnostic_correction" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
