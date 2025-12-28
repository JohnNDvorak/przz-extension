"""
tests/test_phase22_c_accuracy_gate.py
Phase 22: c Accuracy Gate Tests

PURPOSE:
========
Gate tests for c accuracy using the normalized unified bracket approach.

CURRENT STATUS (Phase 22):
==========================
With F(R)/2 normalization:
- κ: c gap = -5.29%
- κ*: c gap = -6.74%

The F(R)/2 normalization is derived from first principles:
- The PRZZ identity has (α+β) = -2Rθ in the denominator
- This introduces a factor of 2 in the normalization

The remaining 5-7% gap comes from non-scalar effects:
- Log factor (1/θ + x + y) contribution to xy coefficient
- Q factor eigenvalue differences (t-dependent vs u-dependent)
- P factor interaction effects

GOAL:
=====
Target: 0.5% accuracy for both benchmarks
Current: 5-7% accuracy
Gap: Further normalization refinement needed (Phase 23+)

REFERENCES:
===========
- src/unified_s12_evaluator_v3.py: Unified bracket with normalization
- src/evaluate.py: difference_quotient_v3 mode
"""

import pytest
import math

from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# Benchmark targets
C_TARGET_KAPPA = 2.137
C_TARGET_KAPPA_STAR = 1.938

# Current accuracy tolerances (will be tightened as normalization improves)
TOLERANCE_PERCENT_CURRENT = 10.0  # 10% for now (actual is 5-7%)
TOLERANCE_PERCENT_GOAL = 0.5  # Ultimate goal


class TestCAccuracyGate:
    """Gate tests for c accuracy using normalized unified bracket."""

    def test_c_accuracy_kappa_within_10_percent(self):
        """c accuracy for kappa should be within 10%."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalize_scalar_baseline=True,
        )

        c_computed = result.total
        c_gap_pct = abs((c_computed - C_TARGET_KAPPA) / C_TARGET_KAPPA) * 100

        print(f"\nκ benchmark:")
        print(f"  c target: {C_TARGET_KAPPA:.6f}")
        print(f"  c computed: {c_computed:.6f}")
        print(f"  c gap: {c_gap_pct:.2f}%")

        assert c_gap_pct < TOLERANCE_PERCENT_CURRENT, \
            f"c gap {c_gap_pct:.2f}% exceeds {TOLERANCE_PERCENT_CURRENT}% tolerance"

    def test_c_accuracy_kappa_star_within_10_percent(self):
        """c accuracy for kappa_star should be within 10%."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.1167,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalize_scalar_baseline=True,
        )

        c_computed = result.total
        c_gap_pct = abs((c_computed - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR) * 100

        print(f"\nκ* benchmark:")
        print(f"  c target: {C_TARGET_KAPPA_STAR:.6f}")
        print(f"  c computed: {c_computed:.6f}")
        print(f"  c gap: {c_gap_pct:.2f}%")

        assert c_gap_pct < TOLERANCE_PERCENT_CURRENT, \
            f"c gap {c_gap_pct:.2f}% exceeds {TOLERANCE_PERCENT_CURRENT}% tolerance"


class TestDiagnosticOutput:
    """Tests that print diagnostic information for debugging."""

    def test_full_diagnostic_output_kappa(self):
        """Print full diagnostic breakdown for kappa benchmark."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalize_scalar_baseline=True,
        )

        print("\n" + "="*60)
        print("DIAGNOSTIC: κ benchmark (difference_quotient_v3)")
        print("="*60)
        print(f"c target: {C_TARGET_KAPPA:.6f}")
        print(f"c computed: {result.total:.6f}")
        print(f"c gap: {(result.total - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100:.2f}%")
        print()
        print(f"S12 (normalized): {result.per_term['_S12_unified_total']:.6f}")
        print(f"S12 (unnormalized): {result.per_term['_S12_unnormalized']:.6f}")
        print(f"S34: {result.per_term['_S34_total']:.6f}")
        print(f"F(R)/2: {result.per_term['_scalar_baseline_factor']:.6f}")
        print()
        print(f"ABD decomposition:")
        print(f"  A: {result.per_term['_abd_A']:.6f}")
        print(f"  B: {result.per_term['_abd_B']:.6f}")
        print(f"  D: {result.per_term['_abd_D']:.2e}")
        print(f"  B/A: {result.per_term['_abd_B_over_A']:.6f}")
        print("="*60)

        # This test always passes - it's for diagnostic output
        assert True

    def test_full_diagnostic_output_kappa_star(self):
        """Print full diagnostic breakdown for kappa_star benchmark."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.1167,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalize_scalar_baseline=True,
        )

        print("\n" + "="*60)
        print("DIAGNOSTIC: κ* benchmark (difference_quotient_v3)")
        print("="*60)
        print(f"c target: {C_TARGET_KAPPA_STAR:.6f}")
        print(f"c computed: {result.total:.6f}")
        print(f"c gap: {(result.total - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR * 100:.2f}%")
        print()
        print(f"S12 (normalized): {result.per_term['_S12_unified_total']:.6f}")
        print(f"S12 (unnormalized): {result.per_term['_S12_unnormalized']:.6f}")
        print(f"S34: {result.per_term['_S34_total']:.6f}")
        print(f"F(R)/2: {result.per_term['_scalar_baseline_factor']:.6f}")
        print()
        print(f"ABD decomposition:")
        print(f"  A: {result.per_term['_abd_A']:.6f}")
        print(f"  B: {result.per_term['_abd_B']:.6f}")
        print(f"  D: {result.per_term['_abd_D']:.2e}")
        print(f"  B/A: {result.per_term['_abd_B_over_A']:.6f}")
        print("="*60)

        assert True


class TestComparisonWithEmpirical:
    """Compare unified bracket results with empirical approach."""

    def test_compare_with_empirical_kappa(self):
        """Compare unified vs empirical for kappa."""
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result_unified = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polynomials,
            mirror_mode="difference_quotient_v3",
            normalize_scalar_baseline=True,
        )

        result_empirical = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polynomials,
            mirror_mode="empirical_scalar",
        )

        print("\n" + "="*60)
        print("COMPARISON: Unified vs Empirical (κ)")
        print("="*60)
        print(f"c target: {C_TARGET_KAPPA:.6f}")
        print()
        print(f"Unified (v3):")
        print(f"  c: {result_unified.total:.6f}")
        print(f"  gap: {(result_unified.total - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100:.2f}%")
        print()
        print(f"Empirical:")
        print(f"  c: {result_empirical.total:.6f}")
        print(f"  gap: {(result_empirical.total - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100:.2f}%")
        print("="*60)

        # Both should be reasonably close to target
        assert abs(result_unified.total - C_TARGET_KAPPA) / C_TARGET_KAPPA < 0.1
        assert abs(result_empirical.total - C_TARGET_KAPPA) / C_TARGET_KAPPA < 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
