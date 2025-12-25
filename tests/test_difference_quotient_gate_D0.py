"""
tests/test_difference_quotient_gate_D0.py
Phase 21B: Gate Tests for D = 0 via Unified Difference Quotient

PURPOSE:
========
These are the CRITICAL gate tests for Phase 21B. They verify that the
unified S12 evaluator (using the difference quotient structure) produces:
    D = 0  (to numerical precision)
    B/A = 5  (exactly)

MATHEMATICAL BASIS:
===================
The key insight is the symmetry property in the micro-case:
    I1(+R) = exp(2R) × I1(-R)

This means:
    S12_combined = I1(+R) - exp(2R) × I1(-R) = 0

And therefore:
    D = S12_combined + I34 = 0 + 0 = 0  (in micro-case)

USAGE:
======
    pytest tests/test_difference_quotient_gate_D0.py -v
"""

import pytest
import numpy as np

from src.unified_s12_evaluator import (
    compute_I1_at_R,
    compute_S12_unified_v2,
    verify_symmetry,
    run_dual_benchmark_v2,
)
from src.abd_diagnostics import check_derived_structure_gate, run_dual_benchmark_gate


# =============================================================================
# GATE TOLERANCES
# =============================================================================

# Very strict tolerance since the symmetry holds to machine precision
D_TOLERANCE = 1e-10
BA_TOLERANCE = 1e-10

# Benchmark parameters
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167


# =============================================================================
# SYMMETRY TESTS
# =============================================================================


class TestSymmetryProperty:
    """Test the fundamental symmetry I1(+R) = exp(2R) × I1(-R)."""

    def test_symmetry_holds_kappa(self):
        """κ: Symmetry should hold to machine precision."""
        result = verify_symmetry(R=KAPPA_R)

        assert result["symmetry_holds"], (
            f"Symmetry broken: ratio={result['ratio']}, "
            f"expected={result['expected_ratio']}, "
            f"rel_error={result['ratio_rel_error']}"
        )

    def test_symmetry_holds_kappa_star(self):
        """κ*: Symmetry should hold to machine precision."""
        result = verify_symmetry(R=KAPPA_STAR_R)

        assert result["symmetry_holds"], (
            f"Symmetry broken: ratio={result['ratio']}, "
            f"expected={result['expected_ratio']}, "
            f"rel_error={result['ratio_rel_error']}"
        )

    def test_combined_is_zero_kappa(self):
        """κ: Combined term [I1+ - exp(2R)*I1-] should be ~0."""
        result = verify_symmetry(R=KAPPA_R)

        combined = result["combined"]
        I1_plus = result["I1_plus"]

        # Combined should be negligible relative to I1_plus
        rel = abs(combined) / abs(I1_plus) if I1_plus != 0 else 0
        assert rel < 1e-10, f"Combined not ~0: {combined} (rel={rel})"

    def test_combined_is_zero_kappa_star(self):
        """κ*: Combined term should be ~0."""
        result = verify_symmetry(R=KAPPA_STAR_R)

        combined = result["combined"]
        I1_plus = result["I1_plus"]

        rel = abs(combined) / abs(I1_plus) if I1_plus != 0 else 0
        assert rel < 1e-10, f"Combined not ~0: {combined} (rel={rel})"


# =============================================================================
# D = 0 GATE TESTS
# =============================================================================


class TestDIsZero:
    """Test that D = 0 in the unified structure."""

    def test_D_is_zero_kappa(self):
        """κ: D should be ~0 in unified structure."""
        result = compute_S12_unified_v2(R=KAPPA_R)

        assert abs(result.abd.D) < D_TOLERANCE, (
            f"D should be ~0, got {result.abd.D}"
        )

    def test_D_is_zero_kappa_star(self):
        """κ*: D should be ~0 in unified structure."""
        result = compute_S12_unified_v2(R=KAPPA_STAR_R)

        assert abs(result.abd.D) < D_TOLERANCE, (
            f"D should be ~0, got {result.abd.D}"
        )


# =============================================================================
# B/A = 5 GATE TESTS
# =============================================================================


class TestBOverAIsFive:
    """Test that B/A = 5 in the unified structure."""

    def test_BA_is_five_kappa(self):
        """κ: B/A should be exactly 5."""
        result = compute_S12_unified_v2(R=KAPPA_R)

        assert abs(result.abd.B_over_A - 5.0) < BA_TOLERANCE, (
            f"B/A should be 5, got {result.abd.B_over_A}"
        )

    def test_BA_is_five_kappa_star(self):
        """κ*: B/A should be exactly 5."""
        result = compute_S12_unified_v2(R=KAPPA_STAR_R)

        assert abs(result.abd.B_over_A - 5.0) < BA_TOLERANCE, (
            f"B/A should be 5, got {result.abd.B_over_A}"
        )


# =============================================================================
# DUAL BENCHMARK GATE
# =============================================================================


class TestDualBenchmarkGate:
    """Test that BOTH benchmarks pass simultaneously."""

    def test_dual_benchmark_passes(self):
        """Both κ and κ* should pass the D=0, B/A=5 gates."""
        kappa, kappa_star = run_dual_benchmark_v2()

        # Both D values should be ~0
        assert abs(kappa.abd.D) < D_TOLERANCE, f"κ: D = {kappa.abd.D}"
        assert abs(kappa_star.abd.D) < D_TOLERANCE, f"κ*: D = {kappa_star.abd.D}"

        # Both B/A values should be 5
        assert abs(kappa.abd.B_over_A - 5.0) < BA_TOLERANCE, f"κ: B/A = {kappa.abd.B_over_A}"
        assert abs(kappa_star.abd.B_over_A - 5.0) < BA_TOLERANCE, f"κ*: B/A = {kappa_star.abd.B_over_A}"

    def test_dual_gate_function(self):
        """The run_dual_benchmark_gate helper should report PASS."""
        kappa, kappa_star = run_dual_benchmark_v2()

        passed, message = run_dual_benchmark_gate(
            kappa.abd,
            kappa_star.abd,
            d_tol=D_TOLERANCE,
            ba_tol=BA_TOLERANCE,
        )

        assert passed, f"Dual gate failed:\n{message}"


# =============================================================================
# DERIVED STRUCTURE GATE
# =============================================================================


class TestDerivedStructure:
    """Test that results match the derived (D=0) structure."""

    def test_is_derived_structure_kappa(self):
        """κ: Should match derived structure."""
        result = compute_S12_unified_v2(R=KAPPA_R)

        assert result.abd.is_derived_structure(tol=D_TOLERANCE), (
            f"Not derived structure: D={result.abd.D}, B/A={result.abd.B_over_A}"
        )

    def test_is_derived_structure_kappa_star(self):
        """κ*: Should match derived structure."""
        result = compute_S12_unified_v2(R=KAPPA_STAR_R)

        assert result.abd.is_derived_structure(tol=D_TOLERANCE), (
            f"Not derived structure: D={result.abd.D}, B/A={result.abd.B_over_A}"
        )


# =============================================================================
# QUADRATURE STABILITY
# =============================================================================


class TestQuadratureStability:
    """Test that results are stable under quadrature refinement."""

    def test_D_stable_under_refinement(self):
        """D should remain ~0 at higher quadrature."""
        for n in [20, 40, 60]:
            result = compute_S12_unified_v2(R=KAPPA_R, n_quad=n)
            assert abs(result.abd.D) < D_TOLERANCE, (
                f"D not stable at n={n}: D={result.abd.D}"
            )

    def test_symmetry_improves_with_refinement(self):
        """Symmetry error should decrease with more quadrature points."""
        errors = []
        for n in [20, 40, 80]:
            result = verify_symmetry(R=KAPPA_R, n_quad=n)
            errors.append(result["ratio_rel_error"])

        # Error should decrease monotonically
        assert errors[1] <= errors[0], f"Error should decrease: {errors}"
        assert errors[2] <= errors[1], f"Error should decrease: {errors}"


# =============================================================================
# DOCUMENTATION
# =============================================================================


class TestPhase21BAchievement:
    """Document what Phase 21B achieved."""

    def test_print_summary(self):
        """Print summary of the Phase 21B achievement."""
        print("\n" + "=" * 70)
        print("PHASE 21B ACHIEVEMENT: D = 0 via Unified Difference Quotient")
        print("=" * 70)

        for name, R in [("kappa", KAPPA_R), ("kappa_star", KAPPA_STAR_R)]:
            result = compute_S12_unified_v2(R=R, benchmark=name)

            print(f"\n{name.upper()} (R={R}):")
            print(f"  S12_combined = {result.S12_combined:.2e}")
            print(f"  I1+/I1- ratio = {result.ratio:.10f}")
            print(f"  Expected ratio = {result.expected_ratio:.10f}")
            print(f"  D = {result.abd.D:.2e}")
            print(f"  B/A = {result.abd.B_over_A:.6f}")

        print("\n" + "=" * 70)
        print("CONCLUSION: Unified approach achieves D=0, B/A=5 to machine precision")
        print("by exploiting the symmetry I1(+R) = exp(2R) × I1(-R).")
        print("=" * 70 + "\n")

        assert True
