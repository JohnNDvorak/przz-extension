#!/usr/bin/env python3
"""
Gate 1: K=3 → K=2 Reduction Test

If P₃ ≡ 0, then all pairs involving P₃ should vanish:
- (1,3), (2,3), (3,3) → 0

This is a brutal check specifically against "Case C wrong sign / normalization" bugs.
It doesn't require external truth data - it's purely internal consistency.

Per GPT guidance:
"Set P₃ ≡ 0. Then the entire evaluator MUST reduce to K=2 theory (no Case C terms)."

Created: 2025-12-28
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import load_przz_polynomials, Polynomial
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper


def get_przz_polynomials_with_zero_p3():
    """Load PRZZ baseline polynomials but with P3 = 0."""
    P1, P2, P3_orig, Q = load_przz_polynomials()

    # Create zero P3 (P3(x) = x * P_tilde(x) where P_tilde = [0, 0, 0])
    P3_zero = Polynomial(np.array([0.0]))  # P(x) = 0

    return {
        "P1": P1,
        "P2": P2,
        "P3": P3_zero,
        "Q": Q,
    }


def get_przz_polynomials_normal():
    """Load normal PRZZ baseline polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Q": Q,
    }


class TestK2Reduction:
    """Test that P3=0 eliminates all Case C pairs."""

    @pytest.fixture
    def zero_p3_polys(self):
        return get_przz_polynomials_with_zero_p3()

    @pytest.fixture
    def normal_polys(self):
        return get_przz_polynomials_normal()

    def test_pair_33_vanishes_with_zero_p3(self, zero_p3_polys):
        """Pair (3,3) should be exactly 0 when P3=0."""
        R = 1.3036
        theta = 4 / 7

        I1_33 = compute_I1_unified_paper(
            R, theta, ell1=3, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60,
        )

        I2_33 = compute_I2_unified_paper(
            R, theta, ell1=3, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )

        print(f"\nPair (3,3) with P3=0:")
        print(f"  I1_33 = {I1_33.I1_value:.10e}")
        print(f"  I2_33 = {I2_33.I2_value:.10e}")

        # Should be numerically zero (within floating point tolerance)
        assert abs(I1_33.I1_value) < 1e-12, f"I1_33 should be 0, got {I1_33.I1_value:.6e}"
        assert abs(I2_33.I2_value) < 1e-12, f"I2_33 should be 0, got {I2_33.I2_value:.6e}"

    def test_pair_13_vanishes_with_zero_p3(self, zero_p3_polys):
        """Pair (1,3) should be exactly 0 when P3=0."""
        R = 1.3036
        theta = 4 / 7

        I1_13 = compute_I1_unified_paper(
            R, theta, ell1=1, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60,
        )

        I2_13 = compute_I2_unified_paper(
            R, theta, ell1=1, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )

        print(f"\nPair (1,3) with P3=0:")
        print(f"  I1_13 = {I1_13.I1_value:.10e}")
        print(f"  I2_13 = {I2_13.I2_value:.10e}")

        assert abs(I1_13.I1_value) < 1e-12, f"I1_13 should be 0, got {I1_13.I1_value:.6e}"
        assert abs(I2_13.I2_value) < 1e-12, f"I2_13 should be 0, got {I2_13.I2_value:.6e}"

    def test_pair_23_vanishes_with_zero_p3(self, zero_p3_polys):
        """Pair (2,3) should be exactly 0 when P3=0."""
        R = 1.3036
        theta = 4 / 7

        I1_23 = compute_I1_unified_paper(
            R, theta, ell1=2, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60,
        )

        I2_23 = compute_I2_unified_paper(
            R, theta, ell1=2, ell2=3,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )

        print(f"\nPair (2,3) with P3=0:")
        print(f"  I1_23 = {I1_23.I1_value:.10e}")
        print(f"  I2_23 = {I2_23.I2_value:.10e}")

        assert abs(I1_23.I1_value) < 1e-12, f"I1_23 should be 0, got {I1_23.I1_value:.6e}"
        assert abs(I2_23.I2_value) < 1e-12, f"I2_23 should be 0, got {I2_23.I2_value:.6e}"

    def test_non_p3_pairs_unchanged(self, zero_p3_polys, normal_polys):
        """Pairs (1,1), (1,2), (2,2) should be unchanged when P3=0."""
        R = 1.3036
        theta = 4 / 7

        pairs_to_check = [
            (1, 1, "A×A"),
            (1, 2, "A×B"),
            (2, 2, "B×B"),
        ]

        print("\nNon-P3 pairs comparison:")
        for ell1, ell2, case_name in pairs_to_check:
            # With normal P3
            I1_normal = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=normal_polys,
                n_quad_u=60, n_quad_t=60,
            )

            # With zero P3
            I1_zero = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=zero_p3_polys,
                n_quad_u=60, n_quad_t=60,
            )

            rel_diff = abs(I1_normal.I1_value - I1_zero.I1_value) / abs(I1_normal.I1_value + 1e-15)

            print(f"  ({ell1},{ell2}) {case_name}: normal={I1_normal.I1_value:.8e}, zero_p3={I1_zero.I1_value:.8e}, rel_diff={rel_diff:.2e}")

            # Should be identical (within numerical precision)
            assert rel_diff < 1e-10, f"Pair ({ell1},{ell2}) changed when P3=0: rel_diff={rel_diff:.6e}"


class TestK2ReductionSummary:
    """Summary test for K=2 reduction gate."""

    def test_full_k2_reduction_summary(self):
        """Comprehensive K=2 reduction check."""
        R = 1.3036
        theta = 4 / 7

        zero_p3_polys = get_przz_polynomials_with_zero_p3()
        normal_polys = get_przz_polynomials_normal()

        print("\n" + "=" * 70)
        print("GATE 1: K=3 → K=2 REDUCTION (P3 ≡ 0)")
        print("=" * 70)

        # Check P3-involving pairs vanish
        p3_pairs = [(1, 3), (2, 3), (3, 3)]
        all_vanish = True

        print("\nP3-involving pairs (should vanish):")
        for ell1, ell2 in p3_pairs:
            I1 = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=zero_p3_polys,
                n_quad_u=60, n_quad_t=60,
            )
            status = "PASS" if abs(I1.I1_value) < 1e-12 else "FAIL"
            print(f"  ({ell1},{ell2}): I1 = {I1.I1_value:.6e} [{status}]")
            if abs(I1.I1_value) >= 1e-12:
                all_vanish = False

        # Check non-P3 pairs unchanged
        non_p3_pairs = [(1, 1), (1, 2), (2, 2)]
        all_unchanged = True

        print("\nNon-P3 pairs (should be unchanged):")
        for ell1, ell2 in non_p3_pairs:
            I1_normal = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=normal_polys,
                n_quad_u=60, n_quad_t=60,
            )
            I1_zero = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=zero_p3_polys,
                n_quad_u=60, n_quad_t=60,
            )
            rel_diff = abs(I1_normal.I1_value - I1_zero.I1_value) / abs(I1_normal.I1_value + 1e-15)
            status = "PASS" if rel_diff < 1e-10 else "FAIL"
            print(f"  ({ell1},{ell2}): rel_diff = {rel_diff:.2e} [{status}]")
            if rel_diff >= 1e-10:
                all_unchanged = False

        print("\n" + "=" * 70)
        overall = "PASS" if (all_vanish and all_unchanged) else "FAIL"
        print(f"GATE 1 OVERALL: {overall}")
        print("=" * 70)

        assert all_vanish, "Some P3-involving pairs did not vanish"
        assert all_unchanged, "Some non-P3 pairs changed unexpectedly"


if __name__ == "__main__":
    # Run quick check
    print("\n" + "=" * 70)
    print("GATE 1: K=3 → K=2 REDUCTION - Quick Check")
    print("=" * 70)

    R = 1.3036
    theta = 4 / 7
    zero_p3_polys = get_przz_polynomials_with_zero_p3()

    print("\nChecking P3-involving pairs vanish with P3=0...")
    for ell1, ell2 in [(1, 3), (2, 3), (3, 3)]:
        I1 = compute_I1_unified_paper(
            R, theta, ell1=ell1, ell2=ell2,
            polynomials=zero_p3_polys,
            n_quad_u=60, n_quad_t=60,
        )
        status = "PASS" if abs(I1.I1_value) < 1e-12 else "FAIL"
        print(f"  ({ell1},{ell2}): I1 = {I1.I1_value:.6e} [{status}]")
