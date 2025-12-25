"""
tests/test_plus5_main_only_gate.py
Phase 20.2: Main-Only Gate Tests (Currently xfail)

PURPOSE:
========
These tests define the Phase 20.2 success criteria:
- Main-only B/A should be within 1% of 5 (=2K-1 for K=3)
- This should happen WITHOUT J₁,₅ (error term)

Currently xfail because:
- Main-only B/A is ~4.28 (not 5)
- The exact main-term extraction (j1_main_term_exact.py) is not yet implemented

When these tests pass, Phase 20.2 is complete.

USAGE:
======
    pytest tests/test_plus5_main_only_gate.py -v

The xfail tests should fail initially, then pass once exact extraction is implemented.
"""

import pytest
import numpy as np

from src.ratios.plus5_harness import (
    compute_plus5_signature_split,
    Plus5SplitResult,
)


class TestMainOnlyGate:
    """
    Main-only B/A gate tests.

    These tests define the Phase 20.2 success criteria:
    Main-only B/A should hit 5 (=2K-1) WITHOUT needing J₁,₅.

    Currently xfail because Euler-Maclaurin approximation gives B/A ≈ 4.28.
    """

    @pytest.mark.xfail(
        reason="Phase 20.2: exact main-term extraction not yet implemented. "
               "Current Euler-Maclaurin gives B/A_main ≈ 4.28, need ~5.0.",
        strict=True  # Fail if test unexpectedly passes
    )
    def test_kappa_main_only_hits_5(self):
        """
        κ: B/A main-only should be within 1% of 5.

        Current state: B/A_main ≈ 4.28 (-14.4% gap)
        Target state: B/A_main ≈ 5.00 (<1% gap)

        This test will pass when j1_main_term_exact.py is implemented
        and wired into the computation.
        """
        result = compute_plus5_signature_split("kappa")

        target = 5.0
        tolerance = 0.01  # 1%

        gap = abs(result.B_over_A_main_only - target) / target

        assert gap < tolerance, (
            f"κ main-only B/A = {result.B_over_A_main_only:.4f}, "
            f"gap = {gap*100:.2f}% (target: <1%)"
        )

    @pytest.mark.xfail(
        reason="Phase 20.2: exact main-term extraction not yet implemented. "
               "Current Euler-Maclaurin gives B/A_main ≈ 4.22, need ~5.0.",
        strict=True
    )
    def test_kappa_star_main_only_hits_5(self):
        """
        κ*: B/A main-only should be within 1% of 5.

        Current state: B/A_main ≈ 4.22 (-15.6% gap)
        Target state: B/A_main ≈ 5.00 (<1% gap)
        """
        result = compute_plus5_signature_split("kappa_star")

        target = 5.0
        tolerance = 0.01  # 1%

        gap = abs(result.B_over_A_main_only - target) / target

        assert gap < tolerance, (
            f"κ* main-only B/A = {result.B_over_A_main_only:.4f}, "
            f"gap = {gap*100:.2f}% (target: <1%)"
        )


class TestBenchmarkIndependence:
    """
    Tests that B/A main-only is benchmark-independent (structural constant).

    The "+5" should be a universal structural constant, not dependent on
    which polynomial set is used.
    """

    def test_main_only_similar_across_benchmarks(self):
        """
        B/A main-only should be benchmark-independent (structural constant).

        Current state: κ gives 4.28, κ* gives 4.22 (1.4% diff)
        This ALREADY passes - benchmarks are within 5% of each other.

        This tests that the "+5" is truly a structural constant,
        not an artifact of polynomial-specific effects.

        Note: This test passes even though both values are ~4.28 not 5.
        The key observation is that they're SIMILAR across benchmarks.
        """
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # Should be within 5% of each other
        ratio = kappa.B_over_A_main_only / kappa_star.B_over_A_main_only

        assert 0.95 < ratio < 1.05, (
            f"Benchmark ratio = {ratio:.4f}, "
            f"κ B/A = {kappa.B_over_A_main_only:.4f}, "
            f"κ* B/A = {kappa_star.B_over_A_main_only:.4f}"
        )


class TestNoJ15Required:
    """
    Tests that J₁,₅ is NOT required to hit target after Phase 20.2.
    """

    @pytest.mark.xfail(
        reason="Phase 20.2: J₁,₅ is currently required. "
               "After exact extraction, it should NOT be required.",
        strict=True
    )
    def test_kappa_j15_not_required(self):
        """
        κ: J₁,₅ should NOT be required for B/A ≈ 5 after Phase 20.2.

        Current state: j15_required_for_target = True
        Target state: j15_required_for_target = False
        """
        result = compute_plus5_signature_split("kappa")

        assert result.j15_required_for_target is False, (
            f"J₁,₅ is still required! "
            f"Main-only B/A = {result.B_over_A_main_only:.4f}, "
            f"gap = {result.gap_percent_main_only:.2f}%"
        )

    @pytest.mark.xfail(
        reason="Phase 20.2: J₁,₅ is currently required. "
               "After exact extraction, it should NOT be required.",
        strict=True
    )
    def test_kappa_star_j15_not_required(self):
        """
        κ*: J₁,₅ should NOT be required for B/A ≈ 5 after Phase 20.2.
        """
        result = compute_plus5_signature_split("kappa_star")

        assert result.j15_required_for_target is False, (
            f"J₁,₅ is still required! "
            f"Main-only B/A = {result.B_over_A_main_only:.4f}, "
            f"gap = {result.gap_percent_main_only:.2f}%"
        )


class TestCurrentState:
    """
    Non-xfail tests documenting the current state.

    These tests verify the current behavior is stable while we work
    on improving it. They will NOT be removed when Phase 20.2 succeeds.
    """

    def test_kappa_main_only_current_value(self):
        """Document current κ main-only B/A (should be ~4.28)."""
        result = compute_plus5_signature_split("kappa")

        # Current value is approximately 4.28
        assert 4.2 < result.B_over_A_main_only < 4.4, (
            f"Unexpected change in κ main-only B/A: {result.B_over_A_main_only:.4f}"
        )

    def test_kappa_star_main_only_current_value(self):
        """Document current κ* main-only B/A (should be ~4.22)."""
        result = compute_plus5_signature_split("kappa_star")

        # Current value is approximately 4.22
        assert 4.1 < result.B_over_A_main_only < 4.3, (
            f"Unexpected change in κ* main-only B/A: {result.B_over_A_main_only:.4f}"
        )

    def test_j15_contribution_magnitude(self):
        """Document current J₁,₅ contribution (should be ~0.65-0.70)."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # J15 contributes ~0.65-0.70 to B/A
        assert 0.6 < kappa.j15_contribution_ratio < 0.75
        assert 0.6 < kappa_star.j15_contribution_ratio < 0.75

    def test_gap_from_5_is_significant(self):
        """Document that main-only gap from 5 is ~14-16%."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # Gap is 14-16%
        assert 10 < abs(kappa.gap_percent_main_only) < 20
        assert 10 < abs(kappa_star.gap_percent_main_only) < 20


class TestPhase20Documentation:
    """Document Phase 20.2 goals and progress."""

    def test_phase_20_2_goal(self):
        """
        PHASE 20.2 GOAL:

        Make MAIN_TERM_ONLY produce B/A = 2K-1 (=5 for K=3) without J₁,₅.

        Current state:
        - Main-only B/A ≈ 4.28 (κ) and 4.22 (κ*)
        - Gap is ~14-16% from target 5
        - J₁,₅ adds ~0.67 to reach B/A ≈ 4.95

        Target state:
        - Main-only B/A ≈ 5.00 (within 1%)
        - J₁,₅ contribution becomes ~0 or negligible
        - j15_required_for_target = False

        Implementation path:
        1. Research PRZZ Section 7 for exact main-term structure
        2. Implement j1_main_term_exact.py with correct combinatorics
        3. Wire into plus5_harness with extraction_mode="exact"
        4. Remove xfail markers when tests pass
        """
        # This test always passes - it's documentation
        result = compute_plus5_signature_split("kappa")

        print(f"\nCURRENT STATE (Phase 20.2 not complete):")
        print(f"  Main-only B/A: {result.B_over_A_main_only:.4f}")
        print(f"  Gap from 5: {result.gap_percent_main_only:.2f}%")
        print(f"  J15 contribution: {result.j15_contribution_ratio:.4f}")
        print(f"  J15 required: {result.j15_required_for_target}")
        print(f"\nTARGET STATE:")
        print(f"  Main-only B/A: ~5.00")
        print(f"  Gap from 5: <1%")
        print(f"  J15 required: False")
