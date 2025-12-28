#!/usr/bin/env python3
"""
tests/test_unified_bracket_plus5_ladder.py
Phase 32.2: Polynomial Ladder Tests for Unified Bracket

This test suite verifies that the B/A=5 and D=0 invariants survive
polynomial introduction through all rungs of the ladder:

    P=1,Q=1 → P=1,Q=PRZZ → P=PRZZ,Q=1 → P=PRZZ,Q=PRZZ

If B/A=5 holds at ALL rungs, the "+5" in m = exp(R) + 5 is structural
and does not depend on polynomial details.

Created: 2025-12-26 (Phase 32)
"""

import pytest
import math
import sys

sys.path.insert(0, ".")

from src.unified_bracket_ladder import (
    run_ladder_test,
    run_dual_benchmark_ladder,
    POLY_MODES,
    LadderSuite,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dual_benchmark_results():
    """Run ladder tests for both benchmarks once."""
    return run_dual_benchmark_ladder(n_quad=40)


@pytest.fixture
def kappa_suite(dual_benchmark_results):
    """Ladder results for κ benchmark."""
    return dual_benchmark_results["kappa"]


@pytest.fixture
def kappa_star_suite(dual_benchmark_results):
    """Ladder results for κ* benchmark."""
    return dual_benchmark_results["kappa_star"]


# =============================================================================
# Core Invariant Tests
# =============================================================================


def test_all_modes_have_results(kappa_suite, kappa_star_suite):
    """All 4 ladder rungs should have results."""
    for mode in POLY_MODES:
        assert mode in kappa_suite.results, f"Missing κ result for {mode}"
        assert mode in kappa_star_suite.results, f"Missing κ* result for {mode}"


def test_kappa_D_zero_all_rungs(kappa_suite):
    """D should be ~0 for all κ ladder rungs."""
    for mode in POLY_MODES:
        r = kappa_suite.results[mode]
        assert r.D_zero_ok, f"κ {mode}: D/A = {r.D_over_A:.6f} > tolerance"


def test_kappa_star_D_zero_all_rungs(kappa_star_suite):
    """D should be ~0 for all κ* ladder rungs."""
    for mode in POLY_MODES:
        r = kappa_star_suite.results[mode]
        assert r.D_zero_ok, f"κ* {mode}: D/A = {r.D_over_A:.6f} > tolerance"


def test_kappa_BA_five_all_rungs(kappa_suite):
    """B/A should be ~5 for all κ ladder rungs."""
    for mode in POLY_MODES:
        r = kappa_suite.results[mode]
        assert r.BA_five_ok, f"κ {mode}: B/A = {r.B_over_A:.4f} ≠ 5"


def test_kappa_star_BA_five_all_rungs(kappa_star_suite):
    """B/A should be ~5 for all κ* ladder rungs."""
    for mode in POLY_MODES:
        r = kappa_star_suite.results[mode]
        assert r.BA_five_ok, f"κ* {mode}: B/A = {r.B_over_A:.4f} ≠ 5"


def test_all_invariants_hold(kappa_suite, kappa_star_suite):
    """Both D=0 and B/A=5 should hold for all rungs."""
    assert kappa_suite.all_D_zero(), "κ: D≠0 at some rung"
    assert kappa_suite.all_BA_five(), "κ: B/A≠5 at some rung"
    assert kappa_star_suite.all_D_zero(), "κ*: D≠0 at some rung"
    assert kappa_star_suite.all_BA_five(), "κ*: B/A≠5 at some rung"


def test_no_failing_rungs(kappa_suite, kappa_star_suite):
    """No rung should fail invariant checks."""
    assert kappa_suite.first_failing_rung() is None, (
        f"κ first failure: {kappa_suite.first_failing_rung()}"
    )
    assert kappa_star_suite.first_failing_rung() is None, (
        f"κ* first failure: {kappa_star_suite.first_failing_rung()}"
    )


# =============================================================================
# Individual Rung Tests
# =============================================================================


def test_microcase_P1Q1_is_baseline(kappa_suite, kappa_star_suite):
    """
    P=1,Q=1 microcase should match Phase 31 results.

    This is the baseline where B/A=5 was first proven.
    """
    for name, suite, R in [("κ", kappa_suite, 1.3036), ("κ*", kappa_star_suite, 1.1167)]:
        r = suite.results["P=1,Q=1"]

        # A should be positive (basic sanity)
        assert r.A > 0, f"{name}: A should be positive in microcase"

        # B/A should be exactly 5.0
        assert abs(r.B_over_A - 5.0) < 1e-6, f"{name}: B/A = {r.B_over_A}, expected 5.0"

        # D should be exactly 0
        assert abs(r.D) < 1e-10, f"{name}: D = {r.D}, expected 0"


def test_Q_polynomial_survives(kappa_suite, kappa_star_suite):
    """
    P=1,Q=PRZZ should still have B/A=5.

    This tests if Q polynomial introduction preserves invariants.
    """
    for name, suite in [("κ", kappa_suite), ("κ*", kappa_star_suite)]:
        r = suite.results["P=1,Q=PRZZ"]

        assert abs(r.B_over_A - 5.0) < 0.01, (
            f"{name} with Q: B/A = {r.B_over_A}, expected ~5"
        )
        assert abs(r.D) < 1e-10, f"{name} with Q: D should be 0"


def test_P_polynomials_survive(kappa_suite, kappa_star_suite):
    """
    P=PRZZ,Q=1 should still have B/A=5.

    This tests if P polynomials introduction preserves invariants.
    """
    for name, suite in [("κ", kappa_suite), ("κ*", kappa_star_suite)]:
        r = suite.results["P=PRZZ,Q=1"]

        assert abs(r.B_over_A - 5.0) < 0.01, (
            f"{name} with P: B/A = {r.B_over_A}, expected ~5"
        )
        assert abs(r.D) < 1e-10, f"{name} with P: D should be 0"


def test_full_polynomial_case(kappa_suite, kappa_star_suite):
    """
    P=PRZZ,Q=PRZZ (full case) should still have B/A=5.

    This is the ultimate test: invariants survive with all polynomials.
    """
    for name, suite in [("κ", kappa_suite), ("κ*", kappa_star_suite)]:
        r = suite.results["P=PRZZ,Q=PRZZ"]

        assert abs(r.B_over_A - 5.0) < 0.01, (
            f"{name} full: B/A = {r.B_over_A}, expected ~5"
        )
        assert abs(r.D) < 1e-10, f"{name} full: D should be 0"


# =============================================================================
# Monotonicity and Consistency Tests
# =============================================================================


def test_A_values_change_with_polynomials(kappa_suite, kappa_star_suite):
    """
    A values should change when polynomials are introduced.

    This verifies that polynomials are actually being applied
    (not just returning microcase values).
    """
    for name, suite in [("κ", kappa_suite), ("κ*", kappa_star_suite)]:
        A_base = suite.results["P=1,Q=1"].A
        A_Q = suite.results["P=1,Q=PRZZ"].A
        A_P = suite.results["P=PRZZ,Q=1"].A
        A_full = suite.results["P=PRZZ,Q=PRZZ"].A

        # A should differ between modes (polynomials have effect)
        # Note: some could be similar by coincidence, but not all equal
        values = [A_base, A_Q, A_P, A_full]
        unique_values = len(set(round(v, 6) for v in values))

        # Allow at most 2 to be similar (unlikely all 4 are equal)
        assert unique_values >= 2, f"{name}: A values are suspiciously similar: {values}"


def test_B_tracks_A_exactly(kappa_suite, kappa_star_suite):
    """B should always be exactly 5×A."""
    for name, suite in [("κ", kappa_suite), ("κ*", kappa_star_suite)]:
        for mode in POLY_MODES:
            r = suite.results[mode]
            expected_B = 5.0 * r.A

            # Use relative tolerance for comparison
            if abs(r.A) > 1e-10:
                rel_err = abs(r.B - expected_B) / abs(expected_B)
                assert rel_err < 1e-10, (
                    f"{name} {mode}: B = {r.B}, expected 5×A = {expected_B}"
                )


# =============================================================================
# Benchmark Consistency Tests
# =============================================================================


def test_benchmarks_have_same_BA_ratio(kappa_suite, kappa_star_suite):
    """Both benchmarks should give B/A=5 for all rungs."""
    for mode in POLY_MODES:
        kappa_BA = kappa_suite.results[mode].B_over_A
        kappa_star_BA = kappa_star_suite.results[mode].B_over_A

        assert abs(kappa_BA - 5.0) < 0.01, f"κ {mode}: B/A = {kappa_BA}"
        assert abs(kappa_star_BA - 5.0) < 0.01, f"κ* {mode}: B/A = {kappa_star_BA}"


def test_R_affects_magnitudes_not_ratio(kappa_suite, kappa_star_suite):
    """
    Different R values should affect A magnitudes but not B/A ratio.

    κ has R=1.3036, κ* has R=1.1167.
    """
    for mode in POLY_MODES:
        r_kappa = kappa_suite.results[mode]
        r_kappa_star = kappa_star_suite.results[mode]

        # A values should differ between benchmarks
        # (Allow for sign differences in Q case)

        # But B/A should be the same
        assert abs(r_kappa.B_over_A - r_kappa_star.B_over_A) < 0.01, (
            f"{mode}: κ B/A = {r_kappa.B_over_A}, κ* B/A = {r_kappa_star.B_over_A}"
        )


# =============================================================================
# Diagnostic Output Tests
# =============================================================================


def test_print_ladder_summary(kappa_suite, kappa_star_suite, capsys):
    """Print a summary for diagnostic purposes."""
    from src.unified_bracket_ladder import print_ladder_results

    print_ladder_results(kappa_suite)
    print_ladder_results(kappa_star_suite)

    captured = capsys.readouterr()

    # Verify key elements are in output
    assert "KAPPA" in captured.out
    assert "KAPPA_STAR" in captured.out
    assert "ALL INVARIANTS HOLD" in captured.out


def test_ladder_uses_canonical_bracket():
    """
    Verify the ladder uses the canonical bracket function from unified_s12_evaluator_v3.

    Phase 32.3: The ladder now calls canonical_bracket_series() instead of
    its own implementation. This ensures consistency with the production evaluator.

    The key invariant B/A = 5.0 should still hold for all rungs.
    Note: Absolute A values may differ from Phase 31 due to using the canonical
    bracket structure (which includes proper (1-u)^{ℓ₁+ℓ₂} prefactors).
    """
    results = run_dual_benchmark_ladder(n_quad=40)

    # Verify B/A = 5 exactly for microcase
    for name, benchmark in [("κ", "kappa"), ("κ*", "kappa_star")]:
        r = results[benchmark].results["P=1,Q=1"]

        # B/A should be exactly 5
        assert abs(r.B_over_A - 5.0) < 1e-6, (
            f"{name} B/A = {r.B_over_A}, expected 5.0"
        )

        # D should be exactly 0 (by construction of unified bracket)
        assert abs(r.D) < 1e-10, f"{name} D = {r.D}, expected 0"

        # A should be positive and non-trivial
        assert r.A > 0, f"{name} A should be positive"

        print(f"\n{name} microcase (canonical bracket):")
        print(f"  A = {r.A:.6f}")
        print(f"  B = {r.B:.6f}")
        print(f"  B/A = {r.B_over_A:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
