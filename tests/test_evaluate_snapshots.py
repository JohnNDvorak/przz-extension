"""
tests/test_evaluate_snapshots.py
Phase 19.6: Snapshot Tests for evaluate.py Refactoring Safety

PURPOSE:
========
These tests capture the CURRENT output of key evaluate.py functions.
Any refactoring of evaluate.py MUST preserve these values.

DO NOT change snapshot values unless:
1. You are fixing a documented bug
2. You have explicit approval
3. The change improves accuracy toward PRZZ targets

The snapshots serve as a safety net during the planned refactoring
of the 6700-line evaluate.py file.

USAGE:
======
    pytest tests/test_evaluate_snapshots.py -v

If snapshots fail after a refactor:
1. Run the diagnostics to understand the change
2. Determine if the change is an improvement or regression
3. Update snapshots with documented justification

CAPTURED: 2025-12-24 (Phase 19.6.1)
"""

import pytest
import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# =============================================================================
# SNAPSHOT VALUES - DO NOT CHANGE WITHOUT JUSTIFICATION
# =============================================================================

# Captured 2025-12-24 with compute_c_paper_with_mirror (n=60, hybrid mode)
SNAPSHOTS = {
    "kappa": {
        "R": 1.3036,
        "c_snapshot": 2.1085354094,
        "c_target": 2.137454406,  # PRZZ target
        "gap_percent": -1.35,
    },
    "kappa_star": {
        "R": 1.1167,
        "c_snapshot": 1.9145864789,
        "c_target": 1.938,  # PRZZ target (approximate)
        "gap_percent": -1.21,
    },
}

# Tolerance for snapshot matching
# This should be VERY tight - any change indicates behavioral change
SNAPSHOT_RTOL = 1e-8  # Relative tolerance
SNAPSHOT_ATOL = 1e-10  # Absolute tolerance

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_polynomial_dict(benchmark: str):
    """Get polynomial dictionary for evaluation."""
    if benchmark == "kappa":
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    else:
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def compute_snapshot_result(benchmark: str):
    """Compute result for snapshot comparison."""
    polys = get_polynomial_dict(benchmark)
    R = SNAPSHOTS[benchmark]["R"]

    result = compute_c_paper_with_mirror(
        theta=4.0 / 7.0,
        R=R,
        n=60,
        polynomials=polys,
        pair_mode="hybrid",
        use_factorial_normalization=True,
        mode="main",
        n_quad_a=40,
        K=3,
        mirror_mode="empirical_scalar",
    )

    return result


# =============================================================================
# SNAPSHOT TESTS - CRITICAL SAFETY CHECKS
# =============================================================================


class TestComputeCPaperWithMirrorSnapshots:
    """Snapshot tests for compute_c_paper_with_mirror."""

    def test_kappa_c_snapshot(self):
        """
        SNAPSHOT: kappa c value must match.

        Captured: 2025-12-24
        Value: 2.1085354094
        Source: compute_c_paper_with_mirror(n=60, hybrid, empirical_scalar)
        """
        result = compute_snapshot_result("kappa")
        expected = SNAPSHOTS["kappa"]["c_snapshot"]

        assert np.isclose(result.total, expected, rtol=SNAPSHOT_RTOL, atol=SNAPSHOT_ATOL), (
            f"SNAPSHOT MISMATCH: kappa c changed!\n"
            f"  Expected: {expected:.10f}\n"
            f"  Got:      {result.total:.10f}\n"
            f"  Delta:    {result.total - expected:+.2e}\n"
            f"\n"
            f"If this is intentional, update SNAPSHOTS['kappa']['c_snapshot']"
        )

    def test_kappa_star_c_snapshot(self):
        """
        SNAPSHOT: kappa_star c value must match.

        Captured: 2025-12-24
        Value: 1.9145864789
        Source: compute_c_paper_with_mirror(n=60, hybrid, empirical_scalar)
        """
        result = compute_snapshot_result("kappa_star")
        expected = SNAPSHOTS["kappa_star"]["c_snapshot"]

        assert np.isclose(result.total, expected, rtol=SNAPSHOT_RTOL, atol=SNAPSHOT_ATOL), (
            f"SNAPSHOT MISMATCH: kappa_star c changed!\n"
            f"  Expected: {expected:.10f}\n"
            f"  Got:      {result.total:.10f}\n"
            f"  Delta:    {result.total - expected:+.2e}\n"
            f"\n"
            f"If this is intentional, update SNAPSHOTS['kappa_star']['c_snapshot']"
        )


class TestGapSnapshots:
    """Snapshot tests for gap behavior."""

    def test_kappa_gap_snapshot(self):
        """Gap should be approximately -1.35%."""
        result = compute_snapshot_result("kappa")
        c_target = SNAPSHOTS["kappa"]["c_target"]

        gap = (result.total - c_target) / c_target * 100
        expected_gap = SNAPSHOTS["kappa"]["gap_percent"]

        # Gap should be within 0.1% of snapshot
        assert abs(gap - expected_gap) < 0.1, (
            f"Gap changed significantly: expected {expected_gap:+.2f}%, got {gap:+.2f}%"
        )

    def test_kappa_star_gap_snapshot(self):
        """Gap should be approximately -1.21%."""
        result = compute_snapshot_result("kappa_star")
        c_target = SNAPSHOTS["kappa_star"]["c_target"]

        gap = (result.total - c_target) / c_target * 100
        expected_gap = SNAPSHOTS["kappa_star"]["gap_percent"]

        # Gap should be within 0.1% of snapshot
        assert abs(gap - expected_gap) < 0.1, (
            f"Gap changed significantly: expected {expected_gap:+.2f}%, got {gap:+.2f}%"
        )


class TestRatioSnapshots:
    """Snapshot tests for ratio behavior."""

    def test_ratio_between_benchmarks_snapshot(self):
        """
        The ratio c(kappa)/c(kappa_star) should be stable.

        This ratio is more stable than absolute values and
        tests relative correctness between benchmarks.
        """
        result_kappa = compute_snapshot_result("kappa")
        result_kappa_star = compute_snapshot_result("kappa_star")

        ratio = result_kappa.total / result_kappa_star.total

        # Expected ratio from targets: 2.137/1.938 ≈ 1.103
        # Computed ratio from snapshots: 2.1085/1.9146 ≈ 1.1013
        expected_ratio = SNAPSHOTS["kappa"]["c_snapshot"] / SNAPSHOTS["kappa_star"]["c_snapshot"]

        assert abs(ratio - expected_ratio) < 1e-5, (
            f"Ratio changed: expected {expected_ratio:.6f}, got {ratio:.6f}"
        )


class TestQuadratureStability:
    """Test that results are stable under quadrature refinement."""

    def test_kappa_quadrature_stability(self):
        """Result should not change significantly with more quadrature points."""
        polys = get_polynomial_dict("kappa")
        R = SNAPSHOTS["kappa"]["R"]

        # Compare n=60 (default) with n=80
        result_60 = compute_c_paper_with_mirror(
            theta=4.0 / 7.0, R=R, n=60, polynomials=polys,
            pair_mode="hybrid", use_factorial_normalization=True,
            mode="main", n_quad_a=40, K=3, mirror_mode="empirical_scalar",
        )
        result_80 = compute_c_paper_with_mirror(
            theta=4.0 / 7.0, R=R, n=80, polynomials=polys,
            pair_mode="hybrid", use_factorial_normalization=True,
            mode="main", n_quad_a=40, K=3, mirror_mode="empirical_scalar",
        )

        # Should be within 0.01% of each other
        rel_diff = abs(result_80.total - result_60.total) / result_60.total
        assert rel_diff < 0.0001, (
            f"Quadrature unstable: n=60 gives {result_60.total:.6f}, "
            f"n=80 gives {result_80.total:.6f}"
        )


class TestModeConsistency:
    """Test consistency between different evaluation modes."""

    def test_hybrid_vs_ordered_consistency(self):
        """Hybrid and ordered modes should give similar results."""
        polys = get_polynomial_dict("kappa")
        R = SNAPSHOTS["kappa"]["R"]

        result_hybrid = compute_c_paper_with_mirror(
            theta=4.0 / 7.0, R=R, n=60, polynomials=polys,
            pair_mode="hybrid", use_factorial_normalization=True,
            mode="main", n_quad_a=40, K=3, mirror_mode="empirical_scalar",
        )
        result_ordered = compute_c_paper_with_mirror(
            theta=4.0 / 7.0, R=R, n=60, polynomials=polys,
            pair_mode="ordered", use_factorial_normalization=True,
            mode="main", n_quad_a=40, K=3, mirror_mode="empirical_scalar",
        )

        # Should be within 0.1% of each other
        rel_diff = abs(result_ordered.total - result_hybrid.total) / result_hybrid.total
        assert rel_diff < 0.001, (
            f"Mode inconsistency: hybrid gives {result_hybrid.total:.6f}, "
            f"ordered gives {result_ordered.total:.6f}"
        )


class TestReturnsCorrectStructure:
    """Test that compute_c_paper_with_mirror returns correct structure."""

    def test_result_has_total(self):
        """Result should have total attribute."""
        result = compute_snapshot_result("kappa")
        assert hasattr(result, "total")
        assert np.isfinite(result.total)

    def test_result_has_per_term(self):
        """Result should have per_term breakdown."""
        result = compute_snapshot_result("kappa")
        assert hasattr(result, "per_term")
        assert result.per_term is not None

    def test_result_has_n(self):
        """Result should have n (quadrature points)."""
        result = compute_snapshot_result("kappa")
        assert hasattr(result, "n")
        assert result.n == 60


class TestSnapshotDocumentation:
    """Document the snapshot state for reference."""

    def test_print_current_state(self):
        """
        Print current state for documentation.

        This test always passes but prints useful information.
        Run with -v to see output.
        """
        print("\n" + "=" * 70)
        print("SNAPSHOT STATE (Phase 19.6.1)")
        print("=" * 70)

        for benchmark in ["kappa", "kappa_star"]:
            result = compute_snapshot_result(benchmark)
            snap = SNAPSHOTS[benchmark]

            gap = (result.total - snap["c_target"]) / snap["c_target"] * 100

            print(f"\n{benchmark.upper()}:")
            print(f"  R = {snap['R']}")
            print(f"  c_computed = {result.total:.10f}")
            print(f"  c_snapshot = {snap['c_snapshot']:.10f}")
            print(f"  c_target   = {snap['c_target']:.10f}")
            print(f"  gap        = {gap:+.2f}%")
            print(f"  match      = {np.isclose(result.total, snap['c_snapshot'], rtol=1e-8)}")

        print()
        print("=" * 70)
