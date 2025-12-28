#!/usr/bin/env python3
"""
tests/test_m_needed_consistency_gate.py
Phase 31B.2: Dual-Benchmark Consistency Gate

This test verifies that the residual error is a STABLE phenomenon:
the ratio m_needed/m_empirical should match between κ and κ*.

If the ratio is consistent, we know:
1. The remaining gap is systematic, not benchmark-dependent
2. The empirical formula is under/over-estimating by a stable factor
3. We haven't introduced any benchmark-specific bugs

If the ratio is NOT consistent:
1. Something upstream is broken (polynomial loading, S34 computation)
2. The mirror formula may need R-dependent correction
3. We need to investigate further

Created: 2025-12-26 (Phase 31B)
"""

import pytest
import sys

sys.path.insert(0, ".")


@pytest.fixture
def dual_diagnostics():
    """Compute diagnostics for both benchmarks."""
    from src.evaluator.diagnostics import compute_dual_benchmark_diagnostics
    return compute_dual_benchmark_diagnostics(n_quad=60)


def test_m_ratio_is_consistent(dual_diagnostics):
    """
    The ratio m_needed/m_empirical should match between κ and κ*.

    This is THE key gate for Phase 31B:
    - If consistent: residual is stable, can be derived
    - If inconsistent: upstream bug or formula needs R-dependence
    """
    from src.evaluator.diagnostics import check_m_ratio_consistency

    kappa = dual_diagnostics["kappa"]
    kappa_star = dual_diagnostics["kappa_star"]

    is_consistent, diff_pct = check_m_ratio_consistency(
        kappa, kappa_star, tolerance_pct=0.5
    )

    print(f"\nM-ratio consistency check:")
    print(f"  κ  m_ratio: {kappa.m_ratio:.6f}")
    print(f"  κ* m_ratio: {kappa_star.m_ratio:.6f}")
    print(f"  Difference: {diff_pct:.4f}%")
    print(f"  Consistent: {is_consistent}")

    assert is_consistent, (
        f"m_ratio differs by {diff_pct:.4f}% between benchmarks "
        f"(κ: {kappa.m_ratio:.6f}, κ*: {kappa_star.m_ratio:.6f}). "
        f"This suggests benchmark-specific error."
    )


def test_m_ratio_is_close_to_one(dual_diagnostics):
    """
    The ratio m_needed/m_empirical should be close to 1.0.

    The empirical formula m = exp(R) + 5 achieves ~1-2% gap,
    so m_ratio should be within ~1-2% of 1.0.
    """
    for name, diag in dual_diagnostics.items():
        print(f"\n{name}: m_ratio = {diag.m_ratio:.6f} ({diag.m_adjustment_pct:+.4f}%)")

        assert 0.97 < diag.m_ratio < 1.03, (
            f"{name}: m_ratio {diag.m_ratio:.6f} is too far from 1.0 "
            f"(adjustment needed: {diag.m_adjustment_pct:+.4f}%)"
        )


def test_m_adjustment_is_small(dual_diagnostics):
    """
    The adjustment needed to hit target should be small (<3%).

    This quantifies "how wrong" the empirical formula is.
    """
    for name, diag in dual_diagnostics.items():
        print(f"\n{name}: m adjustment = {diag.m_adjustment_pct:+.4f}%")

        assert abs(diag.m_adjustment_pct) < 3.0, (
            f"{name}: adjustment {diag.m_adjustment_pct:+.4f}% is too large"
        )


def test_decomposition_is_valid(dual_diagnostics):
    """
    Verify that c = S12_plus + m*S12_minus + S34 holds exactly.
    """
    for name, diag in dual_diagnostics.items():
        assembled = diag.S12_plus + diag.m_empirical * diag.S12_minus + diag.S34

        print(f"\n{name}:")
        print(f"  S12_plus + m*S12_minus + S34 = {assembled:.6f}")
        print(f"  c_computed                   = {diag.c_computed:.6f}")

        assert abs(assembled - diag.c_computed) < 1e-10, (
            f"{name}: decomposition doesn't match "
            f"(assembled={assembled:.6f}, computed={diag.c_computed:.6f})"
        )


def test_both_gaps_are_similar(dual_diagnostics):
    """
    Both benchmarks should have similar c_gap (within ~0.5%).

    This was the Phase 30 bug: κ* had 9% gap vs κ's 1.3%.
    After fix, both should be ~1-1.5%.
    """
    kappa = dual_diagnostics["kappa"]
    kappa_star = dual_diagnostics["kappa_star"]

    gap_diff = abs(kappa.c_gap_pct - kappa_star.c_gap_pct)

    print(f"\nC-gap comparison:")
    print(f"  κ  c_gap: {kappa.c_gap_pct:+.4f}%")
    print(f"  κ* c_gap: {kappa_star.c_gap_pct:+.4f}%")
    print(f"  Difference: {gap_diff:.4f}%")

    assert gap_diff < 0.5, (
        f"C-gap differs by {gap_diff:.4f}% between benchmarks "
        f"(κ: {kappa.c_gap_pct:+.4f}%, κ*: {kappa_star.c_gap_pct:+.4f}%)"
    )


def test_s34_is_not_hardcoded(dual_diagnostics):
    """
    S34 should differ between κ and κ* (different polynomials).

    This guards against Phase 30 bug of hardcoding S34 = -0.6.
    """
    kappa_s34 = dual_diagnostics["kappa"].S34
    kappa_star_s34 = dual_diagnostics["kappa_star"].S34

    print(f"\nS34 values:")
    print(f"  κ  S34: {kappa_s34:.6f}")
    print(f"  κ* S34: {kappa_star_s34:.6f}")

    # S34 should be different because polynomials are different
    assert abs(kappa_s34 - kappa_star_s34) > 0.1, (
        f"S34 values are suspiciously similar "
        f"(κ: {kappa_s34:.6f}, κ*: {kappa_star_s34:.6f}). "
        f"Check for hardcoding bug."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
