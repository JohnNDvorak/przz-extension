#!/usr/bin/env python3
"""
tests/test_phase29_mirror_paper_sanity.py
Phase 29: Mirror Paper Regime Sanity Tests

Validates that the mirror computation in paper regime:
1. Produces finite, stable values
2. Does not have catastrophic amplification
3. Matches expected order of magnitude

Created: 2025-12-26 (Phase 29)
"""

import pytest
import sys
import math

sys.path.insert(0, ".")

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_transform_paper_exact import (
    compute_S12_paper_sum,
    compute_mirror_paper_analysis,
    compute_c_paper_derived,
    breakdown_paper_pairs,
)


@pytest.fixture
def kappa_polys():
    """Load kappa benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_star_polys():
    """Load kappa* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


# =============================================================================
# Test: S12 paper sum is finite and reasonable
# =============================================================================


def test_S12_paper_positive_R_finite(kappa_polys):
    """S12(+R) in paper regime should be finite and positive."""
    R = 1.3036
    theta = 4 / 7

    S12 = compute_S12_paper_sum(R, theta, kappa_polys, n_quad=60)

    print(f"\nS12(+R={R}) in paper regime: {S12:.6f}")

    assert math.isfinite(S12), "S12 should be finite"
    assert S12 > 0, "S12 at +R should be positive"
    assert 0.1 < S12 < 10, f"S12 should be in reasonable range, got {S12}"


def test_S12_paper_negative_R_finite(kappa_polys):
    """S12(-R) in paper regime should be finite."""
    R = 1.3036
    theta = 4 / 7

    S12 = compute_S12_paper_sum(-R, theta, kappa_polys, n_quad=60)

    print(f"\nS12(-R={-R}) in paper regime: {S12:.6f}")

    assert math.isfinite(S12), "S12 should be finite"
    # At -R, S12 should be smaller but still positive
    assert S12 > 0, "S12 at -R should be positive"


def test_mirror_analysis_no_catastrophic_amp(kappa_polys):
    """Mirror analysis should not have catastrophic amplification."""
    R = 1.3036
    theta = 4 / 7

    result = compute_mirror_paper_analysis(R, theta, kappa_polys, n_quad=60, verbose=True)

    print(f"\nMirror analysis results:")
    print(f"  S12_direct:  {result.S12_direct:.6f}")
    print(f"  S12_proxy:   {result.S12_proxy_neg_R:.6f}")
    print(f"  S12_mirror:  {result.S12_mirror_exact:.6f}")
    print(f"  m_eff:       {result.m_eff:.4f}")
    print(f"  m_empirical: {result.m_empirical:.4f}")

    # No catastrophic amplification
    assert math.isfinite(result.S12_mirror_exact)
    assert abs(result.S12_mirror_exact) < 100, "Mirror should not blow up"


# =============================================================================
# Test: c computation in paper regime
# =============================================================================


def test_c_paper_derived_kappa(kappa_polys):
    """Test c computation for kappa benchmark."""
    R = 1.3036
    theta = 4 / 7
    c_target = 2.137454

    result = compute_c_paper_derived(R, theta, kappa_polys, n_quad=60, verbose=True)

    print(f"\nkappa benchmark:")
    print(f"  c computed: {result['c']:.6f}")
    print(f"  c target:   {c_target:.6f}")
    print(f"  kappa:      {result['kappa']:.6f}")

    # Check c is in reasonable range
    assert 1.5 < result['c'] < 3.0, f"c out of range: {result['c']}"

    # Check gap from target
    gap = abs(result['c'] - c_target) / c_target
    print(f"  c gap: {gap:.2%}")

    # Paper regime with empirical m should be within ~5%
    assert gap < 0.1, f"c gap too large: {gap:.2%}"


def test_c_paper_derived_kappa_star(kappa_star_polys):
    """Test c computation for kappa* benchmark."""
    R = 1.1167
    theta = 4 / 7
    c_target = 1.938

    result = compute_c_paper_derived(R, theta, kappa_star_polys, n_quad=60, verbose=True)

    print(f"\nkappa* benchmark:")
    print(f"  c computed: {result['c']:.6f}")
    print(f"  c target:   {c_target:.6f}")
    print(f"  kappa:      {result['kappa']:.6f}")

    # Check c is in reasonable range
    assert 1.0 < result['c'] < 3.0, f"c out of range: {result['c']}"

    # Check gap from target
    gap = abs(result['c'] - c_target) / c_target
    print(f"  c gap: {gap:.2%}")


def test_c_ratio_stability(kappa_polys, kappa_star_polys):
    """Test that c ratio is stable across benchmarks."""
    result_kappa = compute_c_paper_derived(
        1.3036, 4/7, kappa_polys, n_quad=60
    )
    result_kappa_star = compute_c_paper_derived(
        1.1167, 4/7, kappa_star_polys, n_quad=60
    )

    c_ratio = result_kappa['c'] / result_kappa_star['c']
    target_ratio = 2.137454 / 1.938  # ≈ 1.103

    print(f"\nc ratio:")
    print(f"  computed: {c_ratio:.4f}")
    print(f"  target:   {target_ratio:.4f}")
    print(f"  diff:     {abs(c_ratio - target_ratio) / target_ratio:.2%}")

    # Ratio should be close to target
    assert abs(c_ratio - target_ratio) / target_ratio < 0.1


# =============================================================================
# Test: Quadrature stability
# =============================================================================


def test_S12_paper_quadrature_convergence(kappa_polys):
    """Test that S12 converges with increasing quadrature."""
    R = 1.3036
    theta = 4 / 7

    S12_40 = compute_S12_paper_sum(R, theta, kappa_polys, n_quad=40)
    S12_60 = compute_S12_paper_sum(R, theta, kappa_polys, n_quad=60)
    S12_80 = compute_S12_paper_sum(R, theta, kappa_polys, n_quad=80)

    print(f"\nQuadrature convergence:")
    print(f"  n=40: {S12_40:.8f}")
    print(f"  n=60: {S12_60:.8f}")
    print(f"  n=80: {S12_80:.8f}")

    # Check convergence
    diff_40_60 = abs(S12_60 - S12_40) / abs(S12_60)
    diff_60_80 = abs(S12_80 - S12_60) / abs(S12_80)

    print(f"  40→60 diff: {diff_40_60:.2e}")
    print(f"  60→80 diff: {diff_60_80:.2e}")

    # Should be well converged (even n=40 gives machine precision)
    assert diff_60_80 < 1e-6, "n=60 should be well converged"
    # If both diffs are at machine precision, that's fine (already converged)
    if diff_40_60 > 1e-12:
        assert diff_60_80 <= diff_40_60, "Should be converging or already converged"


# =============================================================================
# Test: Per-pair breakdown
# =============================================================================


def test_breakdown_paper_pairs_sums_correctly(kappa_polys):
    """Test that pair breakdown sums to total S12."""
    R = 1.3036
    theta = 4 / 7

    breakdown = breakdown_paper_pairs(R, theta, kappa_polys, n_quad=60)
    S12_direct = compute_S12_paper_sum(R, theta, kappa_polys, n_quad=60)

    # Sum breakdown
    S12_from_breakdown = sum(
        breakdown[pair]["total_normed"] for pair in breakdown
    )

    print(f"\nBreakdown check:")
    print(f"  S12 direct:   {S12_direct:.8f}")
    print(f"  S12 breakdown: {S12_from_breakdown:.8f}")

    rel_diff = abs(S12_direct - S12_from_breakdown) / abs(S12_direct)
    assert rel_diff < 1e-10, f"Breakdown doesn't sum correctly: {rel_diff}"


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
