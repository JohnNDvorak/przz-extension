#!/usr/bin/env python3
"""
tests/test_phase29_unified_paper_matches_dsl_paper.py
Phase 29: Validate unified_paper matches OLD DSL with kernel_regime="paper"

This test validates that the new unified_paper backend produces values
equivalent to the OLD DSL term makers when called with kernel_regime="paper".

CRITICAL DIAGNOSTIC PAIRS (from Phase 28):
- (2,2): should shrink by ~4x vs raw (ratio ~0.24)
- (1,3), (2,3): should FLIP SIGN vs raw

Created: 2025-12-26 (Phase 29)
"""

import pytest
import sys
import math

sys.path.insert(0, ".")

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_i1_paper import compute_I1_unified_paper, compare_raw_vs_paper
from src.unified_i1_general import compute_I1_unified_general
from src.unified_i2_paper import compute_I2_unified_paper, compare_I2_raw_vs_paper

# OLD DSL imports for comparison
from src.terms_k3_d1 import (
    make_I1_11, make_I1_22, make_I1_33,
    make_I1_12, make_I1_13, make_I1_23,
    make_I2_11, make_I2_22, make_I2_33,
    make_I2_12, make_I2_13, make_I2_23,
)
from src.evaluate import evaluate_term


# =============================================================================
# Test fixtures
# =============================================================================


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
# Test: (1,3) microcase - the first validation target
# =============================================================================


def test_unified_paper_13_matches_dsl_paper(kappa_polys):
    """
    PHASE 29.1a CRITICAL: Validate (1,3) pair with paper regime.

    The (1,3) pair should FLIP SIGN in paper vs raw regime.
    This is the primary microcase validation.
    """
    R = 1.3036
    theta = 4 / 7

    # Compute using unified_paper
    paper_result = compute_I1_unified_paper(
        R, theta, 1, 3, kappa_polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )

    # Compute using OLD DSL with paper regime
    term = make_I1_13(theta, R, kernel_regime="paper")
    dsl_result = evaluate_term(term, kappa_polys, n=60, R=R, theta=theta)

    # The values should match
    rel_diff = abs(paper_result.I1_value - dsl_result.value) / max(abs(dsl_result.value), 1e-15)

    print(f"\n(1,3) Paper Regime Comparison:")
    print(f"  unified_paper: {paper_result.I1_value:.10e}")
    print(f"  OLD DSL paper: {dsl_result.value:.10e}")
    print(f"  relative diff: {rel_diff:.2e}")

    assert rel_diff < 1e-4, f"(1,3) mismatch: rel_diff={rel_diff:.2e}"


def test_paper_13_flips_sign_vs_raw(kappa_polys):
    """Verify (1,3) flips sign in paper vs raw regime."""
    R = 1.3036
    theta = 4 / 7

    comparison = compare_raw_vs_paper(R, theta, 1, 3, kappa_polys, n_quad=60)

    print(f"\n(1,3) Raw vs Paper:")
    print(f"  raw:   {comparison['raw']:.6e}")
    print(f"  paper: {comparison['paper']:.6e}")
    print(f"  sign_match: {comparison['sign_match']}")

    # Sign should FLIP
    assert not comparison['sign_match'], "(1,3) should flip sign in paper regime"


# =============================================================================
# Test: (2,2) microcase - magnitude attenuation
# =============================================================================


def test_unified_paper_22_matches_dsl_paper(kappa_polys):
    """
    PHASE 29.1b: Validate (2,2) shows ~4x attenuation in paper regime.
    """
    R = 1.3036
    theta = 4 / 7

    # Compute using unified_paper
    paper_result = compute_I1_unified_paper(
        R, theta, 2, 2, kappa_polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )

    # Compute using OLD DSL with paper regime
    term = make_I1_22(theta, R, kernel_regime="paper")
    dsl_result = evaluate_term(term, kappa_polys, n=60, R=R, theta=theta)

    rel_diff = abs(paper_result.I1_value - dsl_result.value) / max(abs(dsl_result.value), 1e-15)

    print(f"\n(2,2) Paper Regime Comparison:")
    print(f"  unified_paper: {paper_result.I1_value:.10e}")
    print(f"  OLD DSL paper: {dsl_result.value:.10e}")
    print(f"  relative diff: {rel_diff:.2e}")

    assert rel_diff < 1e-4, f"(2,2) mismatch: rel_diff={rel_diff:.2e}"


def test_paper_22_shows_4x_attenuation(kappa_polys):
    """Verify (2,2) shrinks by ~4x in paper vs raw regime."""
    R = 1.3036
    theta = 4 / 7

    comparison = compare_raw_vs_paper(R, theta, 2, 2, kappa_polys, n_quad=60)

    print(f"\n(2,2) Raw vs Paper:")
    print(f"  raw:   {comparison['raw']:.6e}")
    print(f"  paper: {comparison['paper']:.6e}")
    print(f"  ratio (paper/raw): {comparison['ratio']:.4f}")

    # Ratio should be ~0.24 (4x attenuation)
    assert 0.1 < comparison['ratio'] < 0.5, f"(2,2) attenuation ratio out of range: {comparison['ratio']}"
    assert comparison['sign_match'], "(2,2) should NOT flip sign"


# =============================================================================
# Test: (2,3) - another sign flip case
# =============================================================================


def test_paper_23_flips_sign_vs_raw(kappa_polys):
    """Verify (2,3) flips sign in paper vs raw regime."""
    R = 1.3036
    theta = 4 / 7

    comparison = compare_raw_vs_paper(R, theta, 2, 3, kappa_polys, n_quad=60)

    print(f"\n(2,3) Raw vs Paper:")
    print(f"  raw:   {comparison['raw']:.6e}")
    print(f"  paper: {comparison['paper']:.6e}")
    print(f"  sign_match: {comparison['sign_match']}")

    # Sign should FLIP
    assert not comparison['sign_match'], "(2,3) should flip sign in paper regime"


# =============================================================================
# Test: All 6 triangle pairs validation
# =============================================================================


@pytest.mark.parametrize("ell1,ell2", [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
])
def test_unified_paper_matches_dsl_paper_all_pairs(kappa_polys, ell1, ell2):
    """Validate unified_paper matches OLD DSL paper for all pairs."""
    R = 1.3036
    theta = 4 / 7

    # Compute using unified_paper
    paper_result = compute_I1_unified_paper(
        R, theta, ell1, ell2, kappa_polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )

    # Get the make function for OLD DSL
    make_fns = {
        (1, 1): make_I1_11,
        (1, 2): make_I1_12,
        (1, 3): make_I1_13,
        (2, 2): make_I1_22,
        (2, 3): make_I1_23,
        (3, 3): make_I1_33,
    }

    make_fn = make_fns[(ell1, ell2)]
    term = make_fn(theta, R, kernel_regime="paper")
    dsl_result = evaluate_term(term, kappa_polys, n=60, R=R, theta=theta)

    rel_diff = abs(paper_result.I1_value - dsl_result.value) / max(abs(dsl_result.value), 1e-15)

    print(f"\n({ell1},{ell2}) Paper Regime:")
    print(f"  unified_paper: {paper_result.I1_value:.6e}")
    print(f"  OLD DSL paper: {dsl_result.value:.6e}")
    print(f"  relative diff: {rel_diff:.2e}")

    # Allow slightly larger tolerance for higher-order pairs
    tol = 1e-3 if (ell1 == 3 or ell2 == 3) else 1e-4
    assert rel_diff < tol, f"({ell1},{ell2}) mismatch: rel_diff={rel_diff:.2e}"


# =============================================================================
# Test: kappa* benchmark
# =============================================================================


@pytest.mark.parametrize("ell1,ell2", [
    (1, 1),
    (2, 2),
    (1, 3),
])
def test_unified_paper_kappa_star_subset(kappa_star_polys, ell1, ell2):
    """Validate unified_paper on kappa* benchmark (subset of pairs)."""
    R = 1.1167
    theta = 4 / 7

    paper_result = compute_I1_unified_paper(
        R, theta, ell1, ell2, kappa_star_polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )

    # Get OLD DSL comparison
    make_fns = {
        (1, 1): make_I1_11,
        (1, 3): make_I1_13,
        (2, 2): make_I1_22,
    }

    make_fn = make_fns[(ell1, ell2)]
    term = make_fn(theta, R, kernel_regime="paper")
    dsl_result = evaluate_term(term, kappa_star_polys, n=60, R=R, theta=theta)

    rel_diff = abs(paper_result.I1_value - dsl_result.value) / max(abs(dsl_result.value), 1e-15)

    print(f"\nkappa* ({ell1},{ell2}) Paper Regime:")
    print(f"  unified_paper: {paper_result.I1_value:.6e}")
    print(f"  OLD DSL paper: {dsl_result.value:.6e}")
    print(f"  relative diff: {rel_diff:.2e}")

    tol = 1e-3
    assert rel_diff < tol, f"kappa* ({ell1},{ell2}) mismatch: rel_diff={rel_diff:.2e}"


# =============================================================================
# Test: omega mapping
# =============================================================================


def test_omega_mapping():
    """Verify omega = ell - 1 mapping."""
    from src.unified_i1_paper import omega_for_ell

    assert omega_for_ell(1) == 0, "P1 should use omega=0 (Case B)"
    assert omega_for_ell(2) == 1, "P2 should use omega=1 (Case C)"
    assert omega_for_ell(3) == 2, "P3 should use omega=2 (Case C)"


# =============================================================================
# Test: I2 paper regime validation
# =============================================================================


@pytest.mark.parametrize("ell1,ell2", [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
])
def test_unified_paper_I2_matches_dsl_paper(kappa_polys, ell1, ell2):
    """Validate unified I2 paper matches OLD DSL paper for all pairs."""
    R = 1.3036
    theta = 4 / 7

    # Compute using unified_paper
    paper_result = compute_I2_unified_paper(
        R, theta, ell1, ell2, kappa_polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40,
        include_Q=True,
    )

    # Get the make function for OLD DSL
    make_fns = {
        (1, 1): make_I2_11,
        (1, 2): make_I2_12,
        (1, 3): make_I2_13,
        (2, 2): make_I2_22,
        (2, 3): make_I2_23,
        (3, 3): make_I2_33,
    }

    make_fn = make_fns[(ell1, ell2)]
    term = make_fn(theta, R, kernel_regime="paper")
    dsl_result = evaluate_term(term, kappa_polys, n=60, R=R, theta=theta)

    rel_diff = abs(paper_result.I2_value - dsl_result.value) / max(abs(dsl_result.value), 1e-15)

    print(f"\nI2({ell1},{ell2}) Paper Regime:")
    print(f"  unified_paper: {paper_result.I2_value:.6e}")
    print(f"  OLD DSL paper: {dsl_result.value:.6e}")
    print(f"  relative diff: {rel_diff:.2e}")

    tol = 1e-3 if (ell1 == 3 or ell2 == 3) else 1e-4
    assert rel_diff < tol, f"I2({ell1},{ell2}) mismatch: rel_diff={rel_diff:.2e}"


def test_I2_22_paper_shows_attenuation(kappa_polys):
    """Verify I2(2,2) shows attenuation in paper vs raw regime."""
    R = 1.3036
    theta = 4 / 7

    comparison = compare_I2_raw_vs_paper(R, theta, 2, 2, kappa_polys, n_quad=60)

    print(f"\nI2(2,2) Raw vs Paper:")
    print(f"  raw:   {comparison['raw']:.6e}")
    print(f"  paper: {comparison['paper']:.6e}")
    print(f"  ratio (paper/raw): {comparison['ratio']:.4f}")

    # I2 should also show attenuation for (2,2)
    assert 0.01 < comparison['ratio'] < 1.0, f"I2(2,2) attenuation unexpected: {comparison['ratio']}"


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
