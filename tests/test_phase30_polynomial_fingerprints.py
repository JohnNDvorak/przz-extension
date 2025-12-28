#!/usr/bin/env python3
"""
tests/test_phase30_polynomial_fingerprints.py
Phase 30.2: Polynomial Fingerprint Validation

Validates that:
1. κ and κ* polynomials have different fingerprints
2. Fingerprints are stable (regression anchors)
3. Polynomial degrees match expectations

Created: 2025-12-26 (Phase 30)
"""

import pytest
import sys
import hashlib

sys.path.insert(0, ".")

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)


# =============================================================================
# Fingerprint utilities
# =============================================================================


def get_poly_degree(poly) -> int:
    """Get the degree of a polynomial."""
    if hasattr(poly, 'degree'):
        return poly.degree
    if hasattr(poly, 'tilde_coeffs'):
        return len(poly.tilde_coeffs)
    if hasattr(poly, 'coeffs'):
        return len(poly.coeffs) - 1
    if hasattr(poly, 'coeffs_in_basis_terms'):
        # Degree is 2*(number of terms - 1) + 1 for odd Legendre-like basis
        return 2 * (len(poly.coeffs_in_basis_terms) - 1) + 1
    return -1


def get_poly_coeffs_for_fingerprint(poly) -> str:
    """Get polynomial coefficients as a string for fingerprinting."""
    if hasattr(poly, 'tilde_coeffs'):
        coeffs = poly.tilde_coeffs
    elif hasattr(poly, 'coeffs'):
        coeffs = poly.coeffs
    elif hasattr(poly, 'coeffs_in_basis_terms'):
        coeffs = poly.coeffs_in_basis_terms
    else:
        coeffs = []
    # Round to 8 decimal places for stability
    return str([round(float(c), 8) for c in coeffs])


def compute_poly_fingerprint(polys: dict) -> str:
    """Compute stable fingerprint from polynomial coefficients."""
    data = []
    for key in sorted(polys.keys()):
        coeffs = get_poly_coeffs_for_fingerprint(polys[key])
        data.append(f"{key}:{coeffs}")
    return hashlib.sha256("|".join(data).encode()).hexdigest()[:16]


# =============================================================================
# Fixtures
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
# Tests: Fingerprint distinctness
# =============================================================================


def test_kappa_and_kappa_star_have_different_fingerprints(kappa_polys, kappa_star_polys):
    """κ and κ* polynomials must have different fingerprints."""
    fp_k = compute_poly_fingerprint(kappa_polys)
    fp_ks = compute_poly_fingerprint(kappa_star_polys)

    print(f"\nFingerprints:")
    print(f"  κ:  {fp_k}")
    print(f"  κ*: {fp_ks}")

    assert fp_k != fp_ks, (
        "κ and κ* should have different polynomial fingerprints! "
        "If they match, the wrong polynomial loader is being used."
    )


# =============================================================================
# Tests: Polynomial degree expectations
# =============================================================================


def test_kappa_P2_degree_is_3(kappa_polys):
    """κ benchmark P2 should have degree 3."""
    degree = get_poly_degree(kappa_polys["P2"])
    assert degree == 3, f"κ P2 degree should be 3, got {degree}"


def test_kappa_P3_degree_is_3(kappa_polys):
    """κ benchmark P3 should have degree 3."""
    degree = get_poly_degree(kappa_polys["P3"])
    assert degree == 3, f"κ P3 degree should be 3, got {degree}"


def test_kappa_star_P2_degree_is_2(kappa_star_polys):
    """κ* benchmark P2 should have degree 2 (simpler structure)."""
    degree = get_poly_degree(kappa_star_polys["P2"])
    assert degree == 2, f"κ* P2 degree should be 2, got {degree}"


def test_kappa_star_P3_degree_is_2(kappa_star_polys):
    """κ* benchmark P3 should have degree 2 (simpler structure)."""
    degree = get_poly_degree(kappa_star_polys["P3"])
    assert degree == 2, f"κ* P3 degree should be 2, got {degree}"


def test_kappa_vs_kappa_star_polynomial_degrees_differ():
    """κ and κ* have different polynomial degrees (expected design)."""
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials()
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()

    # P2 and P3 should have different degrees
    assert get_poly_degree(P2_k) != get_poly_degree(P2_ks), (
        "P2 degrees should differ between κ and κ*"
    )
    assert get_poly_degree(P3_k) != get_poly_degree(P3_ks), (
        "P3 degrees should differ between κ and κ*"
    )

    print(f"\nPolynomial degree comparison:")
    print(f"  P2: κ={get_poly_degree(P2_k)}, κ*={get_poly_degree(P2_ks)}")
    print(f"  P3: κ={get_poly_degree(P3_k)}, κ*={get_poly_degree(P3_ks)}")


# =============================================================================
# Tests: Fingerprint stability (regression anchors)
# =============================================================================

# These fingerprints were computed on 2025-12-26 and should remain stable
EXPECTED_KAPPA_FINGERPRINT = "175bffb87a77b2d8"
EXPECTED_KAPPA_STAR_FINGERPRINT = "ce0e7f4edf7d4971"


def test_kappa_fingerprint_is_stable(kappa_polys):
    """κ fingerprint should match known stable value."""
    fp = compute_poly_fingerprint(kappa_polys)
    print(f"\nκ fingerprint: {fp}")
    assert fp == EXPECTED_KAPPA_FINGERPRINT, (
        f"κ fingerprint changed! Expected {EXPECTED_KAPPA_FINGERPRINT}, got {fp}. "
        "This may indicate polynomial coefficient changes."
    )


def test_kappa_star_fingerprint_is_stable(kappa_star_polys):
    """κ* fingerprint should match known stable value."""
    fp = compute_poly_fingerprint(kappa_star_polys)
    print(f"\nκ* fingerprint: {fp}")
    assert fp == EXPECTED_KAPPA_STAR_FINGERPRINT, (
        f"κ* fingerprint changed! Expected {EXPECTED_KAPPA_STAR_FINGERPRINT}, got {fp}. "
        "This may indicate polynomial coefficient changes."
    )


# =============================================================================
# Tests: Coefficient sanity
# =============================================================================


def test_kappa_P1_boundary_conditions(kappa_polys):
    """P1 should satisfy P1(0) = 0, P1(1) = 1 (boundary constraints)."""
    P1 = kappa_polys["P1"]
    # Boundary conditions from PRZZ
    P1_0 = float(P1.eval(0))
    P1_1 = float(P1.eval(1))
    assert abs(P1_0) < 1e-10, f"P1(0) should be 0, got {P1_0}"
    assert abs(P1_1 - 1.0) < 1e-10, f"P1(1) should be 1, got {P1_1}"


def test_kappa_star_P1_boundary_conditions(kappa_star_polys):
    """P1 should satisfy P1(0) = 0, P1(1) = 1 (boundary constraints)."""
    P1 = kappa_star_polys["P1"]
    P1_0 = float(P1.eval(0))
    P1_1 = float(P1.eval(1))
    assert abs(P1_0) < 1e-10, f"P1(0) should be 0, got {P1_0}"
    assert abs(P1_1 - 1.0) < 1e-10, f"P1(1) should be 1, got {P1_1}"


def test_kappa_Q_normalized(kappa_polys):
    """Q(0) should be close to 1 (normalization constraint)."""
    Q = kappa_polys["Q"]
    Q0 = float(Q.eval(0))
    # PRZZ printed coefficients sum to ~0.999999, not exactly 1.0
    assert abs(Q0 - 1.0) < 0.001, f"Q(0) should be ≈1.0, got {Q0}"


def test_kappa_star_Q_normalized(kappa_star_polys):
    """Q(0) should be close to 1 (normalization constraint)."""
    Q = kappa_star_polys["Q"]
    Q0 = float(Q.eval(0))
    assert abs(Q0 - 1.0) < 0.001, f"Q(0) should be ≈1.0, got {Q0}"


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
