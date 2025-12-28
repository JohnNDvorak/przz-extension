"""
tests/test_mirror_transform_microcases.py
Phase 27: Mirror Transform Microcase Tests

Tests for the derived mirror transform implementation using P=Q=1 oracles
and comparison against -R proxy approach.

Created: 2025-12-26 (Phase 27)
"""

import pytest
import math

from src.mirror_transform_derived import (
    compute_I1_mirror_derived,
    compute_I2_mirror_derived,
    compute_I1_mirror_P1Q1,
    compare_mirror_to_proxy,
)
from src.unified_i1_general import (
    compute_I1_unified_general,
    compute_I1_unified_general_P1Q1,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def kappa_polynomials():
    """Load kappa benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_star_polynomials():
    """Load kappa* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_params():
    """Parameters for kappa benchmark."""
    return {"R": 1.3036, "theta": 4 / 7}


@pytest.fixture
def kappa_star_params():
    """Parameters for kappa* benchmark."""
    return {"R": 1.1167, "theta": 4 / 7}


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestBasicMirrorFunctionality:
    """Test that mirror functions run without error."""

    def test_I1_mirror_11_finite(self, kappa_polynomials, kappa_params):
        """Mirror I₁(1,1) should be finite."""
        result = compute_I1_mirror_derived(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            n_quad_u=40,
            n_quad_t=40,
        )
        assert math.isfinite(result.I1_mirror_value)

    def test_I2_mirror_11_finite(self, kappa_polynomials, kappa_params):
        """Mirror I₂(1,1) should be finite."""
        result = compute_I2_mirror_derived(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            n_quad_u=40,
            n_quad_t=40,
        )
        assert math.isfinite(result.I2_mirror_value)

    def test_I1_mirror_P1Q1_11_finite(self, kappa_params):
        """Mirror I₁(1,1) with P=Q=1 should be finite."""
        result = compute_I1_mirror_P1Q1(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            n_quad_u=40,
            n_quad_t=40,
        )
        assert math.isfinite(result)

    @pytest.mark.parametrize("pair_key", ["11", "22", "33", "12", "13", "23"])
    def test_I1_mirror_all_pairs_finite(self, kappa_polynomials, kappa_params, pair_key):
        """All mirror I₁ values should be finite."""
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])
        result = compute_I1_mirror_derived(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=ell1,
            ell2=ell2,
            polynomials=kappa_polynomials,
            n_quad_u=40,
            n_quad_t=40,
        )
        assert math.isfinite(result.I1_mirror_value), f"Mirror I1({ell1},{ell2}) not finite"


# =============================================================================
# P=Q=1 MICROCASE TESTS
# =============================================================================

class TestP1Q1MicrocaseOracle:
    """Test P=Q=1 microcase for direct vs mirror comparison."""

    def test_direct_vs_mirror_P1Q1_sign_relation(self, kappa_params):
        """
        With P=Q=1, test the relationship between direct and mirror I₁.

        The mirror should have a specific sign/magnitude relationship
        to the direct term based on the exponential sign flip.
        """
        R, theta = kappa_params["R"], kappa_params["theta"]

        # Direct P=Q=1
        direct = compute_I1_unified_general_P1Q1(
            R=R, theta=theta, ell1=1, ell2=1, n_quad_u=60, n_quad_t=60,
        )

        # Mirror P=Q=1
        mirror = compute_I1_mirror_P1Q1(
            R=R, theta=theta, ell1=1, ell2=1, n_quad_u=60, n_quad_t=60,
        )

        # Both should be finite
        assert math.isfinite(direct)
        assert math.isfinite(mirror)

        # Log for diagnostic
        print(f"\nP=Q=1 (1,1) comparison:")
        print(f"  Direct: {direct:.8e}")
        print(f"  Mirror: {mirror:.8e}")
        print(f"  Ratio (mirror/direct): {mirror/direct if direct != 0 else 'inf':.6f}")

    def test_P1Q1_mirror_different_from_direct(self, kappa_params):
        """Mirror P=Q=1 should be different from direct (sign flip matters)."""
        R, theta = kappa_params["R"], kappa_params["theta"]

        direct = compute_I1_unified_general_P1Q1(
            R=R, theta=theta, ell1=1, ell2=1, n_quad_u=60, n_quad_t=60,
        )
        mirror = compute_I1_mirror_P1Q1(
            R=R, theta=theta, ell1=1, ell2=1, n_quad_u=60, n_quad_t=60,
        )

        # Should not be equal (the exp sign flip changes the result)
        assert direct != mirror, "Mirror should differ from direct"


# =============================================================================
# COMPARISON TO PROXY TESTS
# =============================================================================

class TestMirrorVsProxy:
    """Compare derived mirror to the -R proxy approach."""

    def test_compare_11_reports_m_eff(self, kappa_polynomials, kappa_params):
        """Test that comparison returns m_eff for (1,1)."""
        result = compare_mirror_to_proxy(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            n_quad=40,
        )
        assert result.m_eff is not None or result.I1_proxy_minusR == 0

        # Log for diagnostic
        print(f"\n(1,1) Mirror comparison:")
        print(f"  Direct(+R): {result.I1_direct_plusR:.8e}")
        print(f"  Mirror derived: {result.I1_mirror_derived:.8e}")
        print(f"  Proxy(-R): {result.I1_proxy_minusR:.8e}")
        if result.m_eff is not None:
            print(f"  m_eff (mirror/proxy): {result.m_eff:.6f}")
        print(f"  m_empirical (exp(R)+5): {result.m_empirical:.6f}")

    @pytest.mark.parametrize("pair_key", ["11", "22", "33", "12", "13", "23"])
    def test_compare_all_pairs(self, kappa_polynomials, kappa_params, pair_key):
        """Compare derived mirror to proxy for all pairs."""
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])
        result = compare_mirror_to_proxy(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=ell1,
            ell2=ell2,
            polynomials=kappa_polynomials,
            n_quad=40,
        )

        # Just check finite values
        assert math.isfinite(result.I1_direct_plusR)
        assert math.isfinite(result.I1_mirror_derived)
        assert math.isfinite(result.I1_proxy_minusR)


# =============================================================================
# DIAGNOSTIC OUTPUT TEST (run with -s flag)
# =============================================================================

class TestDiagnosticOutput:
    """Verbose diagnostic tests - run with pytest -s to see output."""

    def test_full_mirror_diagnostic(self, kappa_polynomials, kappa_params):
        """Print full diagnostic table for all pairs."""
        R, theta = kappa_params["R"], kappa_params["theta"]

        print("\n" + "=" * 80)
        print("MIRROR TRANSFORM DIAGNOSTIC - KAPPA BENCHMARK (R=1.3036)")
        print("=" * 80)
        print(f"\nEmpirical m = exp(R) + 5 = {math.exp(R) + 5:.6f}")
        print(f"exp(2R) = {math.exp(2*R):.6f}")
        print(f"exp(2R/theta) = {math.exp(2*R/theta):.6f}")

        print("\n" + "-" * 80)
        print(f"{'Pair':<6} {'Direct(+R)':<14} {'Mirror':<14} {'Proxy(-R)':<14} {'m_eff':<10}")
        print("-" * 80)

        for pair in ["11", "22", "33", "12", "13", "23"]:
            ell1, ell2 = int(pair[0]), int(pair[1])
            result = compare_mirror_to_proxy(
                R=R, theta=theta, ell1=ell1, ell2=ell2,
                polynomials=kappa_polynomials, n_quad=60,
            )

            m_str = f"{result.m_eff:.4f}" if result.m_eff is not None else "N/A"
            print(f"({ell1},{ell2})  {result.I1_direct_plusR:>12.6e}  "
                  f"{result.I1_mirror_derived:>12.6e}  "
                  f"{result.I1_proxy_minusR:>12.6e}  {m_str}")

        print("-" * 80)
        print("Note: m_eff = mirror_derived / proxy(-R)")
        print("=" * 80)


# =============================================================================
# STABILITY TESTS
# =============================================================================

class TestQuadratureStability:
    """Test stability under quadrature refinement."""

    def test_mirror_11_stable_under_refinement(self, kappa_polynomials, kappa_params):
        """Mirror I₁(1,1) should stabilize with quadrature refinement."""
        results = []
        for n in [40, 60, 80]:
            result = compute_I1_mirror_derived(
                R=kappa_params["R"],
                theta=kappa_params["theta"],
                ell1=1,
                ell2=1,
                polynomials=kappa_polynomials,
                n_quad_u=n,
                n_quad_t=n,
            )
            results.append(result.I1_mirror_value)

        # Check convergence (60 vs 80 should be close)
        if results[1] != 0:
            rel_change = abs(results[2] - results[1]) / abs(results[1])
            assert rel_change < 0.01, f"Mirror I₁(1,1) not converging: {rel_change:.2%}"
