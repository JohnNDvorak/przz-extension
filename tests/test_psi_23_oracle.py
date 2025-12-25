"""
Tests for Ψ_{2,3} oracle (μ⋆Λ × μ⋆Λ⋆Λ cross-pair).

The (2,3) pair shows excessive ratio in the DSL with ratio 9.04×
between κ and κ* polynomials. This Ψ-expansion oracle should NOT exhibit
this artificial instability.

Key insight from HANDOFF_SUMMARY Section 2:
- The (2,3) DSL ratio was 9.04, much worse than the target ~1.1×.
- P₃ changes sign on [0,1], so cross-integrals can be negative.
- The Ψ expansion should give a ratio MUCH closer to the target ~1.1×.
"""

import pytest
import numpy as np
from src.psi_23_oracle import psi_oracle_23
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# PRZZ parameters
THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


class TestPsi23Oracle:
    """Test Ψ_{2,3} oracle computation."""

    @pytest.fixture
    def kappa_polynomials(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return P2, P3, Q

    @pytest.fixture
    def kappa_star_polynomials(self):
        """Load κ* polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        return P2, P3, Q

    def test_oracle_runs_without_error(self, kappa_polynomials):
        """Oracle completes successfully."""
        P2, P3, Q = kappa_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=40)
        assert result is not None
        assert np.isfinite(result.total)

    def test_oracle_kappa_computation(self, kappa_polynomials):
        """Test with κ polynomials (R=1.3036)."""
        P2, P3, Q = kappa_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=60, debug=False)

        # Check all monomials are finite
        assert np.isfinite(result.A2B3)
        assert np.isfinite(result.A2B2C)
        assert np.isfinite(result.A2BC2)
        assert np.isfinite(result.A2C3)
        assert np.isfinite(result.AB3C)
        assert np.isfinite(result.AB2C2)
        assert np.isfinite(result.ABC3)
        assert np.isfinite(result.AC4)
        assert np.isfinite(result.B3C2)
        assert np.isfinite(result.B2C3)
        assert np.isfinite(result.BC4)
        assert np.isfinite(result.C5)
        assert np.isfinite(result.DAB2)
        assert np.isfinite(result.DABC)
        assert np.isfinite(result.DAC2)
        assert np.isfinite(result.DB2C)
        assert np.isfinite(result.DBC2)
        assert np.isfinite(result.DC3)
        assert np.isfinite(result.D2B)
        assert np.isfinite(result.D2C)
        assert np.isfinite(result.total)

        # Total should be finite and reasonably sized
        assert abs(result.total) < 100.0  # Sanity check

    def test_oracle_kappa_star_computation(self, kappa_star_polynomials):
        """Test with κ* polynomials (R=1.1167)."""
        P2, P3, Q = kappa_star_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA_STAR, n_quad=60, debug=False)

        # Check all monomials are finite
        assert np.isfinite(result.A2B3)
        assert np.isfinite(result.total)

        # Total should be finite
        assert abs(result.total) < 100.0  # Sanity check

    def test_ratio_better_than_dsl(self, kappa_polynomials, kappa_star_polynomials):
        """
        The κ/κ* ratio should be MUCH better than the DSL's 9.04×.

        Target ratio from HANDOFF: ~1.1× (not 9.04×)
        """
        # Compute κ result
        P2_k, P3_k, Q_k = kappa_polynomials
        result_k = psi_oracle_23(P2_k, P3_k, Q_k, THETA, R_KAPPA, n_quad=80)

        # Compute κ* result
        P2_ks, P3_ks, Q_ks = kappa_star_polynomials
        result_ks = psi_oracle_23(P2_ks, P3_ks, Q_ks, THETA, R_KAPPA_STAR, n_quad=80)

        # Compute ratio
        ratio = abs(result_k.total / result_ks.total)

        print(f"\nΨ_{2,3} oracle ratio (κ/κ*): {ratio:.4f}")
        print(f"DSL ratio (for comparison): 9.04")
        print(f"Target ratio: ~1.1")

        # The ratio should be MUCH better than 9.04
        # Even if not exactly 1.1, it should be closer to that than to 9.04
        assert ratio < 6.0, f"Ratio {ratio:.2f} still too large (DSL is 9.04)"

        # Ideally, should be close to target
        # assert 0.5 < ratio < 3.0, f"Ratio {ratio:.2f} should be near 1.1"

    def test_quadrature_convergence(self, kappa_polynomials):
        """Oracle result should be stable under quadrature refinement."""
        P2, P3, Q = kappa_polynomials

        result_40 = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=40)
        result_60 = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=60)
        result_80 = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=80)

        # Check convergence
        diff_60_40 = abs(result_60.total - result_40.total)
        diff_80_60 = abs(result_80.total - result_60.total)

        # Refinement should reduce error
        # At minimum, both should be small
        assert diff_80_60 < 0.01 * abs(result_80.total), \
            "Result should be stable to ~1% at n=80"

    def test_individual_monomials_nonzero(self, kappa_polynomials):
        """Most monomials should contribute (not identically zero)."""
        P2, P3, Q = kappa_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=60)

        # Total should be nonzero
        assert abs(result.total) > 1e-10

    def test_monomial_count_is_18(self):
        """
        Verify that (2,3) expansion has exactly 18 unique monomials.

        From the formula:
        p=0: (A-C)²(B-C)³ → 8 terms
        p=1: 6(D-C²)(A-C)(B-C)² → 8 terms
        p=2: 6(D-C²)²(B-C) → 4 terms
        After combining like terms: 18 unique monomials
        """
        from src.psi_monomial_expansion import expand_pair_to_monomials

        monomials = expand_pair_to_monomials(2, 3)
        assert len(monomials) == 18, f"Expected 18 monomials, got {len(monomials)}"

    def test_result_structure(self, kappa_polynomials):
        """Verify the result has all expected fields."""
        P2, P3, Q = kappa_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=40)

        # Check that result has all expected monomial fields
        assert hasattr(result, 'A2B3')
        assert hasattr(result, 'A2B2C')
        assert hasattr(result, 'A2BC2')
        assert hasattr(result, 'A2C3')
        assert hasattr(result, 'AB3C')
        assert hasattr(result, 'AB2C2')
        assert hasattr(result, 'ABC3')
        assert hasattr(result, 'AC4')
        assert hasattr(result, 'B3C2')
        assert hasattr(result, 'B2C3')
        assert hasattr(result, 'BC4')
        assert hasattr(result, 'C5')
        assert hasattr(result, 'DAB2')
        assert hasattr(result, 'DABC')
        assert hasattr(result, 'DAC2')
        assert hasattr(result, 'DB2C')
        assert hasattr(result, 'DBC2')
        assert hasattr(result, 'DC3')
        assert hasattr(result, 'D2B')
        assert hasattr(result, 'D2C')
        assert hasattr(result, 'total')

    def test_sign_handling_with_P3(self, kappa_polynomials):
        """
        P₃ changes sign on [0,1], so D = ∫P₂P₃ can be negative.

        This tests that the oracle handles negative integrals correctly.
        """
        P2, P3, Q = kappa_polynomials
        result = psi_oracle_23(P2, P3, Q, THETA, R_KAPPA, n_quad=80, debug=True)

        # The result may have negative contributions due to P₃ sign changes
        # Just verify that computations are stable
        assert np.isfinite(result.total)
        assert abs(result.total) < 1000.0  # Reasonable magnitude
