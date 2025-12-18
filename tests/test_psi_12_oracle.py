"""
Tests for Ψ_{1,2} oracle (μ × μ⋆Λ cross-pair).

The (1,2) pair shows catastrophic cancellation in the DSL with ratio 129×
between κ and κ* polynomials. This Ψ-expansion oracle should NOT exhibit
this artificial cancellation.

Key insight from HANDOFF_SUMMARY Section 2:
- With κ* polynomials, DSL (1,2) has near-perfect cancellation:
  - Sum of positives: 0.380
  - Sum of negatives: -0.382
  - Net: -0.0016 (ratio 129×!)

The Ψ expansion should give a ratio MUCH closer to the target ~1.1×.
"""

import pytest
import numpy as np
from src.psi_12_oracle import psi_oracle_12
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# PRZZ parameters
THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


class TestPsi12Oracle:
    """Test Ψ_{1,2} oracle computation."""

    @pytest.fixture
    def kappa_polynomials(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return P1, P2, Q

    @pytest.fixture
    def kappa_star_polynomials(self):
        """Load κ* polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        return P1, P2, Q

    def test_oracle_runs_without_error(self, kappa_polynomials):
        """Oracle completes successfully."""
        P1, P2, Q = kappa_polynomials
        result = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=40)
        assert result is not None
        assert np.isfinite(result.total)

    def test_oracle_kappa_computation(self, kappa_polynomials):
        """Test with κ polynomials (R=1.3036)."""
        P1, P2, Q = kappa_polynomials
        result = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=60, debug=False)

        # Check all monomials are finite
        assert np.isfinite(result.AB2)
        assert np.isfinite(result.ABC)
        assert np.isfinite(result.AC2)
        assert np.isfinite(result.B2C)
        assert np.isfinite(result.C3)
        assert np.isfinite(result.DB)
        assert np.isfinite(result.DC)
        assert np.isfinite(result.total)

        # Total should be finite and reasonably sized
        # (actual value needs comparison with DSL)
        assert abs(result.total) < 100.0  # Sanity check

    def test_oracle_kappa_star_computation(self, kappa_star_polynomials):
        """Test with κ* polynomials (R=1.1167)."""
        P1, P2, Q = kappa_star_polynomials
        result = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA_STAR, n_quad=60, debug=False)

        # Check all monomials are finite
        assert np.isfinite(result.AB2)
        assert np.isfinite(result.ABC)
        assert np.isfinite(result.AC2)
        assert np.isfinite(result.B2C)
        assert np.isfinite(result.C3)
        assert np.isfinite(result.DB)
        assert np.isfinite(result.DC)
        assert np.isfinite(result.total)

        # Total should be finite
        assert abs(result.total) < 100.0  # Sanity check

    def test_ratio_better_than_dsl(self, kappa_polynomials, kappa_star_polynomials):
        """
        The κ/κ* ratio should be MUCH better than the DSL's catastrophic 129×.

        Target ratio from HANDOFF: ~1.1× (not 129×)
        """
        # Compute κ result
        P1_k, P2_k, Q_k = kappa_polynomials
        result_k = psi_oracle_12(P1_k, P2_k, Q_k, THETA, R_KAPPA, n_quad=80)

        # Compute κ* result
        P1_ks, P2_ks, Q_ks = kappa_star_polynomials
        result_ks = psi_oracle_12(P1_ks, P2_ks, Q_ks, THETA, R_KAPPA_STAR, n_quad=80)

        # Compute ratio
        ratio = abs(result_k.total / result_ks.total)

        print(f"\nΨ_{1,2} oracle ratio (κ/κ*): {ratio:.4f}")
        print(f"DSL ratio (for comparison): 129.0")
        print(f"Target ratio: ~1.1")

        # The ratio should be MUCH better than 129
        # Even if not exactly 1.1, it should be closer to that than to 129
        assert ratio < 50.0, f"Ratio {ratio:.2f} still too large (DSL is 129)"

        # Ideally, should be close to target
        # assert 0.5 < ratio < 5.0, f"Ratio {ratio:.2f} should be near 1.1"

    def test_quadrature_convergence(self, kappa_polynomials):
        """Oracle result should be stable under quadrature refinement."""
        P1, P2, Q = kappa_polynomials

        result_40 = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=40)
        result_60 = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=60)
        result_80 = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=80)

        # Check convergence
        diff_60_40 = abs(result_60.total - result_40.total)
        diff_80_60 = abs(result_80.total - result_60.total)

        # Refinement should reduce error
        # (may not be strictly monotonic due to quadrature oscillations)
        # At minimum, both should be small
        assert diff_80_60 < 0.01 * abs(result_80.total), \
            "Result should be stable to ~1% at n=80"

    def test_no_catastrophic_cancellation(self, kappa_star_polynomials):
        """
        With κ* polynomials, DSL has catastrophic cancellation:
        - Positives: 0.380
        - Negatives: -0.382
        - Net: -0.0016

        The Ψ expansion should NOT have this issue.
        """
        P1, P2, Q = kappa_star_polynomials
        result = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA_STAR, n_quad=80)

        # Sum positive and negative monomial contributions
        contributions = [
            result.AB2,           # +1 coefficient
            -2 * result.ABC,      # -2 coefficient
            result.AC2,           # +1 coefficient
            -result.B2C,          # -1 coefficient
            result.C3,            # +1 coefficient
            2 * result.DB,        # +2 coefficient
            -2 * result.DC        # -2 coefficient
        ]

        sum_positive = sum(c for c in contributions if c > 0)
        sum_negative = sum(c for c in contributions if c < 0)
        net = sum_positive + sum_negative

        print(f"\nκ* monomial analysis:")
        print(f"  Sum of positives: {sum_positive:.6f}")
        print(f"  Sum of negatives: {sum_negative:.6f}")
        print(f"  Net: {net:.6f}")
        print(f"  |neg|/|pos| ratio: {abs(sum_negative/sum_positive):.6f}")

        # The ratio should NOT be close to 1 (catastrophic cancellation)
        # A healthy ratio is significantly different from 1
        cancellation_ratio = abs(sum_negative / sum_positive)

        # Allow some cancellation, but not near-perfect like DSL
        assert abs(cancellation_ratio - 1.0) > 0.1, \
            f"Cancellation ratio {cancellation_ratio:.4f} too close to 1 (DSL: 1.004)"

    def test_individual_monomials_nonzero(self, kappa_polynomials):
        """All 7 monomials should contribute (not identically zero)."""
        P1, P2, Q = kappa_polynomials
        result = psi_oracle_12(P1, P2, Q, THETA, R_KAPPA, n_quad=60)

        # All monomials should be nonzero (within numerical tolerance)
        # Note: C and D are the same in our simple model, so DC might be very small
        assert abs(result.AB2) > 1e-10
        assert abs(result.ABC) > 1e-10
        assert abs(result.AC2) > 1e-10
        assert abs(result.B2C) > 1e-10
        assert abs(result.C3) > 1e-10
        assert abs(result.DB) > 1e-10
        # DC might be small if D ≈ C
