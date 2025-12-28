"""
tests/test_g_i2_weighted_q1_gate.py
Phase 46.2: First-Principles g_I2 Derivation Tests

DERIVATION PRINCIPLE:
====================

I2 has no log factor, so it lacks internal cross-term correction.
Therefore g_I2 = g_baseline = 1 + θ/(2K(2K+1)) BY CONSTRUCTION.

The u-moment under I2's kernel is computed for diagnostics but is NOT
the Beta(2,2K) moment and should NOT be used for g_I2 derivation.

Q=1 GATE:
=========

When Q=1, g_I2 = g_baseline exactly (by definition).

Created: 2025-12-27 (Phase 46.2)
Updated: 2025-12-27 - Fixed: g_I2 = g_baseline by construction
"""
import pytest
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star, Polynomial
from src.unified_s12.g_i2_weighted import (
    derive_g_i2_weighted,
    derive_g_values_weighted,
    compute_i2_kernel_integrals,
)


class TestQ1Gate:
    """Test that g_I2 = g_baseline by construction."""

    def get_q1_polynomials(self):
        """Load PRZZ polynomials but replace Q with Q=1."""
        P1, P2, P3, _ = load_przz_polynomials()
        Q_unity = Polynomial(np.array([1.0]))  # Q(x) = 1
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}

    def test_g_i2_equals_baseline_when_q1(self):
        """
        g_I2 = g_baseline by construction (I2 lacks log factor).

        This should ALWAYS pass because g_I2 is defined as g_baseline.
        """
        polys = self.get_q1_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_i2_weighted(R, theta, polys, K, n_quad=60)

        g_baseline = 1 + theta / (2 * K * (2 * K + 1))

        gap_pct = abs(result.g_I2 / g_baseline - 1) * 100

        print(f"\nQ=1 g_I2 derivation (by construction):")
        print(f"  g_I2 (derived): {result.g_I2:.8f}")
        print(f"  g_baseline: {g_baseline:.8f}")
        print(f"  Gap: {gap_pct:.4f}%")

        # Should be exactly 0% (g_I2 = g_baseline by construction)
        assert gap_pct < 0.01, (
            f"g_I2 should equal g_baseline by construction.\n"
            f"  g_I2: {result.g_I2:.8f}\n"
            f"  g_baseline: {g_baseline:.8f}\n"
            f"  Gap: {gap_pct:.4f}%"
        )

    def test_u_moment_is_diagnostic_only(self):
        """
        The u-moment is NOT the Beta(2, 2K) moment.

        This test documents that u-moment under I2's kernel weighting
        is fundamentally different from the Beta moment. The u-moment
        is computed for diagnostics but should NOT be used for derivation.
        """
        polys = self.get_q1_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_i2_weighted(R, theta, polys, K, n_quad=60)

        beta_moment = 1 / (2 * K * (2 * K + 1))

        print(f"\nU-moment diagnostic (NOT used for derivation):")
        print(f"  u_moment: {result.u_moment:.8f}")
        print(f"  Beta(2,2K): {beta_moment:.8f}")
        print(f"  Note: These are fundamentally different quantities")
        print(f"  u_moment is mean of u under P(u)² weighting")
        print(f"  Beta moment arises from (1-u)^{{K-1}} coefficient extraction")

        # Document that u-moment is NOT close to Beta moment
        # This is expected behavior, not a failure
        assert result.u_moment > 0.5, "u_moment should be around 0.7-0.8"
        assert beta_moment < 0.1, "Beta moment should be around 0.024"


class TestRealQDerivation:
    """Test g_I2 derivation with real Q polynomial."""

    def get_real_q_polynomials(self):
        """Load full PRZZ polynomials including real Q."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_g_i2_equals_baseline_with_real_q(self):
        """
        g_I2 = g_baseline by construction, regardless of Q.

        The first-principles derivation gives g_I2 = g_baseline.
        Q effects are NOT captured in this derivation.
        """
        polys = self.get_real_q_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_i2_weighted(R, theta, polys, K, n_quad=60)

        g_baseline = 1 + theta / (2 * K * (2 * K + 1))

        print(f"\nReal Q g_I2 derivation (by construction):")
        print(f"  g_I2 (derived): {result.g_I2:.8f}")
        print(f"  g_baseline: {g_baseline:.8f}")
        print(f"  Gap from baseline: {result.g_gap_from_baseline_pct:.4f}%")

        # g_I2 = g_baseline by construction
        assert result.g_I2 > 1.0, "g_I2 should be > 1"
        assert result.g_I2 < 1.1, "g_I2 should be < 1.1"
        assert abs(result.g_I2 - g_baseline) < 1e-10, "g_I2 should equal g_baseline"

    def test_g_i2_differs_from_calibrated_value(self):
        """
        The first-principles g_I2 = g_baseline differs from calibrated value.

        This documents the ~0.58% gap between g_baseline (1.0136) and
        g_I2_calibrated (1.0195). This gap represents Q polynomial
        differential attenuation that cannot be derived from first principles.
        """
        polys = self.get_real_q_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_i2_weighted(R, theta, polys, K, n_quad=60)

        # Calibrated value from 2-benchmark solve
        g_I2_calibrated = 1.01945154

        gap_pct = (result.g_I2 / g_I2_calibrated - 1) * 100

        print(f"\nComparison to calibrated g_I2:")
        print(f"  g_I2 (first-principles): {result.g_I2:.8f}")
        print(f"  g_I2 (calibrated): {g_I2_calibrated:.8f}")
        print(f"  Gap: {gap_pct:.2f}%")
        print(f"\n  Note: This ~0.6% gap is EXPECTED.")
        print(f"  Calibrated value includes Q polynomial effects")
        print(f"  that are not captured in first-principles derivation.")

        # Document that first-principles differs from calibrated
        # This is expected - not a failure
        assert gap_pct < 0, "First-principles should underestimate"
        assert abs(gap_pct) < 1, "Gap should be under 1%"


class TestBothComponentsWeighted:
    """Test complete g_I1 and g_I2 derivation."""

    def get_q1_polynomials(self):
        """Load PRZZ polynomials but replace Q with Q=1."""
        P1, P2, P3, _ = load_przz_polynomials()
        Q_unity = Polynomial(np.array([1.0]))
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}

    def get_real_q_polynomials(self):
        """Load full PRZZ polynomials including real Q."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_q1_both_components(self):
        """With Q=1, g_I1 from log factor split, g_I2 = g_baseline."""
        polys = self.get_q1_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_values_weighted(R, theta, polys, K, n_quad=60)

        g_baseline = result.g_baseline

        print(f"\nQ=1 complete derivation:")
        print(f"  g_I1 (from log factor split): {result.g_I1:.8f}")
        print(f"  g_I2 (by construction): {result.g_I2:.8f}")
        print(f"  g_baseline: {g_baseline:.8f}")
        print(f"  g_I1 gap from 1.0: {result.g_I1_gap_from_one * 100:.4f}%")
        print(f"  g_I2 gap from baseline: {result.g_I2_gap_from_baseline * 100:.4f}%")

        # g_I1 from log factor split - tolerance is 5% for Q=1
        assert result.g_I1_gap_from_one < 0.05, f"g_I1 should be close to 1.0"
        # g_I2 = g_baseline by construction - should be exact
        assert result.g_I2_gap_from_baseline < 1e-10, f"g_I2 should equal g_baseline"

    def test_real_q_both_components(self):
        """With real Q, document first-principles derived values."""
        polys = self.get_real_q_polynomials()
        theta = 4 / 7
        K = 3
        R = 1.3036

        result = derive_g_values_weighted(R, theta, polys, K, n_quad=60)

        g_baseline = result.g_baseline

        print(f"\nReal Q complete derivation (first-principles):")
        print(f"  g_I1 (from log factor split): {result.g_I1:.8f}")
        print(f"  g_I2 (by construction): {result.g_I2:.8f}")
        print(f"  g_baseline: {g_baseline:.8f}")

        # Document calibrated values for comparison
        print(f"\n  Comparison to calibrated values:")
        print(f"    g_I1 (calibrated): 1.00091428")
        print(f"    g_I2 (calibrated): 1.01945154")
        print(f"\n  First-principles uses g_I1 from log factor split,")
        print(f"  and g_I2 = g_baseline. The ~0.4% benchmark gap")
        print(f"  comes from Q effects not captured here.")

        # Check reasonableness
        assert 0.9 < result.g_I1 < 1.2, "g_I1 should be reasonable"
        assert 1.0 < result.g_I2 < 1.1, "g_I2 should be reasonable"
        # g_I2 should equal g_baseline exactly
        assert abs(result.g_I2 - g_baseline) < 1e-10, "g_I2 = g_baseline"


class TestKappaStarBenchmark:
    """Test derivation on κ* benchmark."""

    def get_q1_polynomials_kappa_star(self):
        """Load κ* polynomials but replace Q with Q=1."""
        P1, P2, P3, _ = load_przz_polynomials_kappa_star()
        Q_unity = Polynomial(np.array([1.0]))
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}

    def test_kappa_star_q1_g_i2(self):
        """Q=1 κ* g_I2 = g_baseline by construction."""
        polys = self.get_q1_polynomials_kappa_star()
        theta = 4 / 7
        K = 3
        R = 1.1167  # κ* R value

        result = derive_g_i2_weighted(R, theta, polys, K, n_quad=60)

        g_baseline = 1 + theta / (2 * K * (2 * K + 1))

        gap_pct = abs(result.g_I2 / g_baseline - 1) * 100

        print(f"\nQ=1 κ* g_I2 derivation (by construction):")
        print(f"  g_I2 (derived): {result.g_I2:.8f}")
        print(f"  g_baseline: {g_baseline:.8f}")
        print(f"  Gap: {gap_pct:.4f}%")

        # g_I2 = g_baseline by construction - should be exact
        assert gap_pct < 0.01, (
            f"g_I2 should equal g_baseline by construction.\n"
            f"  g_I2: {result.g_I2:.8f}\n"
            f"  g_baseline: {g_baseline:.8f}\n"
            f"  Gap: {gap_pct:.4f}%"
        )
