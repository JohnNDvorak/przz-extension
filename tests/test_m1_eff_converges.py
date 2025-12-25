"""
Gate test for m1_eff(L) convergence.

GPT Phase 3: Derive m1 from first principles via L-sweep.

CURRENT STATUS:
===============
The full L-sweep requires implementing compute_I1_combined_at_L which
evaluates the pre-identity bracket at finite L. This is NOT YET DONE.

These tests verify the structure is in place and document the empirical
m1 = exp(R) + 5 formula as a known calibration.

EMPIRICAL FORMULA (K=3):
========================
m1 = exp(R) + 5 = exp(R) + (2K - 1)

This formula was found empirically to match PRZZ benchmarks:
- κ (R=1.3036): m1 ≈ 8.68
- κ* (R=1.1167): m1 ≈ 8.05

GOAL:
=====
Once compute_I1_combined_at_L is implemented:
1. Verify m1_eff(L) converges as L → ∞
2. Compare converged value against exp(R) + 5
3. Replace empirical formula with principled derivation

See run_m1_from_combined_identity_L_sweep.py for the diagnostic script.
"""

import pytest
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.operator_post_identity import compute_I1_operator_post_identity_pair


@pytest.fixture(scope="module")
def polys_kappa():
    """Load PRZZ polynomials for kappa benchmark."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture(scope="module")
def polys_kappa_star():
    """Load PRZZ polynomials for kappa* benchmark."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestM1EmpiricalFormula:
    """Document and verify the empirical m1 = exp(R) + 5 formula."""

    def test_m1_empirical_formula_kappa(self):
        """Verify empirical m1 formula for κ benchmark."""
        R = 1.3036
        K = 3

        m1_empirical = np.exp(R) + 5  # = exp(R) + (2K-1)
        m1_formula = np.exp(R) + (2 * K - 1)

        assert np.isclose(m1_empirical, m1_formula), \
            f"Formula mismatch: exp(R)+5 = {m1_empirical}, exp(R)+(2K-1) = {m1_formula}"

        # Verify it's in reasonable range
        assert 8.0 < m1_empirical < 9.0, \
            f"m1 out of expected range for κ: {m1_empirical}"

    def test_m1_empirical_formula_kappa_star(self):
        """Verify empirical m1 formula for κ* benchmark."""
        R = 1.1167
        K = 3

        m1_empirical = np.exp(R) + 5
        m1_formula = np.exp(R) + (2 * K - 1)

        assert np.isclose(m1_empirical, m1_formula)

        # Verify it's in reasonable range
        assert 7.5 < m1_empirical < 8.5, \
            f"m1 out of expected range for κ*: {m1_empirical}"


class TestPostIdentityReference:
    """Verify post-identity I1 values (L=∞ reference)."""

    def test_i1_plus_minus_finite_kappa(self, polys_kappa):
        """I1+ and I1- should be finite from post-identity."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result_plus = compute_I1_operator_post_identity_pair(
            theta, R, 1, 1, n, polys_kappa
        )
        result_minus = compute_I1_operator_post_identity_pair(
            theta, -R, 1, 1, n, polys_kappa
        )

        assert np.isfinite(result_plus.I1_value), f"I1+ not finite: {result_plus.I1_value}"
        assert np.isfinite(result_minus.I1_value), f"I1- not finite: {result_minus.I1_value}"

        # Both should be positive for (1,1)
        assert result_plus.I1_value > 0, f"I1+ should be positive: {result_plus.I1_value}"
        assert result_minus.I1_value > 0, f"I1- should be positive: {result_minus.I1_value}"

    def test_i1_plus_minus_finite_kappa_star(self, polys_kappa_star):
        """I1+ and I1- should be finite from post-identity."""
        theta = 4.0 / 7.0
        R = 1.1167
        n = 40

        result_plus = compute_I1_operator_post_identity_pair(
            theta, R, 1, 1, n, polys_kappa_star
        )
        result_minus = compute_I1_operator_post_identity_pair(
            theta, -R, 1, 1, n, polys_kappa_star
        )

        assert np.isfinite(result_plus.I1_value)
        assert np.isfinite(result_minus.I1_value)
        assert result_plus.I1_value > 0
        assert result_minus.I1_value > 0


class TestNaiveMirrorWeight:
    """Document the naive exp(2R) mirror weight from combined identity."""

    def test_naive_vs_empirical_kappa(self, polys_kappa):
        """Compare naive exp(2R) with empirical exp(R)+5."""
        R = 1.3036

        m1_naive = np.exp(2 * R)
        m1_empirical = np.exp(R) + 5

        # The naive weight is much larger than the empirical one
        ratio = m1_naive / m1_empirical
        print(f"\nκ: naive exp(2R) = {m1_naive:.4f}, empirical = {m1_empirical:.4f}, ratio = {ratio:.3f}")

        # Document that they're different
        assert ratio > 1.5, \
            f"Naive should be larger than empirical: ratio = {ratio}"

    def test_naive_vs_empirical_kappa_star(self, polys_kappa_star):
        """Compare naive exp(2R) with empirical exp(R)+5."""
        R = 1.1167

        m1_naive = np.exp(2 * R)
        m1_empirical = np.exp(R) + 5

        ratio = m1_naive / m1_empirical
        print(f"\nκ*: naive exp(2R) = {m1_naive:.4f}, empirical = {m1_empirical:.4f}, ratio = {ratio:.3f}")

        assert ratio > 1.1, \
            f"Naive should be larger than empirical: ratio = {ratio}"


@pytest.mark.skip(reason="compute_I1_combined_at_L not yet implemented")
class TestM1EffConvergence:
    """Gate tests for m1_eff convergence.

    These tests are SKIPPED until compute_I1_combined_at_L is implemented.
    When implemented, they will verify:
    1. m1_eff(L) stabilizes at large L
    2. Converged value matches empirical exp(R)+5
    """

    def test_m1_eff_converges_kappa(self, polys_kappa):
        """m1_eff should stabilize at large L for κ benchmark."""
        from run_m1_from_combined_identity_L_sweep import run_L_sweep

        R = 1.3036
        results = run_L_sweep(4/7, R, 40, polys_kappa)

        # Check convergence: last two L values should be within 1%
        m1_50 = results[-2].m1_eff
        m1_100 = results[-1].m1_eff

        if m1_50 is None or m1_100 is None:
            pytest.skip("m1_eff not computed (I1_combined not implemented)")

        rel_change = abs(m1_100 - m1_50) / abs(m1_50)
        assert rel_change < 0.01, f"m1_eff not converged: {rel_change:.2%}"

    def test_m1_eff_matches_empirical(self, polys_kappa):
        """Converged m1_eff should match empirical exp(R)+5."""
        from run_m1_from_combined_identity_L_sweep import run_L_sweep

        R = 1.3036
        results = run_L_sweep(4/7, R, 40, polys_kappa)

        m1_eff = results[-1].m1_eff
        if m1_eff is None:
            pytest.skip("m1_eff not computed")

        m1_empirical = np.exp(R) + 5

        rel_diff = abs(m1_eff - m1_empirical) / m1_empirical
        assert rel_diff < 0.05, \
            f"m1_eff={m1_eff:.4f} doesn't match exp(R)+5={m1_empirical:.4f}"


class TestM1StatusDocumentation:
    """Document the current m1 derivation status."""

    def test_m1_is_calibrated_not_derived(self):
        """Document that m1 = exp(R)+5 is CALIBRATION, not derivation.

        This test passes to document the known limitation.
        When m1 is derived from first principles, update this test.
        """
        # The empirical formula works but is not derived
        R_kappa = 1.3036
        R_kappa_star = 1.1167

        m1_kappa = np.exp(R_kappa) + 5
        m1_kappa_star = np.exp(R_kappa_star) + 5

        # Document the values
        print(f"\nm1 Status: CALIBRATED (not derived)")
        print(f"  κ:  m1 = exp({R_kappa}) + 5 = {m1_kappa:.4f}")
        print(f"  κ*: m1 = exp({R_kappa_star}) + 5 = {m1_kappa_star:.4f}")
        print(f"  Formula: m1 = exp(R) + (2K-1) for K=3")

        # This test always passes - it's documentation
        assert True, "m1 derivation is pending"
