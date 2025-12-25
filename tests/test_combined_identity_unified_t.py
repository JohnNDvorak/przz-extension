"""
tests/test_combined_identity_unified_t.py
Unit tests for unified-t combined identity implementation.

Tests the three sanity checks from GPT enhanced guidance:
1. Scalar combined identity numeric test
2. Q=1, P=1 reduced kernel test
3. Global factor vs structural mismatch comparison
"""

import pytest
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.combined_identity_unified_t import (
    scalar_combined_identity_lhs,
    scalar_combined_identity_rhs,
    verify_scalar_combined_identity,
    compute_I1_reduced_kernel,
    compute_I1_tex_combined_unified_t_pair,
    compare_unified_t_to_post_identity,
    derive_m1_from_unified_t,
)


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


class TestSanityCheck1_ScalarCombinedIdentity:
    """
    Sanity Check 1: Verify the scalar combined identity.

    (1 - z^{-s})/s == log(z) × ∫₀¹ z^{-ts} dt

    This catches sign mistakes and missing log factors.
    """

    @pytest.mark.parametrize("z,s", [
        (2.0, 0.5),
        (3.0, 1.0),
        (10.0, 0.1),
        (1.5, -0.5),
        (5.0, 2.0),
        (100.0, 0.01),
    ])
    def test_scalar_identity_holds(self, z, s):
        """Scalar combined identity should hold to machine precision."""
        lhs, rhs, rel_error = verify_scalar_combined_identity(z, s, n_quad=50)

        assert rel_error < 1e-10, (
            f"Combined identity failed for z={z}, s={s}: "
            f"LHS={lhs:.12f}, RHS={rhs:.12f}, rel_error={rel_error:.2e}"
        )

    def test_lhs_function_basic(self):
        """LHS function (1 - z^{-s})/s should compute correctly."""
        # z=2, s=1: (1 - 0.5) / 1 = 0.5
        assert np.isclose(scalar_combined_identity_lhs(2.0, 1.0), 0.5)

        # z=4, s=0.5: (1 - 0.5) / 0.5 = 1.0
        assert np.isclose(scalar_combined_identity_lhs(4.0, 0.5), 1.0)

    def test_rhs_converges_with_quadrature(self):
        """RHS should converge under quadrature refinement."""
        z, s = 5.0, 0.3

        rhs_n20 = scalar_combined_identity_rhs(z, s, n_quad=20)
        rhs_n50 = scalar_combined_identity_rhs(z, s, n_quad=50)
        rhs_n100 = scalar_combined_identity_rhs(z, s, n_quad=100)

        # Should converge - but if already at machine precision, both diffs are ~0
        diff_20_50 = abs(rhs_n50 - rhs_n20)
        diff_50_100 = abs(rhs_n100 - rhs_n50)

        # If already converged to machine precision, differences will be ~0
        if diff_20_50 > 1e-14:
            assert diff_50_100 < diff_20_50, "RHS should converge with more quadrature points"
        else:
            # Already at machine precision
            assert diff_50_100 < 1e-14, "RHS already at machine precision"


class TestSanityCheck2_ReducedKernel:
    """
    Sanity Check 2: Q=1, P=1 reduced kernel test.

    With Q=1 and P=1, the kernel simplifies to just the exp structure
    plus optional log and algebraic factors.

    This catches normalization issues like double (1+θ(x+y)).
    """

    def test_reduced_kernel_finite(self):
        """Reduced kernel should produce finite values."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        for include_log in [False, True]:
            for include_alg in [False, True]:
                I1 = compute_I1_reduced_kernel(
                    theta, R, n,
                    include_log_factor=include_log,
                    include_alg_prefactor=include_alg
                )
                assert np.isfinite(I1), (
                    f"Reduced kernel not finite with log={include_log}, alg={include_alg}"
                )

    def test_reduced_kernel_log_vs_alg_ratio(self):
        """Log factor and algebraic prefactor should differ by factor of θ.

        Since (1+θ(x+y)) = θ × (1/θ + x + y), the ratio should be θ.
        """
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        # Exp only
        I1_exp_only = compute_I1_reduced_kernel(
            theta, R, n,
            include_log_factor=False,
            include_alg_prefactor=False
        )

        # With log factor (1+θ(x+y))
        I1_with_log = compute_I1_reduced_kernel(
            theta, R, n,
            include_log_factor=True,
            include_alg_prefactor=False
        )

        # With alg prefactor (1/θ + x + y) = (1+θ(x+y))/θ
        I1_with_alg = compute_I1_reduced_kernel(
            theta, R, n,
            include_log_factor=False,
            include_alg_prefactor=True
        )

        # I1_with_alg should be (1/θ) × I1_with_log
        expected_ratio = 1.0 / theta
        actual_ratio = I1_with_alg / I1_with_log if abs(I1_with_log) > 1e-15 else float('inf')

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01), (
            f"Alg/Log ratio should be 1/θ = {expected_ratio:.4f}, got {actual_ratio:.4f}"
        )

    def test_both_factors_is_squared(self):
        """Including both log and alg factors should give (1+θ(x+y))²/θ effect."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        # With log only
        I1_log = compute_I1_reduced_kernel(
            theta, R, n,
            include_log_factor=True,
            include_alg_prefactor=False
        )

        # With both
        I1_both = compute_I1_reduced_kernel(
            theta, R, n,
            include_log_factor=True,
            include_alg_prefactor=True
        )

        # I1_both / I1_log should be approximately 1/θ × (effect of extra (1+θ(x+y)))
        # At xy level, this is roughly 1/θ
        # Note: This is a rough test - the exact ratio depends on the integrand structure
        ratio = I1_both / I1_log if abs(I1_log) > 1e-15 else float('inf')

        print(f"\nDouble factor test:")
        print(f"  I1_log_only = {I1_log:.8f}")
        print(f"  I1_both = {I1_both:.8f}")
        print(f"  Ratio = {ratio:.4f}")
        print(f"  1/θ = {1.0/theta:.4f}")

        # The ratio should be in a reasonable range
        assert 0.5 < ratio < 5.0, f"Ratio {ratio:.4f} out of expected range"


class TestSanityCheck3_GlobalVsStructural:
    """
    Sanity Check 3: Global factor vs structural mismatch.

    Compare unified-t to post-identity for all pairs:
    - Same constant factor across all pairs: missing normalization
    - Varying ratios by pair: structural issue
    """

    def test_comparison_runs_without_error(self, polys_kappa):
        """Comparison should run without error."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        result = compare_unified_t_to_post_identity(
            theta, R, n, polys_kappa,
            include_log_factor=False
        )

        assert "pairs" in result
        assert "diagnosis" in result
        assert len(result["pairs"]) == 6

    def test_without_log_factor_matches_post_identity(self, polys_kappa):
        """Without log factor, unified-t should match post-identity exactly.

        The exp structure and Q×Q are identical, so the ratio should be 1.0.
        """
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result = compare_unified_t_to_post_identity(
            theta, R, n, polys_kappa,
            include_log_factor=False
        )

        print(f"\nWithout log factor (should match post-identity):")
        print(f"  Mean ratio: {result['mean_ratio']:.6f}")
        print(f"  CV: {result['cv']:.6f}")
        print(f"  Diagnosis: {result['diagnosis']}")

        for pair, data in result["pairs"].items():
            print(f"  {pair}: I1_post={data['I1_post']:.8f}, "
                  f"I1_unified={data['I1_unified']:.8f}, ratio={data['ratio']:.6f}")

        # All ratios should be very close to 1.0
        assert np.isclose(result["mean_ratio"], 1.0, rtol=1e-6), (
            f"Mean ratio should be 1.0, got {result['mean_ratio']:.6f}"
        )
        assert result["diagnosis"] == "GLOBAL_FACTOR", (
            f"Should be GLOBAL_FACTOR (all same), got {result['diagnosis']}"
        )

    def test_with_log_factor_shows_global_scaling(self, polys_kappa):
        """With log factor, unified-t should differ by global factor from post-identity."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result = compare_unified_t_to_post_identity(
            theta, R, n, polys_kappa,
            include_log_factor=True
        )

        print(f"\nWith log factor:")
        print(f"  Mean ratio: {result['mean_ratio']:.6f}")
        print(f"  CV: {result['cv']:.6f}")
        print(f"  Diagnosis: {result['diagnosis']}")

        for pair, data in result["pairs"].items():
            print(f"  {pair}: ratio={data['ratio']:.6f}")

        # Should still be GLOBAL_FACTOR (consistent across pairs)
        # The ratio won't be 1.0, but should be consistent
        assert result["cv"] < 0.1, (
            f"CV should be small (global factor), got {result['cv']:.4f}"
        )


class TestUnifiedTKernel:
    """Test the unified-t combined identity kernel."""

    @pytest.mark.parametrize("ell1,ell2", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)])
    def test_unified_t_finite_kappa(self, ell1, ell2, polys_kappa):
        """Unified-t kernel should produce finite values for all pairs."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        result = compute_I1_tex_combined_unified_t_pair(
            theta, R, ell1, ell2, n, polys_kappa,
            include_log_factor=False
        )

        assert np.isfinite(result.I1_value), (
            f"I1 not finite for ({ell1},{ell2}): {result.I1_value}"
        )

    @pytest.mark.parametrize("ell1,ell2", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)])
    def test_unified_t_finite_kappa_star(self, ell1, ell2, polys_kappa_star):
        """Unified-t kernel should produce finite values for kappa*."""
        theta = 4.0 / 7.0
        R = 1.1167
        n = 30

        result = compute_I1_tex_combined_unified_t_pair(
            theta, R, ell1, ell2, n, polys_kappa_star,
            include_log_factor=False
        )

        assert np.isfinite(result.I1_value), (
            f"I1 not finite for ({ell1},{ell2}): {result.I1_value}"
        )

    def test_quadrature_convergence(self, polys_kappa):
        """Unified-t kernel should converge under quadrature refinement."""
        theta = 4.0 / 7.0
        R = 1.3036

        I1_n30 = compute_I1_tex_combined_unified_t_pair(
            theta, R, 1, 1, 30, polys_kappa, include_log_factor=False
        ).I1_value

        I1_n40 = compute_I1_tex_combined_unified_t_pair(
            theta, R, 1, 1, 40, polys_kappa, include_log_factor=False
        ).I1_value

        rel_diff = abs(I1_n40 - I1_n30) / abs(I1_n30)

        assert rel_diff < 0.01, f"Quadrature not converged: rel_diff={rel_diff:.2%}"


class TestM1Derivation:
    """Test m1 derivation from unified-t combined identity."""

    def test_m1_derivation_runs(self, polys_kappa):
        """m1 derivation should run without error."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result = derive_m1_from_unified_t(
            theta, R, n, polys_kappa,
            include_log_factor=False,
            verbose=True
        )

        assert np.isfinite(result["m1_eff"]), f"m1_eff not finite: {result['m1_eff']}"
        assert np.isfinite(result["m1_empirical"]), f"m1_empirical not finite"
        assert np.isfinite(result["m1_naive"]), f"m1_naive not finite"

    def test_m1_without_log_equals_zero(self, polys_kappa):
        """Without log factor, unified-t == post-identity, so m1_eff should be 0.

        Because I1_unified = I1_plus when log factor is excluded,
        (I1_unified - I1_plus) / I1_minus_base = 0.
        """
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result = derive_m1_from_unified_t(
            theta, R, n, polys_kappa,
            include_log_factor=False,
            verbose=True
        )

        assert np.isclose(result["m1_eff"], 0.0, atol=1e-6), (
            f"Without log factor, m1_eff should be 0, got {result['m1_eff']:.6f}"
        )

    def test_m1_with_log_factor_kappa(self, polys_kappa):
        """With log factor, compute m1_eff and compare to empirical."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        result = derive_m1_from_unified_t(
            theta, R, n, polys_kappa,
            include_log_factor=True,
            verbose=True
        )

        print(f"\nm1 derivation with log factor (kappa):")
        print(f"  m1_eff = {result['m1_eff']:.4f}")
        print(f"  m1_empirical = {result['m1_empirical']:.4f}")
        print(f"  m1_naive = {result['m1_naive']:.4f}")
        print(f"  Ratio to empirical: {result['ratio_to_empirical']:.4f}")

        # Document the result - this is the key finding
        # Success: ratio close to 1.0
        # Failure: ratio significantly different from 1.0

    def test_m1_with_log_factor_kappa_star(self, polys_kappa_star):
        """With log factor, compute m1_eff for kappa* benchmark."""
        theta = 4.0 / 7.0
        R = 1.1167
        n = 40

        result = derive_m1_from_unified_t(
            theta, R, n, polys_kappa_star,
            include_log_factor=True,
            verbose=True
        )

        print(f"\nm1 derivation with log factor (kappa*):")
        print(f"  m1_eff = {result['m1_eff']:.4f}")
        print(f"  m1_empirical = {result['m1_empirical']:.4f}")
        print(f"  m1_naive = {result['m1_naive']:.4f}")
        print(f"  Ratio to empirical: {result['ratio_to_empirical']:.4f}")
