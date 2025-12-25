"""
tests/test_regularized_matches_post_identity.py
Gate Test: Regularized Path Must Match Post-Identity Operator

This test suite verifies that the Phase 5 refactored u-regularized implementation
computes the EXACT SAME I₁ values as the post-identity operator approach.

Key properties verified:
1. Regularized I₁ = Post-identity I₁ to machine precision (< 1e-10)
2. L-invariance: regularized I₁ does NOT scale with L
3. Both benchmarks (κ and κ*) pass

This gate must pass before proceeding to Phase 6 (derived m₁).

GPT Guidance (2025-12-22):
- The u-regularization is a reparameterization of the TeX t-integral, not new math
- If implemented correctly, it should reproduce the same TeX I₁ object
- Present mismatch was due to implementation bugs, not mathematical differences
"""

import pytest
import numpy as np

from src.combined_identity_regularized import (
    compute_I1_combined_regularized_at_L,
    get_A_alpha_affine_coeffs_regularized,
    get_A_beta_affine_coeffs_regularized,
    get_exp_affine_coeffs_regularized,
    compute_QQexp_series_regularized_at_u,
)
from src.operator_post_identity import (
    compute_I1_operator_post_identity_pair,
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
    get_exp_affine_coeffs,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


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


class TestAffineCoefficientsMatch:
    """Verify that regularized and post-identity affine coefficients match under t = 1-u."""

    @pytest.mark.parametrize("u_reg", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_alpha_coeffs_match(self, u_reg):
        """A_α affine coefficients should match under t = 1-u transformation."""
        theta = 4.0 / 7.0
        t = 1 - u_reg  # TeX t

        u0_reg, x_reg, y_reg = get_A_alpha_affine_coeffs_regularized(u_reg, theta)
        u0_post, x_post, y_post = get_A_alpha_affine_coeffs(t, theta)

        assert np.isclose(u0_reg, u0_post), f"u0 mismatch: {u0_reg} vs {u0_post}"
        assert np.isclose(x_reg, x_post), f"x_coeff mismatch: {x_reg} vs {x_post}"
        assert np.isclose(y_reg, y_post), f"y_coeff mismatch: {y_reg} vs {y_post}"

    @pytest.mark.parametrize("u_reg", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_beta_coeffs_match(self, u_reg):
        """A_β affine coefficients should match under t = 1-u transformation."""
        theta = 4.0 / 7.0
        t = 1 - u_reg

        u0_reg, x_reg, y_reg = get_A_beta_affine_coeffs_regularized(u_reg, theta)
        u0_post, x_post, y_post = get_A_beta_affine_coeffs(t, theta)

        assert np.isclose(u0_reg, u0_post), f"u0 mismatch: {u0_reg} vs {u0_post}"
        assert np.isclose(x_reg, x_post), f"x_coeff mismatch: {x_reg} vs {x_post}"
        assert np.isclose(y_reg, y_post), f"y_coeff mismatch: {y_reg} vs {y_post}"

    @pytest.mark.parametrize("u_reg", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_exp_coeffs_match(self, u_reg):
        """Exp affine coefficients should match under t = 1-u transformation."""
        theta = 4.0 / 7.0
        R = 1.3036
        t = 1 - u_reg

        u0_reg, x_reg, y_reg = get_exp_affine_coeffs_regularized(u_reg, theta, R)
        u0_post, x_post, y_post = get_exp_affine_coeffs(t, theta, R)

        assert np.isclose(u0_reg, u0_post), f"exp_u0 mismatch: {u0_reg} vs {u0_post}"
        assert np.isclose(x_reg, x_post), f"exp_x mismatch: {x_reg} vs {x_post}"
        assert np.isclose(y_reg, y_post), f"exp_y mismatch: {y_reg} vs {y_post}"


class TestGateRegularizedMatchesPostIdentity:
    """GATE TEST: Regularized must equal post-identity for all pairs."""

    @pytest.mark.parametrize("ell1,ell2", [
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
    ])
    def test_kappa_benchmark(self, polys_kappa, ell1, ell2):
        """Gate: regularized I₁ = post-identity I₁ for kappa benchmark."""
        R = 1.3036
        L = 50.0

        result_reg = compute_I1_combined_regularized_at_L(
            theta=4/7, R=R, L=L, n=25,
            polynomials=polys_kappa, ell1=ell1, ell2=ell2, n_quad_reg=25
        )

        result_post = compute_I1_operator_post_identity_pair(
            theta=4/7, R=R, ell1=ell1, ell2=ell2, n=25,
            polynomials=polys_kappa
        )

        rel_error = abs(result_reg.I1_combined - result_post.I1_value) / (
            abs(result_post.I1_value) + 1e-100
        )

        assert rel_error < 1e-10, (
            f"Gate FAILED for ({ell1},{ell2}) on kappa: "
            f"reg={result_reg.I1_combined:.10e}, post={result_post.I1_value:.10e}, "
            f"rel_error={rel_error:.2e}"
        )

    @pytest.mark.parametrize("ell1,ell2", [
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
    ])
    def test_kappa_star_benchmark(self, polys_kappa_star, ell1, ell2):
        """Gate: regularized I₁ = post-identity I₁ for kappa* benchmark."""
        R = 1.1167
        L = 50.0

        result_reg = compute_I1_combined_regularized_at_L(
            theta=4/7, R=R, L=L, n=25,
            polynomials=polys_kappa_star, ell1=ell1, ell2=ell2, n_quad_reg=25
        )

        result_post = compute_I1_operator_post_identity_pair(
            theta=4/7, R=R, ell1=ell1, ell2=ell2, n=25,
            polynomials=polys_kappa_star
        )

        rel_error = abs(result_reg.I1_combined - result_post.I1_value) / (
            abs(result_post.I1_value) + 1e-100
        )

        assert rel_error < 1e-10, (
            f"Gate FAILED for ({ell1},{ell2}) on kappa*: "
            f"reg={result_reg.I1_combined:.10e}, post={result_post.I1_value:.10e}, "
            f"rel_error={rel_error:.2e}"
        )


class TestLInvariance:
    """Verify that regularized I₁ is completely L-invariant."""

    @pytest.mark.parametrize("ell1,ell2", [(1, 1), (2, 2), (3, 3)])
    def test_L_invariance_kappa(self, polys_kappa, ell1, ell2):
        """Regularized I₁ should NOT change with L."""
        R = 1.3036
        L_values = [10, 50, 200, 500]

        results = []
        for L in L_values:
            result = compute_I1_combined_regularized_at_L(
                theta=4/7, R=R, L=L, n=20,
                polynomials=polys_kappa, ell1=ell1, ell2=ell2, n_quad_reg=20
            )
            results.append(result.I1_combined)

        # All values should be essentially identical
        mean_val = sum(results) / len(results)
        max_deviation = max(abs(r - mean_val) / (abs(mean_val) + 1e-100) for r in results)

        assert max_deviation < 1e-10, (
            f"L-dependence detected for ({ell1},{ell2}): max_deviation={max_deviation:.2e}, "
            f"values={results}"
        )

    @pytest.mark.parametrize("L", [10, 50, 100, 500])
    def test_L_invariance_sweep(self, polys_kappa, L):
        """For any L, regularized should match post-identity."""
        R = 1.3036

        result_reg = compute_I1_combined_regularized_at_L(
            theta=4/7, R=R, L=L, n=20,
            polynomials=polys_kappa, ell1=1, ell2=1, n_quad_reg=20
        )

        result_post = compute_I1_operator_post_identity_pair(
            theta=4/7, R=R, ell1=1, ell2=1, n=20,
            polynomials=polys_kappa
        )

        rel_error = abs(result_reg.I1_combined - result_post.I1_value) / (
            abs(result_post.I1_value) + 1e-100
        )

        assert rel_error < 1e-10, (
            f"Mismatch at L={L}: rel_error={rel_error:.2e}"
        )


class TestSeriesExtractionCorrectness:
    """Verify the TruncatedSeries extraction gives correct results."""

    def test_QQexp_series_at_u_zero(self, polys_kappa):
        """At u=0 (t=1 in TeX), check series structure."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        R = 1.3036
        u = 0.0

        series = compute_QQexp_series_regularized_at_u(Q, u, theta, R)

        # At u=0:
        # A_α = 1 + θy (u0=1, x_coeff=0, y_coeff=θ)
        # A_β = 1 + θx (u0=1, x_coeff=θ, y_coeff=0)
        # exp_u0 = 2R, exp_lin = θR

        # Verify series is finite and has expected structure
        c00 = series.extract(())
        cx = series.extract(("x",))
        cy = series.extract(("y",))
        cxy = series.extract(("x", "y"))

        assert np.isfinite(c00), f"c00 not finite: {c00}"
        assert np.isfinite(cx), f"cx not finite: {cx}"
        assert np.isfinite(cy), f"cy not finite: {cy}"
        assert np.isfinite(cxy), f"cxy not finite: {cxy}"

        # At u=0, the eigenvalues are symmetric in x↔y swap,
        # so cx and cy should be equal
        assert np.isclose(cx, cy, rtol=1e-10), f"cx={cx} should equal cy={cy} at u=0"

    def test_QQexp_series_symmetry(self, polys_kappa):
        """QQexp should have x↔y symmetry at u=0.5 (eigenvalues symmetric)."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        R = 1.3036
        u = 0.5

        series = compute_QQexp_series_regularized_at_u(Q, u, theta, R)

        # At u=0.5:
        # A_α = 0.5 + θ(0.5y - 0.5x) = 0.5 + 0.5θ(y-x)
        # A_β = 0.5 + θ(0.5x - 0.5y) = 0.5 + 0.5θ(x-y)
        # These are swapped under x↔y, so cx should equal cy

        cx = series.extract(("x",))
        cy = series.extract(("y",))

        # Due to the swap symmetry in eigenvalues, cx and cy should be equal
        assert np.isclose(cx, cy, rtol=1e-10), (
            f"At u=0.5, cx={cx} should equal cy={cy} due to eigenvalue symmetry"
        )
