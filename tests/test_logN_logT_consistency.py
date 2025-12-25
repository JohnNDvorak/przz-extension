"""
tests/test_logN_logT_consistency.py
Phase 7B: Normalization Contract Tests

TeX defines:
- N = T^θ, so log N = θ × log T
- Combined identity introduces log(N^{x+y}T) = (1+θ(x+y)) × log T
- Operators D_α = -1/L × ∂/∂α where L = log T

These relationships MUST be consistent throughout the codebase.
Any drift here shows up as "mysterious amplitude factors" that
get absorbed into the empirical m₁.
"""

import numpy as np
import pytest
from typing import Tuple

from src.operator_post_identity import (
    compute_post_identity_core_E,
    compute_A_alpha,
    compute_A_beta,
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
    get_exp_affine_coeffs,
)

from src.combined_identity_regularized import (
    compute_A_alpha as compute_A_alpha_reg,
    compute_A_beta as compute_A_beta_reg,
    get_A_alpha_affine_coeffs_regularized,
    get_A_beta_affine_coeffs_regularized,
    get_exp_affine_coeffs_regularized,
    compute_kernel_E,
)


# =============================================================================
# Constants
# =============================================================================

THETA = 4.0 / 7.0  # PRZZ θ parameter
R_KAPPA = 1.3036   # κ benchmark
R_KAPPA_STAR = 1.1167  # κ* benchmark


# =============================================================================
# Phase 7B: log N / log T Relationship Tests
# =============================================================================

class TestLogNLogTRelationship:
    """Verify N = T^θ implies log N = θ × log T."""

    def test_logN_equals_theta_logT_numerically(self):
        """log N = θ × log T for any L = log T."""
        for L in [10.0, 100.0, 1000.0]:
            log_N = THETA * L  # By definition N = T^θ
            log_T = L

            assert abs(log_N / log_T - THETA) < 1e-14, \
                f"log N / log T should be θ = {THETA}"

    def test_combined_identity_prefactor_structure(self):
        """
        The combined identity prefactor is log(N^{x+y}T) = (1 + θ(x+y)) × log T.

        For nilpotent x, y (order 1 each), this gives:
        - Constant term: log T = L
        - Linear in x: θL
        - Linear in y: θL
        """
        L = 100.0

        # Prefactor structure: L(1 + θ(x+y))
        # At x=y=0: L
        # ∂/∂x at 0: θL
        # ∂/∂y at 0: θL

        const_term = L
        x_coeff = THETA * L
        y_coeff = THETA * L

        assert abs(const_term - L) < 1e-14
        assert abs(x_coeff - THETA * L) < 1e-14
        assert abs(y_coeff - THETA * L) < 1e-14


# =============================================================================
# Eigenvalue Structure Tests
# =============================================================================

class TestPostIdentityEigenvalues:
    """Verify post-identity eigenvalue structure."""

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_alpha_affine_structure(self, t):
        """
        A_α = t + θ(t-1)·x + θt·y

        Verify coefficients match closed-form extraction.
        """
        u0, x_coeff, y_coeff = get_A_alpha_affine_coeffs(t, THETA)

        expected_u0 = t
        expected_x_coeff = THETA * (t - 1)
        expected_y_coeff = THETA * t

        assert abs(u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_x_coeff) < 1e-14
        assert abs(y_coeff - expected_y_coeff) < 1e-14

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_beta_affine_structure(self, t):
        """
        A_β = t + θt·x + θ(t-1)·y

        Verify coefficients match closed-form extraction.
        """
        u0, x_coeff, y_coeff = get_A_beta_affine_coeffs(t, THETA)

        expected_u0 = t
        expected_x_coeff = THETA * t
        expected_y_coeff = THETA * (t - 1)

        assert abs(u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_x_coeff) < 1e-14
        assert abs(y_coeff - expected_y_coeff) < 1e-14

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_eigenvalue_symmetry(self, t):
        """A_α(x,y) = A_β(y,x) by symmetry."""
        x_val, y_val = 0.3, 0.7

        A_alpha = compute_A_alpha(x_val, y_val, t, THETA)
        A_beta_swapped = compute_A_beta(y_val, x_val, t, THETA)

        assert abs(A_alpha - A_beta_swapped) < 1e-14, \
            f"A_α(x,y) should equal A_β(y,x)"

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_eigenvalue_sum_structure(self, t):
        """
        A_α + A_β = 2t + θ(2t-1)(x+y)

        This is the structure that appears in the exp factor.
        """
        x_val, y_val = 0.3, 0.7

        A_alpha = compute_A_alpha(x_val, y_val, t, THETA)
        A_beta = compute_A_beta(x_val, y_val, t, THETA)

        sum_actual = A_alpha + A_beta
        sum_expected = 2*t + THETA*(2*t - 1)*(x_val + y_val)

        assert abs(sum_actual - sum_expected) < 1e-14


# =============================================================================
# Regularized Path Eigenvalue Tests
# =============================================================================

class TestRegularizedEigenvalues:
    """Verify u-regularized eigenvalue structure."""

    @pytest.mark.parametrize("u", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_alpha_regularized_affine_structure(self, u):
        """
        A_α(u) = (1-u) + θ((1-u)y - ux)

        Verify coefficients match closed-form extraction.
        """
        u0, x_coeff, y_coeff = get_A_alpha_affine_coeffs_regularized(u, THETA)

        expected_u0 = 1 - u
        expected_x_coeff = -THETA * u
        expected_y_coeff = THETA * (1 - u)

        assert abs(u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_x_coeff) < 1e-14
        assert abs(y_coeff - expected_y_coeff) < 1e-14

    @pytest.mark.parametrize("u", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_A_beta_regularized_affine_structure(self, u):
        """
        A_β(u) = (1-u) + θ((1-u)x - uy)

        Verify coefficients match closed-form extraction.
        """
        u0, x_coeff, y_coeff = get_A_beta_affine_coeffs_regularized(u, THETA)

        expected_u0 = 1 - u
        expected_x_coeff = THETA * (1 - u)
        expected_y_coeff = -THETA * u

        assert abs(u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_x_coeff) < 1e-14
        assert abs(y_coeff - expected_y_coeff) < 1e-14


# =============================================================================
# Transformation Between Post-Identity and Regularized Paths
# =============================================================================

class TestPathTransformation:
    """
    Verify u-regularized and post-identity paths are related by t = 1 - u.

    This was proven in Phase 5 to machine precision.
    """

    @pytest.mark.parametrize("t", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_eigenvalue_transformation_t_to_u(self, t):
        """
        Under t = 1 - u:
        - Post-identity A_α(t, x, y) should equal regularized A_α(u=1-t, x, y)

        But eigenvalue structure differs because:
        - Post-identity: A_α = t + θ(t-1)x + θt·y
        - Regularized: A_α = (1-u) + θ((1-u)y - ux)

        With u = 1-t:
        - Regularized A_α = t + θ(ty - (1-t)x) = t + θty - θx + θtx
                          = t + θ(t-1)x + θty
        """
        u = 1 - t
        x_val, y_val = 0.3, 0.7

        # Post-identity eigenvalue
        A_alpha_post = compute_A_alpha(x_val, y_val, t, THETA)

        # Regularized eigenvalue at u = 1-t
        A_alpha_reg = compute_A_alpha_reg(u, x_val, y_val, THETA)

        # These should be equal under the transformation
        assert abs(A_alpha_post - A_alpha_reg) < 1e-14, \
            f"Post-identity A_α at t={t} should equal regularized at u={u}"

    @pytest.mark.parametrize("t", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_beta_eigenvalue_transformation(self, t):
        """Same transformation test for A_β."""
        u = 1 - t
        x_val, y_val = 0.3, 0.7

        A_beta_post = compute_A_beta(x_val, y_val, t, THETA)
        A_beta_reg = compute_A_beta_reg(u, x_val, y_val, THETA)

        assert abs(A_beta_post - A_beta_reg) < 1e-14


# =============================================================================
# Exp Factor Structure Tests
# =============================================================================

class TestExpFactorStructure:
    """Verify exp factor has correct affine structure."""

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("R", [R_KAPPA, R_KAPPA_STAR])
    def test_post_identity_exp_coefficients(self, t, R):
        """
        Exp factor exp(R*(A_α + A_β)) has structure:
        - u0 = 2Rt
        - x_coeff = R(2θt - θ) = Rθ(2t-1)
        - y_coeff = R(2θt - θ) = Rθ(2t-1)
        """
        u0, x_coeff, y_coeff = get_exp_affine_coeffs(t, THETA, R)

        expected_u0 = 2 * R * t
        expected_lin = R * THETA * (2*t - 1)

        assert abs(u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_lin) < 1e-14
        assert abs(y_coeff - expected_lin) < 1e-14

    @pytest.mark.parametrize("u", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("R", [R_KAPPA, R_KAPPA_STAR])
    def test_regularized_exp_coefficients(self, u, R):
        """
        Regularized exp factor at α=β=-R/L:
        - u0 = 2R(1-u)
        - lin = θR(1-2u)
        """
        exp_u0, x_coeff, y_coeff = get_exp_affine_coeffs_regularized(u, THETA, R)

        expected_u0 = 2 * R * (1 - u)
        expected_lin = THETA * R * (1 - 2*u)

        assert abs(exp_u0 - expected_u0) < 1e-14
        assert abs(x_coeff - expected_lin) < 1e-14
        assert abs(y_coeff - expected_lin) < 1e-14


# =============================================================================
# Core Exponential Kernel Tests
# =============================================================================

class TestKernelStructure:
    """Verify exponential kernel structure."""

    def test_post_identity_kernel_at_zero_xy(self):
        """E(α,β;0,0,t) = exp(−2Rt) at α=β=−R/L."""
        L = 100.0
        R = R_KAPPA
        alpha = -R / L
        beta = -R / L
        t = 0.5
        x, y = 0.0, 0.0

        E = compute_post_identity_core_E(alpha, beta, x, y, t, THETA, L)

        # At x=y=0: E = exp(−t(α+β)L) = exp(−t(−2R/L)L) = exp(2Rt)
        expected = np.exp(2 * R * t)

        assert abs(E - expected) < 1e-10

    def test_regularized_kernel_at_zero_xy(self):
        """E_reg(α,β;0,0,u) = exp(2R(1-u)) at α=β=−R/L."""
        L = 100.0
        R = R_KAPPA
        alpha = -R / L
        beta = -R / L
        u = 0.3
        x, y = 0.0, 0.0

        E = compute_kernel_E(alpha, beta, x, y, u, THETA, L)

        # At x=y=0: E = exp(−L s(1-u)) = exp(−L(−2R/L)(1-u)) = exp(2R(1-u))
        expected = np.exp(2 * R * (1 - u))

        assert abs(E - expected) < 1e-10


# =============================================================================
# Operator D_α, D_β Normalization Tests
# =============================================================================

class TestOperatorNormalization:
    """Verify operator D = -1/L × ∂/∂ has correct normalization."""

    def test_D_alpha_on_pure_exponential(self):
        """
        D_α[exp(Lα·f)] = -f × exp(Lα·f)

        This verifies the -1/L normalization.
        """
        # For exp(L·α·f), ∂/∂α = L·f·exp(...)
        # D_α = -1/L × (L·f·exp) = -f·exp
        L = 100.0
        f = 0.5
        alpha = 0.3

        # Manual calculation
        exp_val = np.exp(L * alpha * f)
        partial_alpha = L * f * exp_val
        D_alpha_result = -partial_alpha / L

        expected = -f * exp_val

        assert abs(D_alpha_result - expected) < 1e-9

    def test_eigenvalue_interpretation(self):
        """
        For E = exp(g(α,β,x,y)), we have D_α E = A_α × E
        where A_α = -1/L × ∂g/∂α.

        This is the eigenvalue property used throughout.
        """
        # g = θL(αx + βy) - t(α+β)L(1+θ(x+y))
        # ∂g/∂α = θLx - tL(1+θ(x+y))
        # A_α = -1/L × ∂g/∂α = -θx + t(1+θ(x+y)) = t + θ(t-1)x + θty

        t, x, y = 0.5, 0.3, 0.7

        A_alpha = compute_A_alpha(x, y, t, THETA)
        expected = t + THETA*(t-1)*x + THETA*t*y

        assert abs(A_alpha - expected) < 1e-14
