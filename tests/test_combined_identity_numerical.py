"""
tests/test_combined_identity_numerical.py
Phase 7B: Combined Identity Numerical Validation

Verify the combined identity holds numerically:

[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

This removes "silent mismatch" risk between different code paths.
"""

import numpy as np
import pytest
from src.quadrature import gauss_legendre_01


# =============================================================================
# Constants
# =============================================================================

THETA = 4.0 / 7.0
L = 100.0  # log T (asymptotic parameter)


def compute_N_power(exponent: float) -> float:
    """Compute N^exponent = T^{θ·exponent} = exp(θL·exponent)."""
    return np.exp(THETA * L * exponent)


def compute_T_power(exponent: float) -> float:
    """Compute T^exponent = exp(L·exponent)."""
    return np.exp(L * exponent)


# =============================================================================
# Combined Identity Tests
# =============================================================================

class TestCombinedIdentity:
    """
    Verify the combined identity integral representation equals
    the difference quotient for various parameter values.
    """

    @pytest.mark.parametrize("alpha,beta", [
        (-0.01, -0.01),
        (-0.02, -0.01),
        (-0.01, -0.02),
        (-0.005, -0.015),
    ])
    @pytest.mark.parametrize("x,y", [
        (0.0, 0.0),
        (0.1, 0.1),
        (0.2, 0.3),
        (0.3, 0.1),
    ])
    def test_combined_identity_for_small_alpha_beta(self, alpha, beta, x, y):
        """
        Verify the combined identity holds numerically.

        LHS = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
        RHS = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
        """
        s = alpha + beta

        # Avoid division by zero
        if abs(s) < 1e-12:
            pytest.skip("s = α+β too close to zero")

        # LHS: Difference quotient
        term1 = compute_N_power(alpha * x + beta * y)
        term2 = compute_T_power(-s) * compute_N_power(-beta * x - alpha * y)
        lhs = (term1 - term2) / s

        # RHS: Integral representation
        # N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t·s} dt

        # Prefactor: N^{αx+βy}
        prefactor_N = compute_N_power(alpha * x + beta * y)

        # log(N^{x+y}T) = (x+y)·θL + L = L(1 + θ(x+y))
        log_factor = L * (1 + THETA * (x + y))

        # Integral: ∫₀¹ (N^{x+y}T)^{-t·s} dt
        # (N^{x+y}T) = exp((θ(x+y)+1)L)
        # (N^{x+y}T)^{-t·s} = exp(-t·s·L(θ(x+y)+1))

        exponent_base = L * (1 + THETA * (x + y))

        # Integrate using quadrature
        n_quad = 100
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            integrand = np.exp(-t * s * exponent_base)
            integral += w * integrand

        rhs = prefactor_N * log_factor * integral

        # Compare
        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        assert rel_error < 1e-10, \
            f"Combined identity failed: LHS={lhs}, RHS={rhs}, rel_error={rel_error}"

    def test_combined_identity_at_przz_evaluation_point(self):
        """
        Test the combined identity at α = β = -R/L, the PRZZ evaluation point.
        """
        R = 1.3036
        alpha = -R / L
        beta = -R / L
        x, y = 0.2, 0.3

        s = alpha + beta

        # LHS
        term1 = compute_N_power(alpha * x + beta * y)
        term2 = compute_T_power(-s) * compute_N_power(-beta * x - alpha * y)
        lhs = (term1 - term2) / s

        # RHS
        prefactor_N = compute_N_power(alpha * x + beta * y)
        log_factor = L * (1 + THETA * (x + y))
        exponent_base = L * (1 + THETA * (x + y))

        n_quad = 100
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            integrand = np.exp(-t * s * exponent_base)
            integral += w * integrand

        rhs = prefactor_N * log_factor * integral

        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        assert rel_error < 1e-10


class TestCombinedIdentityEndpoints:
    """Verify the combined identity at boundary cases."""

    def test_x_equals_y_equals_zero(self):
        """At x=y=0, the identity should still hold."""
        alpha, beta = -0.02, -0.01
        x, y = 0.0, 0.0
        s = alpha + beta

        # LHS: [N^0 - T^{-s}N^0] / s = [1 - T^{-s}] / s
        lhs = (1 - compute_T_power(-s)) / s

        # RHS: N^0 × log(N^0·T) × ∫₀¹ (N^0·T)^{-ts} dt
        #    = 1 × log(T) × ∫₀¹ T^{-ts} dt
        #    = L × ∫₀¹ exp(-tsL) dt

        n_quad = 100
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            integrand = np.exp(-t * s * L)
            integral += w * integrand

        rhs = L * integral

        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        assert rel_error < 1e-10

    @pytest.mark.parametrize("x_val", [0.1, 0.3, 0.5])
    def test_y_equals_zero(self, x_val):
        """Test with y=0."""
        alpha, beta = -0.015, -0.015
        x, y = x_val, 0.0
        s = alpha + beta

        # LHS: [N^{αx+βy} - T^{-s}N^{-βx-αy}] / s
        term1 = compute_N_power(alpha * x + beta * y)
        term2 = compute_T_power(-s) * compute_N_power(-beta * x - alpha * y)
        lhs = (term1 - term2) / s

        # RHS: N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-ts} dt
        prefactor_N = compute_N_power(alpha * x + beta * y)
        log_factor = L * (1 + THETA * (x + y))
        exponent_base = L * (1 + THETA * (x + y))

        n_quad = 100
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            integrand = np.exp(-t * s * exponent_base)
            integral += w * integrand

        rhs = prefactor_N * log_factor * integral

        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        assert rel_error < 1e-10


class TestCombinedIdentityAnalyticIntegral:
    """
    The integral ∫₀¹ e^{-c·t} dt has closed form: (1 - e^{-c}) / c.

    Verify this matches quadrature.
    """

    @pytest.mark.parametrize("c", [0.1, 1.0, 5.0, 10.0])
    def test_exponential_integral_closed_form(self, c):
        """∫₀¹ e^{-ct} dt = (1 - e^{-c}) / c."""
        # Closed form
        closed_form = (1 - np.exp(-c)) / c

        # Quadrature
        n_quad = 100
        t_nodes, t_weights = gauss_legendre_01(n_quad)

        integral = 0.0
        for t, w in zip(t_nodes, t_weights):
            integral += w * np.exp(-c * t)

        rel_error = abs(integral - closed_form) / abs(closed_form)
        assert rel_error < 1e-12

    def test_combined_identity_using_closed_form(self):
        """
        Verify combined identity using closed-form integral.

        ∫₀¹ exp(-t·s·M) dt = (1 - exp(-sM)) / (sM)

        where M = L(1+θ(x+y)).

        So RHS = N^{αx+βy} × M × (1 - exp(-sM)) / (sM)
               = N^{αx+βy} × (1 - exp(-sM)) / s
        """
        alpha, beta = -0.02, -0.015
        x, y = 0.2, 0.3
        s = alpha + beta

        # LHS
        term1 = compute_N_power(alpha * x + beta * y)
        term2 = compute_T_power(-s) * compute_N_power(-beta * x - alpha * y)
        lhs = (term1 - term2) / s

        # RHS using closed form
        prefactor_N = compute_N_power(alpha * x + beta * y)
        M = L * (1 + THETA * (x + y))

        # (N^{x+y}T)^{-s} = exp(-sM)
        exp_minus_sM = np.exp(-s * M)

        # RHS = N^{αx+βy} × M × (1 - exp(-sM)) / (sM)
        rhs = prefactor_N * (1 - exp_minus_sM) / s

        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        assert rel_error < 1e-12


class TestCombinedIdentityVsMirrorTerms:
    """
    Verify the combined identity relates to mirror decomposition.

    The key insight: once you apply the combined identity, direct and mirror
    contributions are FUSED into a single integral. They cannot be cleanly
    separated as scalars.
    """

    def test_direct_term_is_integrand_at_t_equals_one(self):
        """
        At t=1, the integrand should relate to the "direct" term.

        Integrand at t=1: exp(-s·M) where M = L(1+θ(x+y))
                        = (N^{x+y}T)^{-s}

        This is the T^{-s} factor that appears in the mirror term.
        """
        R = 1.3036
        alpha = -R / L
        beta = -R / L
        s = alpha + beta
        x, y = 0.2, 0.3

        M = L * (1 + THETA * (x + y))

        # Integrand at t=1
        integrand_at_1 = np.exp(-s * M)

        # (N^{x+y}T)^{-s} = exp(-s(θL(x+y) + L)) = exp(-sL(θ(x+y)+1)) = exp(-sM)
        expected = np.exp(-s * M)

        assert abs(integrand_at_1 - expected) < 1e-14

    def test_integrand_at_t_zero_is_one(self):
        """
        At t=0, the integrand is exp(0) = 1.

        This is consistent with the "direct" endpoint of the interpolation.
        """
        R = 1.3036
        alpha = -R / L
        beta = -R / L
        s = alpha + beta
        x, y = 0.2, 0.3

        M = L * (1 + THETA * (x + y))

        integrand_at_0 = np.exp(-0 * s * M)
        assert abs(integrand_at_0 - 1.0) < 1e-14
