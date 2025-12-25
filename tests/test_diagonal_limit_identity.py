"""
tests/test_diagonal_limit_identity.py
Phase 14 Task 2: Tests for diagonal specialization engine.

The key identity from the paper:
∂/∂α [f(α,γ)/ζ(1-α+γ)]|_{γ=α} = -f(α,α)

This "neat formula" is the pole cancellation that makes the diagonal limit well-defined.
"""

import pytest
import numpy as np
from src.ratios.diagonalize import (
    EULER_MASCHERONI,
    zeta_1_plus_eps,
    inv_zeta_1_plus_eps,
    apply_neat_identity,
    diagonalize_gamma_eq_alpha_delta_eq_beta,
)


class TestZetaLaurentExpansion:
    """Test Laurent expansion of ζ(1+ε) around ε=0."""

    def test_zeta_1_plus_eps_has_pole(self):
        """ζ(1+ε) ~ 1/ε as ε→0."""
        for eps in [0.1, 0.01, 0.001]:
            zeta_val = zeta_1_plus_eps(eps)
            pole_term = 1.0 / eps

            # The pole term should dominate
            ratio = abs(zeta_val / pole_term)
            assert 0.5 < ratio < 2.0, (
                f"At ε={eps}: ζ(1+ε)={zeta_val}, 1/ε={pole_term}, ratio={ratio}"
            )

    def test_zeta_1_plus_eps_constant_is_euler_mascheroni(self):
        """ζ(1+ε) = 1/ε + γ + O(ε) where γ ≈ 0.5772..."""
        # At small ε, ζ(1+ε) - 1/ε should approach γ
        for eps in [0.01, 0.001]:
            zeta_val = zeta_1_plus_eps(eps)
            residual = zeta_val - 1.0 / eps

            # Should be close to Euler-Mascheroni
            assert abs(residual - EULER_MASCHERONI) < 0.1, (
                f"At ε={eps}: residual={residual}, expected γ={EULER_MASCHERONI}"
            )


class TestInverseZetaExpansion:
    """Test Laurent expansion of 1/ζ(1+ε)."""

    def test_inv_zeta_1_plus_eps_starts_with_eps(self):
        """1/ζ(1+ε) = ε - γε² + O(ε³), so first term is ε."""
        for eps in [0.1, 0.01, 0.001]:
            inv_zeta_val = inv_zeta_1_plus_eps(eps)

            # Leading term should be ε
            ratio = inv_zeta_val / eps
            # Should be close to 1 for small ε
            assert 0.9 < ratio < 1.1, (
                f"At ε={eps}: 1/ζ(1+ε)={inv_zeta_val}, ratio to ε={ratio}"
            )

    def test_inv_zeta_second_order_correction(self):
        """1/ζ(1+ε) = ε - γε² + O(ε³)."""
        for eps in [0.01, 0.001]:
            inv_zeta_val = inv_zeta_1_plus_eps(eps, order=2)

            # Check ε - γε²
            expected = eps - EULER_MASCHERONI * eps**2
            rel_error = abs(inv_zeta_val - expected) / abs(expected)
            assert rel_error < 0.1, (
                f"At ε={eps}: got {inv_zeta_val}, expected {expected}"
            )


class TestNeatIdentity:
    """Test the paper's "neat" identity for pole cancellation."""

    def test_neat_identity_with_constant_f(self):
        """
        For f(α,γ) = c (constant):
        ∂/∂α [c/ζ(1-α+γ)]|_{γ=α} = -c

        Since f(α,α) = c, the identity says result = -f(α,α) = -c
        """
        c = 3.0

        def f_const(alpha, gamma):
            return c

        alpha = 0.5
        result = apply_neat_identity(f_const, alpha)

        # Should equal -f(α,α) = -c
        expected = -f_const(alpha, alpha)
        assert abs(result - expected) < 1e-10, (
            f"For constant f={c}: expected {expected}, got {result}"
        )

    def test_neat_identity_with_linear_f(self):
        """
        For f(α,γ) = α + γ:
        f(α,α) = 2α
        Result should be -2α
        """
        def f_linear(alpha, gamma):
            return alpha + gamma

        alpha = 0.3
        result = apply_neat_identity(f_linear, alpha)

        expected = -f_linear(alpha, alpha)  # -2α
        assert abs(result - expected) < 1e-10, (
            f"For linear f: expected {expected}, got {result}"
        )

    def test_neat_identity_with_quadratic_f(self):
        """
        For f(α,γ) = (α+γ)²:
        f(α,α) = (2α)² = 4α²
        Result should be -4α²
        """
        def f_quadratic(alpha, gamma):
            return (alpha + gamma) ** 2

        alpha = 0.4
        result = apply_neat_identity(f_quadratic, alpha)

        expected = -f_quadratic(alpha, alpha)  # -4α²
        assert abs(result - expected) < 1e-10, (
            f"For quadratic f: expected {expected}, got {result}"
        )


class TestDiagonalLimit:
    """Test the full diagonal limit γ=α, δ=β."""

    def test_diagonal_limit_is_finite(self):
        """Diagonalization should produce finite result, not NaN/inf."""
        # Simple test expression
        def expr(alpha, beta, gamma, delta):
            return alpha + beta + gamma + delta

        result = diagonalize_gamma_eq_alpha_delta_eq_beta(
            expr, alpha=0.1, beta=0.2
        )

        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_diagonal_limit_with_pole_structure(self):
        """
        For expressions with 1/ζ(1-α+γ) structure,
        the diagonal limit should still be finite.
        """
        def expr_with_pole(alpha, beta, gamma, delta, eps=None):
            # This simulates a function with pole at γ=α
            # The limit machinery should handle this
            if eps is None:
                eps = gamma - alpha
            if abs(eps) < 1e-14:
                # At exact diagonal, return the limit value
                return alpha + beta  # placeholder
            # Away from diagonal
            zeta_factor = 1.0 / eps  # simulates 1/ζ(1-α+γ) pole
            return (alpha + gamma) * zeta_factor

        # The diagonalization should handle this
        result = diagonalize_gamma_eq_alpha_delta_eq_beta(
            expr_with_pole, alpha=0.1, beta=0.2, mode="limit"
        )

        assert np.isfinite(result), f"Result should be finite, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
