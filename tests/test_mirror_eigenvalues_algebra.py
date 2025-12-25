"""
tests/test_mirror_eigenvalues_algebra.py
Phase 12.1: Micro-tests for mirror eigenvalue algebra.

These tests validate the complement structure:
    A_α^mirror(t) = 1 - A_β(t)
    A_β^mirror(t) = 1 - A_α(t)
"""

import pytest
import numpy as np
from src.mirror_operator_exact import (
    get_mirror_eigenvalues_with_swap,
    get_mirror_eigenvalues_complement_t,
    MirrorEigenvalues,
)
from src.operator_post_identity import (
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
)


THETA = 4.0 / 7.0


class TestComplementAlgebra:
    """Verify the complement relationship A_α^mirror = 1 - A_β."""

    def test_complement_formula_at_t_0(self):
        """At t=0, check complement structure."""
        t = 0.0
        # Direct eigenvalues at t=0:
        # A_α(0) = 0 + θ(-1)x + 0y = -θx
        # A_β(0) = 0 + 0x + θ(-1)y = -θy
        u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, THETA)
        u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, THETA)

        assert abs(u0_a - 0.0) < 1e-10
        assert abs(x_a - THETA * (t - 1)) < 1e-10  # θ(-1) = -θ
        assert abs(y_a - THETA * t) < 1e-10  # 0

        # Mirror eigenvalues: 1 - A_β = 1 - (-θy) = 1 + θy
        # Expected: u0 = 1, x = 0, y = θ
        eig = get_mirror_eigenvalues_complement_t(t, THETA)
        expected_u0_alpha = 1 - u0_b  # 1 - 0 = 1
        expected_x_alpha = -x_b  # -0 = 0
        expected_y_alpha = -y_b  # -(-θ) = θ

        assert abs(eig.u0_alpha - expected_u0_alpha) < 1e-10
        assert abs(eig.x_alpha - expected_x_alpha) < 1e-10
        assert abs(eig.y_alpha - expected_y_alpha) < 1e-10

    def test_complement_formula_at_t_1(self):
        """At t=1, check complement structure."""
        t = 1.0
        # Direct eigenvalues at t=1:
        # A_α(1) = 1 + 0x + θy
        # A_β(1) = 1 + θx + 0y
        u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, THETA)
        u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, THETA)

        assert abs(u0_a - 1.0) < 1e-10
        assert abs(x_a - 0.0) < 1e-10  # θ(1-1) = 0
        assert abs(y_a - THETA) < 1e-10  # θ×1 = θ

        # Mirror eigenvalues: 1 - A_β = 1 - (1 + θx) = -θx
        # Expected: u0 = 0, x = -θ, y = 0
        eig = get_mirror_eigenvalues_complement_t(t, THETA)
        expected_u0_alpha = 1 - u0_b  # 1 - 1 = 0
        expected_x_alpha = -x_b  # -θ
        expected_y_alpha = -y_b  # 0

        assert abs(eig.u0_alpha - expected_u0_alpha) < 1e-10
        assert abs(eig.x_alpha - expected_x_alpha) < 1e-10
        assert abs(eig.y_alpha - expected_y_alpha) < 1e-10

    def test_complement_formula_at_t_half(self):
        """At t=0.5, check complement structure."""
        t = 0.5
        u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, THETA)
        u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, THETA)

        # A_α(0.5) = 0.5 + θ(-0.5)x + θ(0.5)y = 0.5 - 0.5θx + 0.5θy
        # A_β(0.5) = 0.5 + θ(0.5)x + θ(-0.5)y = 0.5 + 0.5θx - 0.5θy

        eig = get_mirror_eigenvalues_complement_t(t, THETA)

        # A_α^mirror = 1 - A_β = 0.5 - 0.5θx + 0.5θy
        # (This is the SAME as A_α at t=0.5!)
        assert abs(eig.u0_alpha - 0.5) < 1e-10
        assert abs(eig.x_alpha - (-THETA * 0.5)) < 1e-10
        assert abs(eig.y_alpha - (THETA * 0.5)) < 1e-10

    def test_complement_is_1_minus_beta(self):
        """Verify A_α^mirror = 1 - A_β numerically for several t values."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, THETA)
            eig = get_mirror_eigenvalues_complement_t(t, THETA)

            # A_α^mirror should equal 1 - A_β evaluated at (x, y)
            # 1 - A_β = 1 - (u0_b + x_b*x + y_b*y) = (1-u0_b) - x_b*x - y_b*y
            expected_u0 = 1 - u0_b
            expected_x = -x_b
            expected_y = -y_b

            assert abs(eig.u0_alpha - expected_u0) < 1e-10, f"u0 mismatch at t={t}"
            assert abs(eig.x_alpha - expected_x) < 1e-10, f"x mismatch at t={t}"
            assert abs(eig.y_alpha - expected_y) < 1e-10, f"y mismatch at t={t}"

    def test_complement_is_1_minus_alpha_for_beta(self):
        """Verify A_β^mirror = 1 - A_α numerically for several t values."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, THETA)
            eig = get_mirror_eigenvalues_complement_t(t, THETA)

            # A_β^mirror should equal 1 - A_α
            expected_u0 = 1 - u0_a
            expected_x = -x_a
            expected_y = -y_a

            assert abs(eig.u0_beta - expected_u0) < 1e-10, f"u0 mismatch at t={t}"
            assert abs(eig.x_beta - expected_x) < 1e-10, f"x mismatch at t={t}"
            assert abs(eig.y_beta - expected_y) < 1e-10, f"y mismatch at t={t}"


class TestQArgumentRange:
    """Verify Q polynomial arguments stay in a reasonable range."""

    def test_phase10_static_range(self):
        """Phase 10 static eigenvalues: Q args in [0, θ]."""
        eig = get_mirror_eigenvalues_with_swap(THETA)

        # A_α^mirror = θy, so Q(A_α^mirror) for y ∈ [0,1] gives Q(0) to Q(θ)
        # θ ≈ 0.571, so Q args are in [0, 0.571] - safe zone
        max_arg_alpha = eig.u0_alpha + eig.x_alpha * 1 + eig.y_alpha * 1
        min_arg_alpha = eig.u0_alpha + eig.x_alpha * 0 + eig.y_alpha * 0

        assert min_arg_alpha >= 0
        assert max_arg_alpha <= 1.0  # Should be θ ≈ 0.571

    def test_phase12_complement_range(self):
        """Phase 12 complement eigenvalues: check Q argument ranges.

        The complement structure A^mirror = 1 - A gives:
        - At t=0: range [1, 1+θ] ≈ [1, 1.57]
        - At t=1: range [-θ, 0] ≈ [-0.57, 0]
        - At t=0.5: range [0.21, 0.79] (well within [0,1])

        Key insight: Phase 9 had arguments going to 1.8+ causing Q blowup.
        Phase 12 stays within [−0.6, 1.6] - manageable for Q polynomial.
        """
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            eig = get_mirror_eigenvalues_complement_t(t, THETA)

            # Evaluate at corners of [0,1]² to find range
            corners = [(0, 0), (0, 1), (1, 0), (1, 1)]

            alpha_vals = [
                eig.u0_alpha + eig.x_alpha * x + eig.y_alpha * y
                for x, y in corners
            ]
            beta_vals = [
                eig.u0_beta + eig.x_beta * x + eig.y_beta * y
                for x, y in corners
            ]

            # The complement structure stays bounded (vs Phase 9's blowup to 1.8+)
            # Allow up to 1+θ ≈ 1.57 which is theoretical max at t=0
            assert max(alpha_vals) <= 1.6, f"A_α^mirror too large at t={t}: {max(alpha_vals)}"
            assert min(alpha_vals) >= -0.6, f"A_α^mirror too negative at t={t}: {min(alpha_vals)}"
            assert max(beta_vals) <= 1.6, f"A_β^mirror too large at t={t}: {max(beta_vals)}"
            assert min(beta_vals) >= -0.6, f"A_β^mirror too negative at t={t}: {min(beta_vals)}"


class TestPhase10VsPhase12Comparison:
    """Compare Phase 10 (static) vs Phase 12 (t-dependent) eigenvalues."""

    def test_phase10_has_no_t_dependence(self):
        """Phase 10 eigenvalues should be the same for all t."""
        eig0 = get_mirror_eigenvalues_with_swap(THETA)

        # Just verify it returns consistent values
        assert eig0.u0_alpha == 0.0
        assert eig0.u0_beta == 0.0
        assert abs(eig0.y_alpha - THETA) < 1e-10
        assert abs(eig0.x_beta - THETA) < 1e-10

    def test_phase12_has_t_dependence(self):
        """Phase 12 eigenvalues should vary with t."""
        eig0 = get_mirror_eigenvalues_complement_t(0.0, THETA)
        eig1 = get_mirror_eigenvalues_complement_t(1.0, THETA)

        # At t=0 and t=1, the u0 values should differ
        assert eig0.u0_alpha != eig1.u0_alpha, "u0_alpha should vary with t"
        assert eig0.u0_beta != eig1.u0_beta, "u0_beta should vary with t"

        # The coefficients should also vary
        assert eig0.x_alpha != eig1.x_alpha, "x_alpha should vary with t"
        assert eig0.y_alpha != eig1.y_alpha, "y_alpha should vary with t"

    def test_symmetry_at_t_half(self):
        """At t=0.5, mirror eigenvalues have special symmetry."""
        t = 0.5
        eig = get_mirror_eigenvalues_complement_t(t, THETA)

        # At t=0.5: A_α^mirror = 1 - A_β
        # A_β(0.5) = 0.5 + 0.5θx - 0.5θy
        # So A_α^mirror = 0.5 - 0.5θx + 0.5θy = A_α(0.5)!
        # This means at t=0.5, the complement equals the direct!

        u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, THETA)

        assert abs(eig.u0_alpha - u0_a) < 1e-10
        assert abs(eig.x_alpha - x_a) < 1e-10
        assert abs(eig.y_alpha - y_a) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
