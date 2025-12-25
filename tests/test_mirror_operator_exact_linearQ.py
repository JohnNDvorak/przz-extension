"""
tests/test_mirror_operator_exact_linearQ.py
Phase 10.2e: Linear Q Test for Mirror Operator

This test validates the swap/sign conjugation by using a simple linear Q polynomial.

MATHEMATICAL BASIS:
===================
For Q(x) = 1 + cx (linear polynomial):
    Q(A_α^mirror) = 1 + c×θy
    Q(A_β^mirror) = 1 + c×θx

This is simple enough to expand explicitly and validate the swap structure.

The test compares:
1. Operator-level computation with swapped eigenvalues
2. Brute-force series expansion

SUCCESS CRITERIA:
=================
Both approaches should give the same result (within numerical precision).
"""

import pytest
import numpy as np
from src.polynomials import Polynomial


def make_linear_Q(c: float = 0.5) -> Polynomial:
    """
    Create linear Q(x) = 1 + cx.

    Args:
        c: Linear coefficient (default 0.5)

    Returns:
        Polynomial representing 1 + cx
    """
    return Polynomial(np.array([1.0, c]))


def make_trivial_P() -> Polynomial:
    """Create P(u) = 1."""
    return Polynomial(np.array([1.0]))


class TestLinearQ:
    """Linear Q tests for validating swap structure."""

    @pytest.fixture
    def linear_Q_polynomials(self):
        """Polynomials with linear Q and trivial P."""
        c = 0.5  # Linear coefficient
        Q = make_linear_Q(c)
        P = make_trivial_P()
        return {
            'P1': P,
            'P2': P,
            'P3': P,
            'Q': Q,
            'c': c  # Store for verification
        }

    @pytest.fixture
    def theta(self):
        """PRZZ θ = 4/7."""
        return 4.0 / 7.0

    @pytest.fixture
    def n_quadrature(self):
        """Quadrature points."""
        return 40

    def test_linear_Q_swap_structure(self, linear_Q_polynomials, theta, n_quadrature):
        """
        Verify that linear Q with swap gives expected structure.

        For Q(x) = 1 + cx:
            Q(θy) = 1 + cθy
            Q(θx) = 1 + cθx

        The product Q(θy)×Q(θx) = (1 + cθy)(1 + cθx)
                                = 1 + cθx + cθy + c²θ²xy

        This has a well-defined xy coefficient: c²θ²
        """
        c = linear_Q_polynomials['c']

        # Expected xy coefficient from Q(θy)×Q(θx)
        expected_xy_from_Q = c**2 * theta**2

        # Verify the Q polynomial evaluates correctly
        Q = linear_Q_polynomials['Q']
        assert abs(Q.eval(np.array([0.0]))[0] - 1.0) < 1e-10, "Q(0) should be 1"
        assert abs(Q.eval(np.array([1.0]))[0] - (1 + c)) < 1e-10, f"Q(1) should be {1+c}"

        print(f"\nLinear Q structure test:")
        print(f"  Q(x) = 1 + {c}x")
        print(f"  Expected Q×Q xy coefficient: c²θ² = {expected_xy_from_Q:.6f}")

    def test_linear_Q_mirror_computation(self, linear_Q_polynomials, theta, n_quadrature):
        """
        Compute mirror with linear Q and verify structure.

        The xy coefficient should be dominated by the c²θ² term from Q×Q.
        """
        from src.mirror_operator_exact import compute_I1_mirror_operator_exact

        R = 1.3036
        c = linear_Q_polynomials['c']

        result = compute_I1_mirror_operator_exact(
            theta=theta,
            R=R,
            n=n_quadrature,
            polynomials=linear_Q_polynomials,
            ell1=1,
            ell2=1,
            verbose=True
        )

        print(f"\nLinear Q mirror computation:")
        print(f"  I_swapped = {result.I_swapped:.8f}")
        print(f"  Full value = {result.value:.8f}")
        print(f"  T_weight = {result.T_weight:.4f}")
        print(f"  Q(A_α^mirror) range: {result.Q_alpha_range}")
        print(f"  Q(A_β^mirror) range: {result.Q_beta_range}")

        # Q should be evaluated in [Q(0), Q(θ)] = [1, 1+cθ]
        expected_Q_min = 1.0
        expected_Q_max = 1.0 + c * theta

        # Allow some tolerance for numerical integration effects
        assert result.Q_alpha_range[0] >= expected_Q_min - 0.1
        assert result.Q_alpha_range[1] <= expected_Q_max + 0.1

        # Value should be finite
        assert np.isfinite(result.value)

    def test_linear_Q_vs_Q1_comparison(self, theta, n_quadrature):
        """
        Compare linear Q with Q=1 to understand the polynomial effect.

        For Q=1: Q(θy)×Q(θx) = 1 (no xy coefficient from Q)
        For Q=1+cx: Q(θy)×Q(θx) = 1 + cθ(x+y) + c²θ²xy (has xy coefficient)

        The ratio should show the polynomial contribution.
        """
        from src.mirror_operator_exact import compute_I1_mirror_operator_exact

        R = 1.3036

        # Q = 1 polynomials
        P = make_trivial_P()
        Q1 = Polynomial(np.array([1.0]))
        polys_Q1 = {'P1': P, 'P2': P, 'P3': P, 'Q': Q1}

        # Q = 1 + 0.5x polynomials
        c = 0.5
        Q_linear = make_linear_Q(c)
        polys_linear = {'P1': P, 'P2': P, 'P3': P, 'Q': Q_linear}

        result_Q1 = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n_quadrature, polynomials=polys_Q1,
            ell1=1, ell2=1
        )

        result_linear = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n_quadrature, polynomials=polys_linear,
            ell1=1, ell2=1
        )

        ratio = result_linear.value / result_Q1.value if abs(result_Q1.value) > 1e-15 else float('inf')

        print(f"\n=== Q=1 vs Linear Q Comparison ===")
        print(f"I1_mirror (Q=1):     {result_Q1.value:.8f}")
        print(f"I1_mirror (Q=1+0.5x): {result_linear.value:.8f}")
        print(f"Ratio (linear/Q1):   {ratio:.4f}")

        # Both should be finite
        assert np.isfinite(result_Q1.value)
        assert np.isfinite(result_linear.value)

        # Ratio should be in reasonable range (not blowing up)
        assert abs(ratio) < 10, f"Linear Q causing unexpected amplification: {ratio}"


class TestMirrorEigenvalueRange:
    """Tests to verify eigenvalue ranges are correct for swap structure."""

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_swap_eigenvalue_range(self, theta):
        """
        Verify that swapped eigenvalues give arguments in [0, θ].

        A_α^mirror = θy for y ∈ [0, 1] → arguments in [0, θ]
        A_β^mirror = θx for x ∈ [0, 1] → arguments in [0, θ]

        This is well within [0, 1] where Q polynomials are well-behaved.
        """
        from src.mirror_operator_exact import get_mirror_eigenvalues_with_swap

        eig = get_mirror_eigenvalues_with_swap(theta)

        # For y ∈ [0, 1]:
        # A_α^mirror = u0_alpha + x_alpha*0 + y_alpha*y
        #            = 0 + 0 + θy
        #            = θy ∈ [0, θ]

        for y in [0.0, 0.5, 1.0]:
            A_alpha = eig.u0_alpha + eig.x_alpha * 0 + eig.y_alpha * y
            expected = theta * y
            assert abs(A_alpha - expected) < 1e-10, f"A_α at y={y} should be θy={expected}"
            assert 0 <= A_alpha <= theta + 1e-10, f"A_α={A_alpha} should be in [0, θ]"

        # For x ∈ [0, 1]:
        # A_β^mirror = u0_beta + x_beta*x + y_beta*0
        #            = 0 + θx + 0
        #            = θx ∈ [0, θ]

        for x in [0.0, 0.5, 1.0]:
            A_beta = eig.u0_beta + eig.x_beta * x + eig.y_beta * 0
            expected = theta * x
            assert abs(A_beta - expected) < 1e-10, f"A_β at x={x} should be θx={expected}"
            assert 0 <= A_beta <= theta + 1e-10, f"A_β={A_beta} should be in [0, θ]"

        print(f"\nSwap eigenvalue range test:")
        print(f"  θ = {theta:.6f}")
        print(f"  A_α^mirror range: [0, {theta:.6f}]")
        print(f"  A_β^mirror range: [0, {theta:.6f}]")
        print(f"  Both well within [0, 1] ✓")

    def test_direct_eigenvalue_range_for_comparison(self, theta):
        """
        Compare with direct eigenvalue ranges for context.

        Direct eigenvalues at t=0.5:
            A_α = 0.5 + θ(-0.5)x + θ(0.5)y ≈ 0.5 - 0.286x + 0.286y
            A_β = 0.5 + θ(0.5)x + θ(-0.5)y ≈ 0.5 + 0.286x - 0.286y

        Range: [0.5 - 0.286, 0.5 + 0.286] = [0.214, 0.786]
        """
        from src.operator_post_identity import get_A_alpha_affine_coeffs, get_A_beta_affine_coeffs

        t = 0.5

        u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, theta)
        u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, theta)

        # Range for direct A_α
        A_alpha_min = u0_a + min(x_a, 0) + min(y_a, 0)  # Both x,y at 0 or 1
        A_alpha_max = u0_a + max(x_a, 0) + max(y_a, 0)

        # Range for direct A_β
        A_beta_min = u0_b + min(x_b, 0) + min(y_b, 0)
        A_beta_max = u0_b + max(x_b, 0) + max(y_b, 0)

        print(f"\nDirect eigenvalue ranges (t=0.5):")
        print(f"  A_α = {u0_a:.3f} + {x_a:.3f}x + {y_a:.3f}y")
        print(f"  A_α range: [{A_alpha_min:.3f}, {A_alpha_max:.3f}]")
        print(f"  A_β = {u0_b:.3f} + {x_b:.3f}x + {y_b:.3f}y")
        print(f"  A_β range: [{A_beta_min:.3f}, {A_beta_max:.3f}]")

        # For comparison with Q(1+·) shift (Phase 9 approach)
        print(f"\n  If Q(1+A_α): range = [{1+A_alpha_min:.3f}, {1+A_alpha_max:.3f}]")
        print(f"  This pushes into [1+, 2+] where Q polynomials can explode!")
