"""
tests/test_combined_identity_regularized.py
Tests for the u-regularized combined identity (pole-free approach).

Key tests:
1. Q=1 identity test: regularized form matches original bracket
2. Symmetry test: (x,α) ↔ (y,β) swap behaves correctly
3. Eigenvalue computation correctness
4. L-scaling: should NOT diverge like Leibniz method
5. Integration with polynomial infrastructure
"""

import pytest
import numpy as np

from src.combined_identity_regularized import (
    compute_A_alpha,
    compute_A_beta,
    compute_kernel_E,
    compute_bracket_original,
    compute_bracket_regularized,
    compute_QQB_regularized,
    evaluate_QQ_on_kernel_E,
    compute_I1_combined_regularized_at_L,
    analyze_L_convergence_regularized,
    RegularizedI1Result,
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


class TestEigenvalues:
    """Test eigenvalue computations A_α and A_β."""

    def test_A_alpha_at_u_zero(self):
        """At u=0, A_α = 1 + θy (from formula)."""
        theta = 4.0 / 7.0
        x, y = 0.3, 0.4

        A_alpha = compute_A_alpha(u=0.0, x=x, y=y, theta=theta)
        expected = 1.0 + theta * y

        assert np.isclose(A_alpha, expected), f"A_α(0,x,y) = {A_alpha}, expected {expected}"

    def test_A_alpha_at_u_one(self):
        """At u=1, A_α = -θx (from formula)."""
        theta = 4.0 / 7.0
        x, y = 0.3, 0.4

        A_alpha = compute_A_alpha(u=1.0, x=x, y=y, theta=theta)
        expected = -theta * x

        assert np.isclose(A_alpha, expected), f"A_α(1,x,y) = {A_alpha}, expected {expected}"

    def test_A_beta_at_u_zero(self):
        """At u=0, A_β = 1 + θx (from formula)."""
        theta = 4.0 / 7.0
        x, y = 0.3, 0.4

        A_beta = compute_A_beta(u=0.0, x=x, y=y, theta=theta)
        expected = 1.0 + theta * x

        assert np.isclose(A_beta, expected), f"A_β(0,x,y) = {A_beta}, expected {expected}"

    def test_A_beta_at_u_one(self):
        """At u=1, A_β = -θy (from formula)."""
        theta = 4.0 / 7.0
        x, y = 0.3, 0.4

        A_beta = compute_A_beta(u=1.0, x=x, y=y, theta=theta)
        expected = -theta * y

        assert np.isclose(A_beta, expected), f"A_β(1,x,y) = {A_beta}, expected {expected}"

    def test_symmetry_xy_swap(self):
        """Swapping (x,α) ↔ (y,β) should swap A_α ↔ A_β."""
        theta = 4.0 / 7.0
        x, y = 0.3, 0.4
        u = 0.5

        A_alpha_xy = compute_A_alpha(u, x, y, theta)
        A_beta_xy = compute_A_beta(u, x, y, theta)

        A_alpha_yx = compute_A_alpha(u, y, x, theta)  # x↔y swapped
        A_beta_yx = compute_A_beta(u, y, x, theta)

        assert np.isclose(A_alpha_xy, A_beta_yx), \
            f"Symmetry failed: A_α(x,y) = {A_alpha_xy}, A_β(y,x) = {A_beta_yx}"
        assert np.isclose(A_beta_xy, A_alpha_yx), \
            f"Symmetry failed: A_β(x,y) = {A_beta_xy}, A_α(y,x) = {A_alpha_yx}"

    def test_at_xy_zero(self):
        """At x=y=0, A_α = A_β = (1-u)."""
        theta = 4.0 / 7.0
        u = 0.6

        A_alpha = compute_A_alpha(u, x=0, y=0, theta=theta)
        A_beta = compute_A_beta(u, x=0, y=0, theta=theta)

        expected = 1 - u
        assert np.isclose(A_alpha, expected)
        assert np.isclose(A_beta, expected)
        assert np.isclose(A_alpha, A_beta)


class TestKernelE:
    """Test the regularized kernel E."""

    def test_kernel_finite(self):
        """Kernel should be finite for reasonable parameters."""
        theta = 4.0 / 7.0
        L = 100.0
        R = 1.3036
        alpha = -R / L
        beta = -R / L
        x, y = 0.3, 0.4
        u = 0.5

        E = compute_kernel_E(alpha, beta, x, y, u, theta, L)
        assert np.isfinite(E), f"Kernel not finite: {E}"

    def test_kernel_at_u_endpoints(self):
        """Verify kernel at u=0 and u=1 matches boundary conditions."""
        theta = 4.0 / 7.0
        L = 50.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4
        s = alpha + beta

        # At u=1: E(u=1) = T^0 N^{-βx-αy+s(x+y)} = N^{αx+βy}
        E_u1 = compute_kernel_E(alpha, beta, x, y, u=1.0, theta=theta, L=L)
        expected_u1 = np.exp(theta * L * (alpha * x + beta * y))
        assert np.isclose(E_u1, expected_u1, rtol=1e-10), \
            f"E(u=1) = {E_u1}, expected {expected_u1}"

        # At u=0: E(u=0) = T^{-s} N^{-βx-αy} = exp(-Ls) exp(-θL(βx+αy))
        E_u0 = compute_kernel_E(alpha, beta, x, y, u=0.0, theta=theta, L=L)
        expected_u0 = np.exp(-L * s - theta * L * (beta * x + alpha * y))
        assert np.isclose(E_u0, expected_u0, rtol=1e-10), \
            f"E(u=0) = {E_u0}, expected {expected_u0}"


class TestQ1IdentityTest:
    """CRITICAL TEST: For Q=1, regularized form must match original bracket."""

    def test_bracket_regularized_matches_original(self):
        """Regularized bracket should equal original bracket."""
        theta = 4.0 / 7.0
        L = 50.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        B_original = compute_bracket_original(alpha, beta, x, y, theta, L)
        B_regularized = compute_bracket_regularized(alpha, beta, x, y, theta, L, n_quad=50)

        rel_error = abs(B_regularized - B_original) / (abs(B_original) + 1e-100)
        print(f"\nQ=1 Identity Test:")
        print(f"  B_original    = {B_original:.10e}")
        print(f"  B_regularized = {B_regularized:.10e}")
        print(f"  Relative error = {rel_error:.2e}")

        assert rel_error < 1e-6, \
            f"Regularized bracket mismatch: rel_error = {rel_error:.2e}"

    def test_bracket_at_przz_point(self):
        """Test at the PRZZ evaluation point α=β=-R/L."""
        theta = 4.0 / 7.0
        R = 1.3036
        L = 100.0
        alpha = beta = -R / L
        x, y = 0.3, 0.4

        B_original = compute_bracket_original(alpha, beta, x, y, theta, L)
        B_regularized = compute_bracket_regularized(alpha, beta, x, y, theta, L, n_quad=50)

        rel_error = abs(B_regularized - B_original) / (abs(B_original) + 1e-100)

        assert rel_error < 1e-5, \
            f"PRZZ point bracket mismatch: rel_error = {rel_error:.2e}"

    def test_bracket_at_multiple_points(self):
        """Test at multiple random points for robustness."""
        theta = 4.0 / 7.0
        L = 50.0

        np.random.seed(42)
        test_points = [
            (0.1, 0.2, 0.3, 0.4),
            (0.05, 0.15, 0.5, 0.5),
            (-0.05, -0.05, 0.2, 0.8),
            (0.2, 0.1, 0.0, 0.0),
        ]

        for alpha, beta, x, y in test_points:
            if abs(alpha + beta) < 0.01:
                continue  # Skip near-pole points

            B_original = compute_bracket_original(alpha, beta, x, y, theta, L)
            B_regularized = compute_bracket_regularized(alpha, beta, x, y, theta, L, n_quad=50)

            rel_error = abs(B_regularized - B_original) / (abs(B_original) + 1e-100)
            assert rel_error < 1e-5, \
                f"Bracket mismatch at ({alpha},{beta},{x},{y}): rel_error = {rel_error:.2e}"


class TestSymmetryTest:
    """Test that swapping (x,α) ↔ (y,β) behaves correctly."""

    def test_bracket_swap_symmetry(self):
        """Bracket should have correct behavior under (x,α) ↔ (y,β)."""
        theta = 4.0 / 7.0
        L = 50.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        B_xy = compute_bracket_original(alpha, beta, x, y, theta, L)
        B_yx = compute_bracket_original(beta, alpha, y, x, theta, L)

        # The bracket has structure:
        # B(α,β,x,y) = [exp(θL(αx+βy)) - exp(-Ls)exp(-θL(βx+αy))] / s
        # Swapping (α,x) ↔ (β,y):
        # B(β,α,y,x) = [exp(θL(βy+αx)) - exp(-Ls)exp(-θL(αy+βx))] / s
        # = B(α,β,x,y)
        # So the bracket should be symmetric!

        assert np.isclose(B_xy, B_yx, rtol=1e-10), \
            f"Bracket not symmetric: B(α,β,x,y)={B_xy}, B(β,α,y,x)={B_yx}"


class TestQQBRegularized:
    """Test Q(D)Q(D)B using regularized form."""

    def test_Q1_gives_bracket(self, polys_kappa):
        """For Q=1, Q(D)Q(D)B should equal B."""
        from src.polynomials import Polynomial

        # Create Q=1 polynomial
        class Q1Poly:
            def eval(self, x):
                return 1.0

        Q1 = Q1Poly()

        theta = 4.0 / 7.0
        L = 50.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        B_original = compute_bracket_original(alpha, beta, x, y, theta, L)
        QQB = compute_QQB_regularized(Q1, alpha, beta, x, y, theta, L, n_quad=50)

        rel_error = abs(QQB - B_original) / (abs(B_original) + 1e-100)
        assert rel_error < 1e-5, \
            f"Q=1 should give B: QQB={QQB}, B={B_original}, rel_error={rel_error:.2e}"

    def test_QQB_finite_with_real_Q(self, polys_kappa):
        """QQB should be finite with real Q polynomial."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        L = 50.0
        R = 1.3036
        alpha = beta = -R / L
        x, y = 0.3, 0.4

        QQB = compute_QQB_regularized(Q, alpha, beta, x, y, theta, L, n_quad=50)

        assert np.isfinite(QQB), f"QQB not finite: {QQB}"


class TestLScalingRegularized:
    """Test that regularized method doesn't diverge with L."""

    def test_QQB_not_exploding(self, polys_kappa):
        """QQB should not grow exponentially with L like Leibniz method."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        R = 1.3036
        x, y = 0.0, 0.0  # Simplest case

        L_values = [10, 50, 100]
        results = []

        for L in L_values:
            alpha = beta = -R / L
            QQB = compute_QQB_regularized(Q, alpha, beta, x, y, theta, L, n_quad=30)
            results.append((L, QQB))
            print(f"L={L}: QQB = {QQB:.6e}")

        # Check that values don't explode exponentially
        # Unlike Leibniz which gives 1e13, 1e21, 1e24...
        for L, QQB in results:
            assert np.isfinite(QQB), f"QQB not finite at L={L}"

        # Ratio test: values should be bounded, not exponentially growing
        if len(results) >= 2 and abs(results[0][1]) > 1e-100:
            ratio = abs(results[-1][1] / results[0][1])
            print(f"\nL={L_values[-1]}/L={L_values[0]} ratio: {ratio:.4f}")

            # Leibniz gave ratios of 1e10+, this should be much smaller
            assert ratio < 1e6, \
                f"Suspicious exponential growth: ratio = {ratio:.2e}"


class TestFullI1Computation:
    """Test the full I1 computation with regularized method."""

    def test_I1_returns_result(self, polys_kappa):
        """Should return RegularizedI1Result."""
        result = compute_I1_combined_regularized_at_L(
            theta=4/7, R=1.3036, L=50.0, n=15, polynomials=polys_kappa
        )

        assert isinstance(result, RegularizedI1Result)
        assert hasattr(result, 'I1_combined')
        assert hasattr(result, 'L')

    def test_I1_finite(self, polys_kappa):
        """I1 should be finite."""
        result = compute_I1_combined_regularized_at_L(
            theta=4/7, R=1.3036, L=50.0, n=15, polynomials=polys_kappa
        )

        assert np.isfinite(result.I1_combined), f"I1 not finite: {result.I1_combined}"

    def test_I1_L_sweep_bounded(self, polys_kappa):
        """I1 values should be bounded across L values (no divergence)."""
        L_values = [10, 50, 100]
        results = analyze_L_convergence_regularized(
            theta=4/7, R=1.3036, L_values=L_values, n=10, polynomials=polys_kappa
        )

        print("\n=== Regularized I1 L-Sweep ===")
        for result in results:
            print(f"L={result.L}: I1 = {result.I1_combined:.6e}")

        # All should be finite
        assert all(np.isfinite(r.I1_combined) for r in results)

        # Check for bounded behavior (not exponential growth)
        if abs(results[0].I1_combined) > 1e-100:
            ratio = abs(results[-1].I1_combined / results[0].I1_combined)
            print(f"\nRatio L={L_values[-1]}/L={L_values[0]}: {ratio:.4f}")

            # Should not show exponential growth
            assert ratio < 1e6, \
                f"Possible divergence: ratio = {ratio:.2e}"


class TestCompareWithLeibniz:
    """Compare regularized method with Leibniz method (documentation)."""

    def test_document_comparison(self, polys_kappa):
        """Document the difference between Leibniz and regularized methods."""
        from src.combined_identity_finite_L import compute_I1_combined_operator_at_L

        L_values = [10, 50, 100]

        print("\n=== REGULARIZED vs LEIBNIZ Comparison ===")
        print("L\t\tLEIBNIZ\t\t\tREGULARIZED")
        print("-" * 60)

        leibniz_results = []
        regularized_results = []

        for L in L_values:
            # Leibniz method (diverges)
            leibniz = compute_I1_combined_operator_at_L(
                theta=4/7, R=1.3036, L=L, n=10, polynomials=polys_kappa
            )
            leibniz_results.append(leibniz.I1_combined)

            # Regularized method (should be bounded)
            regularized = compute_I1_combined_regularized_at_L(
                theta=4/7, R=1.3036, L=L, n=10, polynomials=polys_kappa
            )
            regularized_results.append(regularized.I1_combined)

            print(f"{L}\t\t{leibniz.I1_combined:.6e}\t\t{regularized.I1_combined:.6e}")

        # Document that regularized is bounded while Leibniz explodes
        if abs(leibniz_results[0]) > 1e-100 and abs(regularized_results[0]) > 1e-100:
            leibniz_ratio = abs(leibniz_results[-1] / leibniz_results[0])
            reg_ratio = abs(regularized_results[-1] / regularized_results[0])

            print(f"\nLeibniz ratio L={L_values[-1]}/L={L_values[0]}: {leibniz_ratio:.2e}")
            print(f"Regularized ratio: {reg_ratio:.4f}")

            # Leibniz explodes (ratio > 1e10), regularized should be bounded
            assert leibniz_ratio > 1e6, "Leibniz should show exponential growth"
            # This test passes to document the comparison
            assert True
