"""
tests/test_i2_mirror_gate.py
Phase 13.1: I₂ Mirror Gate - Simplest litmus test for mirror semantics.

I₂ is the SIMPLEST of the I-terms because:
1. No derivatives (empty vars=())
2. Separable: I₂ = (1/θ) × [∫P₁P₂ du] × [∫Q(t)² exp(2Rt) dt]
3. Mirror structure is explicit in TeX

If I₂ mirror fails, the bug is in basic transform, not I₁ complexity.

EXPECTED RELATIONSHIPS:
========================
Direct I₂(+R):
    = (1/θ) × ∫P₁P₂(u) du × ∫Q(t)² exp(2Rt) dt

Direct I₂ at t' = 1-t (t-flip reference):
    By change of variable s = 1-t:
    = (1/θ) × ∫P₁P₂(u) du × ∫Q(s)² exp(2Rs) ds
    = I₂(+R)  [same integral!]

Mirror I₂ (Phase 13):
    = T_weight × ∫∫ (1/θ) × P₁P₂(u) × Q(1-t)² × exp(-2Rt) du dt
    = exp(2R) × (1/θ) × ∫P₁P₂ du × ∫Q(1-t)² exp(-2Rt) dt

With s = 1-t substitution in t-integral:
    ∫Q(1-t)² exp(-2Rt) dt = ∫Q(s)² exp(-2R(1-s)) ds
                          = exp(-2R) × ∫Q(s)² exp(2Rs) ds

So: Mirror I₂ = exp(2R) × exp(-2R) × I₂(+R) = I₂(+R)

WAIT - this means Mirror I₂ should EQUAL Direct I₂!
This is because I₂ has no x,y dependence, so the mirror eigenvalue
structure doesn't affect it beyond the exp factor transformation.

But empirically we know mirror contribution is m₁ × I₂(-R), not I₂(+R).
So there's a semantic difference between operator-derived and empirical.

This test validates what the operator approach actually gives.
"""

import pytest
import numpy as np
from src.polynomials import load_przz_polynomials
from src.mirror_operator_exact import (
    compute_I2_mirror_operator_exact,
    get_mirror_eigenvalues_complement_t,
)
from src.quadrature import gauss_legendre_01


THETA = 4.0 / 7.0
R = 1.3036
N_QUAD = 40


@pytest.fixture
def polynomials():
    """Load PRZZ polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}


class TestI2DirectStructure:
    """Validate the structure of direct I₂."""

    def test_i2_is_separable(self, polynomials):
        """I₂ should be separable into u-integral × t-integral."""
        u_nodes, u_weights = gauss_legendre_01(N_QUAD)
        t_nodes, t_weights = gauss_legendre_01(N_QUAD)

        Q = polynomials['Q']
        P1 = polynomials['P1']

        # 2D integral
        I2_2d = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            for t, w_t in zip(t_nodes, t_weights):
                P_val = P1.eval(np.array([u]))[0] ** 2
                Q_val = Q.eval(np.array([t]))[0] ** 2
                exp_val = np.exp(2 * R * t)
                I2_2d += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        # Product of 1D integrals
        u_integral = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            u_integral += P_val * w_u

        t_integral = 0.0
        for t, w_t in zip(t_nodes, t_weights):
            Q_val = Q.eval(np.array([t]))[0] ** 2
            exp_val = np.exp(2 * R * t)
            t_integral += Q_val * exp_val * w_t

        I2_product = (1.0 / THETA) * u_integral * t_integral

        # Should match within numerical precision
        rel_error = abs(I2_2d - I2_product) / abs(I2_2d)
        assert rel_error < 1e-10, f"I₂ not separable: rel_error = {rel_error}"


class TestI2MirrorVsDirectTFlip:
    """Compare I₂ mirror to I₂ direct with t-flip."""

    def test_mirror_t_integrand_structure(self, polynomials):
        """
        Verify the mirror t-integrand structure.

        Mirror t-integrand at t: Q(1-t)² × exp(-2Rt)
        Direct t-integrand at 1-t: Q(1-t)² × exp(2R(1-t))

        After multiplying by T_weight = exp(2R):
        Mirror total: exp(2R) × exp(-2Rt) × Q(1-t)² = exp(2R(1-t)) × Q(1-t)²
        = Direct at 1-t

        So integrand-wise, they should match!
        """
        Q = polynomials['Q']

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Mirror integrand (before T_weight)
            Q_at_1_minus_t = Q.eval(np.array([1 - t]))[0] ** 2
            mirror_exp = np.exp(-2 * R * t)
            mirror_integrand = Q_at_1_minus_t * mirror_exp

            # Direct integrand at 1-t
            direct_exp_at_1_minus_t = np.exp(2 * R * (1 - t))
            direct_integrand = Q_at_1_minus_t * direct_exp_at_1_minus_t

            # After T_weight
            T_weight = np.exp(2 * R)
            mirror_with_T = T_weight * mirror_integrand

            # Should match
            rel_error = abs(mirror_with_T - direct_integrand) / abs(direct_integrand)
            assert rel_error < 1e-10, (
                f"At t={t}: mirror_with_T={mirror_with_T}, direct={direct_integrand}"
            )

    def test_mirror_integral_equals_direct_integral(self, polynomials):
        """
        After t-flip and T-weight, mirror I₂ should equal direct I₂.

        This is because the t-substitution s = 1-t transforms the integral back.
        """
        Q = polynomials['Q']
        P1 = polynomials['P1']

        u_nodes, u_weights = gauss_legendre_01(N_QUAD)
        t_nodes, t_weights = gauss_legendre_01(N_QUAD)

        # Direct I₂
        I2_direct = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            for t, w_t in zip(t_nodes, t_weights):
                Q_val = Q.eval(np.array([t]))[0] ** 2
                exp_val = np.exp(2 * R * t)
                I2_direct += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        # Mirror I₂ (Phase 13 style: Q(1-t)² × exp(-2Rt) × T_weight)
        I2_mirror_raw = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            for t, w_t in zip(t_nodes, t_weights):
                Q_val = Q.eval(np.array([1 - t]))[0] ** 2
                exp_val = np.exp(-2 * R * t)
                I2_mirror_raw += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        T_weight = np.exp(2 * R)
        I2_mirror = T_weight * I2_mirror_raw

        # They should be equal!
        rel_error = abs(I2_mirror - I2_direct) / abs(I2_direct)
        assert rel_error < 1e-10, (
            f"I2_mirror={I2_mirror}, I2_direct={I2_direct}, rel_error={rel_error}"
        )


class TestI2MirrorOperatorImplementation:
    """Test the actual compute_I2_mirror_operator_exact implementation."""

    def test_phase13_i2_mirror_equals_direct(self, polynomials):
        """
        With Phase 13 settings (use_t_dependent=True, use_t_flip_exp=True),
        the I₂ mirror should equal the direct I₂.

        This is a consequence of the t-flip symmetry for I₂.
        """
        # Compute direct I₂
        Q = polynomials['Q']
        P1 = polynomials['P1']

        u_nodes, u_weights = gauss_legendre_01(N_QUAD)
        t_nodes, t_weights = gauss_legendre_01(N_QUAD)

        I2_direct = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            for t, w_t in zip(t_nodes, t_weights):
                Q_val = Q.eval(np.array([t]))[0] ** 2
                exp_val = np.exp(2 * R * t)
                I2_direct += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        # Compute mirror I₂ using Phase 13 operator
        result = compute_I2_mirror_operator_exact(
            theta=THETA, R=R, n=N_QUAD,
            polynomials=polynomials, ell1=1, ell2=1,
            use_t_dependent=True,
            use_t_flip_exp=True
        )

        rel_error = abs(result.value - I2_direct) / abs(I2_direct)
        assert rel_error < 0.01, (
            f"I2_mirror={result.value}, I2_direct={I2_direct}, rel_error={rel_error}"
        )

    def test_i2_mirror_vs_empirical_basis(self, polynomials):
        """
        Compare operator-derived I₂ mirror to empirical I₂(-R).

        The empirical approach uses I₂(-R) as the mirror basis.
        The operator approach gives something different.

        This test documents the actual relationship.
        """
        Q = polynomials['Q']
        P1 = polynomials['P1']

        u_nodes, u_weights = gauss_legendre_01(N_QUAD)
        t_nodes, t_weights = gauss_legendre_01(N_QUAD)

        # Direct I₂(+R)
        I2_plus_R = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            for t, w_t in zip(t_nodes, t_weights):
                Q_val = Q.eval(np.array([t]))[0] ** 2
                exp_val = np.exp(2 * R * t)
                I2_plus_R += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        # Empirical basis: I₂(-R)
        I2_minus_R = 0.0
        for u, w_u in zip(u_nodes, u_weights):
            P_val = P1.eval(np.array([u]))[0] ** 2
            for t, w_t in zip(t_nodes, t_weights):
                Q_val = Q.eval(np.array([t]))[0] ** 2
                exp_val = np.exp(-2 * R * t)  # Note: -R
                I2_minus_R += (1.0 / THETA) * P_val * Q_val * exp_val * w_u * w_t

        # Operator-derived mirror (Phase 13)
        result = compute_I2_mirror_operator_exact(
            theta=THETA, R=R, n=N_QUAD,
            polynomials=polynomials, ell1=1, ell2=1,
            use_t_dependent=True,
            use_t_flip_exp=True
        )
        I2_operator_mirror = result.value

        # Document the relationships
        m1_empirical = np.exp(R) + 5
        empirical_mirror_contrib = m1_empirical * I2_minus_R

        print(f"\n--- I₂ Mirror Comparison ---")
        print(f"I2(+R) direct:             {I2_plus_R:.6f}")
        print(f"I2(-R) empirical basis:    {I2_minus_R:.6f}")
        print(f"I2 operator mirror:        {I2_operator_mirror:.6f}")
        print(f"m1 × I2(-R):               {empirical_mirror_contrib:.6f}")
        print(f"Ratio I2_op / I2(+R):      {I2_operator_mirror / I2_plus_R:.4f}")
        print(f"Ratio I2_op / (m1×I2(-R)): {I2_operator_mirror / empirical_mirror_contrib:.4f}")

        # The key finding: operator mirror ≈ I2(+R), NOT m1 × I2(-R)
        assert abs(I2_operator_mirror / I2_plus_R - 1.0) < 0.01, (
            "Operator mirror I₂ should approximately equal direct I₂(+R)"
        )


class TestI2EigenvalueAtXY0:
    """Test eigenvalue behavior at x=y=0 for I₂."""

    def test_complement_eigenvalue_at_xy0(self):
        """
        With complement eigenvalues, at x=y=0:
        A_α^mirror = (1-t), A_β^mirror = (1-t)

        So Q(A_α^mirror)|_{x=y=0} = Q(1-t)
        """
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            eig = get_mirror_eigenvalues_complement_t(t, THETA)

            # At x=y=0
            A_alpha_at_0 = eig.u0_alpha + eig.x_alpha * 0 + eig.y_alpha * 0
            A_beta_at_0 = eig.u0_beta + eig.x_beta * 0 + eig.y_beta * 0

            expected = 1 - t
            assert abs(A_alpha_at_0 - expected) < 1e-10, (
                f"At t={t}: A_alpha_at_0={A_alpha_at_0}, expected={expected}"
            )
            assert abs(A_beta_at_0 - expected) < 1e-10, (
                f"At t={t}: A_beta_at_0={A_beta_at_0}, expected={expected}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
