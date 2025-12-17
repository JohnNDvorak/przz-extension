"""
PRZZ (2,2) Oracle - Exact Implementation of PRZZ TeX Formulas.

This computes the I₁, I₂, I₃, I₄ contributions for the ℓ₁=ℓ₂=1 case
(our (2,2) pair: μ⋆Λ × μ⋆Λ) using the exact formulas from PRZZ TeX
lines 1530-1570.

Key insight: The derivatives d²/dxdy and d/dx, d/dy must be computed
symbolically via the chain rule, NOT via finite differences.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, NamedTuple
from math import exp


class OracleResult22(NamedTuple):
    """Result of (2,2) oracle computation."""
    I1: float
    I2: float
    I3: float
    I4: float
    total: float


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def przz_oracle_22(
    P2,  # P₂ polynomial (for piece 2 = μ⋆Λ)
    Q,   # Q polynomial
    theta: float,
    R: float,
    n_quad: int = 60,
    debug: bool = False
) -> OracleResult22:
    """
    Compute the (2,2) contribution using exact PRZZ formulas.

    PRZZ uses P₁, P₂ in their formulas, but these are the polynomials
    for the two factors in the cross-term. For ℓ₁=ℓ₂=1 (μ⋆Λ × μ⋆Λ),
    both factors use our P₂ polynomial.

    Args:
        P2: The P₂ polynomial (used for both factors in (2,2))
        Q: The Q polynomial
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        debug: Print debug info

    Returns:
        OracleResult22 with I₁, I₂, I₃, I₄ and total
    """
    # Set up quadrature
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Precompute polynomial and derivative values at u-nodes
    # P2(u), P2'(u), P2''(u)
    P2_u = P2.eval(u_nodes)
    P2_prime_u = P2.eval_deriv(u_nodes, 1)
    P2_double_prime_u = P2.eval_deriv(u_nodes, 2)

    # Precompute Q values at t-nodes and derivatives
    # Q(t), Q'(t), Q''(t)
    Q_t = Q.eval(t_nodes)
    Q_prime_t = Q.eval_deriv(t_nodes, 1)
    Q_double_prime_t = Q.eval_deriv(t_nodes, 2)

    # =========================================================================
    # I₂ (simplest - no derivatives)
    # PRZZ line 1548:
    # I₂ = (1/θ) × ∫∫ Q(t)² e^{2Rt} P₁(u)P₂(u) du dt
    # For (2,2): both polynomials are P₂
    # =========================================================================

    # u-integral: ∫ P₂(u)² du
    u_integral_I2 = np.sum(u_weights * P2_u * P2_u)

    # t-integral: ∫ Q(t)² e^{2Rt} dt
    exp_2Rt = np.exp(2 * R * t_nodes)
    t_integral_I2 = np.sum(t_weights * Q_t * Q_t * exp_2Rt)

    I2 = (1.0 / theta) * u_integral_I2 * t_integral_I2

    if debug:
        print(f"I₂: u_int={u_integral_I2:.6f}, t_int={t_integral_I2:.6f}, I₂={I2:.6f}")

    # =========================================================================
    # I₁ (most complex - second derivative d²/dxdy)
    # PRZZ line 1530-1532:
    # I₁ = d²/dxdy [(θ(x+y)+1)/θ × ∫∫ (1-u)² P₂(x+u)P₂(y+u)
    #              × Q(arg_α)Q(arg_β) e^{R[arg_α+arg_β]} du dt] |_{x=y=0}
    # where:
    #   arg_α = θt(x+y) - θy + t = t + θ(t-1)y + θtx  (at x=0: t - θy + θty = t + θy(t-1))
    #   arg_β = θt(x+y) - θx + t = t + θ(t-1)x + θty  (at y=0: t - θx + θtx = t + θx(t-1))
    # At x=y=0: arg_α = arg_β = t
    #
    # Need to compute d²/dxdy of the full expression at x=y=0.
    # Use the product rule: d²/dxdy[ABCDEF...] expands to many terms.
    # =========================================================================

    # Let's denote the integrand (before prefactor) as:
    # F(x,y,u,t) = P₂(x+u)P₂(y+u) Q(arg_α)Q(arg_β) exp(R[arg_α+arg_β])
    #
    # where arg_α(x,y,t) = θt(x+y) - θy + t,  arg_β(x,y,t) = θt(x+y) - θx + t
    #
    # At x=y=0:
    #   arg_α = t,  arg_β = t
    #   ∂arg_α/∂x = θt,  ∂arg_α/∂y = θ(t-1)
    #   ∂arg_β/∂x = θ(t-1),  ∂arg_β/∂y = θt
    #
    # The full expression with prefactor is:
    # G(x,y) = [(θ(x+y)+1)/θ] × ∫∫ (1-u)² F(x,y,u,t) du dt
    #
    # d²G/dxdy = d²/dxdy[(θ(x+y)+1)/θ] × (integral at x=y=0)
    #          + d/dx[(θ(x+y)+1)/θ]|₀ × d/dy(integral)|₀
    #          + d/dy[(θ(x+y)+1)/θ]|₀ × d/dx(integral)|₀
    #          + [(θ(x+y)+1)/θ]|₀ × d²/dxdy(integral)|₀
    #
    # Prefactor derivatives:
    # (θ(x+y)+1)/θ at x=y=0: 1/θ
    # d/dx = d/dy = 1
    # d²/dxdy = 0
    #
    # So: d²G/dxdy = 0 × I₀ + 1 × (dI/dy)₀ + 1 × (dI/dx)₀ + (1/θ) × (d²I/dxdy)₀

    # Now compute the derivatives of F(x,y,u,t) at x=y=0:
    # Let's use shorthand:
    #   P = P₂, P' = dP/du
    #   Qα = Q(arg_α), Qβ = Q(arg_β), Q'α = dQ/d(arg_α), etc.
    #   Eα = exp(R×arg_α), Eβ = exp(R×arg_β)
    #
    # F = P(x+u)P(y+u)QαQβEαEβ
    #
    # ∂F/∂x = P'(x+u)P(y+u)QαQβEαEβ
    #       + P(x+u)P(y+u)[Q'α×(θt)]QβEαEβ
    #       + P(x+u)P(y+u)Qα[Q'β×θ(t-1)]EαEβ
    #       + P(x+u)P(y+u)QαQβ[Eα×Rθt]Eβ
    #       + P(x+u)P(y+u)QαQβEα[Eβ×Rθ(t-1)]
    #
    # At x=y=0: P(u), P'(u), Q(t), Q'(t), E(t)=e^{Rt}
    #
    # ∂F/∂x|₀ = P'(u)P(u)Q²E² + P²×Q'×θt×Q×E² + P²×Q×Q'×θ(t-1)×E²
    #         + P²×Q²×E×Rθt×E + P²×Q²×E×Rθ(t-1)×E
    #       = P'P×Q²e^{2Rt} + P²×Q'Q×θ(2t-1)×e^{2Rt} + P²×Q²×R×θ(2t-1)×e^{2Rt}
    #
    # Similarly for ∂F/∂y|₀ (by symmetry of arg roles swapped):
    # ∂F/∂y|₀ = PP'×Q²e^{2Rt} + P²×Q'Q×θ(2t-1)×e^{2Rt} + P²×Q²×R×θ(2t-1)×e^{2Rt}
    #
    # Note: ∂F/∂x|₀ = ∂F/∂y|₀ by symmetry!

    # For d²F/dxdy, we need second mixed derivative...
    # This is getting complex. Let me compute term by term.

    # Set up 2D quadrature mesh
    I1_total = 0.0

    for it, t in enumerate(t_nodes):
        wt = t_weights[it]

        # Values at this t
        Qt = Q_t[it]
        Qp = Q_prime_t[it]
        Qpp = Q_double_prime_t[it]
        E = exp(R * t)  # e^{Rt}
        E2 = E * E      # e^{2Rt}

        # Derivative coefficients for arg_α, arg_β
        # At x=y=0: arg_α = arg_β = t
        # ∂arg_α/∂x = θt, ∂arg_α/∂y = θ(t-1)
        # ∂arg_β/∂x = θ(t-1), ∂arg_β/∂y = θt
        darg_alpha_dx = theta * t
        darg_alpha_dy = theta * (t - 1)
        darg_beta_dx = theta * (t - 1)
        darg_beta_dy = theta * t

        for iu, u in enumerate(u_nodes):
            wu = u_weights[iu]
            one_minus_u_sq = (1.0 - u) ** 2

            P = P2_u[iu]
            Pp = P2_prime_u[iu]
            Ppp = P2_double_prime_u[iu]

            # F = P(x+u)P(y+u)Q(arg_α)Q(arg_β)e^{R(arg_α+arg_β)}
            # At x=y=0: F₀ = P² Q² e^{2Rt}
            F0 = P * P * Qt * Qt * E2

            # ∂F/∂x at x=y=0 (5 terms from product rule)
            # Term 1: P'(u) × P(u) × Q² × e^{2Rt}
            # Term 2: P² × Q'(t)×darg_α/dx × Q × e^{2Rt}
            # Term 3: P² × Q × Q'(t)×darg_β/dx × e^{2Rt}
            # Term 4: P² × Q² × R×darg_α/dx × e^{2Rt}
            # Term 5: P² × Q² × R×darg_β/dx × e^{2Rt}
            dF_dx = (Pp * P * Qt * Qt * E2 +
                     P * P * Qp * darg_alpha_dx * Qt * E2 +
                     P * P * Qt * Qp * darg_beta_dx * E2 +
                     P * P * Qt * Qt * R * darg_alpha_dx * E2 +
                     P * P * Qt * Qt * R * darg_beta_dx * E2)

            # Simplify: darg_α/dx + darg_β/dx = θt + θ(t-1) = θ(2t-1)
            # So terms 2+3+4+5 can be grouped

            # ∂F/∂y at x=y=0 (by symmetry, swap α↔β roles)
            dF_dy = (P * Pp * Qt * Qt * E2 +
                     P * P * Qp * darg_alpha_dy * Qt * E2 +
                     P * P * Qt * Qp * darg_beta_dy * E2 +
                     P * P * Qt * Qt * R * darg_alpha_dy * E2 +
                     P * P * Qt * Qt * R * darg_beta_dy * E2)

            # ∂²F/∂x∂y at x=y=0
            # This is the second mixed derivative. Apply ∂/∂y to ∂F/∂x.
            # Need to differentiate each term of dF/dx with respect to y.

            # Term by term differentiation of dF/dx with respect to y:

            # (A) d/dy[P'(x+u)P(y+u)Q²e^{2Rt}]
            # = P'(u) × P'(u) × Q² × e^{2Rt}
            term_A = Pp * Pp * Qt * Qt * E2

            # (B) d/dy[P² × Q'×darg_α/dx × Q × e^{2Rt}]
            # darg_α/dx = θt (const in y)
            # Q'(arg_α) and Q(arg_β) both depend on y
            # = P×P' × Q'×θt × Q × e^{2Rt}  (from P(y+u))
            # + P² × Q''×darg_α/dy × θt × Q × e^{2Rt}  (from Q'(arg_α))
            # + P² × Q'×θt × Q'×darg_β/dy × e^{2Rt}  (from Q(arg_β))
            # + P² × Q'×θt × Q × R×darg_β/dy × e^{2Rt}  (from exp)
            term_B = (P * Pp * Qp * darg_alpha_dx * Qt * E2 +
                      P * P * Qpp * darg_alpha_dy * darg_alpha_dx * Qt * E2 +
                      P * P * Qp * darg_alpha_dx * Qp * darg_beta_dy * E2 +
                      P * P * Qp * darg_alpha_dx * Qt * R * darg_beta_dy * E2)

            # (C) d/dy[P² × Q × Q'×darg_β/dx × e^{2Rt}]
            # darg_β/dx = θ(t-1) (const in y)
            term_C = (P * Pp * Qt * Qp * darg_beta_dx * E2 +
                      P * P * Qp * darg_alpha_dy * Qp * darg_beta_dx * E2 +
                      P * P * Qt * Qpp * darg_beta_dy * darg_beta_dx * E2 +
                      P * P * Qt * Qp * darg_beta_dx * R * darg_beta_dy * E2)

            # (D) d/dy[P² × Q² × R×darg_α/dx × e^{2Rt}]
            term_D = (P * Pp * Qt * Qt * R * darg_alpha_dx * E2 +
                      P * P * Qp * darg_alpha_dy * Qt * R * darg_alpha_dx * E2 +
                      P * P * Qt * Qp * darg_beta_dy * R * darg_alpha_dx * E2 +
                      P * P * Qt * Qt * R * darg_alpha_dx * R * darg_beta_dy * E2)

            # (E) d/dy[P² × Q² × R×darg_β/dx × e^{2Rt}]
            term_E = (P * Pp * Qt * Qt * R * darg_beta_dx * E2 +
                      P * P * Qp * darg_alpha_dy * Qt * R * darg_beta_dx * E2 +
                      P * P * Qt * Qp * darg_beta_dy * R * darg_beta_dx * E2 +
                      P * P * Qt * Qt * R * darg_beta_dx * R * darg_beta_dy * E2)

            d2F_dxdy = term_A + term_B + term_C + term_D + term_E

            # Full I₁ integrand contribution
            # d²G/dxdy = 1×(dI/dy) + 1×(dI/dx) + (1/θ)×(d²I/dxdy)
            # where I = ∫∫(1-u)² F du dt

            # Contribution from this (u,t) point:
            # (dI/dx)_contrib = (1-u)² × dF/dx
            # (dI/dy)_contrib = (1-u)² × dF/dy
            # (d²I/dxdy)_contrib = (1-u)² × d²F/dxdy

            integrand = (dF_dx + dF_dy + (1.0/theta) * d2F_dxdy) * one_minus_u_sq

            I1_total += wu * wt * integrand

    I1 = I1_total

    if debug:
        print(f"I₁={I1:.6f}")

    # =========================================================================
    # I₃ (first derivative d/dx)
    # PRZZ line 1562-1563:
    # I₃ = -d/dx[(1+θx)/θ × ∫∫ (1-u)P₂(x+u)P₂(u) Q_α Q_β e^{R[...]} du dt]|_{x=0}
    #
    # For I₃, the Q arguments are different from I₁:
    # arg_α = t + θxt (from line 1558)
    # arg_β = -θx + t + θxt = t + θx(t-1)
    #
    # At x=0: both args = t
    # ∂arg_α/∂x = θt
    # ∂arg_β/∂x = θ(t-1)
    #
    # exp factors: e^{R[t+θxt]} × e^{R[-θx+t+θxt]} = e^{2Rt} at x=0
    # ∂/∂x of exp: R×θt × e^{Rt} for first, R×θ(t-1) × e^{Rt} for second
    #
    # Full derivative:
    # d/dx[G] = d/dx[(1+θx)/θ]|₀ × I₀ + [(1+θx)/θ]|₀ × dI/dx|₀
    #         = 1 × I₀ + (1/θ) × dI/dx
    # where I₀ = ∫∫(1-u)P(u)²Q(t)²e^{2Rt} du dt (the integral at x=0)
    # =========================================================================

    # First compute I₀ for I₃
    # I₀ = ∫∫(1-u)P(u)²Q(t)²e^{2Rt} du dt
    u_integral_I3_base = np.sum(u_weights * (1 - u_nodes) * P2_u * P2_u)
    t_integral_I3_base = np.sum(t_weights * Q_t * Q_t * exp_2Rt)
    I3_base = u_integral_I3_base * t_integral_I3_base

    # Now compute dI/dx at x=0
    # F(x,u,t) = P(x+u)P(u)Q(arg_α)Q(arg_β)e^{R(arg_α+arg_β)}
    # At x=0: F₀ = P(u)²Q(t)²e^{2Rt}
    # dF/dx|₀ = P'(u)P(u)Q²e^{2Rt}
    #         + P²×Q'×θt×Q×e^{2Rt} + P²×Q×Q'×θ(t-1)×e^{2Rt}
    #         + P²×Q²×Rθt×e^{2Rt} + P²×Q²×Rθ(t-1)×e^{2Rt}

    I3_deriv = 0.0
    for it, t in enumerate(t_nodes):
        wt = t_weights[it]
        Qt = Q_t[it]
        Qp = Q_prime_t[it]
        E2 = exp(2 * R * t)

        for iu, u in enumerate(u_nodes):
            wu = u_weights[iu]
            one_minus_u = 1.0 - u

            P = P2_u[iu]
            Pp = P2_prime_u[iu]

            dF_dx = (Pp * P * Qt * Qt * E2 +
                     P * P * Qp * theta * t * Qt * E2 +
                     P * P * Qt * Qp * theta * (t - 1) * E2 +
                     P * P * Qt * Qt * R * theta * t * E2 +
                     P * P * Qt * Qt * R * theta * (t - 1) * E2)

            I3_deriv += wu * wt * one_minus_u * dF_dx

    # I₃ = -[1 × I₀ + (1/θ) × dI/dx]
    I3 = -(I3_base + (1.0/theta) * I3_deriv)

    if debug:
        print(f"I₃: base={I3_base:.6f}, deriv={I3_deriv:.6f}, I₃={I3:.6f}")

    # =========================================================================
    # I₄ (first derivative d/dy, symmetric to I₃)
    # By the symmetry noted in PRZZ, I₄ has the same structure as I₃
    # but with the derivative with respect to y on P(y+u) instead of P(x+u).
    #
    # For (2,2), both polynomials are the same (P₂), so the structure is
    # identical to I₃.
    #
    # I₄ = -[1 × I₀ + (1/θ) × dI/dy]
    #
    # Since the integral and derivative structure is symmetric in x,y
    # for (2,2) (both use P₂), we get I₄ = I₃.
    # =========================================================================

    I4 = I3  # By symmetry for the (2,2) case

    if debug:
        print(f"I₄={I4:.6f} (by symmetry)")

    # Total
    total = I1 + I2 + I3 + I4

    return OracleResult22(I1=I1, I2=I2, I3=I3, I4=I4, total=total)


def compare_oracle_22_vs_dsl(
    P2, Q, theta: float, R: float,
    dsl_c22_raw: float, dsl_c22_terms: Dict,
    n_quad: int = 60
) -> None:
    """Compare oracle vs DSL for (2,2) pair."""
    oracle = przz_oracle_22(P2, Q, theta, R, n_quad, debug=True)

    print("\n" + "="*60)
    print("PRZZ Oracle vs DSL: (2,2) Pair")
    print("="*60)
    print(f"Oracle I₁: {oracle.I1:.6f}")
    print(f"Oracle I₂: {oracle.I2:.6f}")
    print(f"Oracle I₃: {oracle.I3:.6f}")
    print(f"Oracle I₄: {oracle.I4:.6f}")
    print(f"Oracle Total: {oracle.total:.6f}")
    print()
    print(f"DSL c₂₂ (raw): {dsl_c22_raw:.6f}")
    print()

    if dsl_c22_terms:
        print("DSL per-term breakdown:")
        for key, val in dsl_c22_terms.items():
            print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    # Quick test
    from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("="*60)
    print("Testing PRZZ (2,2) Oracle")
    print("="*60)

    theta = 4/7

    # Test with κ polynomials (R=1.3036)
    print("\n--- κ Benchmark (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036
    result_k = przz_oracle_22(P2_k, Q_k, theta, R_kappa, n_quad=80, debug=True)
    print(f"\nTotal (2,2) oracle: {result_k.total:.6f}")

    # Test with κ* polynomials (R=1.1167)
    print("\n--- κ* Benchmark (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167
    result_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_kappa_star, n_quad=80, debug=True)
    print(f"\nTotal (2,2) oracle: {result_ks.total:.6f}")

    # Compare ratios
    print("\n--- Comparison ---")
    print(f"κ / κ* ratio: {result_k.total / result_ks.total:.4f}")
