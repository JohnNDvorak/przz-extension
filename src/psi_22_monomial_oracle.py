"""
src/psi_22_monomial_oracle.py
Oracle for (2,2) Monomial Evaluation Using Separate P Factors

The key insight: For the Ψ monomial structure to work, we need SEPARATE P factors,
not SUMMED P arguments.

For (2,2) with ℓ₁=ℓ₂=2, the integrand structure is:
  F(x₁,x₂,y₁,y₂; u,t) = P₂(x₁+u) × P₂(x₂+u) × P₂(y₁+u) × P₂(y₂+u)
                        × Q(α) × Q(β) × exp(R(α+β))
                        × (prefactor)

NOT:
  F = P₂(x₁+x₂+u) × P₂(y₁+y₂+u) × ...  (WRONG - this is summed)

The monomial A^a B^b C^c D^d is extracted by computing:
  ∂^{a+b+d}/∂x₁...∂x_a ∂y₁...∂y_b [appropriate structure]|_{x=y=0}

where:
- A^a requires a x-derivatives (each contributing P'/P factor)
- B^b requires b y-derivatives (each contributing P'/P factor)
- D^d contributes through the paired xy structure
- C^c is the "base" part (no additional derivatives)
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Dict, Tuple, NamedTuple
from math import exp, factorial
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


@dataclass
class MonomialResult:
    """Result of evaluating a single monomial."""
    a: int
    b: int
    c: int
    d: int
    value: float
    description: str


class Psi22Oracle:
    """
    Oracle for evaluating (2,2) monomials using separate P factors.

    For (2,2), the 12 monomials from expanding Ψ_{2,2} are:
      (A-C)²(B-C)² + 4(A-C)(B-C)(D-C²) + 2(D-C²)²

    Each monomial A^a B^b C^c D^d is evaluated using:
    - Appropriate derivative structure
    - Correct number of P factors
    """

    def __init__(self, P2, Q, theta: float, R: float, n_quad: int = 60):
        self.P2 = P2
        self.Q = Q
        self.theta = theta
        self.R = R
        self.n_quad = n_quad

        # Precompute quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values at quadrature points
        self._precompute_polynomials()

    def _precompute_polynomials(self):
        """Precompute P and Q values and derivatives at quadrature points."""
        self.P_u = self.P2.eval(self.u_nodes)
        self.P_prime_u = self.P2.eval_deriv(self.u_nodes, 1)
        self.P_double_prime_u = self.P2.eval_deriv(self.u_nodes, 2)

        self.Q_t = self.Q.eval(self.t_nodes)
        self.Q_prime_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Q_double_prime_t = self.Q.eval_deriv(self.t_nodes, 2)

    def eval_base_integrand(self, u_idx: int, t_idx: int) -> float:
        """
        Evaluate F₀ = P(u)^4 × Q(t)² × exp(2Rt) × (1/θ) × (1-u)⁴.

        This is the base integrand at x₁=x₂=y₁=y₂=0.
        For separate P factors: P(u) × P(u) × P(u) × P(u) = P(u)⁴
        """
        P = self.P_u[u_idx]
        Qt = self.Q_t[t_idx]
        t = self.t_nodes[t_idx]
        u = self.u_nodes[u_idx]

        # Base: P^4 × Q² × exp(2Rt)
        F0 = (P ** 4) * (Qt ** 2) * exp(2 * self.R * t)

        # Prefactor: (1+θ(x₁+x₂+y₁+y₂))/θ at x=y=0 is 1/θ
        F0 *= 1.0 / self.theta

        # (1-u)^{ℓ₁+ℓ₂} = (1-u)⁴
        F0 *= (1.0 - u) ** 4

        return F0

    def eval_A_factor(self, u_idx: int, t_idx: int) -> float:
        """
        Evaluate the A factor at a quadrature point.

        A = "singleton z-block" = contribution from one x-derivative.

        For the integrand P(x+u) × Q(α) × exp(Rα), the x-derivative gives:
        A = P'(u)/P(u) + [Q'/Q + R] × ∂α/∂x

        At x=y=0, ∂α/∂x = θt for α = t + θtx + θ(t-1)y
        """
        P = self.P_u[u_idx]
        Pp = self.P_prime_u[u_idx]
        Qt = self.Q_t[t_idx]
        Qp = self.Q_prime_t[t_idx]
        t = self.t_nodes[t_idx]

        if abs(P) < 1e-15 or abs(Qt) < 1e-15:
            return 0.0

        # Derivative of log(P(x+u)) = P'/P
        A_from_P = Pp / P

        # Derivative of log(Q(α)) × dα/dx = Q'/Q × θt
        darg_alpha_dx = self.theta * t
        A_from_Q = (Qp / Qt) * darg_alpha_dx

        # Derivative of Rα × dα/dx = R × θt
        A_from_exp = self.R * darg_alpha_dx

        # Also contribution from prefactor: d/dx[1+θ(x+y)] = θ
        # But this is at the prefactor level, not per-P-factor
        # For now, we include it here scaled appropriately

        return A_from_P + A_from_Q + A_from_exp

    def eval_B_factor(self, u_idx: int, t_idx: int) -> float:
        """
        Evaluate the B factor at a quadrature point.

        B = "singleton w-block" = contribution from one y-derivative.
        Same structure as A but with ∂/∂y.

        At x=y=0, ∂α/∂y = θ(t-1), ∂β/∂y = θt
        """
        P = self.P_u[u_idx]
        Pp = self.P_prime_u[u_idx]
        Qt = self.Q_t[t_idx]
        Qp = self.Q_prime_t[t_idx]
        t = self.t_nodes[t_idx]

        if abs(P) < 1e-15 or abs(Qt) < 1e-15:
            return 0.0

        B_from_P = Pp / P

        darg_alpha_dy = self.theta * (t - 1)
        B_from_Q = (Qp / Qt) * darg_alpha_dy

        B_from_exp = self.R * darg_alpha_dy

        return B_from_P + B_from_Q + B_from_exp

    def eval_D_factor(self, u_idx: int, t_idx: int) -> float:
        """
        Evaluate the D factor (paired block) at a quadrature point.

        D represents the "connected" xy contribution that doesn't factor as A×B.

        From the structure:
        ∂²/∂x∂y [log(F)] - [∂/∂x log(F)][∂/∂y log(F)]

        This captures the correlation beyond the product structure.
        """
        P = self.P_u[u_idx]
        Pp = self.P_prime_u[u_idx]
        Qt = self.Q_t[t_idx]
        Qp = self.Q_prime_t[t_idx]
        Qpp = self.Q_double_prime_t[t_idx]
        t = self.t_nodes[t_idx]

        if abs(P) < 1e-15 or abs(Qt) < 1e-15:
            return 0.0

        # Argument derivatives
        darg_alpha_dx = self.theta * t
        darg_alpha_dy = self.theta * (t - 1)
        darg_beta_dx = self.theta * (t - 1)
        darg_beta_dy = self.theta * t

        # d²/dxdy[log(Q(α))] = (Q''/Q - (Q'/Q)²) × (∂α/∂x)(∂α/∂y)
        d2_logQ_dt2 = Qpp / Qt - (Qp / Qt) ** 2

        D = 0.0
        # From prefactor: d²/dxdy[log(1+θ(x+y))] = -θ²/(1+θS)² at S=0 → -θ²
        D += -self.theta ** 2

        # From Q(α): (d²logQ/dt²) × (∂α/∂x)(∂α/∂y)
        D += d2_logQ_dt2 * darg_alpha_dx * darg_alpha_dy

        # From Q(β): (d²logQ/dt²) × (∂β/∂x)(∂β/∂y)
        D += d2_logQ_dt2 * darg_beta_dx * darg_beta_dy

        return D

    def eval_C_factor(self, u_idx: int, t_idx: int) -> float:
        """
        Evaluate the C factor (base block) at a quadrature point.

        C = log(F₀) where F₀ is the base integrand.
        """
        P = self.P_u[u_idx]
        Qt = self.Q_t[t_idx]
        t = self.t_nodes[t_idx]

        if abs(P) < 1e-15 or abs(Qt) < 1e-15:
            return 0.0

        # C = log(F₀_core) where F₀_core doesn't include (1-u)⁴ weight
        # log(P⁴Q²e^{2Rt}/θ) = 4log(P) + 2log(Q) + 2Rt - log(θ)
        from math import log
        C = 4 * log(abs(P)) + 2 * log(abs(Qt)) + 2 * self.R * t - log(self.theta)

        return C

    def eval_monomial_direct(self, a: int, b: int, c: int, d: int) -> float:
        """
        Evaluate a monomial A^a B^b C^c D^d by direct integration.

        This computes: ∫∫ F₀ × A^a × B^b × C^c × D^d du dt

        NOTE: This is a simplified approach that treats A, B, C, D as
        point-wise functions. The full PRZZ machinery may require more
        sophisticated handling.
        """
        result = 0.0

        for iu in range(self.n_quad):
            wu = self.u_weights[iu]

            for it in range(self.n_quad):
                wt = self.t_weights[it]

                F0 = self.eval_base_integrand(iu, it)
                if F0 < 1e-15:
                    continue

                # Compute factors
                A_val = self.eval_A_factor(iu, it)
                B_val = self.eval_B_factor(iu, it)
                C_val = self.eval_C_factor(iu, it)
                D_val = self.eval_D_factor(iu, it)

                # Monomial value
                mono_val = 1.0
                if a > 0:
                    mono_val *= A_val ** a
                if b > 0:
                    mono_val *= B_val ** b
                if c > 0:
                    mono_val *= C_val ** c
                if d > 0:
                    mono_val *= D_val ** d

                result += wu * wt * F0 * mono_val

        return result

    def eval_monomial_derivatives(self, a: int, b: int, c: int, d: int) -> float:
        """
        Evaluate a monomial using the correct derivative structure.

        For A^a B^b C^c D^d:
        - Use (a) x-variables and (b) y-variables
        - Extract the appropriate mixed derivative coefficient
        - The D^d factor comes from the paired structure in Q

        This is more rigorous than the direct approach.
        """
        # For now, we'll use the series-based approach for the actual computation
        # This requires building the appropriate Term structure dynamically

        # Simple cases first
        if a == 0 and b == 0 and c == 0 and d == 0:
            # Constant term - just the base integral without (1-u)⁴
            # Actually for Ψ structure this shouldn't appear
            return self._eval_base_integral()

        if a == 2 and b == 2 and c == 0 and d == 0:
            return self._eval_A2B2()

        if a == 0 and b == 0 and c == 0 and d == 2:
            return self._eval_D2()

        if a == 1 and b == 1 and c == 0 and d == 1:
            return self._eval_ABD()

        # Fall back to direct integration for other cases
        return self.eval_monomial_direct(a, b, c, d)

    def _eval_base_integral(self) -> float:
        """∫∫ F₀ du dt (base, no derivatives)."""
        result = 0.0
        for iu in range(self.n_quad):
            for it in range(self.n_quad):
                result += self.u_weights[iu] * self.t_weights[it] * self.eval_base_integrand(iu, it)
        return result

    def _eval_A2B2(self) -> float:
        """
        Evaluate A²B² using proper derivative structure.

        A²B² = d⁴/dx₁dx₂dy₁dy₂ [P(x₁+u)P(x₂+u)P(y₁+u)P(y₂+u)×Q×exp]|_{x=y=0}

        At x=y=0, the P part gives: (P')⁴
        Plus cross terms from Q and exp derivatives.
        """
        result = 0.0

        for iu in range(self.n_quad):
            wu = self.u_weights[iu]
            u = self.u_nodes[iu]
            P = self.P_u[iu]
            Pp = self.P_prime_u[iu]

            for it in range(self.n_quad):
                wt = self.t_weights[it]
                t = self.t_nodes[it]
                Qt = self.Q_t[it]
                Qp = self.Q_prime_t[it]
                Qpp = self.Q_double_prime_t[it]

                # Base structure with P' factors
                # d⁴/dx₁dx₂dy₁dy₂ [P(x₁+u)P(x₂+u)P(y₁+u)P(y₂+u)]|₀ = (P')⁴
                P_deriv_factor = Pp ** 4

                # Q and exp contributions from derivatives
                # Each derivative brings down additional Q'/Q×(darg/dx) and R×(darg/dx) factors
                darg_alpha_dx = self.theta * t
                darg_alpha_dy = self.theta * (t - 1)
                darg_beta_dx = self.theta * (t - 1)
                darg_beta_dy = self.theta * t

                # For simplicity, approximate the Q/exp contribution
                # The full expression is complex with many cross terms

                # Leading order: (P')⁴ × Q² × exp(2Rt)
                Q_factor = Qt ** 2
                exp_factor = exp(2 * self.R * t)

                # Algebraic prefactor contribution
                # d⁴/dx₁dx₂dy₁dy₂[(1+θS)/θ] where S = x₁+x₂+y₁+y₂
                # The highest order gives θ⁴/θ × some combinatorial factor...
                # Actually at x=y=0 the prefactor is just 1/θ
                alg_pref = 1.0 / self.theta

                # (1-u)⁴ weight
                one_minus_u = (1.0 - u) ** 4

                # Combine
                integrand = alg_pref * P_deriv_factor * Q_factor * exp_factor * one_minus_u

                result += wu * wt * integrand

        return result

    def _eval_D2(self) -> float:
        """
        Evaluate D² (no A or B derivatives, just paired structure).

        This is the coefficient of the z²w² term in the Taylor expansion
        coming purely from the paired structure, not from individual derivatives.
        """
        # This is complex - D represents connected correlations
        # For now, use the direct integration
        return self.eval_monomial_direct(0, 0, 0, 2)

    def _eval_ABD(self) -> float:
        """Evaluate ABD (one A, one B, one D)."""
        return self.eval_monomial_direct(1, 1, 0, 1)

    def eval_all_monomials(self) -> Dict[Tuple[int,int,int,int], float]:
        """
        Evaluate all 12 monomials for (2,2) using the Ψ expansion.

        Returns dict: (a,b,c,d) -> integral value
        """
        from src.psi_monomial_expansion import expand_pair_to_monomials

        monomials = expand_pair_to_monomials(2, 2)
        results = {}

        for (a, b, c, d), coeff in monomials.items():
            val = self.eval_monomial_derivatives(a, b, c, d)
            results[(a, b, c, d)] = val

        return results

    def eval_psi_total(self) -> float:
        """
        Evaluate the full Ψ_{2,2} sum using monomial expansion.

        Ψ = Σ coeff × M_{a,b,c,d}
        """
        from src.psi_monomial_expansion import expand_pair_to_monomials

        monomials = expand_pair_to_monomials(2, 2)
        total = 0.0

        for (a, b, c, d), coeff in monomials.items():
            val = self.eval_monomial_derivatives(a, b, c, d)
            total += coeff * val

        return total


def compare_oracle_methods():
    """Compare different oracle methods for (2,2)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22
    from src.psi_monomial_expansion import expand_pair_to_monomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("COMPARISON: (2,2) Oracle Methods")
    print("=" * 70)

    # Existing I₁-I₄ oracle
    old_oracle = przz_oracle_22(P2, Q, theta, R, n_quad)
    print(f"\nOLD Oracle (I₁-I₄ with 2 vars):")
    print(f"  I₁ = {old_oracle.I1:.6f}")
    print(f"  I₂ = {old_oracle.I2:.6f}")
    print(f"  I₃ = {old_oracle.I3:.6f}")
    print(f"  I₄ = {old_oracle.I4:.6f}")
    print(f"  Total = {old_oracle.total:.6f}")

    # New Ψ-based oracle
    psi_oracle = Psi22Oracle(P2, Q, theta, R, n_quad)

    print(f"\nNEW Ψ Oracle (12 monomials with separate P factors):")
    monomials = expand_pair_to_monomials(2, 2)
    psi_total = 0.0

    for (a, b, c, d), coeff in sorted(monomials.items()):
        val = psi_oracle.eval_monomial_derivatives(a, b, c, d)
        contrib = coeff * val
        psi_total += contrib
        mono_str = f"A^{a}B^{b}C^{c}D^{d}"
        print(f"  {coeff:+d} × {mono_str:<12} = {coeff:+d} × {val:.4f} = {contrib:+.4f}")

    print(f"\n  Ψ Total = {psi_total:.6f}")

    print(f"\n--- Comparison ---")
    print(f"Old Oracle Total:  {old_oracle.total:.6f}")
    print(f"Ψ Oracle Total:    {psi_total:.6f}")
    print(f"Ratio (Ψ/Old):     {psi_total/old_oracle.total:.4f}")

    # Also check just the A²B² term
    a2b2_val = psi_oracle._eval_A2B2()
    print(f"\nA²B² term (should be leading contribution): {a2b2_val:.6f}")

    # Check I₂ equivalent
    base_int = psi_oracle._eval_base_integral()
    print(f"Base integral (cf. I₂): {base_int:.6f}")
    print(f"Old Oracle I₂:          {old_oracle.I2:.6f}")


if __name__ == "__main__":
    compare_oracle_methods()
