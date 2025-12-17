"""
src/psi_block_evaluator.py
Block Evaluators for Ψ-Based Main Term Computation

Key insight from GPT: A, B, C, D are SCALAR VALUES computed at each (u,t) point.
They are log-derivatives of the underlying ξ_P object, NOT derivative operators.

A² means (A)·(A) = product of two singleton evaluations, NOT a second derivative.

The connected blocks are:
  X = A - C  (connected x-block)
  Y = B - C  (connected y-block)
  Z = D - C² (connected paired block)

For pair (ℓ, ℓ̄), the contribution is:
  ∫∫ Σ_p [C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}] × Base(u,t) du dt
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, NamedTuple
from math import exp, log, comb, factorial
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class BlockValues:
    """Scalar block values at a single (u, t) point."""
    A: float  # Singleton x-block: d/dx[log ξ]|_{x=0}
    B: float  # Singleton y-block: d/dy[log ξ]|_{y=0}
    C: float  # Base block: related to log ξ at x=y=0
    D: float  # Paired block: d²/dxdy[log ξ]|_{x=y=0}

    @property
    def X(self) -> float:
        """Connected x-block: A - C"""
        return self.A - self.C

    @property
    def Y(self) -> float:
        """Connected y-block: B - C"""
        return self.B - self.C

    @property
    def Z(self) -> float:
        """Connected paired block: D - C²"""
        return self.D - self.C * self.C


class PsiBlockEvaluator:
    """
    Evaluates the Ψ-based main term using block values.

    The base integrand ξ_P(x, y; u, t) has structure:
      ξ_P = (prefactor) × P(x+u) × P(y+u) × Q(α) × Q(β) × exp(R(α+β))

    where α, β are functions of x, y, t with the standard PRZZ structure.

    The log-derivative blocks A, B, C, D are computed from log(ξ_P).
    """

    def __init__(self, P, Q, theta: float, R: float, n_quad: int = 60):
        """
        Initialize the block evaluator.

        Args:
            P: Polynomial for this piece (P_ℓ)
            Q: Q polynomial
            theta: θ = 4/7
            R: R parameter
            n_quad: Quadrature points
        """
        self.P = P
        self.Q = Q
        self.theta = theta
        self.R = R
        self.n_quad = n_quad

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values
        self._precompute()

    def _precompute(self):
        """Precompute polynomial values at quadrature nodes."""
        self.P_u = self.P.eval(self.u_nodes)
        self.P_prime_u = self.P.eval_deriv(self.u_nodes, 1)

        self.Q_t = self.Q.eval(self.t_nodes)
        self.Q_prime_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Q_double_prime_t = self.Q.eval_deriv(self.t_nodes, 2)

    def compute_blocks(self, iu: int, it: int) -> BlockValues:
        """
        Compute block values A, B, C, D at quadrature point (u_iu, t_it).

        The integrand ξ_P has log:
          log(ξ_P) = log(prefactor) + log(P(x+u)) + log(P(y+u))
                     + log(Q(α)) + log(Q(β)) + R(α+β)

        At x=y=0:
          - α = β = t
          - log(ξ_P)|₀ = log(1/θ) + 2log(P(u)) + 2log(Q(t)) + 2Rt

        The blocks are derivatives of log(ξ_P):
          A = ∂/∂x[log ξ_P]|₀
          B = ∂/∂y[log ξ_P]|₀
          D = ∂²/∂x∂y[log ξ_P]|₀
          C = some "base" factor (see below)
        """
        u = self.u_nodes[iu]
        t = self.t_nodes[it]

        P_val = self.P_u[iu]
        Pp_val = self.P_prime_u[iu]
        Q_val = self.Q_t[it]
        Qp_val = self.Q_prime_t[it]
        Qpp_val = self.Q_double_prime_t[it]

        # Handle near-zero values
        if abs(P_val) < 1e-15 or abs(Q_val) < 1e-15:
            return BlockValues(A=0.0, B=0.0, C=0.0, D=0.0)

        # Argument derivatives at x=y=0
        # α = t + θtx + θ(t-1)y, β = t + θ(t-1)x + θty
        darg_alpha_dx = self.theta * t
        darg_alpha_dy = self.theta * (t - 1)
        darg_beta_dx = self.theta * (t - 1)
        darg_beta_dy = self.theta * t

        # A = ∂/∂x[log ξ_P]|₀
        # = ∂/∂x[log(prefactor)]|₀ + P'/P + (Q'/Q)(∂α/∂x) + (Q'/Q)(∂β/∂x) + R(∂α/∂x + ∂β/∂x)
        # Prefactor = (1+θ(x+y))/θ → ∂/∂x[log pref]|₀ = θ/(1) = θ
        A = self.theta  # from prefactor
        A += Pp_val / P_val  # from P(x+u)
        A += (Qp_val / Q_val) * darg_alpha_dx  # from Q(α)
        A += (Qp_val / Q_val) * darg_beta_dx   # from Q(β)
        A += self.R * (darg_alpha_dx + darg_beta_dx)  # from exp

        # B = ∂/∂y[log ξ_P]|₀ (same structure, different arg derivatives)
        B = self.theta  # from prefactor
        B += Pp_val / P_val  # from P(y+u)
        B += (Qp_val / Q_val) * darg_alpha_dy  # from Q(α)
        B += (Qp_val / Q_val) * darg_beta_dy   # from Q(β)
        B += self.R * (darg_alpha_dy + darg_beta_dy)  # from exp

        # D = ∂²/∂x∂y[log ξ_P]|₀
        # From prefactor: ∂²/∂x∂y[log(1+θS)]|₀ = -θ²
        # From Q terms: involves (Q''/Q - (Q'/Q)²) × (∂α/∂x)(∂α/∂y)
        d2_logQ = Qpp_val / Q_val - (Qp_val / Q_val) ** 2

        D = -self.theta ** 2  # from prefactor
        D += d2_logQ * darg_alpha_dx * darg_alpha_dy  # from Q(α)
        D += d2_logQ * darg_beta_dx * darg_beta_dy    # from Q(β)
        # Note: no P contribution to mixed derivative (P factors are separate)

        # C = the "base" block
        # In the PRZZ/Ψ context, C represents ζ'/ζ(1+s+u) = log-derivative at base
        # For our polynomial approximation, this is related to the Q log-derivative
        # at the base point, scaled appropriately.
        #
        # From the (1,1) validation: -AC and -BC give I₃ and I₄
        # These involve d/dx with a "C subtraction" structure
        #
        # The C block represents the "disconnected" or "base" contribution
        # that gets subtracted in the connected blocks X = A-C, Y = B-C
        #
        # Based on the structure: C should capture the common piece between A and B
        # that comes from the base integrand (not the derivative-specific parts)
        #
        # For the symmetric case: C = (Q'/Q) × θ(2t-1) + R × θ(2t-1)
        # This is the Q+exp contribution that's common to both A and B directions

        common_Q_exp = (Qp_val / Q_val + self.R) * self.theta * (2*t - 1)
        C = common_Q_exp

        return BlockValues(A=A, B=B, C=C, D=D)

    def compute_base_integrand(self, iu: int, it: int) -> float:
        """
        Compute the base integrand value at (u, t).

        Base(u,t) = (1/θ) × P(u)² × Q(t)² × exp(2Rt)

        Note: This is the base for ONE P factor per side (summed structure).
        The (1-u) weight is applied separately per p-config.
        """
        u = self.u_nodes[iu]
        t = self.t_nodes[it]

        P_val = self.P_u[iu]
        Q_val = self.Q_t[it]

        return (1.0 / self.theta) * P_val * P_val * Q_val * Q_val * exp(2 * self.R * t)

    def eval_psi_pconfig(
        self,
        ell: int,
        ellbar: int,
        p: int
    ) -> float:
        """
        Evaluate one p-config contribution for pair (ℓ, ℓ̄).

        Contribution = C(ℓ,p) × C(ℓ̄,p) × p! × ∫∫ Z^p × X^{ℓ-p} × Y^{ℓ̄-p} × Base × weight du dt

        The (1-u) weight exponent depends on the derivative structure.
        For the main term: (1-u)^{ℓ+ℓ̄} is the standard weight.
        """
        coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)

        x_exp = ell - p
        y_exp = ellbar - p
        z_exp = p

        # Weight exponent: this is where Euler-Maclaurin comes in
        # For the standard main term, use (1-u)^{ℓ+ℓ̄}
        weight_exp = ell + ellbar

        total = 0.0

        for iu in range(self.n_quad):
            u = self.u_nodes[iu]
            wu = self.u_weights[iu]
            weight = (1.0 - u) ** weight_exp

            for it in range(self.n_quad):
                wt = self.t_weights[it]

                blocks = self.compute_blocks(iu, it)
                base = self.compute_base_integrand(iu, it)

                if abs(base) < 1e-15:
                    continue

                # Block powers: X^{ℓ-p} × Y^{ℓ̄-p} × Z^p
                block_product = 1.0
                if x_exp > 0:
                    block_product *= blocks.X ** x_exp
                if y_exp > 0:
                    block_product *= blocks.Y ** y_exp
                if z_exp > 0:
                    block_product *= blocks.Z ** z_exp

                total += wu * wt * coeff * block_product * base * weight

        return total

    def eval_psi_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate the full Ψ contribution for pair (ℓ, ℓ̄).

        Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} [p-config contribution]
        """
        max_p = min(ell, ellbar)

        total = 0.0
        contributions = []

        for p in range(max_p + 1):
            contrib = self.eval_psi_pconfig(ell, ellbar, p)
            contributions.append((p, contrib))
            total += contrib

            if verbose:
                coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)
                x_exp = ell - p
                y_exp = ellbar - p
                print(f"  p={p}: coeff={coeff}, X^{x_exp}Y^{y_exp}Z^{p} = {contrib:.6f}")

        if verbose:
            print(f"  Total Ψ_{{{ell},{ellbar}}} = {total:.6f}")

        return total


def validate_11_psi():
    """Validate that the Ψ evaluator matches the oracle for (1,1)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("VALIDATION: Ψ Block Evaluator for (1,1)")
    print("=" * 70)

    # Oracle reference (using P1 for (1,1))
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle (1,1) total: {oracle.total:.6f}")

    # Ψ evaluation
    evaluator = PsiBlockEvaluator(P1, Q, theta, R, n_quad)
    psi_total = evaluator.eval_psi_pair(1, 1, verbose=True)

    print(f"\nΨ evaluator total: {psi_total:.6f}")
    print(f"Oracle total:      {oracle.total:.6f}")
    print(f"Difference:        {abs(psi_total - oracle.total):.2e}")

    ratio = psi_total / oracle.total if abs(oracle.total) > 1e-10 else float('nan')
    print(f"Ratio:             {ratio:.6f}")

    return abs(psi_total - oracle.total) < 0.01 * abs(oracle.total)


def evaluate_22_psi():
    """Evaluate (2,2) using the Ψ block evaluator."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("EVALUATION: Ψ Block Evaluator for (2,2)")
    print("=" * 70)

    # Oracle reference
    oracle = przz_oracle_22(P2, Q, theta, R, n_quad)
    print(f"\nOracle (2,2) total: {oracle.total:.6f}")
    print(f"  (Note: Oracle uses I₁-I₄ structure which may be incomplete for (2,2))")

    # Ψ evaluation with p-config breakdown
    evaluator = PsiBlockEvaluator(P2, Q, theta, R, n_quad)

    print(f"\nΨ p-config decomposition for (2,2):")
    psi_total = evaluator.eval_psi_pair(2, 2, verbose=True)

    print(f"\n--- Comparison ---")
    print(f"Oracle total:      {oracle.total:.6f}")
    print(f"Ψ evaluator total: {psi_total:.6f}")
    print(f"Ratio (Ψ/Oracle):  {psi_total/oracle.total:.4f}")


def evaluate_all_k3():
    """Evaluate all K=3 pairs using Ψ."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("FULL K=3 EVALUATION: Ψ Block Evaluator")
    print("=" * 70)

    polys = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    total_c = 0.0

    for ell, ellbar in pairs:
        # For cross-terms, we need to handle the two P polynomials
        # For now, use P_ell (the left polynomial)
        # TODO: Handle cross-terms properly with both P_ell and P_ellbar
        P = polys[ell]

        evaluator = PsiBlockEvaluator(P, Q, theta, R, n_quad)
        contrib = evaluator.eval_psi_pair(ell, ellbar, verbose=False)

        # Symmetry factor: 2 for off-diagonal pairs
        sym_factor = 1 if ell == ellbar else 2
        total_contrib = sym_factor * contrib

        total_c += total_contrib

        print(f"({ell},{ellbar}): Ψ = {contrib:.6f}, sym×Ψ = {total_contrib:.6f}")

    kappa = 1 - log(total_c) / R if total_c > 0 else float('nan')

    print(f"\nTotal c = {total_c:.6f}")
    print(f"Target c = 2.137")
    print(f"Ratio: {total_c/2.137:.4f}")
    print(f"\nκ = 1 - log(c)/R = {kappa:.6f}")
    print(f"Target κ = 0.417")


if __name__ == "__main__":
    if validate_11_psi():
        print("\n✓ (1,1) validation PASSED")
    else:
        print("\n✗ (1,1) validation FAILED")

    evaluate_22_psi()
    evaluate_all_k3()
