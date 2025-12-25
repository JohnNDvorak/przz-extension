"""
src/przz_iterm_monomial_evaluator.py
PRZZ Monomial Evaluator Using I-Term Integral Structures

Key insight: Each Ψ monomial maps to a specific I-term integral structure:
- AB → I₁-like: d²/dxdy of (prefactor × integrand) with (1-u)² weight
- D  → I₂-like: base integral with NO weight
- AC → I₃-like: d/dx of (prefactor × integrand) with (1-u)¹ weight
- BC → I₄-like: d/dy of (prefactor × integrand) with (1-u)¹ weight

For (1,1): Ψ = AB + D - AC - BC = I₁ + I₂ - |I₃| - |I₄|

For (2,2) and higher, more monomials appear:
- A²B² → d⁴/dx²dy² structure?
- ABD → mixed structure?
- etc.

This evaluator implements the I-term derivative extraction for each monomial,
properly handling the prefactor chain rule and weight structure.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, List, Dict
from math import exp, log, comb, factorial
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class MonomialSpec:
    """Specification for a monomial A^i B^j C^k D^m."""
    i: int  # A power
    j: int  # B power
    k: int  # C power
    m: int  # D power
    coeff: int


def expand_psi_to_monomials(ell: int, ellbar: int) -> Dict[Tuple[int,int,int,int], int]:
    """
    Expand Ψ_{ℓ,ℓ̄} to monomials with combined coefficients.

    Returns dict mapping (i, j, k, m) -> coefficient.
    """
    combined = {}
    max_p = min(ell, ellbar)

    for p in range(max_p + 1):
        psi_coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)
        x_power = ell - p
        y_power = ellbar - p
        z_power = p

        # Expand X^{x_power} × Y^{y_power} × Z^{z_power}
        for a in range(x_power + 1):
            x_coeff = comb(x_power, a) * ((-1) ** (x_power - a))
            c_from_x = x_power - a

            for b in range(y_power + 1):
                y_coeff = comb(y_power, b) * ((-1) ** (y_power - b))
                c_from_y = y_power - b

                for d in range(z_power + 1):
                    z_coeff = comb(z_power, d) * ((-1) ** (z_power - d))
                    c_from_z = 2 * (z_power - d)

                    total_coeff = psi_coeff * x_coeff * y_coeff * z_coeff
                    total_c = c_from_x + c_from_y + c_from_z

                    key = (a, b, total_c, d)
                    combined[key] = combined.get(key, 0) + int(total_coeff)

    # Remove zero coefficients
    return {k: v for k, v in combined.items() if v != 0}


class ItermMonomialEvaluator:
    """
    Evaluates monomials using I-term integral structures.

    Maps each monomial (i, j, k, m) to an appropriate derivative extraction:
    - (1,1,0,0) AB → I₁-type (d²/dxdy)
    - (0,0,0,1) D  → I₂-type (base)
    - (1,0,1,0) AC → I₃-type (d/dx)
    - (0,1,1,0) BC → I₄-type (d/dy)
    - Higher monomials need higher derivative orders
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float, n_quad: int = 60):
        self.P_left = P_left
        self.P_right = P_right
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
        # Left polynomial
        self.P_L = self.P_left.eval(self.u_nodes)
        self.Pp_L = self.P_left.eval_deriv(self.u_nodes, 1)
        self.Ppp_L = self.P_left.eval_deriv(self.u_nodes, 2)

        # Right polynomial
        self.P_R = self.P_right.eval(self.u_nodes)
        self.Pp_R = self.P_right.eval_deriv(self.u_nodes, 1)
        self.Ppp_R = self.P_right.eval_deriv(self.u_nodes, 2)

        # Q polynomial
        self.Q_t = self.Q.eval(self.t_nodes)
        self.Qp_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Qpp_t = self.Q.eval_deriv(self.t_nodes, 2)

    def eval_I2_type(self, ell: int, ellbar: int) -> float:
        """
        I₂-type: Base integral with NO (1-u) weight.

        I₂ = (1/θ) × ∫∫ P_L(u) × P_R(u) × Q(t)² × exp(2Rt) du dt
        """
        # u-integral
        u_int = np.sum(self.u_weights * self.P_L * self.P_R)

        # t-integral
        exp_2Rt = np.exp(2 * self.R * self.t_nodes)
        t_int = np.sum(self.t_weights * self.Q_t * self.Q_t * exp_2Rt)

        return (1.0 / self.theta) * u_int * t_int

    def eval_I1_type(self, ell: int, ellbar: int) -> float:
        """
        I₁-type: Mixed derivative d²/dxdy of (prefactor × F).

        I₁ = ∫∫ [dF/dx + dF/dy + (1/θ)×d²F/dxdy] × (1-u)^{ℓ+ℓ̄} du dt

        where F = P_L(x+u) × P_R(y+u) × Q(α) × Q(β) × exp(R(α+β))
        at x = y = 0.
        """
        weight_exp = ell + ellbar
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E = exp(self.R * t)
            E2 = E * E

            # Argument derivatives
            darg_alpha_dx = self.theta * t
            darg_alpha_dy = self.theta * (t - 1)
            darg_beta_dx = self.theta * (t - 1)
            darg_beta_dy = self.theta * t

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # F = P_L × P_R × Q² × exp(2Rt) at x=y=0
                F0 = P_L * P_R * Qt * Qt * E2

                # dF/dx at x=y=0
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_alpha_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_beta_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_alpha_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_beta_dx * E2)

                # dF/dy at x=y=0
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_alpha_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_beta_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_alpha_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_beta_dy * E2)

                # d²F/dxdy at x=y=0 (simplified - main terms)
                # Term A: P'_L × P'_R × Q² × E²
                term_A = Pp_L * Pp_R * Qt * Qt * E2

                # Additional terms from Q/exp structure (from oracle)
                term_B = (P_L * Pp_R * Qp * darg_alpha_dx * Qt * E2 +
                          P_L * P_R * Qpp * darg_alpha_dy * darg_alpha_dx * Qt * E2 +
                          P_L * P_R * Qp * darg_alpha_dx * Qp * darg_beta_dy * E2 +
                          P_L * P_R * Qp * darg_alpha_dx * Qt * self.R * darg_beta_dy * E2)

                term_C = (P_L * Pp_R * Qt * Qp * darg_beta_dx * E2 +
                          P_L * P_R * Qp * darg_alpha_dy * Qp * darg_beta_dx * E2 +
                          P_L * P_R * Qt * Qpp * darg_beta_dy * darg_beta_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_beta_dx * self.R * darg_beta_dy * E2)

                term_D = (P_L * Pp_R * Qt * Qt * self.R * darg_alpha_dx * E2 +
                          P_L * P_R * Qp * darg_alpha_dy * Qt * self.R * darg_alpha_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_beta_dy * self.R * darg_alpha_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_alpha_dx * self.R * darg_beta_dy * E2)

                term_E = (P_L * Pp_R * Qt * Qt * self.R * darg_beta_dx * E2 +
                          P_L * P_R * Qp * darg_alpha_dy * Qt * self.R * darg_beta_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_beta_dy * self.R * darg_beta_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_beta_dx * self.R * darg_beta_dy * E2)

                d2F_dxdy = term_A + term_B + term_C + term_D + term_E

                # I₁ integrand from prefactor chain rule
                integrand = dF_dx + dF_dy + (1.0 / self.theta) * d2F_dxdy
                total += wu * wt * integrand * weight

        return total

    def eval_I3_type(self, ell: int, ellbar: int) -> float:
        """
        I₃-type: d/dx derivative structure with (1-u)^ℓ weight.

        I₃ = -[base + (1/θ)×dI/dx]

        where base uses (1-u)^ℓ weight and dI/dx is the x-derivative contribution.
        """
        weight_exp_base = ell  # Base uses (1-u)^ℓ
        weight_exp_deriv = ell  # Derivative term uses same weight

        # Base integral with (1-u)^ℓ weight
        u_int_base = np.sum(self.u_weights * (1.0 - self.u_nodes) ** weight_exp_base *
                           self.P_L * self.P_R)
        exp_2Rt = np.exp(2 * self.R * self.t_nodes)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * exp_2Rt)
        I3_base = u_int_base * t_int_base

        # Derivative integral
        I3_deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = exp(2 * self.R * t)

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp_deriv

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]

                # dF/dx at x=y=0
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * self.theta * t * Qt * E2 +
                         P_L * P_R * Qt * Qp * self.theta * (t - 1) * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * t * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * (t - 1) * E2)

                I3_deriv += wu * wt * dF_dx * weight

        # I₃ = -[base + (1/θ)×deriv]
        return -(I3_base + (1.0 / self.theta) * I3_deriv)

    def eval_I4_type(self, ell: int, ellbar: int) -> float:
        """
        I₄-type: d/dy derivative structure with (1-u)^{ℓ̄} weight.

        For diagonal pairs, I₄ = I₃ by symmetry.
        """
        # For now, use symmetry for P_left = P_right case
        if self.P_left is self.P_right:
            return self.eval_I3_type(ellbar, ell)

        # Otherwise compute explicitly (TODO)
        weight_exp = ellbar

        u_int_base = np.sum(self.u_weights * (1.0 - self.u_nodes) ** weight_exp *
                           self.P_L * self.P_R)
        exp_2Rt = np.exp(2 * self.R * self.t_nodes)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * exp_2Rt)
        I4_base = u_int_base * t_int_base

        I4_deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = exp(2 * self.R * t)

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dy at x=y=0
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * self.theta * (t - 1) * Qt * E2 +
                         P_L * P_R * Qt * Qp * self.theta * t * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * (t - 1) * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * t * E2)

                I4_deriv += wu * wt * dF_dy * weight

        return -(I4_base + (1.0 / self.theta) * I4_deriv)

    def eval_monomial(self, i: int, j: int, k: int, m: int, coeff: int,
                      ell: int, ellbar: int) -> float:
        """
        Evaluate a single monomial A^i B^j C^k D^m.

        Maps to appropriate I-term structure:
        - (1,1,0,0) AB → I₁
        - (0,0,0,1) D  → I₂
        - (1,0,1,0) AC → |I₃|
        - (0,1,1,0) BC → |I₄|
        - (0,0,k,0) C^k → need special handling
        - Higher order monomials: need generalization
        """
        # Special case: pure C^k terms (no A, B, D)
        if i == 0 and j == 0 and m == 0:
            if k == 0:
                return 0.0  # No contribution from constant 1
            # Pure C^k terms cancel in (1,1) due to XY + Z structure
            # For now, return 0 (they should cancel in the sum)
            return 0.0

        # AB term (i=1, j=1, k=0, m=0)
        if i == 1 and j == 1 and k == 0 and m == 0:
            return coeff * self.eval_I1_type(ell, ellbar)

        # D term (i=0, j=0, k=0, m=1)
        if i == 0 and j == 0 and k == 0 and m == 1:
            return coeff * self.eval_I2_type(ell, ellbar)

        # AC term (i=1, j=0, k=1, m=0)
        if i == 1 and j == 0 and k == 1 and m == 0:
            # -AC in Ψ → -|I₃| = I₃ (since I₃ is negative)
            # The coeff is already -1, so return |I₃|
            I3_abs = abs(self.eval_I3_type(ell, ellbar))
            return coeff * I3_abs

        # BC term (i=0, j=1, k=1, m=0)
        if i == 0 and j == 1 and k == 1 and m == 0:
            I4_abs = abs(self.eval_I4_type(ell, ellbar))
            return coeff * I4_abs

        # Higher order monomials - need implementation
        # For now, warn and return 0
        print(f"WARNING: Unhandled monomial A^{i}B^{j}C^{k}D^{m}")
        return 0.0

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """Evaluate full Ψ contribution for pair (ℓ, ℓ̄)."""
        monomials = expand_psi_to_monomials(ell, ellbar)

        if verbose:
            print(f"\nPair ({ell},{ellbar}): {len(monomials)} monomials")

        total = 0.0
        for (i, j, k, m), coeff in sorted(monomials.items()):
            contrib = self.eval_monomial(i, j, k, m, coeff, ell, ellbar)
            total += contrib

            if verbose:
                sign = "+" if coeff > 0 else ""
                mono_str = f"A^{i}B^{j}C^{k}D^{m}"
                print(f"  {sign}{coeff} × {mono_str}: {contrib:.6f}")

        if verbose:
            print(f"  Total = {total:.6f}")

        return total


def validate_11():
    """Validate on (1,1) pair."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("I-TERM MONOMIAL EVALUATOR VALIDATION: (1,1)")
    print("=" * 70)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle (1,1):")
    print(f"  I₁ = {oracle.I1:.6f}")
    print(f"  I₂ = {oracle.I2:.6f}")
    print(f"  I₃ = {oracle.I3:.6f}")
    print(f"  I₄ = {oracle.I4:.6f}")
    print(f"  Total = {oracle.total:.6f}")

    # I-term monomial evaluator
    evaluator = ItermMonomialEvaluator(P1, P1, Q, theta, R, n_quad)

    # Test individual I-types
    print(f"\nI-type integrals:")
    I1_val = evaluator.eval_I1_type(1, 1)
    I2_val = evaluator.eval_I2_type(1, 1)
    I3_val = evaluator.eval_I3_type(1, 1)
    I4_val = evaluator.eval_I4_type(1, 1)
    print(f"  I₁-type = {I1_val:.6f} (oracle: {oracle.I1:.6f})")
    print(f"  I₂-type = {I2_val:.6f} (oracle: {oracle.I2:.6f})")
    print(f"  I₃-type = {I3_val:.6f} (oracle: {oracle.I3:.6f})")
    print(f"  I₄-type = {I4_val:.6f} (oracle: {oracle.I4:.6f})")

    # Full pair evaluation
    print(f"\nFull pair evaluation:")
    total = evaluator.eval_pair(1, 1, verbose=True)

    print(f"\nMonomial total: {total:.6f}")
    print(f"Oracle total:   {oracle.total:.6f}")
    print(f"Ratio:          {total/oracle.total:.6f}")

    if abs(total - oracle.total) < 0.01 * abs(oracle.total):
        print("\n✓ PASSED")
        return True
    else:
        print("\n✗ FAILED")
        return False


if __name__ == "__main__":
    validate_11()
