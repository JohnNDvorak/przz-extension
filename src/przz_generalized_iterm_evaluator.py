"""
src/przz_generalized_iterm_evaluator.py
Generalized I-Term Evaluator with Correct Pair-Dependent Weights

Key insight: PRZZ uses ell-indexing starting at 0 (ell=0 is mu piece).
Our indexing starts at 1 (ell=1 is mu piece, uses P1).
So: przz_ell = our_ell - 1

Weight structure (from PRZZ):
- I1 weight = (1-u)^{przz_ell1 + przz_ell2}
- I2 has NO weight
- I3 weight = (1-u)^{przz_ell1}
- I4 weight = (1-u)^{przz_ell2}

For our pair (ell, ellbar):
- przz_ell1 = ell - 1
- przz_ell2 = ellbar - 1
- I1 weight = (1-u)^{ell + ellbar - 2}
- I3 weight = (1-u)^{ell - 1}
- I4 weight = (1-u)^{ellbar - 1}
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple
from math import exp


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class ItermResult(NamedTuple):
    """Result of I-term computation for a pair."""
    I1: float
    I2: float
    I3: float
    I4: float
    total: float
    ell: int
    ellbar: int


class GeneralizedItermEvaluator:
    """
    Compute I1, I2, I3, I4 for any pair (ell, ellbar) with correct PRZZ weights.

    Uses:
    - P_left for the left polynomial (P_{ell})
    - P_right for the right polynomial (P_{ellbar})
    - Q for the Q polynomial
    - Correct pair-dependent weights
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float,
                 ell: int, ellbar: int, n_quad: int = 60):
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.theta = theta
        self.R = R
        self.ell = ell
        self.ellbar = ellbar
        self.n_quad = n_quad

        # PRZZ indexing (0-based)
        self.przz_ell1 = ell - 1
        self.przz_ell2 = ellbar - 1

        # Weight exponents
        self.I1_weight_exp = self.przz_ell1 + self.przz_ell2  # (1-u)^{przz_ell1+przz_ell2}
        self.I3_weight_exp = self.przz_ell1  # (1-u)^{przz_ell1}
        self.I4_weight_exp = self.przz_ell2  # (1-u)^{przz_ell2}

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values
        self._precompute()

    def _precompute(self):
        """Precompute polynomial and derivative values."""
        # Left polynomial
        self.P_L = self.P_left.eval(self.u_nodes)
        self.Pp_L = self.P_left.eval_deriv(self.u_nodes, 1)

        # Right polynomial
        self.P_R = self.P_right.eval(self.u_nodes)
        self.Pp_R = self.P_right.eval_deriv(self.u_nodes, 1)

        # Q polynomial and derivatives
        self.Q_t = self.Q.eval(self.t_nodes)
        self.Qp_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Qpp_t = self.Q.eval_deriv(self.t_nodes, 2)

        # Exponential factor
        self.exp_2Rt = np.exp(2 * self.R * self.t_nodes)

    def eval_I2(self) -> float:
        """
        I2 = (1/theta) * integral of P_L(u) * P_R(u) * Q(t)^2 * exp(2Rt) - NO weight.
        """
        u_int = np.sum(self.u_weights * self.P_L * self.P_R)
        t_int = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)
        return (1.0 / self.theta) * u_int * t_int

    def eval_I1(self) -> float:
        """
        I1 = d^2/dxdy [prefactor * integral] at x=y=0

        With weight (1-u)^{przz_ell1 + przz_ell2}.

        The prefactor is (1 + theta*(x+y))/theta.
        After chain rule expansion:
        I1 = (d integral/dy) + (d integral/dx) + (1/theta)*(d^2 integral/dxdy)
        """
        weight_exp = self.I1_weight_exp
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E = exp(self.R * t)
            E2 = E * E

            # Argument derivatives at x=y=0
            darg_alpha_dx = self.theta * t
            darg_alpha_dy = self.theta * (t - 1)
            darg_beta_dx = self.theta * (t - 1)
            darg_beta_dy = self.theta * t

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

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

                # d^2F/dxdy - main term plus additional contributions
                # (A) from P'_L * P'_R
                term_A = Pp_L * Pp_R * Qt * Qt * E2

                # (B) from Q'/exp derivatives acting on dF/dx structure
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

                # Full I1 integrand from prefactor chain rule
                integrand = dF_dx + dF_dy + (1.0 / self.theta) * d2F_dxdy
                total += wu * wt * integrand * weight

        return total

    def eval_I3(self) -> float:
        """
        I3 = -d/dx [prefactor * integral] at x=0

        With weight (1-u)^{przz_ell1}.

        I3 = -[base_integral + (1/theta)*(d integral/dx)]
        """
        weight_exp = self.I3_weight_exp

        # Base integral with weight
        weight_arr = (1.0 - self.u_nodes) ** weight_exp if weight_exp > 0 else np.ones_like(self.u_nodes)
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)
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
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]

                # dF/dx at x=y=0 (y is not present in I3's integrand - Q args are at y=0)
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * self.theta * t * Qt * E2 +
                         P_L * P_R * Qt * Qp * self.theta * (t - 1) * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * t * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * (t - 1) * E2)

                I3_deriv += wu * wt * dF_dx * weight

        return -(I3_base + (1.0 / self.theta) * I3_deriv)

    def eval_I4(self) -> float:
        """
        I4 = -d/dy [prefactor * integral] at y=0

        With weight (1-u)^{przz_ell2}.

        For diagonal pairs (ell == ellbar), I4 = I3 by symmetry.
        """
        if self.ell == self.ellbar:
            return self.eval_I3()

        weight_exp = self.I4_weight_exp

        # Base integral with weight
        weight_arr = (1.0 - self.u_nodes) ** weight_exp if weight_exp > 0 else np.ones_like(self.u_nodes)
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)
        I4_base = u_int_base * t_int_base

        # Derivative integral
        I4_deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = exp(2 * self.R * t)

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dy at x=y=0 (x is not present in I4's integrand)
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * self.theta * (t - 1) * Qt * E2 +
                         P_L * P_R * Qt * Qp * self.theta * t * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * (t - 1) * E2 +
                         P_L * P_R * Qt * Qt * self.R * self.theta * t * E2)

                I4_deriv += wu * wt * dF_dy * weight

        return -(I4_base + (1.0 / self.theta) * I4_deriv)

    def eval_all(self) -> ItermResult:
        """Compute all I-terms and return result."""
        I1 = self.eval_I1()
        I2 = self.eval_I2()
        I3 = self.eval_I3()
        I4 = self.eval_I4()
        total = I1 + I2 + I3 + I4

        return ItermResult(I1=I1, I2=I2, I3=I3, I4=I4, total=total,
                          ell=self.ell, ellbar=self.ellbar)


def validate_against_oracle():
    """Validate the generalized evaluator against the oracle for (2,2)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("GENERALIZED I-TERM EVALUATOR VALIDATION")
    print("=" * 70)

    # (2,2) should match oracle (oracle was designed for this case)
    print("\n--- (2,2) Pair with P2 ---")
    print("This should match oracle (both use (1-u)^2 for I1, (1-u)^1 for I3)")

    oracle = przz_oracle_22(P2, Q, theta, R, n_quad)
    evaluator = GeneralizedItermEvaluator(P2, P2, Q, theta, R, ell=2, ellbar=2, n_quad=n_quad)
    result = evaluator.eval_all()

    print(f"\nWeight exponents: I1={(2-1)+(2-1)}, I3={2-1}, I4={2-1}")
    print(f"Oracle:    I1={oracle.I1:.6f}, I2={oracle.I2:.6f}, I3={oracle.I3:.6f}, I4={oracle.I4:.6f}, Total={oracle.total:.6f}")
    print(f"Evaluator: I1={result.I1:.6f}, I2={result.I2:.6f}, I3={result.I3:.6f}, I4={result.I4:.6f}, Total={result.total:.6f}")

    match_22 = abs(result.total - oracle.total) < 0.01 * abs(oracle.total)
    print(f"Match: {match_22}")

    # (1,1) with correct weights (different from oracle which uses wrong weights)
    print("\n--- (1,1) Pair with P1 ---")
    print("Correct weights: I1=(1-u)^0=1, I3=(1-u)^0=1 (no weight)")
    print("Oracle uses (1-u)^2 for I1 and (1-u)^1 for I3 (WRONG for this case)")

    oracle_11 = przz_oracle_22(P1, Q, theta, R, n_quad)
    evaluator_11 = GeneralizedItermEvaluator(P1, P1, Q, theta, R, ell=1, ellbar=1, n_quad=n_quad)
    result_11 = evaluator_11.eval_all()

    print(f"\nWeight exponents: I1={(1-1)+(1-1)}, I3={1-1}, I4={1-1}")
    print(f"Oracle (wrong):    I1={oracle_11.I1:.6f}, I2={oracle_11.I2:.6f}, I3={oracle_11.I3:.6f}, Total={oracle_11.total:.6f}")
    print(f"Evaluator (right): I1={result_11.I1:.6f}, I2={result_11.I2:.6f}, I3={result_11.I3:.6f}, Total={result_11.total:.6f}")

    # (3,3) with correct weights
    print("\n--- (3,3) Pair with P3 ---")
    print("Correct weights: I1=(1-u)^4, I3=(1-u)^2")
    print("Oracle uses (1-u)^2 for I1 and (1-u)^1 for I3 (WRONG for this case)")

    oracle_33 = przz_oracle_22(P3, Q, theta, R, n_quad)
    evaluator_33 = GeneralizedItermEvaluator(P3, P3, Q, theta, R, ell=3, ellbar=3, n_quad=n_quad)
    result_33 = evaluator_33.eval_all()

    print(f"\nWeight exponents: I1={(3-1)+(3-1)}, I3={3-1}, I4={3-1}")
    print(f"Oracle (wrong):    I1={oracle_33.I1:.6f}, I2={oracle_33.I2:.6f}, I3={oracle_33.I3:.6f}, Total={oracle_33.total:.6f}")
    print(f"Evaluator (right): I1={result_33.I1:.6f}, I2={result_33.I2:.6f}, I3={result_33.I3:.6f}, Total={result_33.total:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF DIAGONAL PAIRS")
    print("=" * 70)
    print(f"(1,1) with P1: Total = {result_11.total:.6f}")
    print(f"(2,2) with P2: Total = {result.total:.6f}")
    print(f"(3,3) with P3: Total = {result_33.total:.6f}")
    print()

    return match_22


if __name__ == "__main__":
    validate_against_oracle()
