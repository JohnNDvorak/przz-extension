"""
src/przz_iterm_evaluator_fixed.py
Fixed I-Term Evaluator with Correct PRZZ 0-Based Weight Indexing

BUG IDENTIFIED: The GeneralizedItermEvaluator used our 1-based indexing
for weights, but PRZZ uses 0-based indexing:
  - PRZZ ell=0 is mu piece (our P1)
  - PRZZ ell=1 is mu*Lambda piece (our P2)
  - PRZZ ell=2 is mu*Lambda*Lambda piece (our P3)

INCORRECT (old):
  I1_weight_exp = ell + ellbar           # (1-u)^{ell+ellbar}
  I3_weight_exp = ell                    # (1-u)^{ell}
  I4_weight_exp = ellbar                 # (1-u)^{ellbar}

CORRECT (this file):
  przz_ell = ell - 1                     # Convert to PRZZ 0-based
  przz_ellbar = ellbar - 1
  I1_weight_exp = przz_ell + przz_ellbar # (1-u)^{przz_ell+przz_ellbar}
  I3_weight_exp = przz_ell               # (1-u)^{przz_ell}
  I4_weight_exp = przz_ellbar            # (1-u)^{przz_ellbar}

Weight table:
  Pair    | Our (ell,ellbar) | PRZZ (l1,l2) | I1 weight | I3 weight | I4 weight
  (1,1)   | (1,1)            | (0,0)        | (1-u)^0   | (1-u)^0   | (1-u)^0
  (2,2)   | (2,2)            | (1,1)        | (1-u)^2   | (1-u)^1   | (1-u)^1
  (3,3)   | (3,3)            | (2,2)        | (1-u)^4   | (1-u)^2   | (1-u)^2
  (1,2)   | (1,2)            | (0,1)        | (1-u)^1   | (1-u)^0   | (1-u)^1
  (1,3)   | (1,3)            | (0,2)        | (1-u)^2   | (1-u)^0   | (1-u)^2
  (2,3)   | (2,3)            | (1,2)        | (1-u)^3   | (1-u)^1   | (1-u)^2
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


class PRZZItermEvaluatorFixed:
    """
    Fixed I-term evaluator with correct PRZZ 0-based weight indexing.

    Uses:
    - P_left for the left polynomial (P_{ell})
    - P_right for the right polynomial (P_{ellbar})
    - Q for the Q polynomial
    - PRZZ 0-based indexing for (1-u) weights
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

        # PRZZ 0-based indexing for weights
        self.przz_ell = ell - 1       # Convert our 1-based to PRZZ 0-based
        self.przz_ellbar = ellbar - 1

        # CORRECTED weight exponents using PRZZ indexing
        self.I1_weight_exp = self.przz_ell + self.przz_ellbar
        self.I3_weight_exp = self.przz_ell
        self.I4_weight_exp = self.przz_ellbar

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

        With weight (1-u)^{przz_ell + przz_ellbar}.

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
                term_A = Pp_L * Pp_R * Qt * Qt * E2

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

        With weight (1-u)^{przz_ell}.

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

        With weight (1-u)^{przz_ellbar}.

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


def compute_c_total(P1, P2, P3, Q, R: float, theta: float, n_quad: int = 60) -> float:
    """Compute total c using fixed evaluator."""
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    poly_map = {1: P1, 2: P2, 3: P3}

    c_total = 0.0
    for ell, ellbar in pairs:
        Pl = poly_map[ell]
        Pr = poly_map[ellbar]
        evaluator = PRZZItermEvaluatorFixed(Pl, Pr, Q, theta, R, ell, ellbar, n_quad)

        contrib = evaluator.eval_all().total
        sym = 1 if ell == ellbar else 2
        c_total += sym * contrib

    return c_total


def test_two_benchmark():
    """Run the two-benchmark test."""
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    theta = 4 / 7

    # Benchmark 1: kappa (R=1.3036)
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036

    # Benchmark 2: kappa* (R=1.1167)
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    R_star = 1.1167

    print("=" * 70)
    print("TWO-BENCHMARK TEST WITH FIXED EVALUATOR")
    print("=" * 70)

    print("\n--- Benchmark 1: kappa (R=1.3036) ---")
    c_kappa = compute_c_total(P1, P2, P3, Q, R_kappa, theta)
    print(f"  c computed = {c_kappa:.6f}")
    print(f"  c target = 2.137")
    print(f"  Factor = {2.137 / c_kappa:.4f}")

    print("\n--- Benchmark 2: kappa* (R=1.1167) ---")
    c_star = compute_c_total(P1s, P2s, P3s, Qs, R_star, theta)
    print(f"  c computed = {c_star:.6f}")
    print(f"  c target = 1.938")
    print(f"  Factor = {1.938 / c_star:.4f}")

    print("\n--- SUMMARY ---")
    factor_kappa = 2.137 / c_kappa
    factor_star = 1.938 / c_star
    print(f"  Factor kappa: {factor_kappa:.4f}")
    print(f"  Factor kappa*: {factor_star:.4f}")
    print(f"  Factor ratio: {factor_kappa / factor_star:.4f} (target: ~1.0)")

    # Gate check
    print("\n--- GATE CHECK ---")
    gate1 = 0.9 < factor_kappa < 1.1
    gate2 = 0.9 < factor_star < 1.1
    gate3 = factor_kappa / factor_star > 0.8

    print(f"  Gate 1 (kappa factor in [0.9, 1.1]): {'PASS' if gate1 else 'FAIL'}")
    print(f"  Gate 2 (kappa* factor in [0.9, 1.1]): {'PASS' if gate2 else 'FAIL'}")
    print(f"  Gate 3 (factor ratio > 0.8): {'PASS' if gate3 else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if gate1 and gate2 and gate3 else 'FAIL'}")


def compare_with_oracle():
    """Compare fixed evaluator with oracle for (2,2)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4 / 7
    R = 1.3036

    print("=" * 70)
    print("COMPARING FIXED EVALUATOR vs ORACLE for (2,2)")
    print("=" * 70)
    print()
    print("Oracle uses hardcoded (1-u)^2 for I1, (1-u)^1 for I3/I4")
    print("Fixed evaluator uses PRZZ 0-based: (1-u)^{przz_ell+przz_ellbar}")
    print()

    # Oracle
    oracle = przz_oracle_22(P2, Q, theta, R, n_quad=60)
    print(f"Oracle (2,2):")
    print(f"  I1 = {oracle.I1:.6f}")
    print(f"  I2 = {oracle.I2:.6f}")
    print(f"  I3 = {oracle.I3:.6f}")
    print(f"  I4 = {oracle.I4:.6f}")
    print(f"  Total = {oracle.total:.6f}")

    print()

    # Fixed evaluator
    fixed = PRZZItermEvaluatorFixed(P2, P2, Q, theta, R, ell=2, ellbar=2, n_quad=60)
    print(f"Fixed evaluator (2,2) with przz_ell={fixed.przz_ell}:")
    print(f"  I1 weight = (1-u)^{fixed.I1_weight_exp}")
    print(f"  I3 weight = (1-u)^{fixed.I3_weight_exp}")
    result = fixed.eval_all()
    print(f"  I1 = {result.I1:.6f}")
    print(f"  I2 = {result.I2:.6f}")
    print(f"  I3 = {result.I3:.6f}")
    print(f"  I4 = {result.I4:.6f}")
    print(f"  Total = {result.total:.6f}")

    print()
    print(f"Difference: {abs(oracle.total - result.total):.6f}")
    print(f"Match: {'YES' if abs(oracle.total - result.total) < 0.001 else 'NO'}")


if __name__ == "__main__":
    compare_with_oracle()
    print()
    test_two_benchmark()
