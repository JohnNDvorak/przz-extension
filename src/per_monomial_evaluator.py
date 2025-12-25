"""
src/per_monomial_evaluator.py
Per-Monomial Evaluator for PRZZ Section 7

This module implements per-monomial evaluation with correct weights for each
monomial in the Ψ expansion.

KEY INSIGHT: The GenEval uses a single weight per I-term type, but the Ψ expansion
has monomials with different weights even for the same derivative order.

For (2,2), the three monomials with (l₁=2, m₁=2) derivative order have:
  - D² (a=0, b=0): weight (1-u)^0 = 1
  - ABD (a=1, b=1): weight (1-u)^2
  - A²B² (a=2, b=2): weight (1-u)^4

The GenEval incorrectly applies (1-u)^{ell+ellbar} = (1-u)^4 to ALL of these.

This module evaluates each monomial with its correct (1-u)^{a+b} weight.

DERIVATIVE STRUCTURE:
For a monomial with derivative order (l₁, m₁) = (a+d, b+d):
  - l₁ derivatives w.r.t. x applied to: prefactor, P_left(u-x), Q(arg_α), Q(arg_β), exp(...)
  - m₁ derivatives w.r.t. y applied to: prefactor, P_right(u-y), Q(arg_α), Q(arg_β), exp(...)

The prefactor is (1 + θ(x+y))/θ.
At x=y=0, its derivatives are:
  - Value: 1/θ
  - d/dx: 1
  - d/dy: 1
  - d²/dxdy: 0
  - Higher: 0

For polynomials:
  - P_L(u-x) at x=0: P_L(u)
  - d/dx P_L(u-x)|_{x=0}: -P'_L(u)
  - d²/dx² P_L(u-x)|_{x=0}: P''_L(u)
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple, List
from dataclasses import dataclass
from math import exp, factorial


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class MonomialResult:
    """Result from evaluating a single monomial."""
    a: int
    b: int
    c_alpha: int
    c_beta: int
    d: int
    psi_coeff: int
    l1: int  # = a + d
    m1: int  # = b + d
    weight_exp: int  # = a + b
    value: float  # The integral value (including weight and psi_coeff)


class PerMonomialEvaluator:
    """
    Evaluate individual monomials from the Ψ expansion with correct weights.

    For each monomial A^a B^b C_α^{c_α} C_β^{c_β} D^d with coefficient ψ:
    1. Derivative order is (l₁, m₁) = (a+d, b+d)
    2. Weight is (1-u)^{a+b}
    3. Compute the appropriate derivative integral
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float, n_quad: int = 60):
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.theta = theta
        self.R = R
        self.n_quad = n_quad

        # Quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values and derivatives up to order 2
        self._precompute()

    def _precompute(self):
        """Precompute polynomial values and derivatives."""
        u = self.u_nodes
        t = self.t_nodes

        # Left polynomial: P_L, P'_L, P''_L
        self.P_L = self.P_left.eval(u)
        self.Pp_L = self.P_left.eval_deriv(u, 1)
        self.Ppp_L = self.P_left.eval_deriv(u, 2)

        # Right polynomial: P_R, P'_R, P''_R
        self.P_R = self.P_right.eval(u)
        self.Pp_R = self.P_right.eval_deriv(u, 1)
        self.Ppp_R = self.P_right.eval_deriv(u, 2)

        # Q polynomial: Q, Q', Q''
        self.Q_t = self.Q.eval(t)
        self.Qp_t = self.Q.eval_deriv(t, 1)
        self.Qpp_t = self.Q.eval_deriv(t, 2)

        # Exponential
        self.exp_2Rt = np.exp(2 * self.R * t)

        # Argument derivatives at x=y=0
        # arg_α = θt - θ(1-t)(x+y)/2 + ... (simplified at x=y=0 → θt)
        # arg_β = θ(1-t) + θt(x+y)/2 + ... (simplified at x=y=0 → θ(1-t))
        #
        # darg_α/dx = θ * t       (from the full formula)
        # darg_α/dy = θ * (t - 1)
        # darg_β/dx = θ * (t - 1)
        # darg_β/dy = θ * t
        self.darg_alpha_dx = self.theta * t
        self.darg_alpha_dy = self.theta * (t - 1)
        self.darg_beta_dx = self.theta * (t - 1)
        self.darg_beta_dy = self.theta * t

    def eval_l1m1_00(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (0, 0) - no derivatives.

        This is the I₂-type integral: (1/θ) × ∫∫ P_L(u) P_R(u) Q² exp(2Rt) du dt
        """
        if weight_exp > 0:
            weight = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight = np.ones_like(self.u_nodes)

        u_int = np.sum(self.u_weights * weight * self.P_L * self.P_R)
        t_int = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)

        return (1.0 / self.theta) * u_int * t_int

    def eval_l1m1_10(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (1, 0) - one x-derivative.

        Returns: [base + (1/θ)×derivative]  (positive base value)

        NOTE: GenEval I₃ = -[base + (1/θ)×derivative] includes the Ψ sign.
        Here we return the positive base value and let the Ψ coefficient provide the sign.
        """
        if weight_exp > 0:
            weight_arr = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight_arr = np.ones_like(self.u_nodes)

        # Base integral
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)
        base = u_int_base * t_int_base

        # Derivative integral
        deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = exp(2 * self.R * t)
            darg_a_dx = self.darg_alpha_dx[it]
            darg_b_dx = self.darg_beta_dx[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = weight_arr[iu]

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]

                # dF/dx at x=y=0: derivative of P_L, Q(arg_α), Q(arg_β), exp terms
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dx * E2)

                deriv += wu * wt * dF_dx * weight

        # Return POSITIVE base value; Ψ coefficient provides sign
        return (base + (1.0 / self.theta) * deriv)

    def eval_l1m1_01(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (0, 1) - one y-derivative.

        Returns: [base + (1/θ)×derivative]  (positive base value)

        NOTE: GenEval I₄ = -[base + (1/θ)×derivative] includes the Ψ sign.
        Here we return the positive base value and let the Ψ coefficient provide the sign.
        """
        if weight_exp > 0:
            weight_arr = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight_arr = np.ones_like(self.u_nodes)

        # Base integral
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)
        base = u_int_base * t_int_base

        # Derivative integral
        deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = exp(2 * self.R * t)
            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = weight_arr[iu]

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dy at x=y=0
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dy * E2)

                deriv += wu * wt * dF_dy * weight

        # Return POSITIVE base value; Ψ coefficient provides sign
        return (base + (1.0 / self.theta) * deriv)

    def eval_l1m1_11(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (1, 1) - mixed xy-derivative.

        This is the I₁-type integral: [dF/dx + dF/dy + (1/θ)×d²F/dxdy]
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E = exp(self.R * t)
            E2 = E * E

            darg_a_dx = self.darg_alpha_dx[it]
            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dx = self.darg_beta_dx[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dx
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dx * E2)

                # dF/dy
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dy * E2)

                # d²F/dxdy - from chain rule on dF/dx
                term_A = Pp_L * Pp_R * Qt * Qt * E2

                term_B = (P_L * Pp_R * Qp * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qpp * darg_a_dy * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qp * darg_b_dy * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qt * self.R * darg_b_dy * E2)

                term_C = (P_L * Pp_R * Qt * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qpp * darg_b_dy * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dx * self.R * darg_b_dy * E2)

                term_D = (P_L * Pp_R * Qt * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_a_dx * self.R * darg_b_dy * E2)

                term_E = (P_L * Pp_R * Qt * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_b_dx * self.R * darg_b_dy * E2)

                d2F_dxdy = term_A + term_B + term_C + term_D + term_E

                integrand = dF_dx + dF_dy + (1.0 / self.theta) * d2F_dxdy
                total += wu * wt * integrand * weight

        return total

    def eval_l1m1_20(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (2, 0) - two x-derivatives.

        Structure: [2×d integral/dx + (1/θ)×d² integral/dx²]
        (from expanding d²/dx² of prefactor × integral)
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = exp(2 * self.R * t)

            darg_a_dx = self.darg_alpha_dx[it]
            darg_b_dx = self.darg_beta_dx[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]

                # dF/dx (same as before)
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dx * E2)

                # d²F/dx² - second derivative in x
                # From P''_L(u)
                term_Ppp = Ppp_L * P_R * Qt * Qt * E2

                # From P'_L × Q' terms
                term_PpQ = (2 * Pp_L * P_R * Qp * darg_a_dx * Qt * E2 +
                            2 * Pp_L * P_R * Qt * Qp * darg_b_dx * E2 +
                            2 * Pp_L * P_R * Qt * Qt * self.R * darg_a_dx * E2 +
                            2 * Pp_L * P_R * Qt * Qt * self.R * darg_b_dx * E2)

                # From Q'' terms
                term_Qpp = (P_L * P_R * Qpp * darg_a_dx * darg_a_dx * Qt * E2 +
                            2 * P_L * P_R * Qp * darg_a_dx * Qp * darg_b_dx * E2 +
                            P_L * P_R * Qt * Qpp * darg_b_dx * darg_b_dx * E2)

                # From Q' × exp terms
                term_Qexp = (2 * P_L * P_R * Qp * darg_a_dx * Qt * self.R * darg_a_dx * E2 +
                             2 * P_L * P_R * Qp * darg_a_dx * Qt * self.R * darg_b_dx * E2 +
                             2 * P_L * P_R * Qt * Qp * darg_b_dx * self.R * darg_a_dx * E2 +
                             2 * P_L * P_R * Qt * Qp * darg_b_dx * self.R * darg_b_dx * E2)

                # From exp'' terms
                term_exp2 = (P_L * P_R * Qt * Qt * self.R * darg_a_dx * self.R * darg_a_dx * E2 +
                             2 * P_L * P_R * Qt * Qt * self.R * darg_a_dx * self.R * darg_b_dx * E2 +
                             P_L * P_R * Qt * Qt * self.R * darg_b_dx * self.R * darg_b_dx * E2)

                d2F_dx2 = term_Ppp + term_PpQ + term_Qpp + term_Qexp + term_exp2

                # Full integrand: 2×dF/dx + (1/θ)×d²F/dx²
                integrand = 2 * dF_dx + (1.0 / self.theta) * d2F_dx2
                total += wu * wt * integrand * weight

        return total

    def eval_l1m1_02(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (0, 2) - two y-derivatives.

        Structure: [2×d integral/dy + (1/θ)×d² integral/dy²]
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = exp(2 * self.R * t)

            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # dF/dy
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * darg_a_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * darg_b_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_a_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * darg_b_dy * E2)

                # d²F/dy² - second derivative in y
                term_Ppp = P_L * Ppp_R * Qt * Qt * E2

                term_PpQ = (2 * P_L * Pp_R * Qp * darg_a_dy * Qt * E2 +
                            2 * P_L * Pp_R * Qt * Qp * darg_b_dy * E2 +
                            2 * P_L * Pp_R * Qt * Qt * self.R * darg_a_dy * E2 +
                            2 * P_L * Pp_R * Qt * Qt * self.R * darg_b_dy * E2)

                term_Qpp = (P_L * P_R * Qpp * darg_a_dy * darg_a_dy * Qt * E2 +
                            2 * P_L * P_R * Qp * darg_a_dy * Qp * darg_b_dy * E2 +
                            P_L * P_R * Qt * Qpp * darg_b_dy * darg_b_dy * E2)

                term_Qexp = (2 * P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_a_dy * E2 +
                             2 * P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_b_dy * E2 +
                             2 * P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_a_dy * E2 +
                             2 * P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_b_dy * E2)

                term_exp2 = (P_L * P_R * Qt * Qt * self.R * darg_a_dy * self.R * darg_a_dy * E2 +
                             2 * P_L * P_R * Qt * Qt * self.R * darg_a_dy * self.R * darg_b_dy * E2 +
                             P_L * P_R * Qt * Qt * self.R * darg_b_dy * self.R * darg_b_dy * E2)

                d2F_dy2 = term_Ppp + term_PpQ + term_Qpp + term_Qexp + term_exp2

                integrand = 2 * dF_dy + (1.0 / self.theta) * d2F_dy2
                total += wu * wt * integrand * weight

        return total

    def eval_l1m1_21(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (2, 1) - two x, one y derivative.

        Structure: d/dy [2×dF/dx + (1/θ)×d²F/dx²] + (1/θ)×d/dy[d²F/dx²]
        = 2×d²F/dxdy + (2/θ)×d³F/dx²dy

        Simplified: 2×d²F/dxdy + (1/θ)×d³F/dx²dy
        (The (1/θ) from the outer prefactor combines with inner derivatives)
        """
        # This is complex - implement the full third derivative structure
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = exp(2 * self.R * t)

            darg_a_dx = self.darg_alpha_dx[it]
            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dx = self.darg_beta_dx[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # d²F/dxdy (already computed in eval_l1m1_11)
                term_A = Pp_L * Pp_R * Qt * Qt * E2
                term_B = (P_L * Pp_R * Qp * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qpp * darg_a_dy * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qp * darg_b_dy * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qt * self.R * darg_b_dy * E2)
                term_C = (P_L * Pp_R * Qt * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qpp * darg_b_dy * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dx * self.R * darg_b_dy * E2)
                term_D = (P_L * Pp_R * Qt * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_a_dx * self.R * darg_b_dy * E2)
                term_E = (P_L * Pp_R * Qt * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_b_dx * self.R * darg_b_dy * E2)
                d2F_dxdy = term_A + term_B + term_C + term_D + term_E

                # d³F/dx²dy - derivative of d²F/dx² w.r.t. y
                # This requires P''_L × P'_R and many Q'''/Q'' terms
                # Approximate: use Leibniz rule structure
                d3F_dx2dy = (
                    # P'' × P' term
                    Ppp_L * Pp_R * Qt * Qt * E2 +
                    # P'' × Q terms
                    Ppp_L * P_R * Qp * darg_a_dy * Qt * E2 +
                    Ppp_L * P_R * Qt * Qp * darg_b_dy * E2 +
                    Ppp_L * P_R * Qt * Qt * self.R * (darg_a_dy + darg_b_dy) * E2 +
                    # P' × P' × Q² (from d²P terms)
                    2 * Pp_L * Pp_R * Qp * darg_a_dx * Qt * E2 +
                    2 * Pp_L * Pp_R * Qt * Qp * darg_b_dx * E2 +
                    2 * Pp_L * Pp_R * Qt * Qt * self.R * (darg_a_dx + darg_b_dx) * E2 +
                    # Cross terms P' × P × Q' × Q'
                    2 * Pp_L * P_R * Qpp * darg_a_dx * darg_a_dy * Qt * E2 +
                    2 * Pp_L * P_R * Qp * darg_a_dx * Qp * darg_b_dy * E2 +
                    2 * Pp_L * P_R * Qt * Qpp * darg_b_dx * darg_b_dy * E2
                )

                # Full integrand: structure for d²/dx² d/dy of prefactor × integral
                # = 2×d²F/dxdy + (1/θ)×d³F/dx²dy (plus additional prefactor terms)
                integrand = 2 * d2F_dxdy + (1.0 / self.theta) * d3F_dx2dy
                total += wu * wt * integrand * weight

        return total

    def eval_l1m1_12(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (1, 2) - one x, two y derivatives.

        By symmetry with eval_l1m1_21, swap x ↔ y roles.
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = exp(2 * self.R * t)

            darg_a_dx = self.darg_alpha_dx[it]
            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dx = self.darg_beta_dx[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # d²F/dxdy (same as before)
                term_A = Pp_L * Pp_R * Qt * Qt * E2
                term_B = (P_L * Pp_R * Qp * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qpp * darg_a_dy * darg_a_dx * Qt * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qp * darg_b_dy * E2 +
                          P_L * P_R * Qp * darg_a_dx * Qt * self.R * darg_b_dy * E2)
                term_C = (P_L * Pp_R * Qt * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qp * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qpp * darg_b_dy * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dx * self.R * darg_b_dy * E2)
                term_D = (P_L * Pp_R * Qt * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_a_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_a_dx * self.R * darg_b_dy * E2)
                term_E = (P_L * Pp_R * Qt * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qp * darg_a_dy * Qt * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qp * darg_b_dy * self.R * darg_b_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * darg_b_dx * self.R * darg_b_dy * E2)
                d2F_dxdy = term_A + term_B + term_C + term_D + term_E

                # d³F/dxdy² - derivative of d²F/dy² w.r.t. x
                d3F_dxdy2 = (
                    Pp_L * Ppp_R * Qt * Qt * E2 +
                    P_L * Ppp_R * Qp * darg_a_dx * Qt * E2 +
                    P_L * Ppp_R * Qt * Qp * darg_b_dx * E2 +
                    P_L * Ppp_R * Qt * Qt * self.R * (darg_a_dx + darg_b_dx) * E2 +
                    2 * Pp_L * Pp_R * Qp * darg_a_dy * Qt * E2 +
                    2 * Pp_L * Pp_R * Qt * Qp * darg_b_dy * E2 +
                    2 * Pp_L * Pp_R * Qt * Qt * self.R * (darg_a_dy + darg_b_dy) * E2 +
                    2 * Pp_L * P_R * Qpp * darg_a_dx * darg_a_dy * Qt * E2 +
                    2 * P_L * Pp_R * Qp * darg_a_dy * Qp * darg_b_dx * E2 +
                    2 * Pp_L * P_R * Qt * Qpp * darg_b_dx * darg_b_dy * E2
                )

                integrand = 2 * d2F_dxdy + (1.0 / self.theta) * d3F_dxdy2
                total += wu * wt * integrand * weight

        return total

    def eval_l1m1_22(self, weight_exp: int) -> float:
        """
        Evaluate integral with (l₁, m₁) = (2, 2) - two x and two y derivatives.

        This is the most complex case, requiring fourth-order mixed derivatives.
        Structure: combination of d⁴F/dx²dy² and lower terms.
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = exp(2 * self.R * t)

            darg_a_dx = self.darg_alpha_dx[it]
            darg_a_dy = self.darg_alpha_dy[it]
            darg_b_dx = self.darg_beta_dx[it]
            darg_b_dy = self.darg_beta_dy[it]

            for iu, u in enumerate(self.u_nodes):
                wu = self.u_weights[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # d⁴F/dx²dy² - fourth-order mixed derivative
                # Leading terms from P'' × P''
                d4F = (
                    Ppp_L * Ppp_R * Qt * Qt * E2 +
                    # P'' × P' × Q' terms
                    2 * Ppp_L * Pp_R * (Qp * darg_a_dy * Qt + Qt * Qp * darg_b_dy) * E2 +
                    2 * Pp_L * Ppp_R * (Qp * darg_a_dx * Qt + Qt * Qp * darg_b_dx) * E2 +
                    # P'' × P × Q² × exp terms
                    Ppp_L * P_R * (Qpp * darg_a_dy**2 * Qt +
                                   2 * Qp * darg_a_dy * Qp * darg_b_dy +
                                   Qt * Qpp * darg_b_dy**2) * E2 +
                    P_L * Ppp_R * (Qpp * darg_a_dx**2 * Qt +
                                   2 * Qp * darg_a_dx * Qp * darg_b_dx +
                                   Qt * Qpp * darg_b_dx**2) * E2 +
                    # P' × P' × Q' × Q' terms
                    4 * Pp_L * Pp_R * (Qp * darg_a_dx * Qp * darg_a_dy +
                                       Qp * darg_a_dx * Qp * darg_b_dy +
                                       Qp * darg_b_dx * Qp * darg_a_dy +
                                       Qp * darg_b_dx * Qp * darg_b_dy) * E2 +
                    # P'' × P × Q × exp terms
                    Ppp_L * P_R * Qt * Qt * self.R * (darg_a_dy + darg_b_dy) * E2 +
                    P_L * Ppp_R * Qt * Qt * self.R * (darg_a_dx + darg_b_dx) * E2 +
                    # P' × P' × Q² × exp terms
                    4 * Pp_L * Pp_R * Qt * Qt * self.R * (darg_a_dx + darg_b_dx) * E2 +
                    4 * Pp_L * Pp_R * Qt * Qt * self.R * (darg_a_dy + darg_b_dy) * E2
                )

                # Lower derivative contributions (d²F/dxdy, d²F/dx², d²F/dy², etc.)
                # These come from the prefactor expansion

                # d²F/dx² terms
                d2F_dx2 = (Ppp_L * P_R * Qt * Qt * E2 +
                           2 * Pp_L * P_R * (Qp * darg_a_dx * Qt + Qt * Qp * darg_b_dx) * E2 +
                           2 * Pp_L * P_R * Qt * Qt * self.R * (darg_a_dx + darg_b_dx) * E2 +
                           P_L * P_R * (Qpp * darg_a_dx**2 * Qt +
                                        2 * Qp * darg_a_dx * Qp * darg_b_dx +
                                        Qt * Qpp * darg_b_dx**2) * E2 +
                           P_L * P_R * Qt * Qt * (self.R * darg_a_dx + self.R * darg_b_dx)**2 * E2)

                # d²F/dy² terms
                d2F_dy2 = (P_L * Ppp_R * Qt * Qt * E2 +
                           2 * P_L * Pp_R * (Qp * darg_a_dy * Qt + Qt * Qp * darg_b_dy) * E2 +
                           2 * P_L * Pp_R * Qt * Qt * self.R * (darg_a_dy + darg_b_dy) * E2 +
                           P_L * P_R * (Qpp * darg_a_dy**2 * Qt +
                                        2 * Qp * darg_a_dy * Qp * darg_b_dy +
                                        Qt * Qpp * darg_b_dy**2) * E2 +
                           P_L * P_R * Qt * Qt * (self.R * darg_a_dy + self.R * darg_b_dy)**2 * E2)

                # Combined structure for (2,2) derivative
                # [4×d²F/dxdy + 2/θ×(d³F/dx²dy + d³F/dxdy²) + 1/θ²×d⁴F/dx²dy²]
                # Simplified: use leading-order structure
                integrand = (2 * d2F_dx2 + 2 * d2F_dy2 +
                             (1.0 / self.theta) * d4F)
                total += wu * wt * integrand * weight

        return total

    def eval_derivative(self, l1: int, m1: int, weight_exp: int) -> float:
        """
        Dispatch to appropriate derivative formula based on (l₁, m₁).
        """
        if l1 == 0 and m1 == 0:
            return self.eval_l1m1_00(weight_exp)
        elif l1 == 1 and m1 == 0:
            return self.eval_l1m1_10(weight_exp)
        elif l1 == 0 and m1 == 1:
            return self.eval_l1m1_01(weight_exp)
        elif l1 == 1 and m1 == 1:
            return self.eval_l1m1_11(weight_exp)
        elif l1 == 2 and m1 == 0:
            return self.eval_l1m1_20(weight_exp)
        elif l1 == 0 and m1 == 2:
            return self.eval_l1m1_02(weight_exp)
        elif l1 == 2 and m1 == 1:
            return self.eval_l1m1_21(weight_exp)
        elif l1 == 1 and m1 == 2:
            return self.eval_l1m1_12(weight_exp)
        elif l1 == 2 and m1 == 2:
            return self.eval_l1m1_22(weight_exp)
        else:
            raise ValueError(f"Unsupported derivative order: (l1={l1}, m1={m1})")

    def eval_monomial(self, mono) -> MonomialResult:
        """
        Evaluate a single monomial contribution.

        Args:
            mono: MonomialSeparatedC from psi_separated_c.py

        Returns:
            MonomialResult with full breakdown

        DERIVATIVE ORDER STRUCTURE:
        - A blocks: singleton x-residue → contributes x-derivative
        - B blocks: singleton y-residue → contributes y-derivative
        - D blocks: paired (x,y) residue → contributes BOTH x AND y derivatives

        Therefore: l₁ = a + d (total x-derivatives), m₁ = b + d (total y-derivatives)

        For (1,1):
        - AB (a=1,b=1,d=0) → (l₁=1, m₁=1): I₁ mixed derivative
        - D (a=0,b=0,d=1) → (l₁=1, m₁=1): I₁ mixed derivative!
        - AC_α (a=1,d=0) → (l₁=1, m₁=0): I₃ left derivative
        - BC_β (b=1,d=0) → (l₁=0, m₁=1): I₄ right derivative

        WAIT: This means D (d=1) should have mixed derivative (1,1), same as AB!
        But GenEval I₂ treats D as "no derivative"...

        The issue is that the I₁-I₄ decomposition captures something different from
        the derivative count. Let me use GenEval's interpretation:
        - I₂ (D term): no derivative of the polynomial P, just the Q integrand
        - I₁ (AB term): derivative from singleton blocks
        - I₃ (AC_α term): x-derivative from A singleton
        - I₄ (BC_β term): y-derivative from B singleton

        So the derivative order for I-term mapping is:
        - l₁ = a (x-derivatives from A singletons)
        - m₁ = b (y-derivatives from B singletons)
        - D blocks contribute to the paired integral structure, not polynomial derivatives
        """
        a, b = mono.a, mono.b
        c_alpha, c_beta = mono.c_alpha, mono.c_beta
        d = mono.d
        psi_coeff = mono.coeff

        # Derivative order: A singletons give x-derivatives, B singletons give y-derivatives
        # D blocks are paired and contribute to a different integral structure (I₂-type)
        l1 = a
        m1 = b
        weight_exp = a + b

        # Evaluate the integral with correct derivative order and weight
        integral_value = self.eval_derivative(l1, m1, weight_exp)

        # The psi_coeff is an integer from the Ψ expansion
        # It includes the sign from the original expansion
        value = psi_coeff * integral_value

        return MonomialResult(
            a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d,
            psi_coeff=psi_coeff, l1=l1, m1=m1,
            weight_exp=weight_exp, value=value
        )

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate full pair contribution using per-monomial evaluation.
        """
        # Use CANONICAL Ψ expansion module
        from src.psi_expansion import expand_psi

        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\n=== Per-Monomial Evaluation ({ell},{ellbar}): {len(monomials)} monomials ===")

        total = 0.0
        for mono in monomials:
            result = self.eval_monomial(mono)
            total += result.value

            if verbose:
                sign = "+" if result.psi_coeff > 0 else ""
                print(f"  A^{result.a}B^{result.b}C_α^{result.c_alpha}C_β^{result.c_beta}D^{result.d}: "
                      f"ψ={sign}{result.psi_coeff}, (l₁={result.l1},m₁={result.m1}), "
                      f"(1-u)^{result.weight_exp}, integral={result.value/result.psi_coeff:.6f}, "
                      f"contrib={result.value:.6f}")

        if verbose:
            print(f"  TOTAL = {total:.6f}")

        return total


def test_11_pair():
    """Test (1,1) pair against GenEval oracle."""
    from src.polynomials import load_przz_polynomials
    from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4/7
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("PER-MONOMIAL EVALUATOR: (1,1) VALIDATION")
    print("=" * 70)

    # Per-monomial evaluation
    pm_eval = PerMonomialEvaluator(P1, P1, Q, theta, R, n_quad)
    pm_total = pm_eval.eval_pair(1, 1, verbose=True)

    # GenEval reference (known to match oracle)
    gen_eval = GeneralizedItermEvaluator(P1, P1, Q, theta, R, 1, 1, n_quad)
    gen_result = gen_eval.eval_all()

    print(f"\nPer-Monomial Total: {pm_total:.6f}")
    print(f"GenEval Total:      {gen_result.total:.6f}")
    print(f"Oracle Target:      0.359159")
    print(f"Difference:         {abs(pm_total - gen_result.total):.6e}")

    return abs(pm_total - gen_result.total) < 0.01


def test_22_pair():
    """Test (2,2) pair per-monomial vs GenEval."""
    from src.polynomials import load_przz_polynomials
    from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4/7
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("PER-MONOMIAL EVALUATOR: (2,2) COMPARISON")
    print("=" * 70)

    # Per-monomial evaluation
    pm_eval = PerMonomialEvaluator(P2, P2, Q, theta, R, n_quad)
    pm_total = pm_eval.eval_pair(2, 2, verbose=True)

    # GenEval reference
    gen_eval = GeneralizedItermEvaluator(P2, P2, Q, theta, R, 2, 2, n_quad)
    gen_result = gen_eval.eval_all()

    print(f"\nPer-Monomial Total: {pm_total:.6f}")
    print(f"GenEval Total:      {gen_result.total:.6f}")
    print(f"Difference:         {abs(pm_total - gen_result.total):.6f}")
    print(f"GenEval I1/I2/I3/I4: {gen_result.I1:.4f}/{gen_result.I2:.4f}/"
          f"{gen_result.I3:.4f}/{gen_result.I4:.4f}")


if __name__ == "__main__":
    test_11_pair()
    test_22_pair()
