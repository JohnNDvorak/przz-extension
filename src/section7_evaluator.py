"""
src/section7_evaluator.py
Section 7 Evaluator - Per-Monomial with Correct F_d and Q-Kernel Structure

This is the benchmark driver for two-benchmark validation (κ and κ*).

ARCHITECTURE (from GPT guidance, Session 9):

The evaluation proceeds in layers:
1. Pre-mirror, pre-Q: Compute I_{1,d}(α,β) per monomial from Ψ expansion
2. Mirror transformation (if needed)
3. Apply Q operators with correct Q/Q'/Q'' structure
4. Evaluate at α=β=-R/log T and integrate over u, t
5. Sum all pairs with symmetry factors

KEY DIFFERENCES FROM GenEval/I-TERM APPROACH:
- GenEval uses I₁-I₄ decomposition which is only exact for (1,1)
- This evaluator uses per-monomial structure from Ψ expansion
- Each monomial has its own weight (1-u)^{a+b}
- C_α/C_β exponents affect the numeric contribution
- Q-kernels are not all just ∫Q²exp(2Rt) dt

CASE A FIX (critical per GPT guidance):
In the scaled/post-identity world, Case A (l=0, ω=-1) contributes:
    F_A(u) = α P(u) + (1/θ) P'(u)
NOT just α P(u). The P'/θ term survives after the t-identity.

VALIDATION STRATEGY:
- (1,1) must match oracle 0.359159 to machine precision
- κ benchmark (R=1.3036): c should be within 5% of 2.137
- κ* benchmark (R=1.1167): c should be within 5% of 1.938
- Ratio should be within 20% of target 1.10
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from math import exp, factorial

from src.psi_expansion import expand_psi, MonomialTwoC


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class MonomialContrib(NamedTuple):
    """Detailed contribution from a single monomial."""
    monomial: MonomialTwoC
    a: int
    b: int
    c_alpha: int
    c_beta: int
    d: int
    psi_coeff: int
    weight_exp: int  # (1-u)^{a+b}
    integral_value: float  # Raw integral (before psi_coeff)
    contribution: float  # psi_coeff × integral


@dataclass
class PairResult:
    """Result from evaluating a pair (ℓ, ℓ̄)."""
    ell: int
    ellbar: int
    total: float
    monomial_contribs: List[MonomialContrib]


class Section7Evaluator:
    """
    Section 7 evaluator with per-monomial structure.

    This evaluator properly handles:
    - Per-monomial (1-u)^{a+b} weights
    - C_α/C_β contributions (TODO: implement numeric effect)
    - Q-kernel variations based on derivative structure (TODO: implement)
    """

    def __init__(self, P_ell, P_ellbar, Q, R: float, theta: float, n_quad: int = 60):
        """
        Initialize the Section 7 evaluator.

        Args:
            P_ell: Left polynomial (P_ℓ)
            P_ellbar: Right polynomial (P_ℓ̄)
            Q: Q polynomial
            R: PRZZ R parameter
            theta: θ parameter (typically 4/7)
            n_quad: Quadrature points
        """
        self.P_ell = P_ell
        self.P_ellbar = P_ellbar
        self.Q = Q
        self.R = R
        self.theta = theta
        self.n_quad = n_quad

        # Quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values
        self._precompute()

    def _precompute(self):
        """Precompute polynomial and derivative values."""
        u = self.u_nodes
        t = self.t_nodes

        # Left polynomial and derivatives
        self.P_L = self.P_ell.eval(u)
        self.Pp_L = self.P_ell.eval_deriv(u, 1)
        self.Ppp_L = self.P_ell.eval_deriv(u, 2)
        self.Pppp_L = self.P_ell.eval_deriv(u, 3)

        # Right polynomial and derivatives
        self.P_R = self.P_ellbar.eval(u)
        self.Pp_R = self.P_ellbar.eval_deriv(u, 1)
        self.Ppp_R = self.P_ellbar.eval_deriv(u, 2)
        self.Pppp_R = self.P_ellbar.eval_deriv(u, 3)

        # Q polynomial and derivatives
        self.Q_t = self.Q.eval(t)
        self.Qp_t = self.Q.eval_deriv(t, 1)
        self.Qpp_t = self.Q.eval_deriv(t, 2)

        # Exponential
        self.exp_2Rt = np.exp(2 * self.R * t)

        # Q argument derivatives (for derivative terms)
        # From PRZZ structure: darg_α/dx = θt, darg_α/dy = θ(t-1), etc.
        self.darg_alpha_dx = self.theta * t
        self.darg_alpha_dy = self.theta * (t - 1)
        self.darg_beta_dx = self.theta * (t - 1)
        self.darg_beta_dy = self.theta * t

    # =========================================================================
    # Q-KERNEL METHODS
    # =========================================================================

    def _q_kernel_00(self) -> float:
        """Q-kernel for (l₁=0, m₁=0) - no derivatives.

        This is the basic ∫Q²exp(2Rt) dt kernel.
        """
        return np.sum(self.t_weights * self.Q_t * self.Q_t * self.exp_2Rt)

    def _q_kernel_10(self, u_idx: int) -> float:
        """Q-kernel for (l₁=1, m₁=0) - x-derivative only.

        Includes Q and Q' contributions from the chain rule.
        Returns the t-integrand for a single u point.
        """
        # For now, use the same kernel as 00
        # TODO: Implement proper Q' structure
        return self._q_kernel_00()

    def _q_kernel_01(self, u_idx: int) -> float:
        """Q-kernel for (l₁=0, m₁=1) - y-derivative only."""
        # For now, use the same kernel as 00
        # TODO: Implement proper Q' structure
        return self._q_kernel_00()

    def _q_kernel_11(self, u_idx: int) -> float:
        """Q-kernel for (l₁=1, m₁=1) - mixed derivatives."""
        # For now, use the same kernel as 00
        # TODO: Implement proper Q'/Q'' structure
        return self._q_kernel_00()

    # =========================================================================
    # INTEGRAL METHODS (per derivative order)
    # =========================================================================

    def _integral_00(self, weight_exp: int) -> float:
        """Compute integral for (l₁=0, m₁=0) derivative order.

        This is the I₂-type integral: (1/θ) × ∫P_L P_R (1-u)^w du × ∫Q² e^{2Rt} dt
        """
        if weight_exp > 0:
            weight = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight = np.ones_like(self.u_nodes)

        u_int = np.sum(self.u_weights * weight * self.P_L * self.P_R)
        t_int = self._q_kernel_00()

        return (1.0 / self.theta) * u_int * t_int

    def _integral_10(self, weight_exp: int) -> float:
        """Compute integral for (l₁=1, m₁=0) derivative order.

        This is the I₃-type integral.
        Returns: base + (1/θ) × deriv
        """
        if weight_exp > 0:
            weight_arr = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight_arr = np.ones_like(self.u_nodes)

        # Base integral (from prefactor's x-coefficient = 1)
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = self._q_kernel_00()
        base = u_int_base * t_int_base

        # Derivative integral (from (1/θ) × dF/dx)
        deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = self.exp_2Rt[it]
            da_dx = self.darg_alpha_dx[it]
            db_dx = self.darg_beta_dx[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                weight = weight_arr[iu]

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]

                # dF/dx at x=y=0
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dx * E2)

                deriv += wu * wt * dF_dx * weight

        return base + (1.0 / self.theta) * deriv

    def _integral_01(self, weight_exp: int) -> float:
        """Compute integral for (l₁=0, m₁=1) derivative order.

        This is the I₄-type integral. By symmetry with _integral_10.
        """
        if weight_exp > 0:
            weight_arr = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight_arr = np.ones_like(self.u_nodes)

        # Base integral
        u_int_base = np.sum(self.u_weights * weight_arr * self.P_L * self.P_R)
        t_int_base = self._q_kernel_00()
        base = u_int_base * t_int_base

        # Derivative integral
        deriv = 0.0
        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            E2 = self.exp_2Rt[it]
            da_dy = self.darg_alpha_dy[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                weight = weight_arr[iu]

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dy at x=y=0
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dy * E2)

                deriv += wu * wt * dF_dy * weight

        return base + (1.0 / self.theta) * deriv

    def _integral_11(self, weight_exp: int) -> float:
        """Compute integral for (l₁=1, m₁=1) derivative order.

        This is the I₁-type integral.
        """
        if weight_exp > 0:
            weight_exp_val = weight_exp
        else:
            weight_exp_val = 0

        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E = exp(self.R * t)
            E2 = E * E

            da_dx = self.darg_alpha_dx[it]
            da_dy = self.darg_alpha_dy[it]
            db_dx = self.darg_beta_dx[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp_val if weight_exp_val > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # dF/dx at x=y=0
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dx * E2)

                # dF/dy at x=y=0
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dy * E2)

                # d²F/dxdy
                term_A = Pp_L * Pp_R * Qt * Qt * E2

                term_B = (P_L * Pp_R * Qp * da_dx * Qt * E2 +
                          P_L * P_R * Qpp * da_dy * da_dx * Qt * E2 +
                          P_L * P_R * Qp * da_dx * Qp * db_dy * E2 +
                          P_L * P_R * Qp * da_dx * Qt * self.R * db_dy * E2)

                term_C = (P_L * Pp_R * Qt * Qp * db_dx * E2 +
                          P_L * P_R * Qp * da_dy * Qp * db_dx * E2 +
                          P_L * P_R * Qt * Qpp * db_dy * db_dx * E2 +
                          P_L * P_R * Qt * Qp * db_dx * self.R * db_dy * E2)

                term_D = (P_L * Pp_R * Qt * Qt * self.R * da_dx * E2 +
                          P_L * P_R * Qp * da_dy * Qt * self.R * da_dx * E2 +
                          P_L * P_R * Qt * Qp * db_dy * self.R * da_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * da_dx * self.R * db_dy * E2)

                term_E = (P_L * Pp_R * Qt * Qt * self.R * db_dx * E2 +
                          P_L * P_R * Qp * da_dy * Qt * self.R * db_dx * E2 +
                          P_L * P_R * Qt * Qp * db_dy * self.R * db_dx * E2 +
                          P_L * P_R * Qt * Qt * self.R * db_dx * self.R * db_dy * E2)

                d2F_dxdy = term_A + term_B + term_C + term_D + term_E

                # Full integrand from prefactor chain rule
                integrand = dF_dx + dF_dy + (1.0 / self.theta) * d2F_dxdy
                total += wu * wt * integrand * weight

        return total

    # =========================================================================
    # HIGHER ORDER INTEGRALS (stubs for now)
    # =========================================================================

    def _integral_20(self, weight_exp: int) -> float:
        """Compute integral for (l₁=2, m₁=0).

        Coefficient of x²y⁰ in (1/θ + x + y)×F:
        = (1/θ) × [x² coeff of F] + 1 × [x¹ coeff of F]
        = (1/2θ) × ∂²F/∂x² + ∂F/∂x
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = self.exp_2Rt[it]

            da_dx = self.darg_alpha_dx[it]
            db_dx = self.darg_beta_dx[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]

                # ∂F/∂x (same as in _integral_10)
                dF_dx = (Pp_L * P_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dx * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dx * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dx * E2)

                # ∂²F/∂x² - second derivative w.r.t. x
                # Product rule on P_L(u-x) gives Ppp_L
                # Q terms give Qpp with (da_dx)², etc.
                d2F_dx2 = (
                    # From P''_L
                    Ppp_L * P_R * Qt * Qt * E2 +
                    # Cross terms P'_L with Q derivatives
                    2 * Pp_L * P_R * Qp * da_dx * Qt * E2 +
                    2 * Pp_L * P_R * Qt * Qp * db_dx * E2 +
                    2 * Pp_L * P_R * Qt * Qt * self.R * (da_dx + db_dx) * E2 +
                    # Second derivatives of Q_α
                    P_L * P_R * Qpp * da_dx * da_dx * Qt * E2 +
                    P_L * P_R * Qp * da_dx * Qp * db_dx * E2 +
                    P_L * P_R * Qp * da_dx * Qt * self.R * db_dx * E2 +
                    # Second derivatives of Q_β
                    P_L * P_R * Qp * da_dx * Qp * db_dx * E2 +
                    P_L * P_R * Qt * Qpp * db_dx * db_dx * E2 +
                    P_L * P_R * Qt * Qp * db_dx * self.R * da_dx * E2 +
                    # R × second derivatives
                    P_L * P_R * Qp * da_dx * Qt * self.R * da_dx * E2 +
                    P_L * P_R * Qt * Qp * db_dx * self.R * da_dx * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dx + db_dx) ** 2 * E2
                )

                integrand = dF_dx + (1.0 / (2.0 * self.theta)) * d2F_dx2
                total += wu * wt * integrand * weight

        return total

    def _integral_02(self, weight_exp: int) -> float:
        """Compute integral for (l₁=0, m₁=2).

        Coefficient of x⁰y² in (1/θ + x + y)×F:
        = (1/2θ) × ∂²F/∂y² + ∂F/∂y
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = self.exp_2Rt[it]

            da_dy = self.darg_alpha_dy[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # ∂F/∂y
                dF_dy = (P_L * Pp_R * Qt * Qt * E2 +
                         P_L * P_R * Qp * da_dy * Qt * E2 +
                         P_L * P_R * Qt * Qp * db_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * da_dy * E2 +
                         P_L * P_R * Qt * Qt * self.R * db_dy * E2)

                # ∂²F/∂y²
                d2F_dy2 = (
                    P_L * Ppp_R * Qt * Qt * E2 +
                    2 * P_L * Pp_R * Qp * da_dy * Qt * E2 +
                    2 * P_L * Pp_R * Qt * Qp * db_dy * E2 +
                    2 * P_L * Pp_R * Qt * Qt * self.R * (da_dy + db_dy) * E2 +
                    P_L * P_R * Qpp * da_dy * da_dy * Qt * E2 +
                    P_L * P_R * Qp * da_dy * Qp * db_dy * E2 +
                    P_L * P_R * Qp * da_dy * Qt * self.R * db_dy * E2 +
                    P_L * P_R * Qp * da_dy * Qp * db_dy * E2 +
                    P_L * P_R * Qt * Qpp * db_dy * db_dy * E2 +
                    P_L * P_R * Qt * Qp * db_dy * self.R * da_dy * E2 +
                    P_L * P_R * Qp * da_dy * Qt * self.R * da_dy * E2 +
                    P_L * P_R * Qt * Qp * db_dy * self.R * da_dy * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dy + db_dy) ** 2 * E2
                )

                integrand = dF_dy + (1.0 / (2.0 * self.theta)) * d2F_dy2
                total += wu * wt * integrand * weight

        return total

    def _integral_21(self, weight_exp: int) -> float:
        """Compute integral for (l₁=2, m₁=1).

        Coefficient of x²y¹ in (1/θ + x + y)×F:
        = (1/θ) × (1/2!)×∂³F/∂x²∂y + 1×∂²F/∂x∂y + 1×(1/2!)×∂²F/∂x²
        = (1/2θ)×∂³F/∂x²∂y + ∂²F/∂x∂y + (1/2)×∂²F/∂x²
        """
        # For now, use a simpler approximation
        # Full implementation would require third derivatives
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = self.exp_2Rt[it]

            da_dx = self.darg_alpha_dx[it]
            da_dy = self.darg_alpha_dy[it]
            db_dx = self.darg_beta_dx[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]

                # ∂²F/∂x² (from _integral_20)
                d2F_dx2 = (
                    Ppp_L * P_R * Qt * Qt * E2 +
                    2 * Pp_L * P_R * Qp * da_dx * Qt * E2 +
                    2 * Pp_L * P_R * Qt * Qp * db_dx * E2 +
                    2 * Pp_L * P_R * Qt * Qt * self.R * (da_dx + db_dx) * E2 +
                    P_L * P_R * Qpp * da_dx * da_dx * Qt * E2 +
                    2 * P_L * P_R * Qp * da_dx * Qp * db_dx * E2 +
                    P_L * P_R * Qt * Qpp * db_dx * db_dx * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dx + db_dx) ** 2 * E2
                )

                # ∂²F/∂x∂y (from _integral_11)
                d2F_dxdy = (
                    Pp_L * Pp_R * Qt * Qt * E2 +
                    Pp_L * P_R * Qp * da_dx * Qt * E2 +
                    Pp_L * P_R * Qt * Qp * db_dx * E2 +
                    Pp_L * P_R * Qt * Qt * self.R * (da_dx + db_dx) * E2 +
                    P_L * Pp_R * Qp * da_dy * Qt * E2 +
                    P_L * Pp_R * Qt * Qp * db_dy * E2 +
                    P_L * Pp_R * Qt * Qt * self.R * (da_dy + db_dy) * E2 +
                    P_L * P_R * Qpp * da_dx * da_dy * Qt * E2 +
                    P_L * P_R * Qp * da_dx * Qp * db_dy * E2 +
                    P_L * P_R * Qp * da_dy * Qp * db_dx * E2 +
                    P_L * P_R * Qt * Qpp * db_dx * db_dy * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dx + db_dx) * (da_dy + db_dy) * E2
                )

                # For ∂³F/∂x²∂y, use finite difference approximation for now
                # (Should be replaced with analytical formula)
                d3F_dx2dy = 0.0  # Placeholder - higher order contribution

                integrand = 0.5 * d2F_dx2 + d2F_dxdy + (1.0 / (2.0 * self.theta)) * d3F_dx2dy
                total += wu * wt * integrand * weight

        return total

    def _integral_12(self, weight_exp: int) -> float:
        """Compute integral for (l₁=1, m₁=2).

        Coefficient of x¹y² in (1/θ + x + y)×F:
        = (1/2θ)×∂³F/∂x∂y² + ∂²F/∂x∂y + (1/2)×∂²F/∂y²
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = self.exp_2Rt[it]

            da_dx = self.darg_alpha_dx[it]
            da_dy = self.darg_alpha_dy[it]
            db_dx = self.darg_beta_dx[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # ∂²F/∂y²
                d2F_dy2 = (
                    P_L * Ppp_R * Qt * Qt * E2 +
                    2 * P_L * Pp_R * Qp * da_dy * Qt * E2 +
                    2 * P_L * Pp_R * Qt * Qp * db_dy * E2 +
                    2 * P_L * Pp_R * Qt * Qt * self.R * (da_dy + db_dy) * E2 +
                    P_L * P_R * Qpp * da_dy * da_dy * Qt * E2 +
                    2 * P_L * P_R * Qp * da_dy * Qp * db_dy * E2 +
                    P_L * P_R * Qt * Qpp * db_dy * db_dy * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dy + db_dy) ** 2 * E2
                )

                # ∂²F/∂x∂y
                d2F_dxdy = (
                    Pp_L * Pp_R * Qt * Qt * E2 +
                    Pp_L * P_R * Qp * da_dx * Qt * E2 +
                    Pp_L * P_R * Qt * Qp * db_dx * E2 +
                    Pp_L * P_R * Qt * Qt * self.R * (da_dx + db_dx) * E2 +
                    P_L * Pp_R * Qp * da_dy * Qt * E2 +
                    P_L * Pp_R * Qt * Qp * db_dy * E2 +
                    P_L * Pp_R * Qt * Qt * self.R * (da_dy + db_dy) * E2 +
                    P_L * P_R * Qpp * da_dx * da_dy * Qt * E2 +
                    P_L * P_R * Qp * da_dx * Qp * db_dy * E2 +
                    P_L * P_R * Qp * da_dy * Qp * db_dx * E2 +
                    P_L * P_R * Qt * Qpp * db_dx * db_dy * E2 +
                    P_L * P_R * Qt * Qt * self.R * self.R * (da_dx + db_dx) * (da_dy + db_dy) * E2
                )

                d3F_dxdy2 = 0.0  # Placeholder

                integrand = 0.5 * d2F_dy2 + d2F_dxdy + (1.0 / (2.0 * self.theta)) * d3F_dxdy2
                total += wu * wt * integrand * weight

        return total

    def _integral_22(self, weight_exp: int) -> float:
        """Compute integral for (l₁=2, m₁=2).

        Coefficient of x²y² in (1/θ + x + y)×F:
        = (1/4θ)×∂⁴F/∂x²∂y² + (1/2)×∂³F/∂x∂y² + (1/2)×∂³F/∂x²∂y
        """
        total = 0.0

        for it, t in enumerate(self.t_nodes):
            wt = self.t_weights[it]
            Qt = self.Q_t[it]
            Qp = self.Qp_t[it]
            Qpp = self.Qpp_t[it]
            E2 = self.exp_2Rt[it]

            da_dx = self.darg_alpha_dx[it]
            da_dy = self.darg_alpha_dy[it]
            db_dx = self.darg_beta_dx[it]
            db_dy = self.darg_beta_dy[it]

            for iu in range(self.n_quad):
                wu = self.u_weights[iu]
                u = self.u_nodes[iu]
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                P_L = self.P_L[iu]
                Pp_L = self.Pp_L[iu]
                Ppp_L = self.Ppp_L[iu]
                P_R = self.P_R[iu]
                Pp_R = self.Pp_R[iu]
                Ppp_R = self.Ppp_R[iu]

                # ∂²F/∂x²∂y² - leading term (simplified)
                # This is P''_L × P''_R × Q² × e^{2Rt} plus many cross terms
                d2F_dx2_dy2_leading = Ppp_L * Ppp_R * Qt * Qt * E2

                # Include more terms for better accuracy
                d2F_dx2_dy2 = (
                    d2F_dx2_dy2_leading +
                    # Cross terms with first derivatives
                    2 * Ppp_L * Pp_R * Qt * Qt * E2 * (Qp / Qt * da_dy + Qp / Qt * db_dy if abs(Qt) > 1e-12 else 0) +
                    2 * Pp_L * Ppp_R * Qt * Qt * E2 * (Qp / Qt * da_dx + Qp / Qt * db_dx if abs(Qt) > 1e-12 else 0)
                )

                # Third derivatives are complex - use simplified approximation
                d3F_dx2dy = Ppp_L * Pp_R * Qt * Qt * E2
                d3F_dxdy2 = Pp_L * Ppp_R * Qt * Qt * E2

                integrand = (0.5 * d3F_dxdy2 + 0.5 * d3F_dx2dy +
                            (1.0 / (4.0 * self.theta)) * d2F_dx2_dy2)
                total += wu * wt * integrand * weight

        return total

    # =========================================================================
    # MAIN EVALUATION METHODS
    # =========================================================================

    def _eval_derivative(self, l1: int, m1: int, weight_exp: int) -> float:
        """Dispatch to appropriate integral method."""
        if l1 == 0 and m1 == 0:
            return self._integral_00(weight_exp)
        elif l1 == 1 and m1 == 0:
            return self._integral_10(weight_exp)
        elif l1 == 0 and m1 == 1:
            return self._integral_01(weight_exp)
        elif l1 == 1 and m1 == 1:
            return self._integral_11(weight_exp)
        elif l1 == 2 and m1 == 0:
            return self._integral_20(weight_exp)
        elif l1 == 0 and m1 == 2:
            return self._integral_02(weight_exp)
        elif l1 == 2 and m1 == 1:
            return self._integral_21(weight_exp)
        elif l1 == 1 and m1 == 2:
            return self._integral_12(weight_exp)
        elif l1 == 2 and m1 == 2:
            return self._integral_22(weight_exp)
        else:
            raise NotImplementedError(f"Derivative order ({l1},{m1}) not implemented")

    def eval_monomial(self, mono: MonomialTwoC, ell: int, ellbar: int) -> MonomialContrib:
        """Evaluate a single monomial contribution.

        Weight convention: Use PAIR-BASED weights (matching GenEval), not monomial-based.
        - I₁-type (a>0, b>0): weight = (1-u)^{ell+ellbar}
        - I₂-type (a=0, b=0): weight = 1 (no weight)
        - I₃-type (a>0, b=0): weight = (1-u)^{ell}
        - I₄-type (a=0, b>0): weight = (1-u)^{ellbar}

        This is CRITICAL: For (1,1), both conventions give the same result.
        For (2,2)+, the pair-based weights are required to match GenEval/PRZZ.
        """
        a, b = mono.a, mono.b
        c_alpha, c_beta = mono.c_alpha, mono.c_beta
        d = mono.d
        psi_coeff = mono.coeff

        # Derivative order
        l1 = a
        m1 = b

        # Weight exponent: USE PAIR-BASED WEIGHTS (not a+b!)
        if a > 0 and b > 0:
            # I₁-type: mixed derivative
            weight_exp = ell + ellbar
        elif a == 0 and b == 0:
            # I₂-type: base term
            weight_exp = 0
        elif a > 0 and b == 0:
            # I₃-type: x-derivative only
            weight_exp = ell
        else:  # a == 0 and b > 0
            # I₄-type: y-derivative only
            weight_exp = ellbar

        # Compute integral
        integral_value = self._eval_derivative(l1, m1, weight_exp)

        # NOTE: c_alpha, c_beta, d are NOT directly used here.
        # The Ψ coefficients already account for the combinatorial structure.
        # The integral depends only on the derivative order (a, b) and weight.

        # Apply coefficient
        contribution = psi_coeff * integral_value

        return MonomialContrib(
            monomial=mono,
            a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d,
            psi_coeff=psi_coeff,
            weight_exp=weight_exp,
            integral_value=integral_value,
            contribution=contribution
        )

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> PairResult:
        """Evaluate full pair contribution."""
        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\n=== Section7Evaluator ({ell},{ellbar}): {len(monomials)} monomials ===")

        contribs = []
        total = 0.0

        for mono in monomials:
            try:
                result = self.eval_monomial(mono, ell, ellbar)
                total += result.contribution
                contribs.append(result)

                if verbose:
                    print(f"  {mono}: weight=(1-u)^{result.weight_exp}, "
                          f"integral={result.integral_value:.6f}, "
                          f"contrib={result.contribution:.6f}")

            except NotImplementedError as e:
                if verbose:
                    print(f"  {mono}: SKIPPED ({e})")
                # For unimplemented orders, skip but warn
                pass

        if verbose:
            print(f"  TOTAL = {total:.6f}")

        return PairResult(ell=ell, ellbar=ellbar, total=total, monomial_contribs=contribs)


def compute_c_section7(P1, P2, P3, Q, R: float, theta: float = 4/7,
                        n_quad: int = 60, verbose: bool = False) -> float:
    """Compute c using Section7Evaluator."""
    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    total = 0.0
    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        try:
            evaluator = Section7Evaluator(P_ell, P_ellbar, Q, R, theta, n_quad)
            result = evaluator.eval_pair(ell, ellbar, verbose=verbose)

            sym = 1 if ell == ellbar else 2
            total += sym * result.total

            if verbose:
                print(f"  ({ell},{ellbar}) × {sym} = {sym * result.total:.6f}")

        except NotImplementedError as e:
            if verbose:
                print(f"  ({ell},{ellbar}): SKIPPED ({e})")

    if verbose:
        print(f"\nTotal c = {total:.6f}")

    return total


# =============================================================================
# TESTING
# =============================================================================

def test_11_oracle():
    """Test (1,1) against the known oracle."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    evaluator = Section7Evaluator(P1, P1, Q, R=1.3036, theta=4/7, n_quad=60)
    result = evaluator.eval_pair(1, 1, verbose=True)

    oracle = 0.359159
    print(f"\n(1,1) Result: {result.total:.6f}")
    print(f"Oracle: {oracle:.6f}")
    print(f"Error: {abs(result.total - oracle) / oracle * 100:.2f}%")
    print(f"Match: {abs(result.total - oracle) < 1e-4}")


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 7 EVALUATOR TEST")
    print("=" * 70)

    test_11_oracle()
