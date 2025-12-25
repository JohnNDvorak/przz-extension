"""
src/fd_evaluation.py
Clean F_d Evaluation Module for PRZZ Section 7

This module implements the three F_d cases for d=1:
- Case A (ω = -1, l = 0): Derivative form
- Case B (ω = 0, l = 1): Direct polynomial evaluation
- Case C (ω ≥ 1, l ≥ 2): Kernel integral

These cases apply to both "left" (α-side) and "right" (β-side) factors.
For higher pairs like (2,2), we need all combinations:
  (A,A), (A,B), (A,C), (B,A), (B,B), (B,C), (C,A), (C,B), (C,C)

PRZZ Reference: arXiv:1802.10521, Section 7, equations around line 2360-2374

IMPORTANT NOTES:

1. Case A (l=0, ω=-1) now includes the P'(u)/θ term that survives after the t-identity.
   Previous versions incorrectly dropped this term assuming it vanishes asymptotically.
   After the t-identity (which replaces (α+β)^{-1} structure), the formula becomes:
       F_A(u) = α × P(u) + (1/θ) × P'(u)

2. The MonoMialFdEvaluator applies per-monomial (1-u)^{a+b} weights correctly.
   This is essential for (2,2)+ pairs where different monomials have different weights.

3. Q-operator handling: Currently uses simplified Q² t-integral. For full correctness,
   higher pairs need Q, Q', Q'' terms based on derivative orders.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple
from enum import Enum
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class FdCase(Enum):
    """F_d evaluation case based on ω value."""
    A = "A"  # ω = -1 (l=0): derivative form
    B = "B"  # ω = 0 (l=1): direct polynomial
    C = "C"  # ω > 0 (l>1): kernel integral


def get_fd_case(l: int) -> FdCase:
    """Determine F_d Case from l value (derivative count).

    Args:
        l: l₁ or m₁ derivative count (= a + d or b + d from monomial)

    Returns:
        FdCase enum (A, B, or C)
    """
    omega = l - 1
    if omega == -1:
        return FdCase.A
    elif omega == 0:
        return FdCase.B
    else:
        return FdCase.C


class FdEvaluator:
    """
    Clean F_d evaluator for all three cases.

    For a polynomial P and parameters (R, θ), evaluates:
    - Case A: α × P(u)  [in asymptotic limit where P'/log N → 0]
    - Case B: P(u)
    - Case C: u^ω × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da

    The exponential in Case C comes from:
        (N/n)^{-αau} → exp(-α × log N × u × a) → exp(R × θ × u × a)
    using α = -R/log T and log N = θ × log T.
    """

    def __init__(self, P, R: float, theta: float, n_quad: int = 60):
        """
        Initialize F_d evaluator.

        Args:
            P: Polynomial object with .eval(u) method
            R: PRZZ R parameter
            theta: PRZZ θ parameter (= 4/7 typically)
            n_quad: Quadrature points
        """
        self.P = P
        self.R = R
        self.theta = theta
        self.n_quad = n_quad

        # Quadrature nodes for kernel integral (a-variable)
        self.a_nodes, self.a_weights = gauss_legendre_01(n_quad)

    def eval_case_A(self, u: np.ndarray, alpha: float) -> np.ndarray:
        """
        Case A (ω = -1, l = 0): Derivative form with θ-scaling correction.

        After applying the t-identity (replacing (α+β)^{-1} structure), the
        P'(u)/log N term does NOT vanish. Instead, it becomes P'(u)/θ because:
          - log N = θ × log T
          - The identity absorbs log T factors
          - So (log T)/(log N) = 1/θ survives

        The correct formula in the scaled/post-identity world is:
            F_A(u) = α × P(u) + (1/θ) × P'(u)

        At α = -R (evaluation point), this becomes:
            F_A(u) = -R × P(u) + (1/θ) × P'(u)

        Args:
            u: Quadrature nodes
            alpha: α parameter (typically -R at evaluation point)
        """
        P_val = self.P.eval(u)
        P_deriv = self.P.eval_deriv(u, 1)  # First derivative
        return alpha * P_val + (1.0 / self.theta) * P_deriv

    def eval_case_B(self, u: np.ndarray) -> np.ndarray:
        """
        Case B (ω = 0, l = 1): Direct polynomial evaluation.

        F_B(u) = P(u)
        """
        return self.P.eval(u)

    def eval_case_C(self, u: np.ndarray, omega: int) -> np.ndarray:
        """
        Case C (ω ≥ 1, l ≥ 2): Kernel integral.

        From PRZZ, the kernel structure is:
            K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(R×θ×u×a) da

        Then:
            F_C(u) = u^ω × K_ω(u; R)

        Note: Factorial and sign prefactors are handled at the monomial level.

        Args:
            u: Quadrature nodes
            omega: ω parameter (≥ 1 for Case C)
        """
        n_u = len(u)
        result = np.zeros(n_u)

        for i in range(n_u):
            ui = u[i]

            if ui < 1e-15:
                # u ≈ 0: integral vanishes due to u^ω factor
                result[i] = 0.0
                continue

            # Arguments for P: (1-a)×u for each a node
            args = (1.0 - self.a_nodes) * ui

            # Integrand: P((1-a)u) × a^{ω-1} × exp(R×θ×u×a)
            P_vals = self.P.eval(args)
            a_power = self.a_nodes ** (omega - 1) if omega > 1 else np.ones_like(self.a_nodes)
            exp_factor = np.exp(self.R * self.theta * ui * self.a_nodes)

            integrand = P_vals * a_power * exp_factor
            kernel = np.sum(self.a_weights * integrand)

            # F_C = u^ω × K_ω(u)
            result[i] = (ui ** omega) * kernel

        return result

    def eval(self, u: np.ndarray, l: int, alpha: float = 0.0) -> np.ndarray:
        """
        Evaluate F_d for given l value, dispatching to correct case.

        Args:
            u: Quadrature nodes
            l: Derivative count (l₁ or m₁)
            alpha: α or β parameter (only used for Case A)
        """
        case = get_fd_case(l)

        if case == FdCase.A:
            return self.eval_case_A(u, alpha)
        elif case == FdCase.B:
            return self.eval_case_B(u)
        else:  # Case C
            omega = l - 1
            return self.eval_case_C(u, omega)


@dataclass
class MonomialEvaluation:
    """Result of evaluating a single Ψ monomial."""
    a: int
    b: int
    c_alpha: int
    c_beta: int
    d: int
    psi_coeff: int
    l1: int
    m1: int
    case_left: FdCase
    case_right: FdCase
    u_integral: float
    weight_exponent: int
    contribution: float


class MonoMialFdEvaluator:
    """
    Evaluate F_d × F_d for a monomial from the Ψ expansion.

    Given a monomial A^a × B^b × C_α^{c_α} × C_β^{c_β} × D^d:
    - Compute l₁ = a + d (left derivative count)
    - Compute m₁ = b + d (right derivative count)
    - Evaluate F_d^{left}(l₁) × F_d^{right}(m₁)
    - Apply (1-u)^{a+b} Euler-Maclaurin weight
    """

    def __init__(self, P_left, P_right, Q, R: float, theta: float, n_quad: int = 60):
        """
        Initialize monomial evaluator.

        Args:
            P_left: Left polynomial (P_ℓ)
            P_right: Right polynomial (P_ℓ̄)
            Q: Q polynomial
            R: PRZZ R parameter
            theta: PRZZ θ parameter
            n_quad: Quadrature points
        """
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.R = R
        self.theta = theta
        self.n_quad = n_quad

        # F_d evaluators for left and right
        self.fd_left = FdEvaluator(P_left, R, theta, n_quad)
        self.fd_right = FdEvaluator(P_right, R, theta, n_quad)

        # Quadrature for u ∈ [0,1] and t ∈ [0,1]
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute common values
        self.Q_vals = self.Q.eval(self.t_nodes)
        self.exp_2Rt = np.exp(2 * self.R * self.t_nodes)

        # t-integral of Q²e^{2Rt}
        self.t_integral = np.sum(self.t_weights * self.Q_vals * self.Q_vals * self.exp_2Rt)

    def eval_monomial(self, mono) -> MonomialEvaluation:
        """
        Evaluate a single monomial contribution.

        Args:
            mono: MonomialSeparatedC object from psi_separated_c.py

        Returns:
            MonomialEvaluation with detailed breakdown
        """
        # Extract indices
        a, b = mono.a, mono.b
        c_alpha, c_beta = mono.c_alpha, mono.c_beta
        d = mono.d
        psi_coeff = mono.coeff

        # Derivative counts
        l1 = a + d  # left
        m1 = b + d  # right

        # Determine cases
        case_left = get_fd_case(l1)
        case_right = get_fd_case(m1)

        # For Case A, we need α/β parameters
        # In PRZZ, α = β = -R at x = y = 0
        alpha = -self.R
        beta = -self.R

        # Evaluate F_d on each side
        F_left = self.fd_left.eval(self.u_nodes, l1, alpha)
        F_right = self.fd_right.eval(self.u_nodes, m1, beta)

        # Euler-Maclaurin weight: (1-u)^{a+b}
        # a and b are singleton block counts (A and B powers)
        weight_exp = a + b
        if weight_exp > 0:
            weight = (1.0 - self.u_nodes) ** weight_exp
        else:
            weight = np.ones_like(self.u_nodes)

        # u-integral: ∫ F_left(u) × F_right(u) × (1-u)^{a+b} du
        u_integrand = F_left * F_right * weight
        u_integral = np.sum(self.u_weights * u_integrand)

        # Full contribution: psi_coeff × u_integral × t_integral / θ
        contribution = psi_coeff * u_integral * self.t_integral / self.theta

        return MonomialEvaluation(
            a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d,
            psi_coeff=psi_coeff,
            l1=l1, m1=m1,
            case_left=case_left, case_right=case_right,
            u_integral=u_integral,
            weight_exponent=weight_exp,
            contribution=contribution
        )

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate full contribution from pair (ℓ, ℓ̄) using Ψ expansion.

        Args:
            ell: Left piece index
            ellbar: Right piece index
            verbose: Print per-monomial breakdown

        Returns:
            Total pair contribution
        """
        # Use CANONICAL Ψ expansion module
        from src.psi_expansion import expand_psi

        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\n=== ({ell},{ellbar}) Evaluation: {len(monomials)} monomials ===")

        total = 0.0
        for mono in monomials:
            result = self.eval_monomial(mono)
            total += result.contribution

            if verbose:
                print(f"  A^{result.a}B^{result.b}C_α^{result.c_alpha}C_β^{result.c_beta}D^{result.d}:")
                print(f"    psi={result.psi_coeff:+d}, l1={result.l1}, m1={result.m1}, "
                      f"cases=({result.case_left.value},{result.case_right.value})")
                print(f"    u_int={result.u_integral:.6f}, weight=(1-u)^{result.weight_exponent}")
                print(f"    contrib={result.contribution:.6f}")

        if verbose:
            print(f"  TOTAL = {total:.6f}")

        return total


def test_11_pair():
    """Test (1,1) pair against known oracle values."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    evaluator = MonoMialFdEvaluator(P1, P1, Q, R=1.3036, theta=4/7, n_quad=60)
    result = evaluator.eval_pair(1, 1, verbose=True)

    print(f"\n(1,1) Total: {result:.6f}")
    print(f"Oracle target: 0.359159")
    print(f"Ratio: {result / 0.359159:.4f}")


def test_all_pairs():
    """Test all K=3 pairs and compare to GeneralizedItermEvaluator."""
    from src.polynomials import load_przz_polynomials
    from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    poly_map = {1: P1, 2: P2, 3: P3}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 70)
    print("COMPARING MonoMialFdEvaluator vs GeneralizedItermEvaluator")
    print("=" * 70)

    fd_total = 0.0
    iterm_total = 0.0

    for ell, ellbar in pairs:
        P_left = poly_map[ell]
        P_right = poly_map[ellbar]

        # New F_d evaluator
        fd_eval = MonoMialFdEvaluator(P_left, P_right, Q, R=1.3036, theta=4/7, n_quad=60)
        fd_contrib = fd_eval.eval_pair(ell, ellbar, verbose=False)

        # Old I-term evaluator
        iterm_eval = GeneralizedItermEvaluator(P_left, P_right, Q, 4/7, 1.3036, ell, ellbar, 60)
        iterm_contrib = iterm_eval.eval_all().total

        sym = 1 if ell == ellbar else 2
        fd_total += sym * fd_contrib
        iterm_total += sym * iterm_contrib

        print(f"({ell},{ellbar}): Fd={fd_contrib:.6f}, Iterm={iterm_contrib:.6f}, "
              f"diff={abs(fd_contrib-iterm_contrib):.6f}")

    print()
    print(f"Total: Fd={fd_total:.6f}, Iterm={iterm_total:.6f}")
    print(f"Target c = 2.137")


if __name__ == "__main__":
    test_11_pair()
    print()
    test_all_pairs()
