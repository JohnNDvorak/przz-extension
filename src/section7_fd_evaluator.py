"""
src/section7_fd_evaluator.py
Section 7 F_d evaluator using proper Ψ → (k₁, l₁, m₁) mapping.

This module implements the PRZZ Section 7 evaluation pipeline correctly:
1. Expand Ψ_{ℓ,ℓ̄} to (a,b,c,d) monomials (psi_monomial_expansion.py)
2. Map each monomial to (k₁, l₁, m₁) triple (psi_fd_mapping.py)
3. Evaluate F_d^{left} × F_d^{right} for each triple based on Case A/B/C
4. Sum with Ψ coefficients

Key insight: The omega values determining Case A/B/C come from:
  ω_left = l₁ - 1  where l₁ = a + d
  ω_right = m₁ - 1  where m₁ = b + d

Not from X/Y/Z powers as the old evaluator used.

PRZZ Reference: arXiv:1802.10521, Section 7
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from math import factorial, log, exp
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from src.psi_fd_mapping import (
    FdCase, FdTriple, MonomialMapping,
    map_pair_monomials, get_eval_structure, monomial_to_triple
)


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class FdEvaluation:
    """Result of F_d factor evaluation."""
    case: FdCase
    l1: int
    omega: int
    value: float


class PolynomialWrapper:
    """Wrapper for polynomial with derivative evaluation."""

    def __init__(self, poly):
        self.poly = poly

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.poly.eval(u)

    def deriv(self, u: np.ndarray, order: int = 1) -> np.ndarray:
        return self.poly.eval_deriv(u, order)


class Section7FdEvaluator:
    """
    Evaluator using correct Ψ → F_d mapping.

    For each (ℓ, ℓ̄) pair:
    1. Get all (a,b,c,d) monomials with Ψ coefficients
    2. Map to (k₁, l₁, m₁) triples
    3. For each triple, evaluate:
       - F_d^{left}(l₁, α) based on ω_left = l₁ - 1 (Case A/B/C)
       - F_d^{right}(m₁, β) based on ω_right = m₁ - 1 (Case A/B/C)
    4. Sum: Σ psi_coeff × F_left × F_right × (other factors)
    """

    def __init__(
        self,
        P_left,  # P_ℓ
        P_right,  # P_ℓ̄
        Q,
        theta: float,
        R: float,
        n_quad: int = 60,
        logT: float = 100.0
    ):
        self.P_left = PolynomialWrapper(P_left)
        self.P_right = PolynomialWrapper(P_right)
        self.Q = PolynomialWrapper(Q)
        self.theta = theta
        self.R = R
        self.n_quad = n_quad
        self.logT = logT
        self.logN = theta * logT

        # α and β from PRZZ (shift parameters)
        self.alpha = -R / logT
        self.beta = -R / logT

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.a_nodes, self.a_weights = gauss_legendre_01(n_quad)  # For Case C kernel

        # Precompute polynomial values at nodes
        self._precompute_polys()

    def _precompute_polys(self):
        """Precompute polynomial and derivative values at quadrature nodes."""
        # Left polynomial (up to 3rd derivative for l1 up to 3)
        self.P_left_vals = {}
        for k in range(4):
            if k == 0:
                self.P_left_vals[k] = self.P_left(self.u_nodes)
            else:
                self.P_left_vals[k] = self.P_left.deriv(self.u_nodes, k)

        # Right polynomial
        self.P_right_vals = {}
        for k in range(4):
            if k == 0:
                self.P_right_vals[k] = self.P_right(self.u_nodes)
            else:
                self.P_right_vals[k] = self.P_right.deriv(self.u_nodes, k)

        # Q polynomial
        self.Q_vals = {}
        for k in range(3):
            if k == 0:
                self.Q_vals[k] = self.Q(self.u_nodes)
            else:
                self.Q_vals[k] = self.Q.deriv(self.u_nodes, k)

    def eval_Fd_case_A(self, iu: int, P_vals: Dict, side: str) -> float:
        """
        Case A (ω = -1, l = 0): Derivative structure.

        F_d = U(d,l) / logN × d/dx[N^{αx} × P(u + x)]|_{x=0}
            = U(d,l) / logN × [α × logN × P(u) + P'(u)]
            = U(d,l) × [α × P(u) + P'(u)/logN]

        For d=1, l=(0,): U = 1
        """
        u = self.u_nodes[iu]
        alpha = self.alpha if side == "left" else self.beta
        P_u = P_vals[0][iu]
        Pprime_u = P_vals[1][iu]

        # U(1, (0,)) = 1 (from PRZZ)
        U = 1.0

        return U * (alpha * P_u + Pprime_u / self.logN)

    def eval_Fd_case_B(self, iu: int, P_vals: Dict) -> float:
        """
        Case B (ω = 0, l = 1): Direct polynomial evaluation.

        F_d = V(d,l) × P(u)

        For d=1, l=(1,): V = -1
        """
        P_u = P_vals[0][iu]

        # V(1, (1,)) = -1 (from PRZZ)
        V = -1.0

        return V * P_u

    def eval_Fd_case_C(self, iu: int, P_vals: Dict, omega: int, side: str) -> float:
        """
        Case C (ω > 0, l > 1): Kernel integral.

        F_d = W(d,l) × (-1)^{1-ω}/(ω-1)! × (logN)^ω × u^ω
              × ∫₀¹ P((1-a)u) × a^{ω-1} × (N/n)^{-αau} da

        For d=1, l=(l₁,) with l₁ > 1: W = (-1)^{l₁}
        """
        u = self.u_nodes[iu]
        alpha = self.alpha if side == "left" else self.beta

        # Handle u = 0 case (kernel vanishes)
        if abs(u) < 1e-15:
            return 0.0

        # l₁ = ω + 1 for d=1
        l1 = omega + 1

        # W(1, (l₁,)) = (-1)^{l₁}
        W = (-1.0) ** l1

        # Prefactors
        sign_factor = (-1.0) ** (1 - omega)
        factorial_factor = 1.0 / factorial(omega - 1) if omega > 1 else 1.0
        logN_power = self.logN ** omega
        u_power = u ** omega

        # Compute the a-integral via quadrature
        # P((1-a) × u)
        a_args = (1.0 - self.a_nodes) * u
        # Need to evaluate P at arbitrary points for Case C
        P_at_a = self.P_left(a_args) if side == "left" else self.P_right(a_args)

        # a^{ω-1}
        a_power = self.a_nodes ** (omega - 1) if omega > 1 else np.ones_like(self.a_nodes)

        # (N/n)^{-αau} = exp(-α × logN × u × a)
        exp_factor = np.exp(-alpha * self.logN * u * self.a_nodes)

        # Full integrand
        integrand = P_at_a * a_power * exp_factor

        integral = float(np.sum(self.a_weights * integrand))

        return W * sign_factor * factorial_factor * logN_power * u_power * integral

    def eval_Fd_single(
        self, iu: int, l1: int, omega: int, case: FdCase, side: str
    ) -> float:
        """
        Evaluate a single F_d factor.

        Args:
            iu: u-node index
            l1: l₁ or m₁ value
            omega: ω value (l1 - 1)
            case: FdCase (A, B, or C)
            side: "left" or "right"

        Returns:
            F_d factor value
        """
        P_vals = self.P_left_vals if side == "left" else self.P_right_vals

        if case == FdCase.A:
            return self.eval_Fd_case_A(iu, P_vals, side)
        elif case == FdCase.B:
            return self.eval_Fd_case_B(iu, P_vals)
        else:  # Case C
            return self.eval_Fd_case_C(iu, P_vals, omega, side)

    def eval_triple(self, triple: FdTriple, psi_coeff: int) -> float:
        """
        Evaluate contribution from a single (k₁, l₁, m₁) triple.

        Args:
            triple: FdTriple with case information
            psi_coeff: Sum of Ψ coefficients for this triple

        Returns:
            Contribution to the pair integral
        """
        k1, l1, m1 = triple.k1, triple.l1, triple.m1

        # Accumulate over u-quadrature
        total = 0.0

        for iu in range(self.n_quad):
            u = self.u_nodes[iu]
            wu = self.u_weights[iu]

            # Evaluate F_d factors
            F_left = self.eval_Fd_single(
                iu, l1, triple.omega_left, triple.case_left, "left"
            )
            F_right = self.eval_Fd_single(
                iu, m1, triple.omega_right, triple.case_right, "right"
            )

            # Q(u) factor (for now, just Q²)
            Q_u = self.Q_vals[0][iu]
            Q_sq = Q_u * Q_u

            # exp(2Ru) from the asymptotic structure
            exp_factor = exp(2 * self.R * u)

            # Euler-Maclaurin weight: (1-u)^{k₁} where k₁ is convolution index
            # For Ψ, the convolution power comes from C^{c} contributions
            # The weight exponent should be c for a term with c C's
            # This is already encoded in k₁ = c
            em_weight = (1.0 - u) ** k1 if k1 > 0 else 1.0

            # Combine
            integrand = psi_coeff * F_left * F_right * Q_sq * exp_factor * em_weight

            total += wu * integrand

        # 1/θ prefactor from PRZZ normalization
        total /= self.theta

        return total

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate contribution from pair (ℓ, ℓ̄).

        Uses the correct Ψ → (k₁, l₁, m₁) mapping.

        Args:
            ell: Left piece index (1, 2, or 3)
            ellbar: Right piece index
            verbose: Print debug info

        Returns:
            Pair contribution
        """
        eval_struct = get_eval_structure(ell, ellbar)

        if verbose:
            print(f"\nPair ({ell},{ellbar}): {len(eval_struct)} unique triples")

        total = 0.0

        for (k1, l1, m1), info in eval_struct.items():
            # Construct FdTriple
            triple = FdTriple(
                k1=k1, l1=l1, m1=m1,
                omega_left=info['omega_left'],
                omega_right=info['omega_right'],
                case_left=info['case_left'],
                case_right=info['case_right']
            )

            psi_coeff = info['psi_coeff']

            # Evaluate this triple's contribution
            contrib = self.eval_triple(triple, psi_coeff)
            total += contrib

            if verbose:
                cases = f"{triple.case_left.value},{triple.case_right.value}"
                print(f"  ({k1},{l1},{m1}) Case {cases}: psi={psi_coeff:+d}, contrib={contrib:.6f}")

        if verbose:
            print(f"  Total: {total:.6f}")

        return total


def eval_full_k3_section7(
    polys: Dict,
    theta: float = 4.0/7.0,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = False
) -> Dict:
    """
    Evaluate all K=3 pairs using Section 7 F_d structure.

    Args:
        polys: Dict with 'P1', 'P2', 'P3', 'Q' polynomials
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        verbose: Print debug info

    Returns:
        Dict with 'total_c', 'kappa', and 'per_pair' breakdown
    """
    P1, P2, P3, Q = polys['P1'], polys['P2'], polys['P3'], polys['Q']
    poly_map = {1: P1, 2: P2, 3: P3}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    per_pair = {}
    total_c = 0.0

    for (ell, ellbar) in pairs:
        P_left = poly_map[ell]
        P_right = poly_map[ellbar]

        evaluator = Section7FdEvaluator(P_left, P_right, Q, theta, R, n_quad)
        contrib = evaluator.eval_pair(ell, ellbar, verbose=verbose)

        # Symmetry factor
        sym_factor = 1 if ell == ellbar else 2
        pair_total = sym_factor * contrib

        per_pair[(ell, ellbar)] = pair_total
        total_c += pair_total

        if verbose:
            print(f"({ell},{ellbar}): contrib={contrib:.6f}, sym×{sym_factor} = {pair_total:.6f}")

    kappa = 1 - log(total_c) / R if total_c > 0 else float('nan')

    return {
        'total_c': total_c,
        'kappa': kappa,
        'per_pair': per_pair,
        'R': R,
        'theta': theta
    }


def validate_11_against_oracle():
    """Validate (1,1) against the known I-term oracle."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("="*70)
    print("SECTION 7 F_d VALIDATION: (1,1) pair")
    print("="*70)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nI-term oracle (1,1):")
    print(f"  I1={oracle.I1:.6f}, I2={oracle.I2:.6f}, I3={oracle.I3:.6f}, I4={oracle.I4:.6f}")
    print(f"  Total = {oracle.total:.6f}")

    # Section 7 F_d evaluator
    evaluator = Section7FdEvaluator(P1, P1, Q, theta, R, n_quad)
    section7_total = evaluator.eval_pair(1, 1, verbose=True)

    print(f"\nSection 7 F_d total: {section7_total:.6f}")
    print(f"Oracle total:        {oracle.total:.6f}")
    print(f"Ratio:               {section7_total/oracle.total:.6f}")

    if abs(section7_total - oracle.total) < 0.1 * abs(oracle.total):
        print("\n✓ Within 10% of oracle")
    else:
        print("\n✗ More than 10% deviation from oracle")


def main():
    """Run full K=3 evaluation."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    print("="*70)
    print("SECTION 7 F_d EVALUATION: Full K=3")
    print("="*70)

    result = eval_full_k3_section7(polys, verbose=True)

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total c:   {result['total_c']:.6f}")
    print(f"Target c:  2.137")
    print(f"Ratio:     {result['total_c']/2.137:.4f}")
    print()
    print(f"κ = 1 - log(c)/R = {result['kappa']:.6f}")
    print(f"Target κ:  0.417")


if __name__ == "__main__":
    # First validate (1,1)
    validate_11_against_oracle()
    print()

    # Then run full K=3
    main()
