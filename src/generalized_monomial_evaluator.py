"""
src/generalized_monomial_evaluator.py
Generalized Ψ Monomial Evaluator for All Pairs

This module extends the I-term monomial evaluator to handle arbitrary (a,b,c,d)
monomials from higher pairs like (2,2), (3,3), etc.

Key insight: Each monomial A^a B^b C^c D^d corresponds to a specific derivative
extraction pattern:
  - A^a: extract P^{(a)}_left(u) via d^a/dx^a
  - B^b: extract P^{(b)}_right(u) via d^b/dy^b
  - C^c: convolution index (affects weight exponent)
  - D^d: coupled derivative structure

The integral structure is:
  ∫∫ [derivative extraction] × Q² × exp × (1-u)^{weight} du dt

Weight exponent:
  For pair (ℓ, ℓ̄), base weight is (1-u)^{ℓ+ℓ̄-a-b+c} (tentative)

PRZZ Reference: arXiv:1802.10521, Section 7
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, List
from math import exp, log, comb, factorial
from dataclasses import dataclass

from src.psi_monomial_expansion import expand_pair_to_monomials


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class GeneralizedMonomialEvaluator:
    """
    Evaluates arbitrary (a,b,c,d) monomials using generalized derivative extraction.

    For monomial A^a B^b C^c D^d:
    - Compute d^a/dx^a[P_left(u+x)]|_{x=0} = P^{(a)}_left(u)
    - Compute d^b/dy^b[P_right(u+y)]|_{y=0} = P^{(b)}_right(u)
    - Handle C convolution and D coupling appropriately
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float,
                 ell: int, ellbar: int, n_quad: int = 60):
        """
        Initialize evaluator for pair (ℓ, ℓ̄).

        Args:
            P_left: P_ℓ polynomial
            P_right: P_{ℓ̄} polynomial
            Q: Q polynomial
            theta: θ = 4/7
            R: R parameter
            ell: Left piece index
            ellbar: Right piece index
            n_quad: Quadrature points
        """
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.theta = theta
        self.R = R
        self.ell = ell
        self.ellbar = ellbar
        self.n_quad = n_quad

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial derivatives up to max needed order
        # For (3,3), max power is 3, so need up to 3rd derivative
        self.max_deriv = max(ell, ellbar) + 1
        self._precompute()

    def _precompute(self):
        """Precompute polynomial values and derivatives at quadrature nodes."""
        self.P_left_derivs = {}
        self.P_right_derivs = {}
        self.Q_derivs = {}

        for k in range(self.max_deriv + 1):
            if k == 0:
                self.P_left_derivs[k] = self.P_left.eval(self.u_nodes)
                self.P_right_derivs[k] = self.P_right.eval(self.u_nodes)
            else:
                self.P_left_derivs[k] = self.P_left.eval_deriv(self.u_nodes, k)
                self.P_right_derivs[k] = self.P_right.eval_deriv(self.u_nodes, k)

        for k in range(min(3, self.max_deriv) + 1):
            if k == 0:
                self.Q_derivs[k] = self.Q.eval(self.t_nodes)
            else:
                self.Q_derivs[k] = self.Q.eval_deriv(self.t_nodes, k)

    def eval_base_integral(self, weight_exp: int = 0) -> float:
        """
        Evaluate base integral with (1-u)^weight_exp weight.

        I = (1/θ) × ∫∫ P_left(u) × P_right(u) × Q(t)² × exp(2Rt) × (1-u)^w du dt
        """
        P_L = self.P_left_derivs[0]
        P_R = self.P_right_derivs[0]
        Q_t = self.Q_derivs[0]

        # u-integral with weight
        if weight_exp > 0:
            weight = (1.0 - self.u_nodes) ** weight_exp
            u_int = np.sum(self.u_weights * P_L * P_R * weight)
        else:
            u_int = np.sum(self.u_weights * P_L * P_R)

        # t-integral
        exp_2Rt = np.exp(2 * self.R * self.t_nodes)
        t_int = np.sum(self.t_weights * Q_t * Q_t * exp_2Rt)

        return (1.0 / self.theta) * u_int * t_int

    def eval_derivative_integral(self, a: int, b: int, weight_exp: int = 0) -> float:
        """
        Evaluate integral with d^a/dx^a × d^b/dy^b derivative extraction.

        This extracts P^{(a)}_left(u) × P^{(b)}_right(u) from the prefactor.

        For the PRZZ structure with Q(α)Q(β)exp(R(α+β)), the full derivative
        involves chain rule terms. This method computes the leading term
        P^{(a)} × P^{(b)} × Q² × exp.

        For now, we use a simplified structure that captures the main contribution.
        A full implementation would need to track all chain rule terms.
        """
        if a > self.max_deriv or b > self.max_deriv:
            return 0.0

        P_L_a = self.P_left_derivs[a]
        P_R_b = self.P_right_derivs[b]
        Q_t = self.Q_derivs[0]

        # Main term: P^{(a)}_L × P^{(b)}_R × Q² × exp
        if weight_exp > 0:
            weight = (1.0 - self.u_nodes) ** weight_exp
            u_int = np.sum(self.u_weights * P_L_a * P_R_b * weight)
        else:
            u_int = np.sum(self.u_weights * P_L_a * P_R_b)

        exp_2Rt = np.exp(2 * self.R * self.t_nodes)
        t_int = np.sum(self.t_weights * Q_t * Q_t * exp_2Rt)

        # Apply 1/θ^{a+b} normalization from derivative prefactors
        # This is a heuristic - the exact normalization needs verification
        theta_factor = self.theta ** (a + b)

        return u_int * t_int / theta_factor

    def eval_monomial_ab(self, a: int, b: int, weight_exp: int) -> float:
        """
        Evaluate pure A^a B^b monomial (no C or D).

        The integral extracts P^{(a)} × P^{(b)} through derivatives.
        """
        return self.eval_derivative_integral(a, b, weight_exp)

    def eval_monomial_d_power(self, d: int, weight_exp: int) -> float:
        """
        Evaluate pure D^d monomial.

        D corresponds to (ζ'/ζ)' which is a second-derivative structure.
        For D^d, we have d coupled second derivatives.

        For D^1 (d=1), this is the I₂-type integral.
        For D^d, the structure is more complex.

        Approximation: D^d ~ (I₂-type)^d / (d!)
        This is a placeholder - needs proper derivation.
        """
        if d == 0:
            return 0.0

        # For d=1, use base integral (I₂ structure)
        if d == 1:
            return self.eval_base_integral(weight_exp)

        # For d>1, approximate as I₂^d / d!
        # This is likely wrong but provides a starting point
        I2 = self.eval_base_integral(weight_exp)
        return (I2 ** d) / factorial(d)

    def eval_monomial_with_c(self, a: int, b: int, c: int, d: int,
                             weight_exp: int) -> float:
        """
        Evaluate monomial with C^c convolution.

        C factors modify the convolution structure and add to the weight exponent.
        For C^c, the weight becomes (1-u)^{weight_exp + c}.

        The derivative structure is still A^a B^b-like, but with modified weight.
        """
        effective_weight = weight_exp + c

        if a == 0 and b == 0:
            # Pure C^c × D^d
            if d > 0:
                return self.eval_monomial_d_power(d, effective_weight)
            else:
                return 0.0  # Pure C^c contributes 0 (no derivative extraction)
        else:
            # A^a B^b C^c D^d
            # Main contribution is from A^a B^b with modified weight
            ab_contrib = self.eval_monomial_ab(a, b, effective_weight)

            # D adds additional structure
            if d > 0:
                d_contrib = self.eval_monomial_d_power(d, effective_weight)
                # Heuristic combination - needs proper derivation
                return ab_contrib + d_contrib
            else:
                return ab_contrib

    def eval_monomial(self, a: int, b: int, c: int, d: int, coeff: int) -> float:
        """
        Evaluate monomial A^a B^b C^c D^d with coefficient.

        Weight exponent is determined by the pair indices and monomial structure.
        For pair (ℓ, ℓ̄), the base weight from Euler-Maclaurin is related to
        the convolution count.

        Heuristic: weight_exp = ℓ + ℓ̄ - 2 for the main terms
        (This matches (1,1) where weight is (1-u)^0 for I₂ and (1-u)² for I₁)
        """
        # Special handling for known (1,1) monomials
        if self.ell == 1 and self.ellbar == 1:
            return self._eval_11_monomial(a, b, c, d, coeff)

        # General case weight heuristic
        # The weight exponent depends on the derivative structure
        # Higher derivatives reduce the weight exponent
        base_weight = self.ell + self.ellbar
        deriv_reduction = a + b + 2 * d  # Each A/B reduces by 1, D reduces by 2
        weight_exp = max(0, base_weight - deriv_reduction)

        return coeff * self.eval_monomial_with_c(a, b, c, d, weight_exp)

    def _eval_11_monomial(self, a: int, b: int, c: int, d: int, coeff: int) -> float:
        """
        Evaluate (1,1) monomials using validated I-term structure.

        This uses the known working formulas from przz_iterm_monomial_evaluator.
        """
        # Import the working evaluator
        from src.przz_iterm_monomial_evaluator import ItermMonomialEvaluator

        evaluator = ItermMonomialEvaluator(
            self.P_left, self.P_right, self.Q,
            self.theta, self.R, self.n_quad
        )
        return evaluator.eval_monomial(a, b, c, d, coeff, 1, 1)

    def eval_pair(self, verbose: bool = False) -> float:
        """
        Evaluate full Ψ contribution for this pair.

        Expands Ψ_{ℓ,ℓ̄} to monomials and evaluates each.
        """
        monomials = expand_pair_to_monomials(self.ell, self.ellbar)

        if verbose:
            print(f"\nPair ({self.ell},{self.ellbar}): {len(monomials)} monomials")

        total = 0.0
        for (a, b, c, d), coeff in sorted(monomials.items()):
            contrib = self.eval_monomial(a, b, c, d, coeff)
            total += contrib

            if verbose:
                sign = "+" if coeff > 0 else ""
                print(f"  {sign}{coeff} × A^{a}B^{b}C^{c}D^{d}: {contrib:.6f}")

        if verbose:
            print(f"  Total = {total:.6f}")

        return total


def eval_full_k3(polys: Dict, theta: float = 4.0/7.0, R: float = 1.3036,
                 n_quad: int = 60, verbose: bool = False) -> Dict:
    """
    Evaluate all K=3 pairs using generalized monomial evaluator.
    """
    P1, P2, P3, Q = polys['P1'], polys['P2'], polys['P3'], polys['Q']
    poly_map = {1: P1, 2: P2, 3: P3}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    per_pair = {}
    total_c = 0.0

    for (ell, ellbar) in pairs:
        evaluator = GeneralizedMonomialEvaluator(
            poly_map[ell], poly_map[ellbar], Q,
            theta, R, ell, ellbar, n_quad
        )
        contrib = evaluator.eval_pair(verbose=verbose)

        # Symmetry factor
        sym_factor = 1 if ell == ellbar else 2
        pair_total = sym_factor * contrib

        per_pair[(ell, ellbar)] = pair_total
        total_c += pair_total

        if not verbose:
            print(f"({ell},{ellbar}): {len(expand_pair_to_monomials(ell, ellbar))} monomials, "
                  f"contrib={contrib:.6f}, sym×{sym_factor} = {pair_total:.6f}")

    kappa = 1 - log(total_c) / R if total_c > 0 else float('nan')

    return {
        'total_c': total_c,
        'kappa': kappa,
        'per_pair': per_pair,
        'R': R,
        'theta': theta
    }


def main():
    """Run full K=3 evaluation."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    print("="*70)
    print("GENERALIZED MONOMIAL EVALUATOR: Full K=3")
    print("="*70)

    result = eval_full_k3(polys, verbose=False)

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
    main()
