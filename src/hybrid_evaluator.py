"""
src/hybrid_evaluator.py
Hybrid Evaluator: Series I1 + GenEval I2/I3/I4

This evaluator addresses the mixed-derivative bug in manual evaluators by:
- Using series engine for I1 (mixed derivative d²F/dxdy) - CORRECT
- Using GenEval formulas for I2/I3/I4 - simpler, less room for cross-term errors

The key insight (from Session 11 GPT feedback):
- Series coefficient f[1,1] is the correct ∂²F/∂x∂y|_{0,0} / (1!×1!)
- Manual evaluators have incomplete cross-terms in the mixed derivative
- But I2/I3/I4 are simpler (no mixed derivatives) and likely correct

Mapping from I-terms to coefficients:
- I1: Uses g[1,1] integral with weight (1-u)^{ell+ellbar}
- I2: Uses g[0,0] integral (no weight factor beyond prefactor)
- I3: Uses g[1,0] integral with weight (1-u)^{ell}
- I4: Uses g[0,1] integral with weight (1-u)^{ellbar}
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple
from math import exp

from src.psi_series_evaluator import PsiSeriesEvaluator


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class HybridResult(NamedTuple):
    """Result of hybrid I-term computation."""
    I1: float  # From series engine
    I2: float  # From GenEval
    I3: float  # From GenEval
    I4: float  # From GenEval
    total: float
    ell: int
    ellbar: int


class HybridEvaluator:
    """
    Hybrid evaluator combining series-based I1 with GenEval I2/I3/I4.

    Uses:
    - Series engine for I1 (mixed derivative) - verified correct via coefficient gate
    - GenEval-style formulas for I2, I3, I4 - simpler, less error-prone
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

        # Weight exponents (from Ψ monomial structure)
        self.I1_weight_exp = ell + ellbar  # (1-u)^{ell+ellbar}
        self.I3_weight_exp = ell           # (1-u)^{ell}
        self.I4_weight_exp = ellbar        # (1-u)^{ellbar}

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Series evaluator for I1 (max_order=1 is sufficient for g[1,1])
        self.series_eval = PsiSeriesEvaluator(
            P_left, P_right, Q, R, theta,
            max_order=max(ell, ellbar),  # Need derivatives up to max(ell, ellbar)
            n_quad=n_quad
        )
        self._integral_grid = None

        # Precompute polynomial values for GenEval-style I2/I3/I4
        self._precompute()

    def _get_integral_grid(self):
        """Compute and cache the series integral grid once per evaluator."""
        if self._integral_grid is None:
            self._integral_grid = self.series_eval.compute_integral_grid()
        return self._integral_grid

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

        # Exponential factor
        self.exp_2Rt = np.exp(2 * self.R * self.t_nodes)

    def eval_I1(self) -> float:
        """
        I1 via series engine: Integrates g[1,1] with weight (1-u)^{ell+ellbar}.

        This is the CORRECT mixed derivative computation.
        """
        integral_grid = self._get_integral_grid()
        return integral_grid[(1, 1, self.I1_weight_exp)]

    def eval_I2(self) -> float:
        # g[0,0] integrates to I2 (no (1-u) weight).
        integral_grid = self._get_integral_grid()
        return integral_grid[(0, 0, 0)]

    def eval_I3(self) -> float:
        # g[1,0] is the prefactor-chain-rule integrand; I3 adds an overall minus sign.
        integral_grid = self._get_integral_grid()
        return -integral_grid[(1, 0, self.I3_weight_exp)]

    def eval_I4(self) -> float:
        if self.ell == self.ellbar:
            return self.eval_I3()
        integral_grid = self._get_integral_grid()
        return -integral_grid[(0, 1, self.I4_weight_exp)]

    def eval_all(self) -> HybridResult:
        """Compute all I-terms and return result."""
        I1 = self.eval_I1()
        I2 = self.eval_I2()
        I3 = self.eval_I3()
        I4 = self.eval_I4()
        total = I1 + I2 + I3 + I4

        return HybridResult(
            I1=I1, I2=I2, I3=I3, I4=I4, total=total,
            ell=self.ell, ellbar=self.ellbar
        )


def compute_c_hybrid(P1, P2, P3, Q, R: float, theta: float = 4/7,
                      n_quad: int = 60, verbose: bool = False) -> float:
    """
    Compute total c using HybridEvaluator.

    Returns sum over all pairs (ℓ₁,ℓ₂) with symmetry factors.
    """
    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    total = 0.0
    if verbose:
        print("\n" + "=" * 60)
        print("HYBRID EVALUATOR: c COMPUTATION")
        print("=" * 60)

    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        evaluator = HybridEvaluator(
            P_ell, P_ellbar, Q, theta, R,
            ell, ellbar, n_quad
        )
        result = evaluator.eval_all()

        sym = 1 if ell == ellbar else 2
        pair_contrib = sym * result.total
        total += pair_contrib

        if verbose:
            print(f"\n({ell},{ellbar}) × {sym}:")
            print(f"  I1 (series): {result.I1:.6f}")
            print(f"  I2 (direct): {result.I2:.6f}")
            print(f"  I3 (direct): {result.I3:.6f}")
            print(f"  I4 (direct): {result.I4:.6f}")
            print(f"  Total: {result.total:.6f} × {sym} = {pair_contrib:.6f}")

    if verbose:
        print("\n" + "-" * 60)
        print(f"TOTAL c = {total:.6f}")

    return total


# =============================================================================
# COMPARISON WITH GENEVAL
# =============================================================================

def compare_with_geneval():
    """Compare hybrid evaluator against GenEval for all pairs."""
    from src.polynomials import load_przz_polynomials
    from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("HYBRID VS GENEVAL COMPARISON")
    print("=" * 70)

    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        hybrid = HybridEvaluator(P_ell, P_ellbar, Q, theta, R, ell, ellbar, n_quad)
        geneval = GeneralizedItermEvaluator(P_ell, P_ellbar, Q, theta, R, ell, ellbar, n_quad)

        h = hybrid.eval_all()
        g = geneval.eval_all()

        print(f"\n--- ({ell},{ellbar}) ---")
        print(f"          Hybrid      GenEval      Diff")
        print(f"  I1:     {h.I1:+.6f}   {g.I1:+.6f}   {h.I1 - g.I1:+.6f}")
        print(f"  I2:     {h.I2:+.6f}   {g.I2:+.6f}   {h.I2 - g.I2:+.6f}")
        print(f"  I3:     {h.I3:+.6f}   {g.I3:+.6f}   {h.I3 - g.I3:+.6f}")
        print(f"  I4:     {h.I4:+.6f}   {g.I4:+.6f}   {h.I4 - g.I4:+.6f}")
        print(f"  Total:  {h.total:+.6f}   {g.total:+.6f}   {h.total - g.total:+.6f}")


if __name__ == "__main__":
    compare_with_geneval()
