"""src/psi_22_complete_oracle.py

Complete Ψ-based oracle for the (2,2) pair (12 monomials).

This module exists to provide a stable, testable implementation for the
(2,2) contribution that:
- enumerates the 12 monomials in the one-C expansion (A,B,C,D)
- evaluates each monomial via a shared coefficient-extraction backend
- produces a total in the same ballpark as the I-term oracle.

Implementation strategy
-----------------------

We use `src.psi_series_evaluator.PsiSeriesEvaluator` as a coefficient engine.
It computes the bivariate Taylor coefficients of

    G(x,y) = (1/θ + x + y) · F(x,y)

and returns integrals

    I[a,b,w] = ∬ [x^a y^b]G(x,y) · (1-u)^w du dt.

For Ψ monomials, the current codebase uses the Euler–Maclaurin weight rule

    weight exponent w = a + b,

so each monomial (a,b,c,d) is mapped to the single integral key (a, b, a+b).
In this approximation/model, the C and D powers only affect the Ψ coefficient;
C and D do not select a different kernel.

This mapping is consistent with `PsiSeriesEvaluator.eval_pair(2,2)` and yields
numerically stable values (no P'/P poles, no ad-hoc scaling factors).

Note
----
This is not claiming the full PRZZ TeX Section 7 machinery is complete.
It is the internally-consistent Ψ-series model currently used in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from src.psi_monomial_expansion import expand_pair_to_monomials
from src.psi_series_evaluator import PsiSeriesEvaluator


@dataclass
class MonomialValue:
    """Result of evaluating a single monomial."""

    a: int
    b: int
    c: int
    d: int
    coefficient: int
    raw_integral: float
    contribution: float  # coefficient * raw_integral

    def monomial_str(self) -> str:
        """Return string representation like 'C²D⁰A¹B¹'."""
        return f"C{self.c}D{self.d}A{self.a}B{self.b}"


class Psi22CompleteOracle:
    """Complete (2,2) Ψ oracle backed by the series coefficient engine."""

    def __init__(self, P2, Q, theta: float, R: float, n_quad: int = 60):
        self.P2 = P2
        self.Q = Q
        self.theta = float(theta)
        self.R = float(R)
        self.n_quad = int(n_quad)

        # (2,2) needs coefficients up to a,b <= 2
        self._series_eval = PsiSeriesEvaluator(
            P2,
            P2,
            Q,
            R=self.R,
            theta=self.theta,
            max_order=2,
            n_quad=self.n_quad,
        )
        self._integral_grid: Optional[Dict[Tuple[int, int, int], float]] = None

    @property
    def integral_grid(self) -> Dict[Tuple[int, int, int], float]:
        """Lazy cache for the integral grid I[a,b,w]."""
        if self._integral_grid is None:
            self._integral_grid = self._series_eval.compute_integral_grid()
        return self._integral_grid

    def _eval_base_integral(self) -> float:
        """Base integral (no derivatives, weight exponent 0)."""
        return float(self.integral_grid[(0, 0, 0)])

    def eval_monomial(self, a: int, b: int, c: int, d: int) -> float:
        """Evaluate a single (2,2) monomial A^a B^b C^c D^d.

        Current model: integral depends only on (a,b,weight=a+b).
        """
        key = (int(a), int(b), int(a) + int(b))
        return float(self.integral_grid.get(key, 0.0))

    def compute_all_monomials(
        self, verbose: bool = True
    ) -> Tuple[float, Dict[Tuple[int, int, int, int], MonomialValue]]:
        """Compute all 12 monomials for Ψ_{2,2}.

        Returns:
            total: Sum of all weighted monomial contributions
            results: Dict mapping (a,b,c,d) to MonomialValue
        """
        monomials = expand_pair_to_monomials(2, 2)

        if verbose:
            print("=" * 70)
            print("Ψ_{2,2} Complete Oracle: 12 Monomials")
            print("=" * 70)

        results: Dict[Tuple[int, int, int, int], MonomialValue] = {}
        total = 0.0

        for (a, b, c, d), coeff in sorted(monomials.items()):
            raw = self.eval_monomial(a, b, c, d)
            contrib = coeff * raw
            total += contrib

            mv = MonomialValue(
                a=a,
                b=b,
                c=c,
                d=d,
                coefficient=coeff,
                raw_integral=raw,
                contribution=contrib,
            )
            results[(a, b, c, d)] = mv

            if verbose:
                print(
                    f"  {coeff:+3d} × {mv.monomial_str():<12} = "
                    f"{coeff:+3d} × {raw:10.6f} = {contrib:+10.6f}"
                )

        if verbose:
            print(f"\n  Total Ψ_{{2,2}} = {total:.12f}")

        return total, results
