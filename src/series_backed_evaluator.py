"""
src/series_backed_evaluator.py
Series-Backed Evaluator: All I-terms via Series Coefficient Extraction

This evaluator eliminates all manual derivative algebra by computing
ALL I-terms (I1, I2, I3, I4) via the validated series engine.

Purpose (Phase A of Session 12 plan):
- Create a "derivative-clean" baseline with no hand-derived formulas
- Verify the series engine works for all coefficient extractions
- Establish that the ratio error is NOT in derivative computation

Key insight from Session 11:
- The coefficient gate PASSED: series f[1,1] matches finite difference within 1e-7
- But the two-benchmark gate FAILS: 80% ratio error persists
- Conclusion: the issue is NOT in derivative computation, but in structural assembly

I-term mapping to series coefficients:
- I1 = g[1,1] with weight (1-u)^{ell+ellbar}  (mixed derivative)
- I2 = g[0,0] with weight 0                    (no derivative, just prefactor)
- I3 = -g[1,0] with weight (1-u)^{ell}        (x-derivative only)
- I4 = -g[0,1] with weight (1-u)^{ellbar}     (y-derivative only)

The negative signs on I3/I4 come from the derivative expansion:
- I3 = -d/dx[prefactor * F] at x=0
- I4 = -d/dy[prefactor * F] at y=0
"""

from __future__ import annotations
from typing import Tuple, Dict, NamedTuple

from src.psi_series_evaluator import PsiSeriesEvaluator


class SeriesBackedResult(NamedTuple):
    """Result of series-backed I-term computation."""
    I1: float
    I2: float
    I3: float
    I4: float
    total: float
    ell: int
    ellbar: int


class SeriesBackedEvaluator:
    """
    All I-terms computed via series coefficient extraction.
    No hand-derived derivative algebra remains.

    This evaluator uses PsiSeriesEvaluator.compute_integral_grid() to get
    all g[a,b,weight] integrals, then maps them to I-terms:

    - I1 (mixed): g[1,1] with weight (1-u)^{ell+ellbar}
    - I2 (base):  g[0,0] with weight 0
    - I3 (x-deriv): -g[1,0] with weight (1-u)^{ell}
    - I4 (y-deriv): -g[0,1] with weight (1-u)^{ellbar}
    """

    def __init__(self, P_ell, P_ellbar, Q, R: float, theta: float,
                 ell: int, ellbar: int, n_quad: int = 60):
        """
        Initialize the series-backed evaluator.

        Args:
            P_ell: Left polynomial (piece ell)
            P_ellbar: Right polynomial (piece ellbar)
            Q: Q polynomial
            R: PRZZ R parameter
            theta: theta parameter (typically 4/7)
            ell: Left piece index
            ellbar: Right piece index
            n_quad: Number of quadrature points
        """
        self.P_ell = P_ell
        self.P_ellbar = P_ellbar
        self.Q = Q
        self.R = R
        self.theta = theta
        self.ell = ell
        self.ellbar = ellbar
        self.n_quad = n_quad

        # Create series evaluator with sufficient max_order
        # Need derivatives up to (1,1) for I1, so max_order=1 suffices
        # But use max(ell, ellbar) to be safe for future extensions
        max_order = max(ell, ellbar)
        self.series_eval = PsiSeriesEvaluator(
            P_ell, P_ellbar, Q, R, theta,
            max_order=max_order,
            n_quad=n_quad
        )

        # Cache integral_grid (computed lazily)
        self._integral_grid = None

    @property
    def integral_grid(self) -> Dict[Tuple[int, int, int], float]:
        """Lazily compute and cache the integral grid."""
        if self._integral_grid is None:
            self._integral_grid = self.series_eval.compute_integral_grid()
        return self._integral_grid

    def eval_I1(self) -> float:
        """
        I1 = integral of g[1,1] with weight (1-u)^{ell+ellbar}.

        This is the mixed derivative term, verified by coefficient gate.
        """
        weight_exp = self.ell + self.ellbar
        return self.integral_grid.get((1, 1, weight_exp), 0.0)

    def eval_I2(self) -> float:
        """
        I2 = integral of g[0,0] with weight (1-u)^0 = 1.

        This is the base term with no derivatives.
        Note: g[0,0] = f[0,0]/theta (from prefactor transform).
        """
        return self.integral_grid.get((0, 0, 0), 0.0)

    def eval_I3(self) -> float:
        """
        I3 = -integral of g[1,0] with weight (1-u)^{ell}.

        This is the x-derivative term.
        The negative sign comes from -d/dx[prefactor * F].
        """
        weight_exp = self.ell
        return -self.integral_grid.get((1, 0, weight_exp), 0.0)

    def eval_I4(self) -> float:
        """
        I4 = -integral of g[0,1] with weight (1-u)^{ellbar}.

        This is the y-derivative term.
        The negative sign comes from -d/dy[prefactor * F].
        """
        weight_exp = self.ellbar
        return -self.integral_grid.get((0, 1, weight_exp), 0.0)

    def eval_all(self) -> SeriesBackedResult:
        """Compute all I-terms and return result."""
        I1 = self.eval_I1()
        I2 = self.eval_I2()
        I3 = self.eval_I3()
        I4 = self.eval_I4()
        total = I1 + I2 + I3 + I4

        return SeriesBackedResult(
            I1=I1, I2=I2, I3=I3, I4=I4, total=total,
            ell=self.ell, ellbar=self.ellbar
        )


def compute_c_series_backed(P1, P2, P3, Q, R: float, theta: float = 4/7,
                             n_quad: int = 60, verbose: bool = False) -> float:
    """
    Compute total c using SeriesBackedEvaluator.

    Returns sum over all pairs (ell, ellbar) with symmetry factors.
    """
    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    total = 0.0
    if verbose:
        print("\n" + "=" * 60)
        print("SERIES-BACKED EVALUATOR: c COMPUTATION")
        print("=" * 60)

    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        evaluator = SeriesBackedEvaluator(
            P_ell, P_ellbar, Q, R, theta,
            ell, ellbar, n_quad
        )
        result = evaluator.eval_all()

        sym = 1 if ell == ellbar else 2
        pair_contrib = sym * result.total
        total += pair_contrib

        if verbose:
            print(f"\n({ell},{ellbar}) x {sym}:")
            print(f"  I1 (mixed):   {result.I1:+.6f}")
            print(f"  I2 (base):    {result.I2:+.6f}")
            print(f"  I3 (x-deriv): {result.I3:+.6f}")
            print(f"  I4 (y-deriv): {result.I4:+.6f}")
            print(f"  Total: {result.total:.6f} x {sym} = {pair_contrib:.6f}")

    if verbose:
        print("\n" + "-" * 60)
        print(f"TOTAL c = {total:.6f}")

    return total


# =============================================================================
# COMPARISON WITH HYBRID EVALUATOR
# =============================================================================

def compare_with_hybrid():
    """Compare series-backed evaluator against hybrid evaluator."""
    from src.polynomials import load_przz_polynomials
    from src.hybrid_evaluator import HybridEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("SERIES-BACKED VS HYBRID COMPARISON")
    print("=" * 70)

    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        # Note: HybridEvaluator uses (P_left, P_right, Q, theta, R, ...)
        # SeriesBackedEvaluator uses (P_ell, P_ellbar, Q, R, theta, ...)
        series = SeriesBackedEvaluator(P_ell, P_ellbar, Q, R, theta, ell, ellbar, n_quad)
        hybrid = HybridEvaluator(P_ell, P_ellbar, Q, theta, R, ell, ellbar, n_quad)

        s = series.eval_all()
        h = hybrid.eval_all()

        print(f"\n--- ({ell},{ellbar}) ---")
        print(f"          Series      Hybrid       Diff")
        print(f"  I1:     {s.I1:+.6f}   {h.I1:+.6f}   {s.I1 - h.I1:+.6f}")
        print(f"  I2:     {s.I2:+.6f}   {h.I2:+.6f}   {s.I2 - h.I2:+.6f}")
        print(f"  I3:     {s.I3:+.6f}   {h.I3:+.6f}   {s.I3 - h.I3:+.6f}")
        print(f"  I4:     {s.I4:+.6f}   {h.I4:+.6f}   {s.I4 - h.I4:+.6f}")
        print(f"  Total:  {s.total:+.6f}   {h.total:+.6f}   {s.total - h.total:+.6f}")


if __name__ == "__main__":
    compare_with_hybrid()
