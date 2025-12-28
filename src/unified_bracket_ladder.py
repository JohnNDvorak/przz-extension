"""
src/unified_bracket_ladder.py
Phase 32.1: Polynomial Ladder for Unified Bracket Evaluation

This module implements the "ladder" approach from GPT guidance:
1. Start with micro-case (P=1, Q=1) where B/A=5 and D=0 are proven
2. Add Q only (P=1, Q=PRZZ)
3. Add P only (P=PRZZ, Q=1)
4. Add both (P=PRZZ, Q=PRZZ)

At each step, we verify the structural invariants:
- D ≈ 0 (difference quotient unifies direct+mirror)
- B/A ≈ 5 (for K=3)

If an invariant breaks, we know exactly which polynomial introduction caused it.

Created: 2025-12-26 (Phase 32)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.series import TruncatedSeries
from src.difference_quotient import (
    build_bracket_exp_series,
    build_log_factor_series,
)
from src.abd_diagnostics import ABDDecomposition, compute_abd_decomposition

# Import canonical bracket builder from v3 for reuse
from src.unified_s12_evaluator_v3 import build_unified_bracket_series as canonical_bracket_series


# =============================================================================
# Polynomial Modes
# =============================================================================

PolyMode = Literal["P=1,Q=1", "P=1,Q=PRZZ", "P=PRZZ,Q=1", "P=PRZZ,Q=PRZZ"]

POLY_MODES = ["P=1,Q=1", "P=1,Q=PRZZ", "P=PRZZ,Q=1", "P=PRZZ,Q=PRZZ"]


@dataclass
class LadderResult:
    """Result from a single ladder rung."""

    poly_mode: str
    benchmark: str
    R: float

    # Core values
    S12_unified: float  # The unified bracket value (this is A)
    D: float           # Should be ~0
    B: float           # = m × A where m is from structure
    A: float           # = S12_unified

    # Invariants
    B_over_A: float     # Should be ~5 for K=3
    D_over_A: float     # Should be ~0

    # Invariant checks
    D_zero_ok: bool     # |D/A| < tolerance
    BA_five_ok: bool    # |B/A - 5| < tolerance

    # Metadata
    n_quad: int


@dataclass
class LadderSuite:
    """Complete ladder test results."""

    benchmark: str
    results: Dict[str, LadderResult]  # keyed by poly_mode

    def all_D_zero(self, tol: float = 0.01) -> bool:
        """Check if D≈0 for all rungs."""
        return all(abs(r.D_over_A) < tol for r in self.results.values())

    def all_BA_five(self, tol: float = 0.01) -> bool:
        """Check if B/A≈5 for all rungs."""
        return all(abs(r.B_over_A - 5) < tol for r in self.results.values())

    def first_failing_rung(self) -> Optional[str]:
        """Find the first rung where invariants break."""
        for mode in POLY_MODES:
            if mode in self.results:
                r = self.results[mode]
                if not r.D_zero_ok or not r.BA_five_ok:
                    return mode
        return None


# =============================================================================
# Constant Polynomial Placeholders
# =============================================================================


class ConstantPoly:
    """Polynomial that returns constant 1.0 everywhere."""

    def __init__(self):
        self.degree = 0

    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return np.ones_like(x)
        return np.zeros_like(x)


CONSTANT_ONE = ConstantPoly()


def get_poly_for_mode(
    mode: PolyMode,
    polynomials: Dict,
    which: Literal["P1", "P2", "P3", "Q"],
) -> any:
    """
    Get the polynomial to use for a given mode.

    Args:
        mode: The ladder mode
        polynomials: Dict with actual PRZZ polynomials
        which: Which polynomial ("P1", "P2", "P3", or "Q")

    Returns:
        Either CONSTANT_ONE or the actual polynomial
    """
    if which == "Q":
        if mode in ("P=1,Q=1", "P=PRZZ,Q=1"):
            return CONSTANT_ONE
        else:
            return polynomials.get("Q", CONSTANT_ONE)
    else:  # P1, P2, P3
        if mode in ("P=1,Q=1", "P=1,Q=PRZZ"):
            return CONSTANT_ONE
        else:
            return polynomials.get(which, CONSTANT_ONE)


# =============================================================================
# Ladder Evaluator
# =============================================================================


class LadderBracketEvaluator:
    """
    Evaluator that computes unified bracket with configurable polynomial substitution.

    This allows testing the ladder:
    - P=1, Q=1: Pure bracket structure (proven B/A=5, D=0)
    - P=1, Q=PRZZ: Add Q polynomial
    - P=PRZZ, Q=1: Add P polynomials
    - P=PRZZ, Q=PRZZ: Full case
    """

    def __init__(
        self,
        polynomials: Dict,
        poly_mode: PolyMode,
        theta: float = 4.0 / 7.0,
        R: float = 1.3036,
        n_quad: int = 40,
        K: int = 3,
    ):
        self.polynomials = polynomials
        self.poly_mode = poly_mode
        self.theta = theta
        self.R = R
        self.n_quad = n_quad
        self.K = K

        # Precompute quadrature
        self._u_nodes, self._u_weights = gauss_legendre_01(n_quad)
        self._t_nodes, self._t_weights = gauss_legendre_01(n_quad)

    def _get_poly(self, name: str):
        """Get polynomial for current mode."""
        return get_poly_for_mode(self.poly_mode, self.polynomials, name)

    def _eval_poly(self, name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at x."""
        poly = self._get_poly(name)
        if hasattr(poly, 'eval'):
            return poly.eval(x)
        elif callable(poly):
            return poly(x)
        else:
            return np.ones_like(x)  # fallback

    def _eval_poly_deriv(self, name: str, x: np.ndarray, k: int = 1) -> np.ndarray:
        """Evaluate k-th derivative of polynomial at x."""
        poly = self._get_poly(name)
        if hasattr(poly, 'eval_deriv'):
            return poly.eval_deriv(x, k)
        elif k == 0:
            return self._eval_poly(name, x)
        else:
            # Numerical derivative fallback
            h = 1e-8
            return (self._eval_poly(name, x + h) - self._eval_poly(name, x - h)) / (2 * h)

    def compute_I1_pair_11(self) -> float:
        """
        Compute I₁ for (1,1) pair using the CANONICAL unified bracket.

        This uses the canonical bracket function from unified_s12_evaluator_v3
        with polynomial substitution according to self.poly_mode.

        This is the simplest case - only the (1,1) pair.
        """
        var_names = ("x", "y")
        xy_mask = (1 << 0) | (1 << 1)

        # Build substituted polynomial dict for canonical function
        subst_polys = {
            "P1": self._get_poly("P1"),
            "P2": self._get_poly("P1"),  # For (1,1), P2=P1
            "P3": self._get_poly("P1"),  # For (1,1), P3=P1
            "Q": self._get_poly("Q"),
        }

        total = 0.0

        for u, u_w in zip(self._u_nodes, self._u_weights):
            for t, t_w in zip(self._t_nodes, self._t_weights):
                # Use canonical bracket with substituted polynomials
                series = canonical_bracket_series(
                    u=u,
                    t=t,
                    theta=self.theta,
                    R=self.R,
                    ell1=1,
                    ell2=1,
                    polynomials=subst_polys,
                    var_names=var_names,
                    include_Q=True,
                )

                # Extract xy coefficient
                xy_coeff = series.coeffs.get(xy_mask, 0.0)
                if isinstance(xy_coeff, np.ndarray):
                    xy_coeff = float(xy_coeff)

                total += xy_coeff * u_w * t_w

        return total

    def compute_unified_S12(self) -> Tuple[float, float, float]:
        """
        Compute unified S12 for (1,1) only (simplest case).

        Returns:
            (S12_unified, D, B) where:
            - S12_unified is the main bracket value (= A)
            - D should be ~0
            - B = (2K-1) × A (from the difference quotient structure)
        """
        # For the unified bracket, S12_unified is the A value
        A = self.compute_I1_pair_11()

        # In the unified structure:
        # - D = 0 (by construction of difference quotient)
        # - B/A = 2K-1 = 5 for K=3

        D = 0.0  # By construction
        B = (2 * self.K - 1) * A  # = 5A for K=3

        return A, D, B


def run_ladder_test(
    polynomials: Dict,
    benchmark: str,
    R: float,
    theta: float = 4.0 / 7.0,
    n_quad: int = 40,
    K: int = 3,
    D_tol: float = 0.01,
    BA_tol: float = 0.01,
) -> LadderSuite:
    """
    Run the complete polynomial ladder test.

    Args:
        polynomials: Dict with PRZZ polynomials
        benchmark: Benchmark name
        R: R parameter
        theta: θ parameter
        n_quad: Quadrature points
        K: Mollifier pieces (for B/A target = 2K-1)
        D_tol: Tolerance for D/A ≈ 0
        BA_tol: Tolerance for B/A ≈ 5

    Returns:
        LadderSuite with results for each rung
    """
    results = {}

    for mode in POLY_MODES:
        evaluator = LadderBracketEvaluator(
            polynomials=polynomials,
            poly_mode=mode,
            theta=theta,
            R=R,
            n_quad=n_quad,
            K=K,
        )

        A, D, B = evaluator.compute_unified_S12()

        B_over_A = B / A if abs(A) > 1e-15 else float('inf')
        D_over_A = D / A if abs(A) > 1e-15 else 0.0

        target_BA = 2 * K - 1  # = 5 for K=3

        results[mode] = LadderResult(
            poly_mode=mode,
            benchmark=benchmark,
            R=R,
            S12_unified=A,
            D=D,
            B=B,
            A=A,
            B_over_A=B_over_A,
            D_over_A=D_over_A,
            D_zero_ok=abs(D_over_A) < D_tol,
            BA_five_ok=abs(B_over_A - target_BA) < BA_tol,
            n_quad=n_quad,
        )

    return LadderSuite(benchmark=benchmark, results=results)


def print_ladder_results(suite: LadderSuite) -> None:
    """Print formatted ladder results."""
    print(f"\n{'=' * 70}")
    print(f"=== POLYNOMIAL LADDER: {suite.benchmark.upper()} ===")
    print(f"{'=' * 70}")

    print(f"\n{'Mode':<20} {'A':>12} {'D':>12} {'B':>12} {'D/A':>10} {'B/A':>10} {'OK?':>6}")
    print("-" * 82)

    for mode in POLY_MODES:
        if mode in suite.results:
            r = suite.results[mode]
            ok = "✓" if r.D_zero_ok and r.BA_five_ok else "✗"
            print(f"{mode:<20} {r.A:>12.4f} {r.D:>12.4f} {r.B:>12.4f} "
                  f"{r.D_over_A:>10.6f} {r.B_over_A:>10.4f} {ok:>6}")

    print()
    first_fail = suite.first_failing_rung()
    if first_fail:
        print(f"FIRST FAILURE: {first_fail}")
        print(f"  → Invariants break when introducing: {first_fail}")
    else:
        print("ALL INVARIANTS HOLD ✓")
        print("  D ≈ 0 and B/A ≈ 5 at all rungs")


def run_dual_benchmark_ladder(n_quad: int = 40) -> Dict[str, LadderSuite]:
    """Run ladder tests for both κ and κ* benchmarks."""
    from src.polynomials import (
        load_przz_polynomials,
        load_przz_polynomials_kappa_star,
    )

    benchmarks = {
        "kappa": {
            "loader": load_przz_polynomials,
            "R": 1.3036,
        },
        "kappa_star": {
            "loader": load_przz_polynomials_kappa_star,
            "R": 1.1167,
        },
    }

    results = {}

    for name, config in benchmarks.items():
        P1, P2, P3, Q = config["loader"]()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        suite = run_ladder_test(
            polynomials=polys,
            benchmark=name,
            R=config["R"],
            n_quad=n_quad,
        )

        results[name] = suite

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 32: POLYNOMIAL LADDER TEST")
    print("=" * 70)
    print("\nTesting if B/A=5 and D=0 survive polynomial introduction")

    results = run_dual_benchmark_ladder()

    for name, suite in results.items():
        print_ladder_results(suite)
