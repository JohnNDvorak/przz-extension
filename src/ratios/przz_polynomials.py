"""
src/ratios/przz_polynomials.py
Phase 14D Task D1: Clean polynomial loader for ratios pipeline

PURPOSE:
========
Provide a single "source of truth" for PRZZ polynomials in the ratios pipeline.
This module wraps the existing load_przz_polynomials() functions and provides
callable polynomial objects suitable for numerical integration.

DESIGN:
======
- Does NOT duplicate polynomial coefficients (imports from src/polynomials.py)
- Returns a dataclass with P1, P2, P3 as polynomial objects
- Polynomial objects have .eval(u) method for scalar evaluation
- Includes benchmark-specific R value for convenience

USAGE:
=====
    polys = load_przz_k3_polynomials("kappa")
    # polys.P1, polys.P2, polys.P3 are polynomial objects
    # polys.R is the benchmark R value (1.3036 for kappa, 1.1167 for kappa_star)

    # For integration:
    value = polys.P1.eval(np.array([0.5]))[0]  # P1(0.5)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    P1Polynomial,
    PellPolynomial,
    QPolynomial,
)


# Benchmark R values from PRZZ paper
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167


@dataclass(frozen=True)
class PrzzK3Polynomials:
    """
    PRZZ K=3 polynomials for the ratios pipeline.

    This is the single source of truth for polynomial access in src/ratios/*.

    Attributes:
        P1: P1 polynomial (constrained: P1(0)=0, P1(1)=1)
        P2: P2 polynomial (constrained: P2(0)=0)
        P3: P3 polynomial (constrained: P3(0)=0)
        Q: Q polynomial (constrained: Q(0)=1)
        benchmark: "kappa" or "kappa_star"
        R: Benchmark R value
        theta: θ parameter (4/7)
    """
    P1: P1Polynomial
    P2: PellPolynomial
    P3: PellPolynomial
    Q: QPolynomial
    benchmark: str
    R: float
    theta: float = 4.0 / 7.0


def load_przz_k3_polynomials(
    benchmark: Literal["kappa", "kappa_star"] = "kappa",
    enforce_Q0: bool = False
) -> PrzzK3Polynomials:
    """
    Load PRZZ K=3 polynomials for a given benchmark.

    This is the canonical entry point for the ratios pipeline.

    Args:
        benchmark: "kappa" for main benchmark (R=1.3036) or
                   "kappa_star" for simple zeros benchmark (R=1.1167)
        enforce_Q0: If True, enforce Q(0)=1 exactly (for optimization).
                   If False (default), use paper-literal Q(0)≈0.999999.

    Returns:
        PrzzK3Polynomials dataclass with P1, P2, P3, Q and metadata

    Raises:
        ValueError: If benchmark is not "kappa" or "kappa_star"
    """
    if benchmark == "kappa":
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=enforce_Q0)
        R = KAPPA_R
    elif benchmark == "kappa_star":
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=enforce_Q0)
        R = KAPPA_STAR_R
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'kappa' or 'kappa_star'.")

    return PrzzK3Polynomials(
        P1=P1,
        P2=P2,
        P3=P3,
        Q=Q,
        benchmark=benchmark,
        R=R,
    )


def make_poly_callable(poly: Any) -> callable:
    """
    Convert a polynomial object to a simple callable f(u) -> float.

    This is a convenience wrapper for integration code that expects
    a simple function signature.

    Args:
        poly: A polynomial object with .eval(x) method

    Returns:
        A callable that takes a float and returns a float
    """
    def f(u: float) -> float:
        return float(poly.eval(np.array([u]))[0])
    return f


def get_polynomial_functions(polys: PrzzK3Polynomials) -> dict:
    """
    Extract polynomial callables from a PrzzK3Polynomials dataclass.

    Returns:
        dict with keys 'P1_func', 'P2_func', 'P3_func', 'Q_func'
        each mapping to a callable f(u) -> float
    """
    return {
        'P1_func': make_poly_callable(polys.P1),
        'P2_func': make_poly_callable(polys.P2),
        'P3_func': make_poly_callable(polys.P3),
        'Q_func': make_poly_callable(polys.Q),
    }


# =============================================================================
# Validation helpers
# =============================================================================

def validate_constraints(polys: PrzzK3Polynomials, tol: float = 1e-10) -> dict:
    """
    Validate that polynomial constraints are satisfied.

    Args:
        polys: PrzzK3Polynomials to validate
        tol: Tolerance for constraint checks

    Returns:
        dict with constraint check results
    """
    results = {}

    # P1(0) = 0
    p1_at_0 = float(polys.P1.eval(np.array([0.0]))[0])
    results['P1(0)=0'] = abs(p1_at_0) < tol
    results['P1(0)_value'] = p1_at_0

    # P1(1) = 1
    p1_at_1 = float(polys.P1.eval(np.array([1.0]))[0])
    results['P1(1)=1'] = abs(p1_at_1 - 1.0) < tol
    results['P1(1)_value'] = p1_at_1

    # P2(0) = 0
    p2_at_0 = float(polys.P2.eval(np.array([0.0]))[0])
    results['P2(0)=0'] = abs(p2_at_0) < tol
    results['P2(0)_value'] = p2_at_0

    # P3(0) = 0
    p3_at_0 = float(polys.P3.eval(np.array([0.0]))[0])
    results['P3(0)=0'] = abs(p3_at_0) < tol
    results['P3(0)_value'] = p3_at_0

    # Q(0) = 1 (approximately, paper has Q(0) ≈ 0.999999)
    q_at_0 = float(polys.Q.eval(np.array([0.0]))[0])
    results['Q(0)≈1'] = abs(q_at_0 - 1.0) < 1e-5  # Looser tolerance for Q
    results['Q(0)_value'] = q_at_0

    # All constraints pass
    results['all_pass'] = all(
        results[k] for k in ['P1(0)=0', 'P1(1)=1', 'P2(0)=0', 'P3(0)=0', 'Q(0)≈1']
    )

    return results


if __name__ == "__main__":
    # Quick self-test
    print("Loading kappa polynomials...")
    polys_kappa = load_przz_k3_polynomials("kappa")
    print(f"  Benchmark: {polys_kappa.benchmark}, R={polys_kappa.R}")
    constraints = validate_constraints(polys_kappa)
    print(f"  Constraints: {constraints}")

    print()
    print("Loading kappa_star polynomials...")
    polys_kstar = load_przz_k3_polynomials("kappa_star")
    print(f"  Benchmark: {polys_kstar.benchmark}, R={polys_kstar.R}")
    constraints = validate_constraints(polys_kstar)
    print(f"  Constraints: {constraints}")

    print()
    print("Testing callable conversion...")
    funcs = get_polynomial_functions(polys_kappa)
    print(f"  P1(0.5) = {funcs['P1_func'](0.5):.6f}")
    print(f"  P2(0.5) = {funcs['P2_func'](0.5):.6f}")
    print(f"  P3(0.5) = {funcs['P3_func'](0.5):.6f}")
