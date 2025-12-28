"""
src/evaluator/s12_backend.py
Phase 27: Backend Abstraction for S12 (I₁ + I₂) Computation

This module provides a unified interface for computing I₁ and I₂ integrals,
supporting multiple backend implementations:

1. "unified_general" (default): Phase 26B bivariate series with x^ℓ₁y^ℓ₂ extraction
2. "dsl": Original Term DSL evaluation via terms_k3_d1.py

Phase 26B proved that unified_general matches DSL to ~1e-13 relative error for all pairs.
The backend abstraction allows:
- Production code to use unified_general (cleaner, faster)
- Validation code to compare against DSL
- Future backends (e.g., derived mirror) to plug in seamlessly

Created: 2025-12-26 (Phase 27)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, List
import math

# Type alias for backend selection
BackendType = Literal["unified_general", "dsl"]

# Pair keys for K=3
TRIANGLE_PAIRS = ["11", "22", "33", "12", "13", "23"]


@dataclass
class I1Result:
    """Result from I₁ backend computation."""
    ell1: int
    ell2: int
    value: float
    backend: str
    n_quad: int


@dataclass
class I2Result:
    """Result from I₂ backend computation."""
    ell1: int
    ell2: int
    value: float
    backend: str
    n_quad: int


@dataclass
class S12PairResult:
    """Result for a single (ℓ₁, ℓ₂) pair's S12 = I₁ + I₂ contribution."""
    ell1: int
    ell2: int
    I1_value: float
    I2_value: float
    S12_value: float
    backend: str


@dataclass
class S12AllPairsResult:
    """Result for all 6 triangle pairs."""
    pair_results: Dict[str, S12PairResult]
    total_raw: float  # Sum before factorial normalization
    total_normalized: float  # Sum after factorial and symmetry normalization
    backend: str
    n_quad: int


# =============================================================================
# FACTORIAL NORMALIZATION (S12)
# =============================================================================

def get_s12_factorial_normalization() -> Dict[str, float]:
    """
    Return factorial normalization factors for S12 pairs.

    Factor for pair (ℓ₁, ℓ₂) is 1/(ℓ₁! × ℓ₂!)
    """
    return {
        "11": 1.0,       # 1/1!/1! = 1
        "22": 0.25,      # 1/2!/2! = 1/4
        "33": 1/36,      # 1/3!/3! = 1/36
        "12": 0.5,       # 1/1!/2! = 1/2
        "13": 1/6,       # 1/1!/3! = 1/6
        "23": 1/12,      # 1/2!/3! = 1/12
    }


def get_s12_symmetry_factors() -> Dict[str, float]:
    """
    Return symmetry factors for S12 pairs (triangle×2 convention).

    Off-diagonal pairs get factor of 2.
    """
    return {
        "11": 1.0,
        "22": 1.0,
        "33": 1.0,
        "12": 2.0,
        "13": 2.0,
        "23": 2.0,
    }


# =============================================================================
# UNIFIED GENERAL BACKEND (Phase 26B)
# =============================================================================

def _compute_I1_unified_general(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I₁ using Phase 26B unified general evaluator."""
    from src.unified_i1_general import compute_I1_unified_general

    result = compute_I1_unified_general(
        R=R,
        theta=theta,
        ell1=ell1,
        ell2=ell2,
        polynomials=polynomials,
        n_quad_u=n_quad,
        n_quad_t=n_quad,
        include_Q=True,
        apply_factorial_norm=True,
    )
    return result.I1_value


def _compute_I2_unified_general(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I₂ using Phase 26B unified general evaluator."""
    from src.unified_i2_general import compute_I2_unified_general

    result = compute_I2_unified_general(
        R=R,
        theta=theta,
        ell1=ell1,
        ell2=ell2,
        polynomials=polynomials,
        n_quad_u=n_quad,
        n_quad_t=n_quad,
        include_Q=True,
    )
    return result.I2_value


# =============================================================================
# DSL BACKEND (Original Term DSL)
# =============================================================================

def _compute_I1_dsl(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I₁ using OLD DSL (multi-variable structure).

    The OLD DSL uses ℓ₁+ℓ₂ variables (x1,...,x_ℓ₁, y1,...,y_ℓ₂) and extracts
    d^{ℓ₁+ℓ₂}/dx₁...dy_ℓ₂. This matches Phase 26B's unified_general which
    extracts ℓ₁!ℓ₂! × [x^ℓ₁ y^ℓ₂].

    Note: This is DIFFERENT from the V2 DSL which uses just (x,y) and extracts d²/dxdy.
    """
    from src.terms_k3_d1 import (
        make_I1_11, make_I1_22, make_I1_33,
        make_I1_12, make_I1_13, make_I1_23
    )
    from src.evaluate import evaluate_term

    pair_key = f"{ell1}{ell2}"

    # Use OLD DSL term makers (multi-variable structure matching Phase 26B)
    term_makers = {
        "11": lambda: make_I1_11(theta, R, kernel_regime="paper"),
        "22": lambda: make_I1_22(theta, R, kernel_regime="paper"),
        "33": lambda: make_I1_33(theta, R, kernel_regime="paper"),
        "12": lambda: make_I1_12(theta, R, kernel_regime="paper"),
        "13": lambda: make_I1_13(theta, R, kernel_regime="paper"),
        "23": lambda: make_I1_23(theta, R, kernel_regime="paper"),
    }

    if pair_key not in term_makers:
        raise ValueError(f"Unknown pair: {pair_key}")

    term = term_makers[pair_key]()
    result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
    return result.value


def _compute_I2_dsl(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I₂ using OLD DSL.

    I₂ has no derivatives, so the variable structure issue doesn't affect it.
    However, we use the OLD DSL functions for consistency.
    """
    from src.terms_k3_d1 import (
        make_I2_11, make_I2_22, make_I2_33,
        make_I2_12, make_I2_13, make_I2_23
    )
    from src.evaluate import evaluate_term

    pair_key = f"{ell1}{ell2}"

    # Use OLD DSL term makers
    term_makers = {
        "11": lambda: make_I2_11(theta, R, kernel_regime="paper"),
        "22": lambda: make_I2_22(theta, R, kernel_regime="paper"),
        "33": lambda: make_I2_33(theta, R, kernel_regime="paper"),
        "12": lambda: make_I2_12(theta, R, kernel_regime="paper"),
        "13": lambda: make_I2_13(theta, R, kernel_regime="paper"),
        "23": lambda: make_I2_23(theta, R, kernel_regime="paper"),
    }

    if pair_key not in term_makers:
        raise ValueError(f"Unknown pair: {pair_key}")

    term = term_makers[pair_key]()
    result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
    return result.value


# =============================================================================
# BACKEND DISPATCH
# =============================================================================

def compute_I1_backend(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    backend: BackendType = "unified_general",
    n_quad: int = 60,
) -> I1Result:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) using specified backend.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (typically 4/7)
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        backend: "unified_general" or "dsl"
        n_quad: Number of quadrature points

    Returns:
        I1Result with computed value and metadata
    """
    if backend == "unified_general":
        value = _compute_I1_unified_general(R, theta, ell1, ell2, polynomials, n_quad)
    elif backend == "dsl":
        value = _compute_I1_dsl(R, theta, ell1, ell2, polynomials, n_quad)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return I1Result(ell1=ell1, ell2=ell2, value=value, backend=backend, n_quad=n_quad)


def compute_I2_backend(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    backend: BackendType = "unified_general",
    n_quad: int = 60,
) -> I2Result:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂) using specified backend.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (typically 4/7)
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        backend: "unified_general" or "dsl"
        n_quad: Number of quadrature points

    Returns:
        I2Result with computed value and metadata
    """
    if backend == "unified_general":
        value = _compute_I2_unified_general(R, theta, ell1, ell2, polynomials, n_quad)
    elif backend == "dsl":
        value = _compute_I2_dsl(R, theta, ell1, ell2, polynomials, n_quad)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return I2Result(ell1=ell1, ell2=ell2, value=value, backend=backend, n_quad=n_quad)


def compute_S12_pair(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    backend: BackendType = "unified_general",
    n_quad: int = 60,
) -> S12PairResult:
    """
    Compute S12 = I₁ + I₂ for a single pair.

    Args:
        R, theta, ell1, ell2, polynomials, backend, n_quad: Same as I₁/I₂ functions

    Returns:
        S12PairResult with I₁, I₂, and S12 values
    """
    I1_result = compute_I1_backend(R, theta, ell1, ell2, polynomials, backend, n_quad)
    I2_result = compute_I2_backend(R, theta, ell1, ell2, polynomials, backend, n_quad)

    return S12PairResult(
        ell1=ell1,
        ell2=ell2,
        I1_value=I1_result.value,
        I2_value=I2_result.value,
        S12_value=I1_result.value + I2_result.value,
        backend=backend,
    )


def compute_S12_all_pairs(
    R: float,
    theta: float,
    polynomials: Dict,
    backend: BackendType = "unified_general",
    n_quad: int = 60,
    apply_normalization: bool = True,
) -> S12AllPairsResult:
    """
    Compute S12 for all 6 triangle pairs.

    Args:
        R, theta, polynomials, backend, n_quad: Same as pair functions
        apply_normalization: If True, apply factorial and symmetry normalization

    Returns:
        S12AllPairsResult with all pair results and totals
    """
    factorial_norms = get_s12_factorial_normalization()
    symmetry_factors = get_s12_symmetry_factors()

    pair_results = {}
    total_raw = 0.0
    total_normalized = 0.0

    for pair_key in TRIANGLE_PAIRS:
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])

        result = compute_S12_pair(R, theta, ell1, ell2, polynomials, backend, n_quad)
        pair_results[pair_key] = result

        total_raw += result.S12_value

        if apply_normalization:
            norm = factorial_norms[pair_key] * symmetry_factors[pair_key]
            total_normalized += result.S12_value * norm
        else:
            total_normalized += result.S12_value

    return S12AllPairsResult(
        pair_results=pair_results,
        total_raw=total_raw,
        total_normalized=total_normalized,
        backend=backend,
        n_quad=n_quad,
    )


# =============================================================================
# BACKEND COMPARISON UTILITIES
# =============================================================================

@dataclass
class BackendComparisonResult:
    """Result of comparing two backends for a single pair."""
    ell1: int
    ell2: int
    unified_I1: float
    dsl_I1: float
    unified_I2: float
    dsl_I2: float
    I1_ratio: float
    I2_ratio: float
    I1_rel_err: float
    I2_rel_err: float
    match: bool  # True if both relative errors < threshold


def compare_backends_pair(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
    tolerance: float = 1e-12,
) -> BackendComparisonResult:
    """
    Compare unified_general and DSL backends for a single pair.

    Returns comparison result with ratios and relative errors.
    """
    unified_I1 = _compute_I1_unified_general(R, theta, ell1, ell2, polynomials, n_quad)
    dsl_I1 = _compute_I1_dsl(R, theta, ell1, ell2, polynomials, n_quad)

    unified_I2 = _compute_I2_unified_general(R, theta, ell1, ell2, polynomials, n_quad)
    dsl_I2 = _compute_I2_dsl(R, theta, ell1, ell2, polynomials, n_quad)

    # Compute ratios
    I1_ratio = unified_I1 / dsl_I1 if abs(dsl_I1) > 1e-15 else float('inf')
    I2_ratio = unified_I2 / dsl_I2 if abs(dsl_I2) > 1e-15 else float('inf')

    # Compute relative errors
    I1_rel_err = abs(I1_ratio - 1.0) if I1_ratio != float('inf') else abs(unified_I1 - dsl_I1)
    I2_rel_err = abs(I2_ratio - 1.0) if I2_ratio != float('inf') else abs(unified_I2 - dsl_I2)

    match = I1_rel_err < tolerance and I2_rel_err < tolerance

    return BackendComparisonResult(
        ell1=ell1,
        ell2=ell2,
        unified_I1=unified_I1,
        dsl_I1=dsl_I1,
        unified_I2=unified_I2,
        dsl_I2=dsl_I2,
        I1_ratio=I1_ratio,
        I2_ratio=I2_ratio,
        I1_rel_err=I1_rel_err,
        I2_rel_err=I2_rel_err,
        match=match,
    )


def compare_backends_all_pairs(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    tolerance: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, BackendComparisonResult]:
    """
    Compare unified_general and DSL backends for all 6 triangle pairs.

    Args:
        R, theta, polynomials, n_quad: Evaluation parameters
        tolerance: Maximum relative error for "match"
        verbose: If True, print comparison table

    Returns:
        Dict mapping pair keys to BackendComparisonResult
    """
    results = {}

    for pair_key in TRIANGLE_PAIRS:
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])
        comparison = compare_backends_pair(R, theta, ell1, ell2, polynomials, n_quad, tolerance)
        results[pair_key] = comparison

    if verbose:
        print(f"\nBackend Comparison (R={R}, n={n_quad}):")
        print("-" * 80)
        print(f"{'Pair':<6} {'Unified I1':>14} {'DSL I1':>14} {'I1 Ratio':>12} {'I1 RelErr':>12}")
        print(f"{'':6} {'Unified I2':>14} {'DSL I2':>14} {'I2 Ratio':>12} {'I2 RelErr':>12}")
        print("-" * 80)

        for pair_key, comp in results.items():
            status = "MATCH" if comp.match else "FAIL"
            print(f"({comp.ell1},{comp.ell2})  {comp.unified_I1:>14.8e} {comp.dsl_I1:>14.8e} "
                  f"{comp.I1_ratio:>12.9f} {comp.I1_rel_err:>12.2e}")
            print(f"       {comp.unified_I2:>14.8e} {comp.dsl_I2:>14.8e} "
                  f"{comp.I2_ratio:>12.9f} {comp.I2_rel_err:>12.2e}  [{status}]")
            print()

    return results


def assert_backends_equivalent(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    tolerance: float = 1e-12,
) -> None:
    """
    Assert that unified_general and DSL backends produce equivalent results.

    Raises AssertionError if any pair exceeds tolerance.
    """
    results = compare_backends_all_pairs(R, theta, polynomials, n_quad, tolerance)

    failures = []
    for pair_key, comp in results.items():
        if not comp.match:
            failures.append(
                f"({comp.ell1},{comp.ell2}): I1 rel_err={comp.I1_rel_err:.2e}, "
                f"I2 rel_err={comp.I2_rel_err:.2e}"
            )

    if failures:
        raise AssertionError(
            f"Backend equivalence check failed (tolerance={tolerance}):\n" +
            "\n".join(failures)
        )
