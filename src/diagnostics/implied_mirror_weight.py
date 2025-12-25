"""
src/diagnostics/implied_mirror_weight.py
Phase 18.1: Implied Mirror Weight Diagnostic

PURPOSE:
========
Compute the m1 value that would achieve c = c_target exactly, then compare
it against the empirical formula exp(R) + (2K-1).

This is a diagnostic-first approach: we're NOT fitting new parameters,
we're measuring what value WOULD be required.

IMPORTANT NOTE ON CHANNEL STRUCTURE:
====================================
This diagnostic uses the J1x channel decomposition from j1_euler_maclaurin.py,
which implements a SIMPLIFIED Case B-only approach. The production pipeline
(Term DSL in evaluate.py) uses the FULL Case C structure.

As a result:
- J1x implied m1: ~1.3-1.8 (very different from empirical)
- Empirical m1: ~8.7 (exp(R) + 5)

This 5-6x difference tells us the J1x channels are structurally different
from the production channels. The J1x diagnostic is useful for understanding
the "+5 signature" mechanism, but not for calibrating production m1.

KEY FINDING:
============
The ratio_to_empirical is ~0.15-0.22, meaning J1x channels need only ~15-22%
of the empirical m1 weight. This is expected because J1x doesn't include
Case C cross-term contributions that dominate production.

USAGE:
======
>>> from src.diagnostics.implied_mirror_weight import run_implied_m1_comparison
>>> results = run_implied_m1_comparison(verbose=True)
>>> # Shows implied m1 for both benchmarks with comparison to empirical
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    LaurentMode,
    DEFAULT_LAURENT_MODE,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.m1_policy import M1_FITTED_COEFFICIENT_A, M1_FITTED_COEFFICIENT_B


# Benchmark targets
BENCHMARKS = {
    "kappa": {
        "R": 1.3036,
        "theta": 4.0 / 7.0,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "R": 1.1167,
        "theta": 4.0 / 7.0,
        "c_target": 1.9379524124677437,
        "kappa_target": 0.407511457,
    },
}


@dataclass(frozen=True)
class ImpliedM1Result:
    """Result of implied m1 computation."""

    benchmark: str
    """Benchmark name: 'kappa' or 'kappa_star'."""

    m1_implied: float
    """The m1 value that achieves c = c_target exactly."""

    m1_empirical: float
    """Empirical formula: exp(R) + (2K-1)."""

    m1_fitted: float
    """Fitted formula: A*exp(R) + B from Phase 14."""

    ratio_to_empirical: float
    """m1_implied / m1_empirical (should be ~1.0)."""

    ratio_to_fitted: float
    """m1_implied / m1_fitted."""

    # Channel decomposition
    I12_plus: float
    """Sum of I1+I2 at +R."""

    I12_minus: float
    """Sum of I1+I2 at -R (before m1 scaling)."""

    I34_plus: float
    """Sum of I3+I4 at +R (no mirror)."""

    c_target: float
    """Target c value from PRZZ."""

    c_computed: float
    """c computed with m1_implied (should match c_target)."""

    residual: float
    """c_computed - c_target (should be ~0)."""

    R: float
    """R parameter."""

    K: int
    """Number of pieces."""

    laurent_mode: str
    """Laurent mode used for computation."""


def compute_implied_m1(
    c_target: float,
    I12_plus: float,
    I12_minus: float,
    I34_plus: float,
    R: float,
    K: int = 3,
) -> float:
    """
    Solve for m1 from the mirror assembly formula.

    From: c = I12(+R) + m1 * I12(-R) + I34(+R)
    Solve: m1 = (c_target - I12_plus - I34_plus) / I12_minus

    Args:
        c_target: Target c value from PRZZ
        I12_plus: Sum of I1+I2 at +R
        I12_minus: Sum of I1+I2 at -R (base, before m1 scaling)
        I34_plus: Sum of I3+I4 at +R
        R: R parameter
        K: Number of pieces

    Returns:
        The m1 value that achieves c = c_target exactly

    Raises:
        ValueError: If I12_minus is too small
    """
    if abs(I12_minus) < 1e-15:
        raise ValueError(f"I12_minus is too small ({I12_minus:.2e}), cannot solve for m1")

    numerator = c_target - I12_plus - I34_plus
    return numerator / I12_minus


def compute_implied_m1_with_breakdown(
    benchmark: str,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> ImpliedM1Result:
    """
    Compute implied m1 with full breakdown for a benchmark.

    Uses compute_m1_with_mirror_assembly() to get channel values,
    then solves for implied m1.

    Args:
        benchmark: "kappa" or "kappa_star"
        laurent_mode: LaurentMode for J12/J13/J14 evaluation
        theta: theta parameter
        K: Number of pieces

    Returns:
        ImpliedM1Result with full attribution
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'kappa' or 'kappa_star'.")

    params = BENCHMARKS[benchmark]
    R = params["R"]
    c_target = params["c_target"]

    # Load polynomials for this benchmark
    polys = load_przz_k3_polynomials(benchmark)

    # Compute mirror assembly components
    result = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=K,
        laurent_mode=laurent_mode,
    )

    # Extract channel values
    I12_plus = result["i12_plus_total"]
    I12_minus = result["i12_minus_total"]
    I34_plus = result["i34_plus_total"]

    # Compute implied m1
    m1_implied = compute_implied_m1(
        c_target=c_target,
        I12_plus=I12_plus,
        I12_minus=I12_minus,
        I34_plus=I34_plus,
        R=R,
        K=K,
    )

    # Compute empirical m1
    m1_empirical = np.exp(R) + (2 * K - 1)

    # Compute fitted m1 (from Phase 14)
    m1_fitted = M1_FITTED_COEFFICIENT_A * np.exp(R) + M1_FITTED_COEFFICIENT_B

    # Verify by computing c with implied m1
    c_computed = I12_plus + m1_implied * I12_minus + I34_plus

    return ImpliedM1Result(
        benchmark=benchmark,
        m1_implied=float(m1_implied),
        m1_empirical=float(m1_empirical),
        m1_fitted=float(m1_fitted),
        ratio_to_empirical=float(m1_implied / m1_empirical),
        ratio_to_fitted=float(m1_implied / m1_fitted),
        I12_plus=float(I12_plus),
        I12_minus=float(I12_minus),
        I34_plus=float(I34_plus),
        c_target=float(c_target),
        c_computed=float(c_computed),
        residual=float(c_computed - c_target),
        R=float(R),
        K=K,
        laurent_mode=laurent_mode.value,
    )


def run_implied_m1_comparison(
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    verbose: bool = True,
) -> Dict[str, ImpliedM1Result]:
    """
    Run implied m1 comparison for both benchmarks.

    Args:
        laurent_mode: Laurent mode for computation
        verbose: Print detailed report

    Returns:
        Dict with 'kappa' and 'kappa_star' results
    """
    results = {}

    for benchmark in ["kappa", "kappa_star"]:
        results[benchmark] = compute_implied_m1_with_breakdown(
            benchmark=benchmark,
            laurent_mode=laurent_mode,
        )

    if verbose:
        print_implied_m1_report(results, laurent_mode)

    return results


def print_implied_m1_report(
    results: Dict[str, ImpliedM1Result],
    laurent_mode: LaurentMode,
) -> None:
    """Print formatted report of implied m1 analysis."""
    print()
    print("=" * 70)
    print("PHASE 18.1: IMPLIED MIRROR WEIGHT DIAGNOSTIC")
    print(f"Laurent Mode: {laurent_mode.value}")
    print("=" * 70)
    print()

    # Summary table
    print("-" * 70)
    print(f"{'Benchmark':<12} {'m1_implied':<12} {'m1_emp':<10} "
          f"{'ratio':<8} {'gap%':<8} {'residual':<12}")
    print("-" * 70)

    for name, result in results.items():
        gap_pct = (result.ratio_to_empirical - 1.0) * 100
        print(f"{name:<12} {result.m1_implied:<12.6f} {result.m1_empirical:<10.4f} "
              f"{result.ratio_to_empirical:<8.4f} {gap_pct:+7.2f}% {result.residual:<12.2e}")

    print("-" * 70)
    print()

    # Detailed breakdown
    for name, result in results.items():
        print(f"\n{name.upper()} (R={result.R}):")
        print(f"  Channel decomposition:")
        print(f"    I12(+R) = {result.I12_plus:.6f}")
        print(f"    I12(-R) = {result.I12_minus:.6f}")
        print(f"    I34(+R) = {result.I34_plus:.6f}")
        print(f"  Target c = {result.c_target:.10f}")
        print(f"  Implied m1 = {result.m1_implied:.6f}")
        print(f"    vs empirical exp(R)+5 = {result.m1_empirical:.6f} (ratio: {result.ratio_to_empirical:.4f})")
        print(f"    vs fitted A*exp(R)+B = {result.m1_fitted:.6f} (ratio: {result.ratio_to_fitted:.4f})")

    # Key insights
    kappa = results["kappa"]
    kappa_star = results["kappa_star"]

    print()
    print("KEY INSIGHTS:")
    print("-" * 40)

    ratio_diff = kappa.ratio_to_empirical - kappa_star.ratio_to_empirical
    print(f"  Ratio difference (kappa - kappa*): {ratio_diff:+.4f}")

    if abs(ratio_diff) < 0.02:
        print("  -> Consistent: Both benchmarks require similar m1 scaling")
    else:
        print("  -> Inconsistent: R-dependent factor may be missing")

    avg_ratio = (kappa.ratio_to_empirical + kappa_star.ratio_to_empirical) / 2
    print(f"  Average ratio to empirical: {avg_ratio:.4f}")

    if abs(avg_ratio - 1.0) < 0.05:
        print("  -> Empirical formula exp(R)+5 is close to correct")
    else:
        print(f"  -> Empirical formula needs adjustment by factor {avg_ratio:.4f}")

    print()


if __name__ == "__main__":
    # Run diagnostic
    run_implied_m1_comparison(laurent_mode=LaurentMode.ACTUAL_LOGDERIV, verbose=True)
