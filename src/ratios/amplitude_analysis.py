"""
src/ratios/amplitude_analysis.py
Phase 20.3: exp(R) Coefficient Residual Analysis

PURPOSE:
========
Analyze the exp(R) coefficient (A = I₁₂(-R)) to understand:
1. What contributes to A across different pipelines
2. Whether a single global factor explains benchmark differences
3. How A relates to the target c values

BACKGROUND:
===========
The mirror assembly formula is:
    c = I₁₂(+R) + [exp(R) + 5] × I₁₂(-R) + I₃₄(+R)
    c = A × exp(R) + B

where A = I₁₂(-R) is the exp(R) coefficient.

Phase 20.2 found that D = I₁₂(+R) + I₃₄(+R) ≠ 0, causing B/A ≠ 5.
Phase 20.3 investigates whether there are additional residual patterns
in the A coefficient that explain the ~1.3% gap in the production c.

USAGE:
======
>>> from src.ratios.amplitude_analysis import analyze_exp_coefficient_residual
>>> result = analyze_exp_coefficient_residual("kappa")
>>> print(result['summary'])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    compute_I12_components,
    LaurentMode,
    DEFAULT_LAURENT_MODE,
)


@dataclass(frozen=True)
class ExpCoefficientAnalysis:
    """Result of exp(R) coefficient analysis."""

    benchmark: str
    R: float

    # A coefficient (exp(R) multiplier)
    A: float
    A_target: float  # Implied from c target
    A_ratio: float   # A / A_target
    A_gap_percent: float

    # Per-piece contributions to A
    j11_contribution: float
    j12_contribution: float
    j15_contribution: float  # Should be 0 in main-only

    # Comparison with other pipeline
    A_production: Optional[float]
    A_production_ratio: Optional[float]

    # Overall c metrics
    c_computed: float
    c_target: float
    c_gap_percent: float

    # Summary
    summary: str


@dataclass(frozen=True)
class CrossBenchmarkAnalysis:
    """Comparison of residual patterns between benchmarks."""

    kappa: ExpCoefficientAnalysis
    kappa_star: ExpCoefficientAnalysis

    # Cross-benchmark ratios
    A_ratio_kappa_to_kstar: float
    A_target_ratio: float  # Theoretical expectation

    # Pattern analysis
    single_factor_explains: bool
    single_factor_value: Optional[float]

    summary: str


# Target c values from PRZZ
C_TARGET_KAPPA = 2.13745440613217263636
C_TARGET_KAPPA_STAR = 1.938  # Approximate from PRZZ κ*


def analyze_exp_coefficient_residual(
    benchmark: str,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    include_j15: bool = False,
) -> ExpCoefficientAnalysis:
    """
    Analyze the exp(R) coefficient for a single benchmark.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        laurent_mode: Laurent factor mode
        include_j15: Whether to include J₁,₅ (error term)

    Returns:
        ExpCoefficientAnalysis with detailed breakdown
    """
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R
    theta = 4.0 / 7.0

    # Get c target
    c_target = C_TARGET_KAPPA if benchmark == "kappa" else C_TARGET_KAPPA_STAR

    # Compute mirror assembly
    decomp = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=3,
        laurent_mode=laurent_mode,
        include_j15=include_j15,
    )

    # Extract A and c
    A = decomp['exp_coefficient']
    c_computed = decomp['assembled_total']

    # Compute implied A from c target
    # c = A × exp(R) + B, and B/A ≈ 5 (target)
    # So c = A × exp(R) + 5A = A × (exp(R) + 5)
    # A_target = c_target / (exp(R) + 5)
    A_target = c_target / (np.exp(R) + 5)

    A_ratio = A / A_target if abs(A_target) > 1e-14 else float('inf')
    A_gap_percent = (A_ratio - 1.0) * 100

    c_gap_percent = (c_computed - c_target) / c_target * 100

    # Per-piece contributions to A (from I₁₂(-R))
    i12_minus = decomp['i12_minus_pieces']
    j11_contrib = i12_minus['j11']
    j12_contrib = i12_minus['j12']
    j15_contrib = i12_minus.get('j15', 0.0)

    # Try to get production pipeline A for comparison
    A_production = None
    A_production_ratio = None
    try:
        from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
        from src.evaluate import compute_c_paper_with_mirror

        if benchmark == "kappa":
            P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        else:
            P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        poly_dict = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_c_paper_with_mirror(
            theta=theta, R=R, n=60, polynomials=poly_dict, n_quad_a=40
        )
        A_production = result.per_term.get('_S12_minus_total', None)
        if A_production is not None and abs(A_target) > 1e-14:
            A_production_ratio = A_production / A_target
    except Exception:
        pass

    # Summary
    summary_lines = [
        f"exp(R) Coefficient Analysis for {benchmark.upper()}",
        f"R = {R}",
        "",
        f"A (computed, main-only): {A:.6f}",
        f"A (implied from c target): {A_target:.6f}",
        f"A ratio: {A_ratio:.4f} ({A_gap_percent:+.2f}%)",
        "",
        "Per-piece contributions to A:",
        f"  J11: {j11_contrib:.6f} ({j11_contrib/A*100:.1f}%)",
        f"  J12: {j12_contrib:.6f} ({j12_contrib/A*100:.1f}%)",
        f"  J15: {j15_contrib:.6f}",
    ]

    if A_production is not None:
        summary_lines.extend([
            "",
            f"Production pipeline A: {A_production:.6f}",
            f"Production A / Simplified A: {A_production/A:.4f}",
        ])

    summary_lines.extend([
        "",
        f"c (computed): {c_computed:.6f}",
        f"c (target): {c_target:.6f}",
        f"c gap: {c_gap_percent:+.2f}%",
    ])

    return ExpCoefficientAnalysis(
        benchmark=benchmark,
        R=R,
        A=A,
        A_target=A_target,
        A_ratio=A_ratio,
        A_gap_percent=A_gap_percent,
        j11_contribution=j11_contrib,
        j12_contribution=j12_contrib,
        j15_contribution=j15_contrib,
        A_production=A_production,
        A_production_ratio=A_production_ratio,
        c_computed=c_computed,
        c_target=c_target,
        c_gap_percent=c_gap_percent,
        summary="\n".join(summary_lines),
    )


def compare_amplitude_across_benchmarks(
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> CrossBenchmarkAnalysis:
    """
    Compare residual patterns between κ and κ* benchmarks.

    This helps identify whether a single global factor could explain
    both benchmark differences, or if more complex per-benchmark
    adjustments are needed.

    Returns:
        CrossBenchmarkAnalysis with comparison metrics
    """
    kappa = analyze_exp_coefficient_residual("kappa", laurent_mode)
    kappa_star = analyze_exp_coefficient_residual("kappa_star", laurent_mode)

    # Cross-benchmark A ratio
    A_ratio_computed = kappa.A / kappa_star.A if abs(kappa_star.A) > 1e-14 else float('inf')
    A_ratio_target = kappa.A_target / kappa_star.A_target if abs(kappa_star.A_target) > 1e-14 else float('inf')

    # Check if single factor explains both
    # If A_computed / A_target is similar for both benchmarks, a single factor works
    single_factor_k = kappa.A_ratio
    single_factor_ks = kappa_star.A_ratio

    ratio_diff = abs(single_factor_k - single_factor_ks) / max(single_factor_k, single_factor_ks)
    single_factor_explains = ratio_diff < 0.05  # Within 5%
    single_factor_value = (single_factor_k + single_factor_ks) / 2 if single_factor_explains else None

    # Summary
    summary_lines = [
        "Cross-Benchmark exp(R) Coefficient Analysis",
        "=" * 50,
        "",
        f"κ (R={kappa.R}):",
        f"  A computed: {kappa.A:.6f}",
        f"  A target: {kappa.A_target:.6f}",
        f"  A ratio: {kappa.A_ratio:.4f}",
        "",
        f"κ* (R={kappa_star.R}):",
        f"  A computed: {kappa_star.A:.6f}",
        f"  A target: {kappa_star.A_target:.6f}",
        f"  A ratio: {kappa_star.A_ratio:.4f}",
        "",
        f"A(κ) / A(κ*) computed: {A_ratio_computed:.4f}",
        f"A(κ) / A(κ*) target: {A_ratio_target:.4f}",
        "",
        f"Single factor explains both: {single_factor_explains}",
    ]

    if single_factor_value is not None:
        summary_lines.append(f"Single factor value: {single_factor_value:.4f}")
    else:
        summary_lines.extend([
            f"Ratio difference: {ratio_diff*100:.1f}%",
            "  -> Per-benchmark adjustments may be needed",
        ])

    return CrossBenchmarkAnalysis(
        kappa=kappa,
        kappa_star=kappa_star,
        A_ratio_kappa_to_kstar=A_ratio_computed,
        A_target_ratio=A_ratio_target,
        single_factor_explains=single_factor_explains,
        single_factor_value=single_factor_value,
        summary="\n".join(summary_lines),
    )


def print_amplitude_analysis_report(
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> None:
    """Print comprehensive amplitude analysis report."""
    print()
    print("=" * 70)
    print("PHASE 20.3: exp(R) COEFFICIENT RESIDUAL ANALYSIS")
    print("=" * 70)
    print()

    cross = compare_amplitude_across_benchmarks(laurent_mode)

    print(cross.summary)
    print()

    print("-" * 70)
    print("DETAILED ANALYSIS: κ")
    print("-" * 70)
    print(cross.kappa.summary)
    print()

    print("-" * 70)
    print("DETAILED ANALYSIS: κ*")
    print("-" * 70)
    print(cross.kappa_star.summary)
    print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The exp(R) coefficient A = I₁₂(-R) determines the scaling of c with R.

KEY OBSERVATIONS:
1. A is dominated by J12(-R) which is POSITIVE (unlike J12(+R))
2. The simplified and production pipelines give different A values
3. The ratio A_computed / A_target shows how much A needs to scale

If single_factor_explains = True:
  -> A global normalization factor could fix both benchmarks

If single_factor_explains = False:
  -> Per-benchmark or R-dependent corrections are needed
""")


if __name__ == "__main__":
    print_amplitude_analysis_report()
