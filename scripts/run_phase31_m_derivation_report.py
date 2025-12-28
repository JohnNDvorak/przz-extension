#!/usr/bin/env python3
"""
scripts/run_phase31_m_derivation_report.py
Phase 31.1: Comprehensive M Derivation Status Report

This script prints a complete diagnostic of all m derivation approaches:
1. Empirical baseline: m = exp(R) + 5 accuracy on both benchmarks
2. Unified bracket: D value and B/A ratio
3. Operator Q-shift: m_eff from mirror transform
4. J₁ five-piece decomposition: Per-piece contributions

PURPOSE:
Before deriving m from first principles, we must understand:
- What accuracy does the empirical formula achieve?
- How far is each derivation approach from the target?
- Which components contribute the "+5" constant?

Created: 2025-12-26 (Phase 31)
"""

import sys
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

sys.path.insert(0, ".")

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import compute_c_paper_with_mirror
from src.mirror_transform_paper_exact import compute_mirror_paper_analysis


# =============================================================================
# Benchmark Configurations
# =============================================================================

BENCHMARKS = {
    "kappa": {
        "loader": load_przz_polynomials,
        "loader_name": "load_przz_polynomials()",
        "R": 1.3036,
        "theta": 4 / 7,
        "c_target": 2.13745440613217,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "loader": load_przz_polynomials_kappa_star,
        "loader_name": "load_przz_polynomials_kappa_star()",
        "R": 1.1167,
        "theta": 4 / 7,
        "c_target": 1.9379524124677437,
        "kappa_target": 0.407511457,
    },
}


@dataclass
class MDerivationReport:
    """Complete report on m derivation status."""

    benchmark: str
    R: float
    theta: float

    # Empirical baseline
    m_empirical: float
    c_empirical: float
    c_target: float
    c_gap_pct_empirical: float

    # Unified bracket (if available)
    bracket_D: Optional[float]
    bracket_A: Optional[float]
    bracket_B: Optional[float]
    bracket_ratio_B_over_A: Optional[float]

    # Q-shift mode (if available)
    q_shift_m_eff: Optional[float]
    q_shift_c: Optional[float]
    q_shift_gap_pct: Optional[float]

    # Direct/Proxy ratio (for comparison)
    S12_plus: float
    S12_minus: float
    ratio_direct_proxy: float

    # J₁ decomposition (if available)
    j1_pieces: Optional[Dict[str, float]]


def compute_empirical_baseline(benchmark: str, n_quad: int = 60) -> Dict[str, Any]:
    """Compute empirical m = exp(R) + 5 accuracy."""
    config = BENCHMARKS[benchmark]
    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]

    # Compute using empirical_scalar mode (the baseline)
    result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polys,
        pair_mode="triangle",
        use_factorial_normalization=True,
        mode="main",
        K=3,
        mirror_mode="empirical_scalar",
    )

    c_computed = result.total
    c_gap_pct = 100 * (c_computed - c_target) / c_target
    m_empirical = math.exp(R) + 5

    # Extract S12 components for ratio analysis
    per_term = result.per_term or {}
    S12_plus = per_term.get("_S12_plus_total", float('nan'))
    S12_minus = per_term.get("_S12_minus_total", float('nan'))

    return {
        "m_empirical": m_empirical,
        "c_computed": c_computed,
        "c_target": c_target,
        "c_gap_pct": c_gap_pct,
        "S12_plus": S12_plus,
        "S12_minus": S12_minus,
        "per_term": per_term,
    }


def compute_unified_bracket_status(benchmark: str, n_quad: int = 60) -> Dict[str, Any]:
    """Check unified bracket D=0 status."""
    config = BENCHMARKS[benchmark]
    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    R = config["R"]
    theta = config["theta"]

    try:
        # Try difference_quotient_v3 mode for unified bracket
        result = compute_c_paper_with_mirror(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polys,
            pair_mode="triangle",
            use_factorial_normalization=True,
            mode="main",
            K=3,
            mirror_mode="difference_quotient_v3",
            normalization_mode="scalar",
        )

        per_term = result.per_term or {}

        return {
            "available": True,
            "D": per_term.get("_D_total", None),
            "A": per_term.get("_A_total", None),
            "B": per_term.get("_B_total", None),
            "c_computed": result.total,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def compute_q_shift_status(benchmark: str, n_quad: int = 60) -> Dict[str, Any]:
    """Check operator Q-shift mode accuracy."""
    config = BENCHMARKS[benchmark]
    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]

    try:
        result = compute_c_paper_with_mirror(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polys,
            pair_mode="triangle",
            use_factorial_normalization=True,
            mode="main",
            K=3,
            mirror_mode="operator_q_shift",
        )

        c_computed = result.total
        c_gap_pct = 100 * (c_computed - c_target) / c_target

        per_term = result.per_term or {}
        m_eff = per_term.get("_mirror_multiplier_effective", None)

        return {
            "available": True,
            "c_computed": c_computed,
            "c_gap_pct": c_gap_pct,
            "m_eff": m_eff,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def compute_j1_decomposition(benchmark: str) -> Dict[str, Any]:
    """Compute J₁ five-piece decomposition for the benchmark."""
    config = BENCHMARKS[benchmark]
    R = config["R"]

    try:
        from src.ratios.j1_k3_decomposition import compute_j1_pieces

        # J₁ pieces at α = β = -R (the evaluation point)
        pieces = compute_j1_pieces(alpha=-R, beta=-R)

        return {
            "available": True,
            "j11": float(pieces.j11.real),
            "j12": float(pieces.j12.real),
            "j13": float(pieces.j13.real),
            "j14": float(pieces.j14.real),
            "j15": float(pieces.j15.real),
            "total": float(sum([pieces.j11, pieces.j12, pieces.j13, pieces.j14, pieces.j15]).real),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def compute_full_report(benchmark: str, n_quad: int = 60) -> MDerivationReport:
    """Compute comprehensive derivation status report."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]

    # 1. Empirical baseline
    empirical = compute_empirical_baseline(benchmark, n_quad)

    # 2. Unified bracket
    bracket = compute_unified_bracket_status(benchmark, n_quad)

    # 3. Q-shift mode
    q_shift = compute_q_shift_status(benchmark, n_quad)

    # 4. J₁ decomposition
    j1 = compute_j1_decomposition(benchmark)

    # Compute ratio
    S12_plus = empirical["S12_plus"]
    S12_minus = empirical["S12_minus"]
    ratio = S12_plus / S12_minus if abs(S12_minus) > 1e-15 else float('inf')

    return MDerivationReport(
        benchmark=benchmark,
        R=R,
        theta=theta,
        m_empirical=empirical["m_empirical"],
        c_empirical=empirical["c_computed"],
        c_target=c_target,
        c_gap_pct_empirical=empirical["c_gap_pct"],
        bracket_D=bracket.get("D") if bracket.get("available") else None,
        bracket_A=bracket.get("A") if bracket.get("available") else None,
        bracket_B=bracket.get("B") if bracket.get("available") else None,
        bracket_ratio_B_over_A=(
            bracket.get("B") / bracket.get("A")
            if bracket.get("available") and bracket.get("A") and abs(bracket.get("A")) > 1e-15
            else None
        ),
        q_shift_m_eff=q_shift.get("m_eff") if q_shift.get("available") else None,
        q_shift_c=q_shift.get("c_computed") if q_shift.get("available") else None,
        q_shift_gap_pct=q_shift.get("c_gap_pct") if q_shift.get("available") else None,
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        ratio_direct_proxy=ratio,
        j1_pieces=j1 if j1.get("available") else None,
    )


def print_report(report: MDerivationReport) -> None:
    """Print formatted report."""
    print(f"\n{'=' * 70}")
    print(f"=== M DERIVATION STATUS: {report.benchmark.upper()} ===")
    print(f"{'=' * 70}")

    print(f"\n--- Parameters ---")
    print(f"  R     = {report.R}")
    print(f"  theta = {report.theta:.6f}")

    print(f"\n--- 1. EMPIRICAL BASELINE (m = exp(R) + 5) ---")
    print(f"  m_empirical = exp({report.R}) + 5 = {report.m_empirical:.4f}")
    print(f"  c_computed  = {report.c_empirical:.6f}")
    print(f"  c_target    = {report.c_target:.6f}")
    print(f"  c_gap       = {report.c_gap_pct_empirical:+.2f}%")
    print(f"  STATUS: {'GOOD (<2%)' if abs(report.c_gap_pct_empirical) < 2 else 'NEEDS WORK'}")

    print(f"\n--- 2. UNIFIED BRACKET (D=0 Target) ---")
    if report.bracket_D is not None:
        print(f"  D = {report.bracket_D:.6f}")
        print(f"  A = {report.bracket_A:.6f}")
        print(f"  B = {report.bracket_B:.6f}")
        print(f"  B/A = {report.bracket_ratio_B_over_A:.4f} (target: 5.0)")
        D_ratio = abs(report.bracket_D) / abs(report.bracket_A) if report.bracket_A else float('inf')
        print(f"  D/A = {D_ratio:.4f} (target: <0.01)")
        print(f"  STATUS: {'D≈0 ACHIEVED' if D_ratio < 0.01 else 'D≠0 - needs work'}")
    else:
        print(f"  NOT AVAILABLE (difference_quotient_v3 mode may not be implemented)")

    print(f"\n--- 3. OPERATOR Q-SHIFT MODE ---")
    if report.q_shift_m_eff is not None:
        print(f"  m_eff = {report.q_shift_m_eff:.4f} (target: {report.m_empirical:.4f})")
        print(f"  c_computed = {report.q_shift_c:.6f}")
        print(f"  c_gap = {report.q_shift_gap_pct:+.2f}%")
        m_gap = 100 * (report.q_shift_m_eff - report.m_empirical) / report.m_empirical
        print(f"  m_gap = {m_gap:+.2f}%")
        print(f"  STATUS: {'GOOD' if abs(report.q_shift_gap_pct) < 5 else 'NEEDS WORK'}")
    elif report.q_shift_c is not None:
        print(f"  c_computed = {report.q_shift_c:.6f}")
        print(f"  c_gap = {report.q_shift_gap_pct:+.2f}%")
        print(f"  STATUS: {'GOOD' if abs(report.q_shift_gap_pct) < 5 else 'NEEDS WORK'}")
    else:
        print(f"  NOT AVAILABLE or errored")

    print(f"\n--- 4. DIRECT/PROXY RATIO (for reference) ---")
    print(f"  S12(+R) = {report.S12_plus:.6f}")
    print(f"  S12(-R) = {report.S12_minus:.6f}")
    print(f"  Ratio S12(+R)/S12(-R) = {report.ratio_direct_proxy:.4f}")
    print(f"  NOTE: This ratio ({report.ratio_direct_proxy:.2f}) ≠ m_empirical ({report.m_empirical:.2f})")
    print(f"        m is NOT simply S12(+R)/S12(-R)")

    print(f"\n--- 5. J₁ FIVE-PIECE DECOMPOSITION ---")
    if report.j1_pieces:
        pieces = report.j1_pieces
        print(f"  j11 (1⋆Λ₂ series)     = {pieces.get('j11', 'N/A'):.6f}")
        print(f"  j12 (double ζ'/ζ)     = {pieces.get('j12', 'N/A'):.6f}")
        print(f"  j13 (log with β ζ'/ζ) = {pieces.get('j13', 'N/A'):.6f}")
        print(f"  j14 (log with α ζ'/ζ) = {pieces.get('j14', 'N/A'):.6f}")
        print(f"  j15 (A^{(1,1)} term)   = {pieces.get('j15', 'N/A'):.6f}")
        print(f"  Total J₁             = {pieces.get('total', 'N/A'):.6f}")
        print(f"  NOTE: The '+5' may emerge from piece combinations")
    else:
        print(f"  NOT AVAILABLE (j1_k3_decomposition may need compute_j1_pieces function)")


def print_comparison(kappa: MDerivationReport, kappa_star: MDerivationReport) -> None:
    """Print side-by-side comparison of derivation approaches."""
    print(f"\n{'=' * 70}")
    print("=== CROSS-BENCHMARK COMPARISON ===")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<35} {'kappa':>15} {'kappa*':>15}")
    print("-" * 65)

    print(f"{'R':<35} {kappa.R:>15.4f} {kappa_star.R:>15.4f}")
    print(f"{'m_empirical = exp(R)+5':<35} {kappa.m_empirical:>15.4f} {kappa_star.m_empirical:>15.4f}")
    print(f"{'c_gap (empirical)':<35} {kappa.c_gap_pct_empirical:>14.2f}% {kappa_star.c_gap_pct_empirical:>14.2f}%")
    print(f"{'S12(+R)/S12(-R) ratio':<35} {kappa.ratio_direct_proxy:>15.4f} {kappa_star.ratio_direct_proxy:>15.4f}")

    if kappa.bracket_ratio_B_over_A is not None:
        print(f"{'Bracket B/A (target 5)':<35} {kappa.bracket_ratio_B_over_A:>15.4f} {kappa_star.bracket_ratio_B_over_A or 'N/A':>15}")

    if kappa.q_shift_gap_pct is not None:
        print(f"{'Q-shift c_gap':<35} {kappa.q_shift_gap_pct:>14.2f}% {kappa_star.q_shift_gap_pct or 'N/A':>15}")

    print(f"\n{'=' * 70}")
    print("=== KEY INSIGHTS ===")
    print(f"{'=' * 70}")
    print(f"""
1. The empirical formula m = exp(R) + 5 achieves:
   - kappa:  {kappa.c_gap_pct_empirical:+.2f}% gap
   - kappa*: {kappa_star.c_gap_pct_empirical:+.2f}% gap

2. The ratio S12(+R)/S12(-R) is NOT equal to m:
   - kappa:  ratio = {kappa.ratio_direct_proxy:.2f} vs m = {kappa.m_empirical:.2f}
   - kappa*: ratio = {kappa_star.ratio_direct_proxy:.2f} vs m = {kappa_star.m_empirical:.2f}

3. The "+5" (= 2K-1 for K=3) comes from:
   - J₁ five-piece decomposition structure
   - NOT from a simple ratio of integrals

4. To derive m from first principles, we need:
   - Track A: Unified bracket D=0 → B/A = 5
   - Track B: J₁ piece analysis → derive 2K-1
   - Track C: Operator Q-shift → m_eff matching
""")


def main():
    """Generate comprehensive m derivation report."""
    print("=" * 70)
    print("PHASE 31.1: M DERIVATION STATUS REPORT")
    print("=" * 70)
    print(f"\nPurpose: Baseline all m derivation approaches before first-principles work")
    print(f"Formula: m = exp(R) + (2K-1), for K=3: m = exp(R) + 5")

    # Compute reports for both benchmarks
    print("\nComputing kappa benchmark...")
    kappa_report = compute_full_report("kappa", n_quad=60)

    print("Computing kappa* benchmark...")
    kappa_star_report = compute_full_report("kappa_star", n_quad=60)

    # Print individual reports
    print_report(kappa_report)
    print_report(kappa_star_report)

    # Print comparison
    print_comparison(kappa_report, kappa_star_report)

    print("\n" + "=" * 70)
    print("PHASE 31.1 REPORT COMPLETE")
    print("=" * 70)

    return {
        "kappa": kappa_report,
        "kappa_star": kappa_star_report,
    }


if __name__ == "__main__":
    main()
