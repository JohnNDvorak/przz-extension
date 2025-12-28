#!/usr/bin/env python3
"""
scripts/run_phase31_track_a_bracket.py
Phase 31 Track A: Unified Bracket D=0 Analysis

GOAL:
=====
Verify the unified bracket evaluator achieves D=0, where:
  D = S12(+R) + S34(+R)  (should vanish in unified structure)
  A = S12(-R)            (the main coefficient)
  B = m × A              (the "+5" contribution)

If D=0, then:
  c = A + B = A × (1 + B/A) = A × (1 + m)

And B/A = m should equal exp(R) + 5 ≈ 8.68 for κ.

MICRO-CASE STRATEGY:
===================
Following GPT guidance, test with minimal structure first:
1. P = Q = 1 (isolate bracket machinery)
2. Only (1,1) pair (simplest case)
3. Then extend to full polynomials

Created: 2025-12-26 (Phase 31)
"""

import sys
import math
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, ".")

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)


# =============================================================================
# Benchmark Configurations
# =============================================================================

BENCHMARKS = {
    "kappa": {
        "loader": load_przz_polynomials,
        "R": 1.3036,
        "theta": 4 / 7,
        "c_target": 2.13745440613217,
    },
    "kappa_star": {
        "loader": load_przz_polynomials_kappa_star,
        "R": 1.1167,
        "theta": 4 / 7,
        "c_target": 1.9379524124677437,
    },
}


@dataclass
class BracketAnalysisResult:
    """Result of unified bracket analysis."""
    benchmark: str
    R: float
    theta: float

    # Micro-case (P=Q=1, (1,1) only)
    micro_D: Optional[float]
    micro_A: Optional[float]
    micro_B: Optional[float]
    micro_D_over_A: Optional[float]
    micro_B_over_A: Optional[float]

    # Full polynomial case
    full_D: Optional[float]
    full_A: Optional[float]
    full_B: Optional[float]
    full_D_over_A: Optional[float]
    full_B_over_A: Optional[float]

    # Targets
    m_target: float  # exp(R) + 5


def analyze_micro_case(benchmark: str) -> Dict:
    """Analyze micro-case (P=Q=1, (1,1) only)."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]

    try:
        from src.unified_bracket_evaluator import MicroCaseEvaluator

        evaluator = MicroCaseEvaluator(
            theta=theta,
            R=R,
            n_quad_u=40,
            n_quad_t=40,
        )

        result = evaluator.compute_S12_micro_case(
            include_log_factor=True,
            include_alg_prefactor=True,
        )

        abd = result.abd

        return {
            "available": True,
            "S12_plus": result.S12_plus,
            "S12_minus": result.S12_minus,
            "D": abd.D if abd else None,
            "A": abd.A if abd else None,
            "B": abd.B if abd else None,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def analyze_full_polynomials(benchmark: str, n_quad: int = 60) -> Dict:
    """Analyze with full PRZZ polynomials."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]

    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    try:
        from src.unified_bracket_evaluator import FullS12Evaluator

        evaluator = FullS12Evaluator(
            polynomials=polys,
            theta=theta,
            R=R,
            n_quad_u=n_quad,
            n_quad_t=n_quad,
            use_factorial_normalization=True,
        )

        # Try to get the compute method
        if hasattr(evaluator, 'compute_full_S12'):
            result = evaluator.compute_full_S12()
            abd = result.abd

            return {
                "available": True,
                "S12_plus": result.S12_plus,
                "S12_minus": result.S12_minus,
                "D": abd.D if abd else None,
                "A": abd.A if abd else None,
                "B": abd.B if abd else None,
            }
        else:
            return {
                "available": False,
                "error": "FullS12Evaluator.compute_full_S12() not implemented",
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def analyze_difference_quotient_mode(benchmark: str, n_quad: int = 60) -> Dict:
    """Analyze using difference_quotient mode in evaluate.py."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]

    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Try different difference quotient modes
    results = {}

    for mode in ["difference_quotient", "difference_quotient_v2", "difference_quotient_v3"]:
        try:
            from src.evaluate import compute_c_paper_with_mirror

            result = compute_c_paper_with_mirror(
                theta=theta,
                R=R,
                n=n_quad,
                polynomials=polys,
                pair_mode="triangle",
                use_factorial_normalization=True,
                mode="main",
                K=3,
                mirror_mode=mode,
                normalization_mode="scalar",
            )

            per_term = result.per_term or {}

            results[mode] = {
                "available": True,
                "c_computed": result.total,
                "c_gap_pct": 100 * (result.total - c_target) / c_target,
                "D": per_term.get("_D_total", per_term.get("D")),
                "A": per_term.get("_A_total", per_term.get("A")),
                "B": per_term.get("_B_total", per_term.get("B")),
            }
        except Exception as e:
            results[mode] = {
                "available": False,
                "error": str(e),
            }

    return results


def run_bracket_analysis(benchmark: str, n_quad: int = 60) -> BracketAnalysisResult:
    """Run complete bracket analysis."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]
    m_target = math.exp(R) + 5

    # 1. Micro-case analysis
    micro = analyze_micro_case(benchmark)

    micro_D = micro.get("D") if micro.get("available") else None
    micro_A = micro.get("A") if micro.get("available") else None
    micro_B = micro.get("B") if micro.get("available") else None

    # 2. Full polynomial analysis
    full = analyze_full_polynomials(benchmark, n_quad)

    full_D = full.get("D") if full.get("available") else None
    full_A = full.get("A") if full.get("available") else None
    full_B = full.get("B") if full.get("available") else None

    return BracketAnalysisResult(
        benchmark=benchmark,
        R=R,
        theta=theta,
        micro_D=micro_D,
        micro_A=micro_A,
        micro_B=micro_B,
        micro_D_over_A=abs(micro_D / micro_A) if micro_D is not None and micro_A and abs(micro_A) > 1e-15 else None,
        micro_B_over_A=micro_B / micro_A if micro_B is not None and micro_A and abs(micro_A) > 1e-15 else None,
        full_D=full_D,
        full_A=full_A,
        full_B=full_B,
        full_D_over_A=abs(full_D / full_A) if full_D is not None and full_A and abs(full_A) > 1e-15 else None,
        full_B_over_A=full_B / full_A if full_B is not None and full_A and abs(full_A) > 1e-15 else None,
        m_target=m_target,
    )


def print_result(result: BracketAnalysisResult) -> None:
    """Print formatted result."""
    print(f"\n{'=' * 70}")
    print(f"=== UNIFIED BRACKET ANALYSIS: {result.benchmark.upper()} ===")
    print(f"{'=' * 70}")

    print(f"\n--- Parameters ---")
    print(f"  R = {result.R}, theta = {result.theta:.6f}")
    print(f"  m_target = exp(R) + 5 = {result.m_target:.4f}")

    print(f"\n--- MICRO-CASE (P=Q=1, (1,1) only) ---")
    if result.micro_D is not None:
        print(f"  D = {result.micro_D:.6f}")
        print(f"  A = {result.micro_A:.6f}")
        print(f"  B = {result.micro_B:.6f}")
        print(f"  |D/A| = {result.micro_D_over_A:.6f} (target: <0.01)")
        print(f"  B/A   = {result.micro_B_over_A:.4f} (target: {result.m_target:.2f})")
        print(f"  STATUS: {'D≈0' if result.micro_D_over_A < 0.01 else 'D≠0'}")
    else:
        print(f"  NOT AVAILABLE")

    print(f"\n--- FULL POLYNOMIALS ---")
    if result.full_D is not None:
        print(f"  D = {result.full_D:.6f}")
        print(f"  A = {result.full_A:.6f}")
        print(f"  B = {result.full_B:.6f}")
        print(f"  |D/A| = {result.full_D_over_A:.6f} (target: <0.01)")
        print(f"  B/A   = {result.full_B_over_A:.4f} (target: {result.m_target:.2f})")
        print(f"  STATUS: {'D≈0' if result.full_D_over_A < 0.01 else 'D≠0'}")
    else:
        print(f"  NOT AVAILABLE")


def main():
    """Run Track A: Unified Bracket Analysis."""
    print("=" * 70)
    print("PHASE 31 TRACK A: UNIFIED BRACKET D=0 ANALYSIS")
    print("=" * 70)
    print(f"\nGoal: Verify D=0 in unified bracket structure")
    print(f"If D=0, then c = A(1+m) and B/A should give m = exp(R)+5")

    # Analyze both benchmarks
    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        print(f"\nAnalyzing {benchmark}...")
        results[benchmark] = run_bracket_analysis(benchmark)
        print_result(results[benchmark])

    # Also test difference quotient modes
    print(f"\n{'=' * 70}")
    print("=== DIFFERENCE QUOTIENT MODES ===")
    print(f"{'=' * 70}")

    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n--- {benchmark.upper()} ---")
        dq_results = analyze_difference_quotient_mode(benchmark)
        for mode, res in dq_results.items():
            if res.get("available"):
                print(f"  {mode}:")
                print(f"    c_computed = {res['c_computed']:.6f}")
                print(f"    c_gap = {res['c_gap_pct']:+.2f}%")
                if res.get('D') is not None:
                    print(f"    D = {res['D']:.6f}, A = {res['A']:.6f}, B = {res['B']:.6f}")
            else:
                print(f"  {mode}: NOT AVAILABLE - {res.get('error', 'unknown error')}")

    # Summary
    print(f"\n{'=' * 70}")
    print("=== TRACK A SUMMARY ===")
    print(f"{'=' * 70}")

    kappa = results["kappa"]
    kappa_star = results["kappa_star"]

    if kappa.micro_D is not None:
        print(f"\nMicro-case D/A:")
        print(f"  kappa:  {kappa.micro_D_over_A:.6f} {'✓ D≈0' if kappa.micro_D_over_A < 0.01 else '✗ D≠0'}")
        print(f"  kappa*: {kappa_star.micro_D_over_A:.6f} {'✓ D≈0' if kappa_star.micro_D_over_A < 0.01 else '✗ D≠0'}")
    else:
        print(f"\nMicro-case: Not available")

    if kappa.full_D is not None:
        print(f"\nFull polynomial D/A:")
        print(f"  kappa:  {kappa.full_D_over_A:.6f} {'✓ D≈0' if kappa.full_D_over_A < 0.01 else '✗ D≠0'}")
        print(f"  kappa*: {kappa_star.full_D_over_A:.6f} {'✓ D≈0' if kappa_star.full_D_over_A < 0.01 else '✗ D≠0'}")
    else:
        print(f"\nFull polynomial: Not available")

    print(f"\n{'=' * 70}")
    print("TRACK A COMPLETE")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
