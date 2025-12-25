#!/usr/bin/env python3
"""
scripts/k_sweep.py
Phase 19.5: K-Sweep Harness for B/A = 2K-1 Universality Check

PURPOSE:
========
Test whether the formula B/A = 2K-1 holds universally across K values.

PRZZ uses K=3 (3-piece mollifier). If the +5 signature (B/A = 5) is a
fundamental property of the mirror assembly, it should generalize:
    - K=3: B/A ≈ 5
    - K=4: B/A ≈ 7
    - K=5: B/A ≈ 9

CRITICAL CHECK:
==============
If the gap grows with K, the current derivation is incomplete.
If the gap stays constant or shrinks, we have confidence in the formula.

USAGE:
======
    python scripts/k_sweep.py
    python scripts/k_sweep.py --K 3 4 5 6
    python scripts/k_sweep.py --output k_sweep_results.json

NOTE:
=====
We don't have K>3 polynomials, so we reuse K=3 polynomials as proxy.
The key check is structural formula behavior, not exact polynomial matching.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    DEFAULT_LAURENT_MODE,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


@dataclass
class KSweepResult:
    """Result for a single (benchmark, K) computation."""

    benchmark: str
    K: int
    R: float
    theta: float
    laurent_mode: str

    # Target B/A = 2K-1
    target_B_over_A: int

    # Computed values
    A: float
    B: float
    B_over_A: float

    # Gap analysis
    gap: float
    gap_percent: float

    # Include J15 mode
    include_j15: bool


@dataclass
class KSweepReport:
    """Full K-sweep report."""

    benchmarks: List[str]
    K_values: List[int]
    laurent_mode: str
    results: Dict[str, Dict[int, KSweepResult]]

    # Summary metrics
    universality_holds: bool
    max_gap_percent: float
    gap_trend: str  # "shrinking", "constant", "growing"

    warnings: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert nested dataclasses
        for bench, k_results in d["results"].items():
            for k, result in k_results.items():
                d["results"][bench][k] = asdict(result) if hasattr(result, '__dict__') else result
        return d


def compute_k_result(
    benchmark: str,
    K: int,
    polys=None,
    theta: float = 4.0 / 7.0,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    include_j15: bool = True,
) -> KSweepResult:
    """
    Compute B/A result for a single (benchmark, K) pair.

    Args:
        benchmark: Benchmark name
        K: Number of mollifier pieces
        polys: Polynomials to use (defaults to load from benchmark)
        theta: θ parameter
        laurent_mode: Laurent mode
        include_j15: Whether to include J15 term

    Returns:
        KSweepResult with computed values
    """
    if polys is None:
        polys = load_przz_k3_polynomials(benchmark)

    R = polys.R

    # Compute mirror assembly with specified K
    result = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=K,
        laurent_mode=laurent_mode,
        include_j15=include_j15,
    )

    target = 2 * K - 1
    B_over_A = result["B_over_A"]
    gap = B_over_A - target
    gap_percent = gap / target * 100

    return KSweepResult(
        benchmark=benchmark,
        K=K,
        R=R,
        theta=theta,
        laurent_mode=laurent_mode.value,
        target_B_over_A=target,
        A=result["exp_coefficient"],
        B=result["constant_offset"],
        B_over_A=B_over_A,
        gap=gap,
        gap_percent=gap_percent,
        include_j15=include_j15,
    )


def run_k_sweep(
    benchmarks: List[str] = None,
    K_values: List[int] = None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    include_j15: bool = True,
    verbose: bool = True,
) -> KSweepReport:
    """
    Run K-sweep analysis across benchmarks and K values.

    Args:
        benchmarks: List of benchmark names
        K_values: List of K values to test
        laurent_mode: Laurent mode
        include_j15: Whether to include J15 term
        verbose: Print detailed output

    Returns:
        KSweepReport with full analysis
    """
    if benchmarks is None:
        benchmarks = ["kappa", "kappa_star"]
    if K_values is None:
        K_values = [3, 4, 5]

    results = {}
    warnings = []
    all_gaps = []

    for benchmark in benchmarks:
        polys = load_przz_k3_polynomials(benchmark)
        results[benchmark] = {}

        for K in K_values:
            result = compute_k_result(
                benchmark=benchmark,
                K=K,
                polys=polys,
                laurent_mode=laurent_mode,
                include_j15=include_j15,
            )
            results[benchmark][K] = result
            all_gaps.append((K, result.gap_percent))

            if K > 3:
                warnings.append(
                    f"{benchmark} K={K}: Using K=3 polynomials as proxy"
                )

    # Analyze gap trend
    max_gap = max(abs(gap) for _, gap in all_gaps)

    # Check if gaps grow with K
    gaps_by_K = {}
    for K, gap in all_gaps:
        if K not in gaps_by_K:
            gaps_by_K[K] = []
        gaps_by_K[K].append(abs(gap))

    avg_gaps = {K: sum(gaps) / len(gaps) for K, gaps in gaps_by_K.items()}
    sorted_K = sorted(avg_gaps.keys())

    if len(sorted_K) >= 2:
        first_gap = avg_gaps[sorted_K[0]]
        last_gap = avg_gaps[sorted_K[-1]]

        if last_gap > first_gap * 1.5:
            gap_trend = "growing"
            warnings.append(
                f"WARNING: Gap GROWS with K ({first_gap:.1f}% → {last_gap:.1f}%)"
            )
        elif last_gap < first_gap * 0.7:
            gap_trend = "shrinking"
        else:
            gap_trend = "constant"
    else:
        gap_trend = "unknown"

    # Check universality
    # Universality holds if all gaps are within 10%
    universality_holds = max_gap < 10.0

    if not universality_holds:
        warnings.append(
            f"UNIVERSALITY FAILS: Max gap {max_gap:.1f}% exceeds 10% threshold"
        )

    report = KSweepReport(
        benchmarks=benchmarks,
        K_values=K_values,
        laurent_mode=laurent_mode.value,
        results=results,
        universality_holds=universality_holds,
        max_gap_percent=max_gap,
        gap_trend=gap_trend,
        warnings=warnings,
    )

    if verbose:
        print_k_sweep_report(report)

    return report


def print_k_sweep_report(report: KSweepReport) -> None:
    """Print formatted K-sweep report."""
    print()
    print("=" * 80)
    print("PHASE 19.5: K-SWEEP UNIVERSALITY ANALYSIS")
    print(f"Testing B/A = 2K-1 across K ∈ {{{', '.join(map(str, report.K_values))}}}")
    print(f"Laurent Mode: {report.laurent_mode}")
    print("=" * 80)
    print()

    # Table for each benchmark
    for benchmark in report.benchmarks:
        bench_results = report.results[benchmark]
        R = bench_results[report.K_values[0]].R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 70)
        print(f"{'K':<6} {'Target B/A':<12} {'Computed B/A':<15} {'Gap':<12} {'Gap %':<10}")
        print("-" * 70)

        for K in report.K_values:
            result = bench_results[K]
            print(
                f"{K:<6} {result.target_B_over_A:<12} {result.B_over_A:<15.4f} "
                f"{result.gap:+12.4f} {result.gap_percent:+8.2f}%"
            )

        print("-" * 70)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Gap trend: {report.gap_trend.upper()}")
    print(f"  Max gap: {report.max_gap_percent:.2f}%")
    print(f"  Universality (all gaps < 10%): {'YES' if report.universality_holds else 'NO'}")

    if report.warnings:
        print("\nWARNINGS:")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")

    print()

    # Interpretation
    print("INTERPRETATION:")
    if report.gap_trend == "growing":
        print("  The gap GROWS with K, suggesting the derivation is incomplete.")
        print("  The formula B/A = 2K-1 may not hold at higher K.")
    elif report.gap_trend == "shrinking":
        print("  The gap SHRINKS with K, suggesting good extrapolation.")
        print("  The formula B/A = 2K-1 appears stable.")
    else:
        print("  The gap is CONSTANT across K values.")
        print("  This suggests systematic offset, not K-dependent error.")

    print("=" * 80)


def run_j15_comparison_sweep(
    benchmarks: List[str] = None,
    K_values: List[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run K-sweep comparing with and without J15.

    This shows whether J15 requirement changes with K.
    """
    if benchmarks is None:
        benchmarks = ["kappa", "kappa_star"]
    if K_values is None:
        K_values = [3, 4, 5]

    results = {}

    for benchmark in benchmarks:
        results[benchmark] = {}

        for K in K_values:
            with_j15 = compute_k_result(
                benchmark, K, include_j15=True
            )
            without_j15 = compute_k_result(
                benchmark, K, include_j15=False
            )

            j15_contribution = with_j15.B_over_A - without_j15.B_over_A

            results[benchmark][K] = {
                "with_j15": with_j15.B_over_A,
                "without_j15": without_j15.B_over_A,
                "j15_contribution": j15_contribution,
                "target": 2 * K - 1,
                "with_gap_pct": with_j15.gap_percent,
                "without_gap_pct": without_j15.gap_percent,
            }

    if verbose:
        print()
        print("=" * 80)
        print("J15 CONTRIBUTION ACROSS K VALUES")
        print("=" * 80)

        for benchmark in benchmarks:
            print(f"\n{benchmark.upper()}:")
            print("-" * 70)
            print(f"{'K':<4} {'Target':<8} {'With J15':<12} {'Without':<12} {'J15 effect':<12} {'Gap Δ':<10}")
            print("-" * 70)

            for K in K_values:
                r = results[benchmark][K]
                gap_delta = r["without_gap_pct"] - r["with_gap_pct"]
                print(
                    f"{K:<4} {r['target']:<8} {r['with_j15']:<12.4f} "
                    f"{r['without_j15']:<12.4f} {r['j15_contribution']:+12.4f} "
                    f"{gap_delta:+8.2f}%"
                )

            print("-" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 19.5: K-sweep universality analysis"
    )
    parser.add_argument(
        "--K",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="K values to sweep (default: 3 4 5)",
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["kappa", "kappa_star"],
        help="Benchmarks to test (default: both)",
    )
    parser.add_argument(
        "--mode",
        choices=["actual", "raw", "pole_cancelled"],
        default="actual",
        help="Laurent mode (default: actual)",
    )
    parser.add_argument(
        "--j15-comparison",
        action="store_true",
        help="Show J15 contribution comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Map mode name to enum
    mode_map = {
        "actual": LaurentMode.ACTUAL_LOGDERIV,
        "raw": LaurentMode.RAW_LOGDERIV,
        "pole_cancelled": LaurentMode.POLE_CANCELLED,
    }
    laurent_mode = mode_map[args.mode]

    # Run main sweep
    report = run_k_sweep(
        benchmarks=args.bench,
        K_values=args.K,
        laurent_mode=laurent_mode,
        verbose=True,
    )

    # Optional J15 comparison
    if args.j15_comparison:
        run_j15_comparison_sweep(
            benchmarks=args.bench,
            K_values=args.K,
            verbose=True,
        )

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        json_data = {
            "benchmarks": report.benchmarks,
            "K_values": report.K_values,
            "laurent_mode": report.laurent_mode,
            "universality_holds": report.universality_holds,
            "max_gap_percent": report.max_gap_percent,
            "gap_trend": report.gap_trend,
            "warnings": report.warnings,
            "results": {},
        }

        for bench, k_results in report.results.items():
            json_data["results"][bench] = {}
            for K, result in k_results.items():
                json_data["results"][bench][str(K)] = asdict(result)

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
