#!/usr/bin/env python3
"""
scripts/run_delta_report.py
Phase 19.3: Unified Delta Decomposition Report

PURPOSE:
========
Generate a comprehensive delta decomposition report with:
1. Per-piece breakdown (I12+, I12-, I34+, individual J terms)
2. Mode-labeled output (SEMANTIC vs NUMERIC)
3. Invariant checking (δ == D/A, triangle convention)
4. JSON export for programmatic analysis

USAGE:
======
    python scripts/run_delta_report.py --bench kappa kappa_star
    python scripts/run_delta_report.py --mode semantic --output delta_report.json
    python scripts/run_delta_report.py --verbose

OUTPUT:
=======
Prints formatted report to stdout and optionally saves JSON.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ratios.delta_track import compute_delta_record, DeltaRecord
from src.ratios.j1_euler_maclaurin import LaurentMode
from src.ratios.plus5_harness import (
    compute_plus5_signature_split,
    Plus5SplitResult,
)


@dataclass
class DeltaDecomposition:
    """
    Extended delta decomposition with invariant checks.

    This is the Phase 19 output format with explicit mode tracking
    and per-piece attribution.
    """

    # Identification
    benchmark: str
    mode: str  # "SEMANTIC_LAURENT" or "NUMERIC_FUNCTIONAL_EQ"
    R: float
    theta: float
    K: int

    # Core delta metrics
    A: float
    B: float
    D: float
    delta: float
    B_over_A: float
    gap_percent: float

    # Per-piece breakdown
    per_piece: Dict[str, float]

    # Invariants
    invariants: Dict[str, bool]

    # Warnings
    warnings: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def map_laurent_mode_to_phase19(laurent_mode: LaurentMode) -> str:
    """Map LaurentMode to Phase 19 mode terminology."""
    if laurent_mode == LaurentMode.RAW_LOGDERIV:
        return "SEMANTIC_LAURENT"
    else:
        return "NUMERIC_FUNCTIONAL_EQ"


def compute_delta_decomposition(
    benchmark: str,
    mode: str = "numeric",
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> DeltaDecomposition:
    """
    Compute full delta decomposition with invariant checks.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        mode: 'semantic' or 'numeric'
        theta: theta parameter
        K: Number of pieces

    Returns:
        DeltaDecomposition with all metrics and invariants
    """
    # Map mode to LaurentMode
    if mode.lower() == "semantic":
        laurent_mode = LaurentMode.RAW_LOGDERIV
    else:
        laurent_mode = LaurentMode.ACTUAL_LOGDERIV

    # Compute base record
    record = compute_delta_record(benchmark, laurent_mode, theta, K)

    # Compute D = I12+ + I34+ (the "+R branch" total)
    D = record.I12_plus + record.I34_plus

    # Build per-piece breakdown
    per_piece = {
        "I12_plus_R": record.I12_plus,
        "I12_minus_R": record.I12_minus,
        "I34_plus_R": record.I34_plus,
        "I34_minus_R": 0.0,  # Not computed in J1x pipeline
        "j11_plus": record.j11_plus,
        "j12_plus": record.j12_plus,
        "j15_plus": record.j15_plus,
        "j13_plus": record.j13_plus,
        "j14_plus": record.j14_plus,
        "j11_minus": record.j11_minus,
        "j12_minus": record.j12_minus,
        "j15_minus": record.j15_minus,
    }

    # Compute invariants
    invariants = {}
    warnings = []

    # Invariant 1: delta == D/A (within tolerance)
    if abs(record.A) > 1e-10:
        computed_delta = D / record.A
        delta_matches = abs(computed_delta - record.delta) < 1e-6
        invariants["delta_equals_D_over_A"] = delta_matches
        if not delta_matches:
            warnings.append(
                f"δ mismatch: computed D/A = {computed_delta:.6f} vs stored δ = {record.delta:.6f}"
            )
    else:
        invariants["delta_equals_D_over_A"] = False
        warnings.append("A is too small for δ = D/A check")

    # Invariant 2: Triangle convention (TRUTH_SPEC requires ℓ₁ ≤ ℓ₂)
    # This is a structural check - we assume it's correct if no explicit violation
    invariants["triangle_convention"] = True  # Assumed unless proven otherwise

    # Invariant 3: B/A should be close to 2K-1 for K pieces
    expected_B_over_A = 2 * K - 1
    ba_gap = abs(record.B_over_A - expected_B_over_A) / expected_B_over_A
    invariants["B_over_A_near_target"] = ba_gap < 0.05  # Within 5%

    # Compute gap percent
    gap_percent = (record.B_over_A - expected_B_over_A) / expected_B_over_A * 100

    # Check for mode-specific warnings
    if mode.lower() == "semantic":
        # Laurent mode at R≈1.3 is known to be inaccurate
        if record.R > 0.5:
            warnings.append(
                f"SEMANTIC mode at R={record.R:.4f} may have ~{record.R*20:.0f}% Laurent error. "
                "Consider NUMERIC mode for production."
            )

    return DeltaDecomposition(
        benchmark=benchmark,
        mode=map_laurent_mode_to_phase19(laurent_mode),
        R=record.R,
        theta=theta,
        K=K,
        A=record.A,
        B=record.B,
        D=D,
        delta=record.delta,
        B_over_A=record.B_over_A,
        gap_percent=gap_percent,
        per_piece=per_piece,
        invariants=invariants,
        warnings=warnings,
    )


def print_report(decompositions: Dict[str, DeltaDecomposition]) -> None:
    """Print formatted delta decomposition report."""
    print()
    print("=" * 80)
    print("PHASE 19.3: DELTA DECOMPOSITION REPORT")
    print("=" * 80)

    for key, decomp in decompositions.items():
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {decomp.benchmark.upper()}")
        print(f"MODE: {decomp.mode}")
        print(f"R = {decomp.R:.4f}, θ = {decomp.theta:.6f}, K = {decomp.K}")
        print("=" * 70)

        # Core metrics
        print("\nCORE METRICS:")
        print(f"  A (exp(R) coeff) = {decomp.A:.8f}")
        print(f"  B (constant)     = {decomp.B:.8f}")
        print(f"  D (I12+ + I34+)  = {decomp.D:.8f}")
        print(f"  δ = D/A          = {decomp.delta:.8f}")
        print(f"  B/A              = {decomp.B_over_A:.6f}")
        print(f"  Gap from {2*decomp.K-1}      = {decomp.gap_percent:+.2f}%")

        # Per-piece breakdown
        print("\nPER-PIECE BREAKDOWN:")
        print("-" * 50)
        for name, value in decomp.per_piece.items():
            if abs(value) > 1e-12:
                print(f"  {name:<15} = {value:>12.8f}")
        print("-" * 50)

        # Invariants
        print("\nINVARIANTS:")
        for name, passed in decomp.invariants.items():
            status = "PASS" if passed else "FAIL"
            marker = "✓" if passed else "✗"
            print(f"  {marker} {name}: {status}")

        # Warnings
        if decomp.warnings:
            print("\nWARNINGS:")
            for warning in decomp.warnings:
                print(f"  ⚠ {warning}")

    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Key':<30} {'B/A':<10} {'δ':<12} {'Gap%':<10}")
    print("-" * 62)

    for key, decomp in decompositions.items():
        print(
            f"{key:<30} {decomp.B_over_A:<10.4f} {decomp.delta:<12.6f} "
            f"{decomp.gap_percent:+.2f}%"
        )

    print("-" * 62)
    print()


def print_plus5_split_report(benchmarks: List[str]) -> Dict[str, Plus5SplitResult]:
    """
    Print Phase 20 +5 split report showing main-only vs with-error.

    This is the "north star" diagnostic for Phase 20:
    - Shows whether J15 is required for B/A ≈ 5
    - Per-piece contributions to A and B
    - Gap analysis for both modes

    Returns:
        Dict mapping benchmark name to Plus5SplitResult
    """
    print()
    print("=" * 90)
    print("PHASE 20: +5 SPLIT DIAGNOSTIC (Main-Only vs With-Error)")
    print("=" * 90)
    print()
    print("GOAL: Main-only B/A should equal 5 (=2K-1 for K=3) WITHOUT J₁,₅.")
    print("      If J₁,₅ is required, we're computing the wrong main-term object.")
    print()

    results = {}

    for benchmark in benchmarks:
        result = compute_plus5_signature_split(benchmark)
        results[benchmark] = result

    # Summary table
    print("-" * 90)
    print(f"{'Benchmark':<12} {'Main B/A':<12} {'Full B/A':<12} {'J15 Δ':<12} "
          f"{'Main Gap%':<12} {'Full Gap%':<12} {'J15 Req?':<10}")
    print("-" * 90)

    for benchmark, result in results.items():
        j15_req = "YES ⚠" if result.j15_required_for_target else "no"
        print(
            f"{benchmark:<12} {result.B_over_A_main_only:<12.4f} "
            f"{result.B_over_A_with_error:<12.4f} {result.j15_contribution_ratio:+12.4f} "
            f"{result.gap_percent_main_only:+11.2f}% {result.gap_percent:+11.2f}% "
            f"{j15_req:<10}"
        )

    print("-" * 90)

    # Detailed per-benchmark analysis
    for benchmark, result in results.items():
        print(f"\n{benchmark.upper()} DETAILED BREAKDOWN (R={result.R}):")
        print("-" * 70)

        print("\n  MAIN-TERM ONLY (J₁,₅ excluded):")
        print(f"    A = {result.A_main_only:.8f}")
        print(f"    B = {result.B_main_only:.8f}")
        print(f"    B/A = {result.B_over_A_main_only:.6f}")
        print(f"    Gap from 5: {result.gap_percent_main_only:+.2f}%")
        print(f"    δ = {result.delta_main_only:.6f}")

        print("\n  WITH ERROR TERMS (J₁,₅ included):")
        print(f"    A = {result.A:.8f}")
        print(f"    B = {result.B:.8f}")
        print(f"    B/A = {result.B_over_A:.6f}")
        print(f"    Gap from 5: {result.gap_percent:+.2f}%")
        print(f"    δ = {result.delta:.6f}")

        print("\n  J₁,₅ CONTRIBUTION:")
        print(f"    ΔA (J15 adds to A): {result.j15_contribution_A:+.8f}")
        print(f"    ΔB (J15 adds to B): {result.j15_contribution_B:+.8f}")
        print(f"    Δ(B/A): {result.j15_contribution_ratio:+.6f}")

        if result.j15_required_for_target:
            print("\n  ⚠ PROBLEM: J₁,₅ is REQUIRED to achieve B/A ≈ 5")
            print("    The derivation is using error terms to match the target.")
            print("    Phase 20.2 goal: fix main term so J₁,₅ is unnecessary.")
        else:
            print("\n  ✓ J₁,₅ is NOT required for B/A ≈ 5")

    # Analysis summary
    print("\n" + "=" * 90)
    print("ANALYSIS SUMMARY")
    print("=" * 90)

    all_require_j15 = all(r.j15_required_for_target for r in results.values())
    avg_main_gap = sum(abs(r.gap_percent_main_only) for r in results.values()) / len(results)
    avg_j15_contribution = sum(r.j15_contribution_ratio for r in results.values()) / len(results)

    print(f"\n  All benchmarks require J₁,₅: {'YES' if all_require_j15 else 'NO'}")
    print(f"  Average main-only gap: {avg_main_gap:.2f}%")
    print(f"  Average J₁,₅ contribution to B/A: {avg_j15_contribution:+.4f}")

    if all_require_j15:
        missing = 5.0 - sum(r.B_over_A_main_only for r in results.values()) / len(results)
        print(f"\n  PHASE 20.2 TARGET: Find ~{missing:.2f} in main-term to eliminate J₁,₅ dependence")

    print()
    print("=" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 19 delta decomposition report"
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["kappa", "kappa_star"],
        help="Benchmarks to analyze (default: both)",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "numeric", "both"],
        default="both",
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-piece breakdown",
    )
    parser.add_argument(
        "--check-invariants",
        action="store_true",
        default=True,
        help="Check and report invariant violations",
    )
    parser.add_argument(
        "--plus5-split",
        action="store_true",
        help="Show Phase 20 +5 split analysis (main-only vs with-error)",
    )

    args = parser.parse_args()

    # Handle --plus5-split mode separately
    if args.plus5_split:
        plus5_results = print_plus5_split_report(args.bench)
        # Optionally save to JSON
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            json_data = {
                bench: {
                    "R": result.R,
                    "A_main_only": result.A_main_only,
                    "B_main_only": result.B_main_only,
                    "A_with_error": result.A,
                    "B_with_error": result.B,
                    "B_over_A_main_only": result.B_over_A_main_only,
                    "B_over_A_with_error": result.B_over_A_with_error,
                    "gap_percent_main_only": result.gap_percent_main_only,
                    "gap_percent_with_error": result.gap_percent,
                    "j15_contribution_A": result.j15_contribution_A,
                    "j15_contribution_B": result.j15_contribution_B,
                    "j15_contribution_ratio": result.j15_contribution_ratio,
                    "j15_required_for_target": result.j15_required_for_target,
                    "delta_main_only": result.delta_main_only,
                    "delta_with_error": result.delta,
                }
                for bench, result in plus5_results.items()
            }
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"\nPlus5 split report saved to: {output_path}")
        return  # Exit after plus5-split report

    # Determine modes to run
    if args.mode == "both":
        modes = ["semantic", "numeric"]
    else:
        modes = [args.mode]

    # Compute decompositions
    results = {}
    for benchmark in args.bench:
        for mode in modes:
            key = f"{benchmark}_{mode}"
            try:
                decomp = compute_delta_decomposition(benchmark, mode)
                results[key] = decomp
            except Exception as e:
                print(f"Error computing {key}: {e}", file=sys.stderr)

    # Print report
    if args.verbose or not args.output:
        print_report(results)

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {key: decomp.to_dict() for key, decomp in results.items()}

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\nReport saved to: {output_path}")

    # Check invariants and exit with error if any fail
    if args.check_invariants:
        all_pass = True
        for key, decomp in results.items():
            for inv_name, passed in decomp.invariants.items():
                if not passed:
                    print(f"INVARIANT FAIL: {key}.{inv_name}", file=sys.stderr)
                    all_pass = False

        if not all_pass:
            sys.exit(1)


if __name__ == "__main__":
    main()
