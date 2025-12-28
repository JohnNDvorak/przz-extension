#!/usr/bin/env python3
"""
scripts/run_phase25_gap_attribution.py
Phase 25.1: Gap Attribution Report Runner

PURPOSE:
========
Run the gap attribution harness on both kappa and kappa* benchmarks,
identifying precisely which component(s) account for the 5-7% gap
between unified S12 and empirical DSL evaluators.

USAGE:
======
    python scripts/run_phase25_gap_attribution.py
    python scripts/run_phase25_gap_attribution.py --n 80 --output gap_report.json
    python scripts/run_phase25_gap_attribution.py --bench kappa
    python scripts/run_phase25_gap_attribution.py --verbose

OUTPUT:
=======
Prints formatted gap attribution reports to stdout.
Optionally saves JSON for programmatic analysis.

NON-NEGOTIABLES (from Phase 24):
================================
- Uses normalization_mode="scalar" (first-principles, Phase 22)
- Does NOT use diagnostic_corrected (quarantined, Phase 24)
- Two-benchmark gate: reports on BOTH kappa AND kappa*
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator.gap_attribution import (
    run_dual_benchmark_gap_attribution,
    compute_gap_report,
    print_gap_report,
    print_dual_benchmark_summary,
    save_gap_reports_json,
    KAPPA_R,
    KAPPA_STAR_R,
    THETA,
    KAPPA_C_TARGET,
    KAPPA_STAR_C_TARGET,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def main():
    parser = argparse.ArgumentParser(
        description="Phase 25.1: Gap Attribution Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE:")[0],
    )
    parser.add_argument(
        "--n", type=int, default=60,
        help="Quadrature points (default: 60)"
    )
    parser.add_argument(
        "--bench", type=str, nargs="+", default=["kappa", "kappa_star"],
        choices=["kappa", "kappa_star"],
        help="Which benchmark(s) to run (default: both)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path (optional)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-pair breakdown"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 25.1: GAP ATTRIBUTION REPORT")
    print("=" * 70)
    print()
    print("NON-NEGOTIABLES:")
    print("  - normalization_mode='scalar' (first-principles, Phase 22)")
    print("  - diagnostic_corrected is QUARANTINED (Phase 24)")
    print("  - Two-benchmark gate enforced")
    print()
    print(f"Parameters: n_quad={args.n}")
    print()

    kappa_report = None
    kappa_star_report = None

    # Run requested benchmarks
    if "kappa" in args.bench and "kappa_star" in args.bench:
        # Run both at once
        print("Running BOTH benchmarks (two-benchmark gate)...")
        print()
        kappa_report, kappa_star_report = run_dual_benchmark_gap_attribution(
            n_quad=args.n,
            normalization_mode="scalar",
        )

        # Print individual reports
        print_gap_report(kappa_report)
        print()
        print_gap_report(kappa_star_report)
        print()

        # Print summary comparison
        print_dual_benchmark_summary(kappa_report, kappa_star_report)

    else:
        # Run single benchmark
        if "kappa" in args.bench:
            print("Running KAPPA benchmark only...")
            P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
            polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
            kappa_report = compute_gap_report(
                theta=THETA,
                R=KAPPA_R,
                n_quad=args.n,
                polynomials=polys,
                normalization_mode="scalar",
                benchmark_name="kappa",
                c_target=KAPPA_C_TARGET,
            )
            print_gap_report(kappa_report)

        if "kappa_star" in args.bench:
            print("Running KAPPA* benchmark only...")
            P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
            polys = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
            kappa_star_report = compute_gap_report(
                theta=THETA,
                R=KAPPA_STAR_R,
                n_quad=args.n,
                polynomials=polys,
                normalization_mode="scalar",
                benchmark_name="kappa_star",
                c_target=KAPPA_STAR_C_TARGET,
            )
            print_gap_report(kappa_star_report)

    # Save JSON if requested
    if args.output:
        if kappa_report and kappa_star_report:
            save_gap_reports_json(kappa_report, kappa_star_report, args.output)
        elif kappa_report:
            with open(args.output, 'w') as f:
                json.dump({"kappa": kappa_report.to_dict()}, f, indent=2)
            print(f"Saved kappa report to: {args.output}")
        elif kappa_star_report:
            with open(args.output, 'w') as f:
                json.dump({"kappa_star": kappa_star_report.to_dict()}, f, indent=2)
            print(f"Saved kappa_star report to: {args.output}")

    # Print verbose per-pair breakdown if requested
    if args.verbose:
        print()
        print("=" * 70)
        print("VERBOSE PER-PAIR BREAKDOWN")
        print("=" * 70)
        if kappa_report and kappa_report.unified_per_pair:
            print("\nKAPPA per-pair contributions (unified):")
            for pair_key, value in sorted(kappa_report.unified_per_pair.items()):
                pct = value / kappa_report.unified_S12_total * 100 if kappa_report.unified_S12_total else 0
                print(f"  {pair_key}: {value:>12.6f}  ({pct:>5.1f}% of S12)")

        if kappa_star_report and kappa_star_report.unified_per_pair:
            print("\nKAPPA* per-pair contributions (unified):")
            for pair_key, value in sorted(kappa_star_report.unified_per_pair.items()):
                pct = value / kappa_star_report.unified_S12_total * 100 if kappa_star_report.unified_S12_total else 0
                print(f"  {pair_key}: {value:>12.6f}  ({pct:>5.1f}% of S12)")


if __name__ == "__main__":
    main()
