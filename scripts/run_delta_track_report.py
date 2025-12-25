#!/usr/bin/env python3
"""
Run comprehensive delta tracking report.

This is the "one-shot debugging lens" for the +5 signature.
Whenever something changes, run this to see which piece moved.

Usage:
    python scripts/run_delta_track_report.py
    python scripts/run_delta_track_report.py --output artifacts/delta_report.json
    python scripts/run_delta_track_report.py --convergence
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ratios.delta_track import (
    run_delta_track_report,
    run_convergence_sweep,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive delta tracking report"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--convergence", "-c",
        action="store_true",
        help="Run convergence sweep instead of single-shot report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    if args.convergence:
        # Run convergence sweep for both benchmarks
        print("\n" + "=" * 70)
        print("CONVERGENCE SWEEP (quadrature_n)")
        print("=" * 70)

        for benchmark in ["kappa", "kappa_star"]:
            table = run_convergence_sweep(
                benchmark=benchmark,
                sweep_type="quadrature_n",
                values=[40, 60, 80, 100],
                laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
            )
            print(table.summary_table())
    else:
        # Run one-shot report
        output_path = Path(args.output) if args.output else None
        run_delta_track_report(
            verbose=not args.quiet,
            output_path=output_path,
        )

        if output_path:
            print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
