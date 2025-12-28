#!/usr/bin/env python3
"""
scripts/score_polynomials.py
Phase 47: Polynomial Scoring CLI

Scores polynomial candidates using the validated THETA_CUBED formula.

Usage:
    # Score PRZZ polynomials (default)
    python scripts/score_polynomials.py

    # Score from JSON file
    python scripts/score_polynomials.py --input candidates.json

    # Custom R range
    python scripts/score_polynomials.py --R-min 1.0 --R-max 1.5

    # Fast mode (fewer quadrature points)
    python scripts/score_polynomials.py --fast

    # Output to JSON
    python scripts/score_polynomials.py --output results.json

Created: 2025-12-27 (Phase 47)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomial_scorer import (
    PolynomialScorer,
    score_przz_polynomials,
    FullScoringResult,
    KAPPA_TARGET,
    C_TARGET_KAPPA,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    P1Polynomial,
    PellPolynomial,
    Polynomial,
)
import numpy as np


def load_polynomials_from_json(filepath: str) -> Dict:
    """
    Load polynomial coefficients from JSON file.

    Expected format:
    {
        "P1_tilde": [c0, c1, ...],
        "P2_tilde": [c0, c1, ...],
        "P3_tilde": [c0, c1, ...],
        "Q_mono": [c0, c1, ...]
    }
    """
    with open(filepath) as f:
        data = json.load(f)

    P1 = P1Polynomial(tilde_coeffs=np.array(data["P1_tilde"]))
    P2 = PellPolynomial(tilde_coeffs=np.array(data["P2_tilde"]))
    P3 = PellPolynomial(tilde_coeffs=np.array(data["P3_tilde"]))
    Q = Polynomial(coeffs=np.array(data["Q_mono"]))

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def result_to_dict(result: FullScoringResult) -> Dict:
    """Convert scoring result to JSON-serializable dict."""
    return {
        "summary": {
            "R_opt": result.R_opt,
            "c_opt": result.c_opt,
            "kappa_opt": result.kappa_opt,
            "overall_valid": result.overall_valid,
        },
        "contract": {
            "passed": result.contract.passed,
            "message": result.contract.message,
            "checks": result.contract.checks,
        },
        "ladder": {
            "ladder_valid": result.ladder.ladder_valid,
            "validation_message": result.ladder.validation_message,
            "cases": {
                name: {
                    "c": case.c,
                    "kappa": case.kappa,
                    "f_I1": case.f_I1,
                    "success": case.success,
                }
                for name, case in result.ladder.cases.items()
            },
        },
        "fast_sweep": {
            "R_opt": result.fast_sweep.R_opt,
            "c_opt": result.fast_sweep.c_opt,
            "kappa_opt": result.fast_sweep.kappa_opt,
            "n_points": len(result.fast_sweep.points),
        },
        "confirmation": {
            "R": result.confirmation.R,
            "converged": result.confirmation.converged,
            "max_drift": result.confirmation.max_drift,
            "levels": {
                str(n): {"c": pt.c, "kappa": pt.kappa}
                for n, pt in result.confirmation.levels.items()
            },
        },
        "two_benchmark": {
            "gate_passed": result.two_benchmark.gate_passed,
            "kappa": result.two_benchmark.kappa,
            "kappa_star": result.two_benchmark.kappa_star,
            "ratio": result.two_benchmark.ratio,
            "ratio_target": result.two_benchmark.ratio_target,
            "ratio_gap_pct": result.two_benchmark.ratio_gap_pct,
        },
    }


def print_result(result: FullScoringResult):
    """Print scoring result to console."""
    print("\n" + "=" * 70)
    print("POLYNOMIAL SCORING RESULTS")
    print("=" * 70)

    # Contract
    status = "PASS" if result.contract.passed else "FAIL"
    print(f"\n[Stage A] Contract Validation: {status}")
    if not result.contract.passed:
        print(f"  {result.contract.message}")
    else:
        checks_str = ", ".join(
            f"{k}={v}" for k, v in list(result.contract.checks.items())[:5]
        )
        print(f"  Checks: {checks_str}...")

    # Ladder
    status = "PASS" if result.ladder.ladder_valid else "FAIL"
    print(f"\n[Stage B] Microcase Ladder: {status}")
    for name, case in result.ladder.cases.items():
        if case.success:
            print(f"  {name}: c={case.c:.6f}, kappa={case.kappa:.6f}")
        else:
            print(f"  {name}: FAILED - {case.error}")

    # Fast sweep
    print(f"\n[Stage C] Fast R-Sweep: {len(result.fast_sweep.points)} points")
    print(f"  R_opt = {result.fast_sweep.R_opt:.4f}")
    print(f"  c_opt = {result.fast_sweep.c_opt:.6f}")
    print(f"  kappa_opt = {result.fast_sweep.kappa_opt:.6f}")

    # Confirmation
    status = "CONVERGED" if result.confirmation.converged else "NOT CONVERGED"
    print(f"\n[Stage D] Slow Confirmation: {status}")
    print(f"  max_drift = {result.confirmation.max_drift:.2e}")
    for n, pt in sorted(result.confirmation.levels.items()):
        print(f"  n={n}: c={pt.c:.10f}, kappa={pt.kappa:.10f}")

    # Two-benchmark gate
    status = "PASS" if result.two_benchmark.gate_passed else "FAIL"
    print(f"\n[Gate] Two-Benchmark Gate: {status}")
    kappa = result.two_benchmark.kappa
    kappa_star = result.two_benchmark.kappa_star
    print(f"  kappa: c={kappa['c']:.6f} (target {kappa['target_c']:.6f}, gap {kappa['c_gap_pct']:+.4f}%)")
    print(f"  kappa*: c={kappa_star['c']:.6f} (target {kappa_star['target_c']:.6f}, gap {kappa_star['c_gap_pct']:+.4f}%)")
    print(f"  ratio: {result.two_benchmark.ratio:.4f} (target {result.two_benchmark.ratio_target:.4f}, gap {result.two_benchmark.ratio_gap_pct:+.4f}%)")

    # Summary
    print("\n" + "=" * 70)
    status = "VALID" if result.overall_valid else "INVALID"
    print(f"OVERALL: {status}")
    print(f"  R_opt = {result.R_opt:.4f}")
    print(f"  c_opt = {result.c_opt:.10f}")
    print(f"  kappa_opt = {result.kappa_opt:.10f}")

    # Improvement potential
    c_improvement_pct = (C_TARGET_KAPPA - result.c_opt) / C_TARGET_KAPPA * 100
    kappa_improvement = result.kappa_opt - KAPPA_TARGET
    print(f"\n  vs PRZZ target (kappa={KAPPA_TARGET}):")
    print(f"    c improvement: {c_improvement_pct:+.4f}%")
    print(f"    kappa improvement: {kappa_improvement:+.8f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Score polynomial candidates for kappa optimization"
    )
    parser.add_argument(
        "--input",
        help="JSON file with polynomial coefficients (default: PRZZ polynomials)",
    )
    parser.add_argument(
        "--R-min", type=float, default=0.9, help="Minimum R for sweep (default: 0.9)"
    )
    parser.add_argument(
        "--R-max", type=float, default=1.6, help="Maximum R for sweep (default: 1.6)"
    )
    parser.add_argument(
        "--n-fast", type=int, default=30, help="Quadrature for fast sweep (default: 30)"
    )
    parser.add_argument(
        "--n-slow",
        type=int,
        nargs="+",
        default=[60, 80, 120],
        help="Quadrature levels for confirmation (default: 60 80 120)",
    )
    parser.add_argument(
        "--output", help="Output JSON file (default: print to console)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer quadrature points (30, 40, 50)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only output JSON, no console output"
    )

    args = parser.parse_args()

    # Adjust for fast mode
    if args.fast:
        args.n_fast = 20
        args.n_slow = [30, 40, 50]

    # Load polynomials
    if args.input:
        print(f"Loading polynomials from {args.input}...")
        polynomials = load_polynomials_from_json(args.input)
    else:
        print("Using PRZZ kappa polynomials...")
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Create scorer
    scorer = PolynomialScorer(K=3, theta=4/7)

    # Run full scoring pipeline
    print(f"Running full scoring pipeline...")
    print(f"  R range: [{args.R_min}, {args.R_max}]")
    print(f"  Fast quadrature: n={args.n_fast}")
    print(f"  Confirmation levels: {args.n_slow}")

    result = scorer.score_full(
        polynomials,
        R_range=(args.R_min, args.R_max),
        n_fast=args.n_fast,
        n_slow_levels=args.n_slow,
    )

    # Output
    if args.output:
        output_data = result_to_dict(result)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")

    if not args.quiet:
        print_result(result)

    # Exit code
    return 0 if result.overall_valid else 1


if __name__ == "__main__":
    sys.exit(main())
