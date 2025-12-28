#!/usr/bin/env python3
"""
scripts/run_batch_scoring.py
Phase 47b: Batch Polynomial Scoring

Runs the scorer on all candidate files and produces a summary table.

Usage:
    python scripts/run_batch_scoring.py
    python scripts/run_batch_scoring.py --fast
    python scripts/run_batch_scoring.py --input-dir my_candidates/
    python scripts/run_batch_scoring.py --top 5  # Show detailed results for top 5

Created: 2025-12-28 (Phase 47b)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomial_scorer import PolynomialScorer, C_TARGET_KAPPA, KAPPA_TARGET
from src.polynomials import P1Polynomial, PellPolynomial, Polynomial


@dataclass
class CandidateResult:
    """Result for a single candidate."""
    id: str
    c: float
    kappa: float
    delta_c_pct: float  # % change from baseline
    ladder_pass: bool
    gate_pass: bool
    kappa_star_gap_pct: float
    error: Optional[str] = None


def load_candidate(filepath: Path) -> dict:
    """Load polynomial coefficients from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    P1 = P1Polynomial(tilde_coeffs=np.array(data["P1_tilde"]))
    P2 = PellPolynomial(tilde_coeffs=np.array(data["P2_tilde"]))
    P3 = PellPolynomial(tilde_coeffs=np.array(data["P3_tilde"]))
    Q = Polynomial(coeffs=np.array(data["Q_mono"]))

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def score_candidate(
    scorer: PolynomialScorer,
    filepath: Path,
    n_quad: int = 40,
) -> CandidateResult:
    """Score a single candidate file."""
    cand_id = filepath.stem

    try:
        polynomials = load_candidate(filepath)

        # Run contract validation
        contract = scorer.validate_contract(polynomials)
        if not contract.passed:
            return CandidateResult(
                id=cand_id,
                c=float('nan'),
                kappa=float('nan'),
                delta_c_pct=float('nan'),
                ladder_pass=False,
                gate_pass=False,
                kappa_star_gap_pct=float('nan'),
                error=f"Contract: {contract.message}",
            )

        # Run ladder
        ladder = scorer.run_microcase_ladder(polynomials, n_quad=n_quad)

        # Score at kappa benchmark R
        from src.kappa_engine import KappaEngine
        P1_coeffs = polynomials["P1"].tilde_coeffs.tolist()
        P2_coeffs = polynomials["P2"].tilde_coeffs.tolist()
        P3_coeffs = polynomials["P3"].tilde_coeffs.tolist()
        Q_coeffs = polynomials["Q"].coeffs.tolist()

        engine = KappaEngine(
            P1_coeffs=P1_coeffs,
            P2_coeffs=P2_coeffs,
            P3_coeffs=P3_coeffs,
            Q_coeffs=Q_coeffs,
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=n_quad,
        )
        result = engine.compute_kappa()

        # Also check kappa* for stability filter
        engine_star = KappaEngine(
            P1_coeffs=P1_coeffs,
            P2_coeffs=P2_coeffs,
            P3_coeffs=P3_coeffs,
            Q_coeffs=Q_coeffs,
            theta=4/7,
            K=3,
            R=1.1167,
            n_quad=n_quad,
        )
        result_star = engine_star.compute_kappa()
        kappa_star_target_c = 1.93795241121330
        kappa_star_gap = (result_star.c / kappa_star_target_c - 1) * 100

        delta_c_pct = (result.c / C_TARGET_KAPPA - 1) * 100
        gate_pass = abs(kappa_star_gap) < 10.0  # Soft constraint

        return CandidateResult(
            id=cand_id,
            c=result.c,
            kappa=result.kappa,
            delta_c_pct=delta_c_pct,
            ladder_pass=ladder.ladder_valid,
            gate_pass=gate_pass,
            kappa_star_gap_pct=kappa_star_gap,
        )

    except Exception as e:
        return CandidateResult(
            id=cand_id,
            c=float('nan'),
            kappa=float('nan'),
            delta_c_pct=float('nan'),
            ladder_pass=False,
            gate_pass=False,
            kappa_star_gap_pct=float('nan'),
            error=str(e),
        )


def print_results_table(results: List[CandidateResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("BATCH SCORING RESULTS")
    print("=" * 100)
    print(f"\n{'Candidate':<25} | {'c':>12} | {'Δc (%)':>10} | {'κ':>10} | {'Ladder':>6} | {'κ* Gap':>10} | {'Status':>8}")
    print("-" * 100)

    for r in results:
        if r.error:
            print(f"{r.id:<25} | {'ERROR':>12} | {'-':>10} | {'-':>10} | {'-':>6} | {'-':>10} | {'FAIL':>8}")
            print(f"    Error: {r.error}")
        else:
            ladder_str = "PASS" if r.ladder_pass else "FAIL"
            gate_str = "PASS" if r.gate_pass else "WARN"
            status = "OK" if r.ladder_pass and r.gate_pass else ("WARN" if r.ladder_pass else "FAIL")
            print(f"{r.id:<25} | {r.c:>12.6f} | {r.delta_c_pct:>+10.4f} | {r.kappa:>10.6f} | {ladder_str:>6} | {r.kappa_star_gap_pct:>+10.4f} | {status:>8}")

    print("-" * 100)


def print_summary(results: List[CandidateResult]):
    """Print summary statistics."""
    valid = [r for r in results if not r.error and r.ladder_pass]

    print(f"\nSUMMARY:")
    print(f"  Total candidates: {len(results)}")
    print(f"  Valid (ladder pass): {len(valid)}")
    print(f"  Gate pass (κ* < 10%): {len([r for r in valid if r.gate_pass])}")

    if valid:
        # Find best (lowest c)
        best = min(valid, key=lambda r: r.c)
        print(f"\n  Best candidate: {best.id}")
        print(f"    c = {best.c:.10f}")
        print(f"    Δc = {best.delta_c_pct:+.4f}%")
        print(f"    κ = {best.kappa:.10f}")

        # Improvements
        improvements = [r for r in valid if r.delta_c_pct < 0]
        if improvements:
            print(f"\n  Candidates with c < baseline: {len(improvements)}")
            for r in sorted(improvements, key=lambda x: x.c):
                gate_str = "" if r.gate_pass else " [κ* WARN]"
                print(f"    {r.id}: Δc = {r.delta_c_pct:+.4f}%{gate_str}")
        else:
            print(f"\n  No candidates improved c below baseline")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch scoring on candidate polynomial files"
    )
    parser.add_argument(
        "--input-dir",
        default="candidate_files",
        help="Input directory with candidate JSON files (default: candidate_files)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (n_quad=30)",
    )
    parser.add_argument(
        "--n-quad",
        type=int,
        default=40,
        help="Quadrature points (default: 40)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show detailed results for top N candidates",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    if args.fast:
        args.n_quad = 30

    # Find candidate files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found. Run generate_candidates.py first.")
        return 1

    candidate_files = sorted(input_dir.glob("*.json"))
    if not candidate_files:
        print(f"Error: No .json files found in {input_dir}")
        return 1

    print(f"Found {len(candidate_files)} candidate files in {input_dir}")
    print(f"Scoring with n_quad={args.n_quad}...")

    # Score all candidates
    scorer = PolynomialScorer(K=3, theta=4/7)
    results = []

    for i, filepath in enumerate(candidate_files):
        print(f"  [{i+1}/{len(candidate_files)}] {filepath.stem}...", end=" ", flush=True)
        result = score_candidate(scorer, filepath, n_quad=args.n_quad)
        results.append(result)
        if result.error:
            print("ERROR")
        else:
            print(f"c={result.c:.6f} (Δ={result.delta_c_pct:+.4f}%)")

    # Print table
    print_results_table(results)
    print_summary(results)

    # Output JSON
    if args.output:
        output_data = {
            "n_quad": args.n_quad,
            "baseline_c": C_TARGET_KAPPA,
            "baseline_kappa": KAPPA_TARGET,
            "results": [
                {
                    "id": r.id,
                    "c": r.c if np.isfinite(r.c) else None,
                    "kappa": r.kappa if np.isfinite(r.kappa) else None,
                    "delta_c_pct": r.delta_c_pct if np.isfinite(r.delta_c_pct) else None,
                    "ladder_pass": bool(r.ladder_pass),
                    "gate_pass": bool(r.gate_pass),
                    "kappa_star_gap_pct": r.kappa_star_gap_pct if np.isfinite(r.kappa_star_gap_pct) else None,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
