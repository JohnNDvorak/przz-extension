#!/usr/bin/env python3
"""
GPT Run 16: Quadrature Convergence Sweep

This script determines whether the ~1% c gap in tex_mirror is numerical
(quadrature resolution) or structural (assembly/amplitude model).

Key question: Does increasing quadrature points reduce the gap?

If gap shrinks toward 0: Increase n for production use.
If gap is stable: Gap is structural, Run 14+ is needed.

Usage:
    python run_gpt_run16_quadrature_convergence.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_tex_mirror


THETA = 4.0 / 7.0

# Targets from PRZZ
TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
        "kappa_target": 0.405,
    },
}


@dataclass
class ConvergenceResult:
    """Result for a single quadrature resolution."""
    n: int
    c: float
    kappa: float
    c_target: float
    c_gap_pct: float
    kappa_gap_pct: float
    delta_c_from_prev: float  # Change in c from previous n
    delta_c_pct: float  # Percentage change
    converged: bool  # True if |delta_c_pct| < threshold


def compute_convergence_sweep(
    R: float,
    c_target: float,
    kappa_target: float,
    polynomials: Dict,
    n_values: List[int],
    convergence_threshold: float = 0.1,  # 0.1% = converged
) -> List[ConvergenceResult]:
    """
    Run tex_mirror at multiple quadrature resolutions.

    Args:
        R: The R value for this benchmark
        c_target: Target c value
        kappa_target: Target kappa value
        polynomials: Dictionary with P1, P2, P3, Q
        n_values: List of quadrature point counts to test
        convergence_threshold: Percentage threshold for convergence

    Returns:
        List of ConvergenceResult, one per n value
    """
    results = []
    prev_c = None

    for n in n_values:
        # Compute tex_mirror result
        result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=n,
            polynomials=polynomials,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        c = result.c
        kappa = 1 - np.log(c) / R

        c_gap_pct = 100 * (c - c_target) / c_target
        kappa_gap_pct = 100 * (kappa - kappa_target) / kappa_target

        # Compute delta from previous n
        if prev_c is not None:
            delta_c = c - prev_c
            delta_c_pct = 100 * abs(delta_c) / abs(prev_c)
            converged = delta_c_pct < convergence_threshold
        else:
            delta_c = 0.0
            delta_c_pct = 0.0
            converged = False  # Can't determine convergence from first point

        results.append(ConvergenceResult(
            n=n,
            c=c,
            kappa=kappa,
            c_target=c_target,
            c_gap_pct=c_gap_pct,
            kappa_gap_pct=kappa_gap_pct,
            delta_c_from_prev=delta_c,
            delta_c_pct=delta_c_pct,
            converged=converged,
        ))

        prev_c = c

    return results


def print_convergence_table(
    benchmark_name: str,
    R: float,
    c_target: float,
    results: List[ConvergenceResult],
):
    """Print convergence results in a formatted table."""
    print(f"\nBenchmark: {benchmark_name} (R={R}, c_target={c_target:.6f})")
    print("-" * 80)
    print(f"{'n':<8} {'c':<14} {'κ':<10} {'c gap':<10} {'Δc(n-1)':<12} {'Converged?':<12}")
    print("-" * 80)

    for r in results:
        if r.n == results[0].n:
            delta_str = "—"
            conv_str = "—"
        else:
            delta_str = f"{r.delta_c_pct:.4f}%"
            conv_str = "YES" if r.converged else "NO"

        print(
            f"{r.n:<8} {r.c:<14.8f} {r.kappa:<10.6f} {r.c_gap_pct:+.3f}%    "
            f"{delta_str:<12} {conv_str:<12}"
        )


def analyze_convergence(results: List[ConvergenceResult]) -> Dict:
    """Analyze convergence behavior and determine gap type."""
    # Find first converged point
    first_converged_n = None
    for r in results:
        if r.converged:
            first_converged_n = r.n
            break

    # Get final result
    final = results[-1]

    # Check if gap is shrinking
    gaps = [r.c_gap_pct for r in results]
    gap_trend = gaps[-1] - gaps[0]  # Positive = gap grew, negative = gap shrunk

    # Determine if gap is structural
    # If converged but still have significant gap, it's structural
    is_structural = (
        first_converged_n is not None and
        abs(final.c_gap_pct) > 0.1  # Gap larger than 0.1%
    )

    return {
        "first_converged_n": first_converged_n,
        "final_gap_pct": final.c_gap_pct,
        "gap_trend": gap_trend,
        "is_structural": is_structural,
        "final_c": final.c,
        "final_kappa": final.kappa,
    }


def main():
    print("=" * 80)
    print("GPT Run 16: Quadrature Convergence Sweep")
    print("=" * 80)
    print()
    print("Testing whether ~1% c gap is numerical (quadrature) or structural.")
    print()

    # Load polynomials for both benchmarks
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Quadrature resolutions to test
    n_values = [40, 60, 80, 100, 120, 140]

    analyses = {}

    for bench_key, bench_data in TARGETS.items():
        polys = polys_kappa if bench_key == "kappa" else polys_kappa_star

        results = compute_convergence_sweep(
            R=bench_data["R"],
            c_target=bench_data["c_target"],
            kappa_target=bench_data["kappa_target"],
            polynomials=polys,
            n_values=n_values,
        )

        print_convergence_table(
            benchmark_name=bench_data["name"],
            R=bench_data["R"],
            c_target=bench_data["c_target"],
            results=results,
        )

        analysis = analyze_convergence(results)
        analyses[bench_key] = analysis

    # Print summary
    print()
    print("=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    for bench_key, analysis in analyses.items():
        name = TARGETS[bench_key]["name"]
        print(f"\n{name} Benchmark:")
        print(f"  First converged at: n={analysis['first_converged_n']}")
        print(f"  Final c gap: {analysis['final_gap_pct']:+.3f}%")
        print(f"  Gap trend (first→last): {analysis['gap_trend']:+.4f}%")
        print(f"  Gap type: {'STRUCTURAL' if analysis['is_structural'] else 'NUMERICAL'}")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    both_structural = all(a["is_structural"] for a in analyses.values())
    both_numerical = all(not a["is_structural"] for a in analyses.values())

    if both_structural:
        print("""
Quadrature has CONVERGED but residual gap remains.
The ~1% gap is STRUCTURAL, not numerical resolution.

This means:
1. Increasing n beyond 100 will NOT reduce the gap
2. The gap comes from the amplitude/assembly model
3. Run 14 (TeX-combined mirror) is needed to close the gap

Recommended production setting: n=100 (converged, minimal overhead)
""")
    elif both_numerical:
        print("""
Quadrature has NOT yet converged.
The gap appears to be NUMERICAL (resolution-limited).

This means:
1. Increasing n may further reduce the gap
2. Consider testing n=160, 200 for higher precision
3. Run 14 may still be valuable for understanding

Recommended: Test higher n values before concluding
""")
    else:
        print("""
Mixed results: One benchmark converged, one did not.

Further investigation needed:
1. The benchmarks may have different convergence rates
2. Consider testing higher n values
3. Run 14 should proceed regardless
""")


if __name__ == "__main__":
    main()
