#!/usr/bin/env python3
"""
Phase 41.2: Component Quadrature Convergence

Sweeps n_quad in {40, 60, 80, 120, 160} for S12+, S12-, S34 independently.
Reports convergence deltas per component.

Decision Gate:
- If one component isn't converged -> fix numerics
- If all converged -> proceed to derived functional g(P,Q,...) mode OR audit S34

Created: 2025-12-27 (Phase 41)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_transform_paper_exact import compute_S12_paper_sum
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


@dataclass
class ConvergenceResult:
    """Convergence data for a single component at multiple n_quad values."""

    component: str          # "S12_plus", "S12_minus", "S34"
    benchmark: str
    n_values: List[int]     # [40, 60, 80, 120, 160]
    values: List[float]     # Component values at each n
    deltas: List[float]     # Relative change from previous n
    converged: bool         # True if final delta < threshold

    @property
    def final_value(self) -> float:
        return self.values[-1] if self.values else float('nan')

    @property
    def final_delta(self) -> float:
        return self.deltas[-1] if self.deltas else float('nan')


def compute_S34_component(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int,
) -> float:
    """Compute S34 = I3 + I4 using term DSL."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key] * symmetry_factor[pair_key]
        # I3 and I4 are indices 2 and 3
        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += norm * result.value

    return S34


def run_convergence_sweep(
    benchmark_name: str,
    R: float,
    theta: float,
    polynomials: Dict,
    n_values: List[int] = None,
    convergence_threshold: float = 1e-5,
) -> List[ConvergenceResult]:
    """
    Run quadrature convergence sweep for all components.

    Args:
        benchmark_name: Name of benchmark for labeling
        R: R parameter
        theta: theta parameter
        polynomials: Polynomial dictionary
        n_values: Quadrature point values to test
        convergence_threshold: Threshold for declaring convergence

    Returns:
        List of ConvergenceResult for each component
    """
    if n_values is None:
        n_values = [40, 60, 80, 120, 160]

    results = []

    for component in ["S12_plus", "S12_minus", "S34"]:
        values = []

        for n in n_values:
            if component == "S12_plus":
                val = compute_S12_paper_sum(R, theta, polynomials, n_quad=n)
            elif component == "S12_minus":
                val = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n)
            else:  # S34
                val = compute_S34_component(R, theta, polynomials, n)
            values.append(val)

        # Compute deltas (relative change from previous n)
        deltas = [float('nan')]  # First value has no delta
        for i in range(1, len(values)):
            if abs(values[i-1]) > 1e-15:
                delta = abs((values[i] - values[i-1]) / values[i-1])
            else:
                delta = abs(values[i] - values[i-1])
            deltas.append(delta)

        # Check convergence (last delta below threshold)
        converged = (len(deltas) >= 2 and deltas[-1] < convergence_threshold)

        results.append(ConvergenceResult(
            component=component,
            benchmark=benchmark_name,
            n_values=n_values,
            values=values,
            deltas=deltas,
            converged=converged,
        ))

    return results


def print_convergence_table(results: List[ConvergenceResult]) -> None:
    """Print convergence table for all components."""
    print()
    print("CONVERGENCE TABLE")
    print("-" * 100)

    # Group by benchmark
    benchmarks = {}
    for r in results:
        if r.benchmark not in benchmarks:
            benchmarks[r.benchmark] = []
        benchmarks[r.benchmark].append(r)

    for bm_name, bm_results in benchmarks.items():
        print(f"\nBenchmark: {bm_name}")
        print("-" * 100)

        # Header with n values
        n_vals = bm_results[0].n_values
        header = f"{'Component':<12}"
        for n in n_vals:
            header += f" | n={n:<8}"
        header += " | Converged"
        print(header)
        print("-" * 100)

        for r in bm_results:
            row = f"{r.component:<12}"
            for val in r.values:
                row += f" | {val:<10.6f}"
            row += f" | {'YES' if r.converged else 'NO'}"
            print(row)

        # Print deltas
        print()
        print("Relative deltas (change from previous n):")
        for r in bm_results:
            delta_str = f"  {r.component:<12}:"
            for i, delta in enumerate(r.deltas):
                if i == 0:
                    delta_str += f"  {'---':<10}"
                else:
                    delta_str += f"  {delta:<10.2e}"
            print(delta_str)


def print_convergence_summary(all_results: List[ConvergenceResult]) -> Dict:
    """Print summary of convergence status. Returns summary dict."""
    print()
    print("=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    print()

    not_converged = [r for r in all_results if not r.converged]

    summary = {
        "all_converged": len(not_converged) == 0,
        "not_converged": [(r.benchmark, r.component, r.final_delta) for r in not_converged],
    }

    if not_converged:
        print("WARNING: The following components did NOT converge:")
        for r in not_converged:
            print(f"  - {r.benchmark}/{r.component}: final delta = {r.final_delta:.2e}")
        print()
        print("DECISION: Fix numerics before proceeding to Route A/B")
    else:
        print("All components converged.")
        print()
        print("DECISION: Proceed to derived functional g(P,Q,R,K,theta) mode")
        print("          OR audit S34 derivation for missing normalization")

    return summary


def main():
    """Main entry point."""
    theta = 4 / 7
    n_values = [40, 60, 80, 120, 160]

    print("=" * 80)
    print("PHASE 41.2: COMPONENT QUADRATURE CONVERGENCE")
    print("=" * 80)
    print()
    print(f"Testing n_quad in {n_values}")
    print(f"Convergence threshold: 1e-5 (relative change)")
    print()

    all_results = []

    # Kappa benchmark
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R_kappa = 1.3036

    print("Computing convergence for kappa (R=1.3036)...")
    results_kappa = run_convergence_sweep(
        "kappa", R_kappa, theta, polynomials, n_values
    )
    all_results.extend(results_kappa)

    # Kappa* benchmark
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    R_kappa_star = 1.1167

    print("Computing convergence for kappa* (R=1.1167)...")
    results_kappa_star = run_convergence_sweep(
        "kappa_star", R_kappa_star, theta, polynomials_star, n_values
    )
    all_results.extend(results_kappa_star)

    print_convergence_table(all_results)
    summary = print_convergence_summary(all_results)

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("  If all converged:")
    print("    - Run Phase 41.1 (residual budget) to determine Route A or B")
    print("    - Route A: Derive polynomial-aware correction g(P,Q,R,K,theta)")
    print("    - Route B: Audit S34 for missing normalization/factor")
    print()
    print("  If not converged:")
    print("    - Increase n_quad further (200, 250)")
    print("    - Check for numerical instability in specific pairs")
    print("    - Consider adaptive quadrature for problematic integrals")
    print()

    return all_results, summary


if __name__ == "__main__":
    main()
