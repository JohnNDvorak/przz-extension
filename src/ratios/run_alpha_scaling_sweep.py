"""
src/ratios/run_alpha_scaling_sweep.py
Phase 14K Task K2: Alpha Scaling Regime Experiment

PURPOSE:
========
PRZZ uses α = -R/L for small shifts where L ~ log(T).
This script sweeps L ∈ {20, 40, 80, 160} to see if δ(L) → 0.

INTERPRETATION:
===============
- If δ(L) → 0 as L grows: δ is asymptotic remainder (consistent with O(1/log N))
- If δ(L) doesn't shrink: missing main-term pieces

USAGE:
======
    python -m src.ratios.run_alpha_scaling_sweep

OUTPUT:
=======
- Console table of results
- JSON report in artifacts/alpha_scaling.json
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    LaurentMode,
    DEFAULT_LAURENT_MODE,
)


@dataclass
class ScalingResult:
    """Result of a single L-scaling point."""
    benchmark: str
    L: float
    R_original: float
    R_effective: float  # R/L
    delta: float
    B_over_A: float
    gap_percent: float
    A: float
    D: float


def compute_scaled_metrics(
    benchmark: str,
    L: float,
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> ScalingResult:
    """Compute delta metrics with R scaled by 1/L.

    PRZZ asymptotic regime uses α = -R/L where L ~ log(T).
    This simulates larger L (larger T) to see asymptotic behavior.
    """
    polys = load_przz_k3_polynomials(benchmark)
    R_original = polys.R
    R_effective = R_original / L

    # Compute with scaled R
    decomp = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R_effective,
        polys=polys,
        K=K,
        laurent_mode=DEFAULT_LAURENT_MODE,
    )

    gap = (decomp["B_over_A"] - 5.0) / 5.0 * 100

    return ScalingResult(
        benchmark=benchmark,
        L=L,
        R_original=R_original,
        R_effective=R_effective,
        delta=decomp["delta"],
        B_over_A=decomp["B_over_A"],
        gap_percent=gap,
        A=decomp["exp_coefficient"],
        D=decomp["D"],
    )


def run_L_sweep(
    L_values: List[float] = None,
    verbose: bool = True,
) -> Dict[str, List[ScalingResult]]:
    """Sweep L values for both benchmarks.

    Args:
        L_values: List of L scaling factors (default: [1, 2, 5, 10, 20])
        verbose: Print results to console

    Returns:
        Dictionary mapping benchmark name to list of ScalingResult
    """
    if L_values is None:
        L_values = [1.0, 2.0, 5.0, 10.0, 20.0]

    results = {"kappa": [], "kappa_star": []}

    for benchmark in ["kappa", "kappa_star"]:
        for L in L_values:
            result = compute_scaled_metrics(benchmark, L)
            results[benchmark].append(result)

    if verbose:
        print("=" * 80)
        print("ALPHA SCALING SWEEP: δ(L) Analysis")
        print("=" * 80)
        print()
        print("PRZZ uses α = -R/L where L ~ log(T)")
        print("Sweeping L to see asymptotic behavior of δ")
        print()

        for benchmark in ["kappa", "kappa_star"]:
            polys = load_przz_k3_polynomials(benchmark)
            print(f"--- {benchmark.upper()} (R_original = {polys.R}) ---")
            print()
            print(f"{'L':<8} {'R_eff':<10} {'delta':<12} {'B/A':<12} {'Gap':<10}")
            print("-" * 52)

            for r in results[benchmark]:
                print(f"{r.L:<8.1f} {r.R_effective:<10.4f} {r.delta:<12.6f} "
                      f"{r.B_over_A:<12.6f} {r.gap_percent:+.2f}%")

            print()

    return results


def analyze_scaling_trend(results: Dict[str, List[ScalingResult]], verbose: bool = True) -> Dict:
    """Analyze the trend of δ(L) to determine if it's asymptotic.

    If δ(L) → 0 as L → ∞: delta is O(1/L) asymptotic remainder
    If δ(L) → const: missing main-term pieces
    """
    analysis = {}

    for benchmark, data in results.items():
        L_vals = np.array([r.L for r in data])
        delta_vals = np.array([r.delta for r in data])

        # Check if delta decreases with L
        if len(data) >= 2:
            delta_first = data[0].delta
            delta_last = data[-1].delta

            # Relative change
            if abs(delta_first) > 1e-10:
                relative_change = (delta_last - delta_first) / delta_first
            else:
                relative_change = 0.0

            # Fit power law: delta ~ L^(-alpha)
            # log(delta) ~ -alpha * log(L)
            if all(d > 0 for d in delta_vals):
                log_L = np.log(L_vals)
                log_delta = np.log(delta_vals)
                # Linear regression
                slope, intercept = np.polyfit(log_L, log_delta, 1)
                power_exponent = -slope
            else:
                power_exponent = None

            analysis[benchmark] = {
                "delta_at_L1": delta_first,
                "delta_at_L_max": delta_last,
                "relative_change": relative_change,
                "power_exponent": power_exponent,
                "trend": "decreasing" if delta_last < delta_first else "increasing",
                "is_asymptotic": power_exponent is not None and power_exponent > 0.5,
            }

    if verbose:
        print("=" * 80)
        print("SCALING TREND ANALYSIS")
        print("=" * 80)
        print()

        for benchmark, data in analysis.items():
            print(f"--- {benchmark.upper()} ---")
            print(f"  δ(L=1): {data['delta_at_L1']:.6f}")
            print(f"  δ(L=max): {data['delta_at_L_max']:.6f}")
            print(f"  Relative change: {data['relative_change']*100:+.1f}%")
            print(f"  Trend: {data['trend']}")

            if data['power_exponent'] is not None:
                print(f"  Power law fit: δ ~ L^(-{data['power_exponent']:.2f})")

                if data['is_asymptotic']:
                    print(f"  → ASYMPTOTIC: δ vanishes as L → ∞ (O(1/L^{data['power_exponent']:.1f}))")
                else:
                    print(f"  → WEAK SCALING: δ shrinks slowly")
            else:
                print(f"  Power law fit: N/A (non-positive delta values)")

            print()

        # Overall interpretation
        print("INTERPRETATION:")
        print("-" * 50)

        kappa_asymp = analysis["kappa"]["is_asymptotic"]
        kappa_star_asymp = analysis["kappa_star"]["is_asymptotic"]

        if kappa_asymp and kappa_star_asymp:
            print("  BOTH benchmarks show asymptotic δ(L) → 0 behavior")
            print("  → The gap is O(1/log T) remainder, not missing main-term")
        elif kappa_star_asymp:
            print("  κ* shows asymptotic behavior, κ does not")
            print("  → κ may have structural gap at R=1.3036")
        else:
            print("  Neither benchmark shows clear asymptotic behavior")
            print("  → Gaps may be structural (missing terms or formula issues)")

        print()

    return analysis


def save_scaling_report(filename: str = None) -> str:
    """Save scaling sweep report to JSON."""
    if filename is None:
        os.makedirs("artifacts", exist_ok=True)
        filename = "artifacts/alpha_scaling.json"

    results = run_L_sweep(verbose=False)
    analysis = analyze_scaling_trend(results, verbose=False)

    report = {
        "phase": "14K",
        "description": "Alpha scaling regime experiment",
        "L_values": [1.0, 2.0, 5.0, 10.0, 20.0],
        "results": {
            benchmark: [asdict(r) for r in data]
            for benchmark, data in results.items()
        },
        "analysis": analysis,
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    return filename


def main():
    """Run full alpha scaling sweep."""
    print("=" * 80)
    print("PHASE 14K: ALPHA SCALING REGIME EXPERIMENT")
    print("=" * 80)
    print()

    # Run sweep
    results = run_L_sweep()

    # Analyze trend
    analyze_scaling_trend(results)

    # Save report
    path = save_scaling_report()
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    main()
