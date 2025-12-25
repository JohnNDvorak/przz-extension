#!/usr/bin/env python3
"""
GPT Run 5 Diagnostic: R-Optimum Preservation Sweep

This script checks whether the exp_R_ref mode distorts the shape of κ(R).

Key question:
- For κ polynomials: Does R_opt ≈ 1.3036 under both exp_R and exp_R_ref modes?
- For κ* polynomials: Does R_opt ≈ 1.1167 under both modes?

If exp_R_ref shifts the optimum, it's a calibration artifact.

Usage:
    python run_gpt_run5_r_optimum_sweep.py
"""

import numpy as np
from typing import Dict, List, Tuple

from src.evaluate import compute_c_paper_tex_mirror
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)

THETA = 4.0 / 7.0
R_REF = 1.3036  # Reference R for exp_R_ref mode


def sweep_r_and_find_optimum(
    polynomials: Dict,
    R_values: np.ndarray,
    mode: str,
    benchmark_name: str,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Sweep R values and find the optimal κ.

    Args:
        polynomials: Polynomial dict with P1, P2, P3, Q
        R_values: Array of R values to test
        mode: "exp_R" or "exp_R_ref"
        benchmark_name: For display purposes

    Returns:
        R_opt: R value that maximizes κ
        results: List of (R, kappa, c) tuples
    """
    results = []

    for R in R_values:
        result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polynomials,
            tex_exp_component=mode,
            tex_R_ref=R_REF,
            n_quad_a=40,
        )

        # Compute κ = 1 - log(c) / R
        c = result.c
        if c > 0:
            kappa = 1 - np.log(c) / R
        else:
            kappa = -np.inf  # Invalid

        results.append((R, kappa, c))

    # Find optimum
    valid_results = [(R, k, c) for R, k, c in results if k > -np.inf]
    if valid_results:
        R_opt = max(valid_results, key=lambda x: x[1])[0]
    else:
        R_opt = np.nan

    return R_opt, results


def main():
    print("=" * 80)
    print("GPT Run 5: R-Optimum Preservation Diagnostic")
    print("=" * 80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # R sweep range
    R_values = np.linspace(0.8, 1.6, 41)

    modes = ["exp_R", "exp_R_ref"]
    benchmarks = [
        ("κ", polys_kappa, 1.3036),
        ("κ*", polys_kappa_star, 1.1167),
    ]

    # Run sweeps
    all_results = {}
    for bench_name, polys, R_expected in benchmarks:
        print(f"\n{'-'*40}")
        print(f"Benchmark: {bench_name} (expected R_opt ≈ {R_expected})")
        print(f"{'-'*40}")

        for mode in modes:
            print(f"\n  Mode: {mode}")
            R_opt, results = sweep_r_and_find_optimum(polys, R_values, mode, bench_name)
            all_results[(bench_name, mode)] = (R_opt, results)

            # Find κ at optimum
            kappa_at_opt = [k for R, k, c in results if R == R_opt][0]

            # Check deviation from expected
            deviation = R_opt - R_expected if not np.isnan(R_opt) else np.nan

            print(f"    R_opt = {R_opt:.4f} (deviation = {deviation:+.4f})")
            print(f"    κ at R_opt = {kappa_at_opt:.6f}")

            # Print some key points
            key_Rs = [0.9, 1.0, 1.1167, 1.3036, 1.4, 1.5]
            print(f"    Key R values:")
            for R_key in key_Rs:
                # Find closest R in results
                closest_idx = np.argmin(np.abs(R_values - R_key))
                R_act, kappa, c = results[closest_idx]
                print(f"      R={R_act:.4f}: κ={kappa:.6f}, c={c:.4f}")

    # Summary comparison
    print("\n")
    print("=" * 80)
    print("SUMMARY: R-Optimum Preservation")
    print("=" * 80)
    print()
    print(f"{'Benchmark':<8} {'Mode':<15} {'R_opt':>10} {'Expected':>10} {'Deviation':>12} {'Preserved?':>12}")
    print("-" * 70)

    for bench_name, polys, R_expected in benchmarks:
        for mode in modes:
            R_opt, _ = all_results[(bench_name, mode)]
            deviation = R_opt - R_expected if not np.isnan(R_opt) else np.nan
            preserved = "YES" if abs(deviation) < 0.05 else "NO"
            print(f"{bench_name:<8} {mode:<15} {R_opt:>10.4f} {R_expected:>10.4f} {deviation:>+12.4f} {preserved:>12}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("If exp_R_ref shifts the optimum significantly from the expected R:")
    print("  - This confirms exp_R_ref is a calibration hack")
    print("  - It works at the benchmark R but distorts the optimization landscape")
    print("  - Use exp_R for R-sweep/optimization, exp_R_ref only for benchmark reproduction")
    print()

    # κ(R) shape comparison
    print("=" * 80)
    print("κ(R) SHAPE COMPARISON")
    print("=" * 80)
    print()

    for bench_name, polys, R_expected in benchmarks:
        print(f"\n{bench_name} benchmark:")
        print(f"{'R':>8} {'κ(exp_R)':>12} {'κ(exp_R_ref)':>14} {'Δκ':>10}")
        print("-" * 48)

        _, results_exp_R = all_results[(bench_name, "exp_R")]
        _, results_exp_R_ref = all_results[(bench_name, "exp_R_ref")]

        for i, R in enumerate(R_values[::4]):  # Every 4th point
            idx = list(R_values).index(R) if R in R_values else np.argmin(np.abs(R_values - R))
            k1 = results_exp_R[idx][1]
            k2 = results_exp_R_ref[idx][1]
            dk = k2 - k1
            print(f"{R:>8.4f} {k1:>12.6f} {k2:>14.6f} {dk:>+10.6f}")


if __name__ == "__main__":
    main()
