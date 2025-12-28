#!/usr/bin/env python3
"""
Phase 35A: Run the log factor split analysis to verify correction factor.

This script:
1. Computes the main and cross contributions from the product rule split
2. Measures the correction factor directly from coefficient structure
3. Compares to the theoretical prediction 1 + θ/(2K(2K+1))

Goal: Determine if the measured correction matches the Beta moment prediction,
and whether any residual variation is R-dependent or polynomial-dependent.

Created: 2025-12-26 (Phase 35A)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def run_logfactor_split_analysis():
    """Run the log factor split analysis for both benchmarks."""
    from src.unified_s12.logfactor_split import (
        split_logfactor_for_pair,
        compute_aggregate_correction,
    )

    print("=" * 70)
    print("PHASE 35A: LOG FACTOR SPLIT ANALYSIS")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3
    n_quad = 60

    # Theoretical prediction
    predicted = 1 + theta / (2 * K * (2 * K + 1))
    print(f"THEORETICAL PREDICTION")
    print("-" * 50)
    print(f"  θ = {theta:.6f}")
    print(f"  K = {K}")
    print(f"  2K(2K+1) = {2*K*(2*K+1)}")
    print(f"  θ/(2K(2K+1)) = {theta/(2*K*(2*K+1)):.8f}")
    print(f"  Predicted correction = 1 + θ/(2K(2K+1)) = {predicted:.8f}")
    print()

    # Load polynomials for both benchmarks
    benchmarks = [
        {"name": "κ", "R": 1.3036, "loader": "kappa"},
        {"name": "κ*", "R": 1.1167, "loader": "kappa_star"},
    ]

    for bm in benchmarks:
        print(f"BENCHMARK: {bm['name']} (R={bm['R']})")
        print("-" * 50)

        # Load polynomials
        try:
            if bm["loader"] == "kappa":
                P1, P2, P3, Q = load_przz_polynomials()
            else:
                P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        except Exception as e:
            print(f"  ERROR loading polynomials: {e}")
            print()
            continue

        polynomials = {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "Q": Q,
        }

        # Run aggregate correction analysis
        try:
            result = compute_aggregate_correction(
                theta=theta,
                R=bm["R"],
                K=K,
                polynomials=polynomials,
                n_quad=n_quad,
            )

            print(f"  Total main coeff: {result['total_main']:.8f}")
            print(f"  Total cross (F_x+F_y): {result['total_cross']:.8f}")
            print(f"  Measured correction: {result['measured_correction']:.8f}")
            print(f"  Predicted correction: {result['predicted_correction']:.8f}")
            print(f"  Gap: {result['gap_pct']:+.4f}%")
            print()

            print(f"  Per-pair breakdown:")
            for pair_key, pr in result['pair_results'].items():
                print(f"    {pair_key}: main={pr['main']:+.6f}, "
                      f"cross_from_x(F_y)={pr['cross_from_x']:+.6f}, "
                      f"cross_from_y(F_x)={pr['cross_from_y']:+.6f}, "
                      f"corr={pr['correction']:.6f}")
            print()

        except Exception as e:
            print(f"  ERROR in analysis: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    print("INTERPRETATION")
    print("-" * 50)
    print("""
  If measured_correction ≈ predicted_correction for both benchmarks:
    → The Beta moment derivation is CONFIRMED
    → The ±0.15% "R-dependence" is NOT in the correction factor itself
    → Look elsewhere for the source (assembly, normalization, etc.)

  If measured_correction differs systematically between benchmarks:
    → The correction factor has polynomial dependence
    → Need to investigate P/Q structure effects

  If measured_correction varies with R (same polynomials):
    → True R-dependence exists
    → Need to derive the R-dependent correction term
""")


def run_per_pair_analysis():
    """Run detailed per-pair analysis."""
    from src.unified_s12.logfactor_split import split_logfactor_for_pair

    print()
    print("=" * 70)
    print("PER-PAIR LOG FACTOR SPLIT DETAILS")
    print("=" * 70)
    print()

    theta = 4 / 7
    K = 3
    R = 1.3036
    n_quad = 60

    # Load κ polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Q": Q,
    }

    print(f"Parameters: θ={theta:.4f}, R={R}, K={K}")
    print()

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        try:
            split = split_logfactor_for_pair(
                pair_key, theta, R, K, polynomials, n_quad
            )

            print(f"Pair ({pair_key}):")
            print(f"  Main (1/θ × F_xy):      {split.main_coeff:+.8f}")
            print(f"  Cross from x term (F_y): {split.cross_from_x_term:+.8f}")
            print(f"  Cross from y term (F_x): {split.cross_from_y_term:+.8f}")
            print(f"  Total:                   {split.total_coeff:+.8f}")
            print(f"  Correction factor:       {split.correction_factor:.8f}")
            print(f"  Predicted:               {split.predicted_correction:.8f}")
            print(f"  Gap:                     {split.correction_gap:+.4f}%")
            print()

        except Exception as e:
            print(f"  ERROR for pair {pair_key}: {e}")
            print()


if __name__ == "__main__":
    run_logfactor_split_analysis()
    run_per_pair_analysis()
