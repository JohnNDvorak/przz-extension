#!/usr/bin/env python3
"""
Overnight Optimization Script: P2, P3, and R search

This script performs comprehensive optimization:
1. P2/P3 coefficient search with fixed R=1.3036
2. R optimization (kept in separate leaderboard section)

All results are validated: only κ < 1 (c > 1) results are saved.

Created: 2025-12-31
"""

import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine
from src.polynomials import QPolynomial

# =============================================================================
# CONFIGURATION
# =============================================================================

# Best known P1 coefficients (from theta_aligned_search)
BEST_P1 = [-1.9, 0.98, 1.0, -0.6]

# Starting P2/P3 from a2a3_search
BASELINE_P2 = [1.048274, 1.319912, -0.940058]  # [b0, b1, b2]
BASELINE_P3 = [0.522811, -0.68651, -0.049923]  # [c0, c1, c2]

# Fixed Q in PRZZ basis (DO NOT CHANGE)
Q_BASIS = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}

# Evaluation settings
N_QUAD = 60
THETA = 4/7

# Output file
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"overnight_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_q_mono():
    """Get Q polynomial in monomial basis."""
    Q = QPolynomial(Q_BASIS)
    return Q.to_monomial().coeffs.tolist()

def evaluate_params(P1_tilde, P2_tilde, P3_tilde, R=1.3036, n_quad=N_QUAD):
    """Evaluate parameters and return (kappa, c) or None if invalid."""
    try:
        engine = KappaEngine(
            P1_coeffs=P1_tilde,
            P2_coeffs=P2_tilde,
            P3_coeffs=P3_tilde,
            Q_coeffs=get_q_mono(),
            theta=THETA,
            K=3,
            R=R,
            n_quad=n_quad,
        )
        result = engine.compute_kappa()

        # Validate: κ must be in (0, 1), c must be > 1
        if 0 < result.kappa < 1 and result.c > 1:
            return result.kappa, result.c
        return None
    except Exception as e:
        return None

def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {filename}")

# =============================================================================
# PHASE 1: P2/P3 OPTIMIZATION
# =============================================================================

def optimize_p2p3():
    """Optimize P2 and P3 coefficients with fixed P1 and R."""
    print("=" * 70)
    print("PHASE 1: P2/P3 Coefficient Optimization")
    print("=" * 70)
    print(f"Fixed P1 = {BEST_P1}")
    print(f"Fixed R = 1.3036")
    print()

    # Current best baseline
    baseline = evaluate_params(BEST_P1, BASELINE_P2, BASELINE_P3)
    if baseline:
        print(f"Baseline: κ={baseline[0]:.6f}, c={baseline[1]:.6f}")
    else:
        print("ERROR: Baseline evaluation failed!")
        return []

    best_results = []
    best_kappa = baseline[0]

    # P2 has 3 coefficients: b0, b1, b2
    # P3 has 3 coefficients: c0, c1, c2

    # Search ranges (±50% of baseline values, capped at reasonable range)
    p2_ranges = [
        (BASELINE_P2[0] * 0.5, BASELINE_P2[0] * 1.5) if BASELINE_P2[0] > 0 else (BASELINE_P2[0] * 1.5, BASELINE_P2[0] * 0.5),
        (BASELINE_P2[1] * 0.5, BASELINE_P2[1] * 1.5),
        (BASELINE_P2[2] * 1.5, BASELINE_P2[2] * 0.5) if BASELINE_P2[2] < 0 else (BASELINE_P2[2] * 0.5, BASELINE_P2[2] * 1.5),
    ]

    p3_ranges = [
        (BASELINE_P3[0] * 0.5, BASELINE_P3[0] * 1.5) if BASELINE_P3[0] > 0 else (BASELINE_P3[0] * 1.5, BASELINE_P3[0] * 0.5),
        (BASELINE_P3[1] * 1.5, BASELINE_P3[1] * 0.5) if BASELINE_P3[1] < 0 else (BASELINE_P3[1] * 0.5, BASELINE_P3[1] * 1.5),
        (BASELINE_P3[2] * 1.5, BASELINE_P3[2] * 0.5) if BASELINE_P3[2] < 0 else (BASELINE_P3[2] * 0.5, BASELINE_P3[2] * 1.5),
    ]

    # Stage 1: Individual coefficient sweeps
    print("\nStage 1: Individual P2/P3 coefficient sweeps")

    # P2 sweeps
    for idx in range(3):
        print(f"\nSweeping P2[{idx}] (b{idx})...")
        low, high = p2_ranges[idx]
        for val in np.linspace(low, high, 21):
            P2_test = BASELINE_P2.copy()
            P2_test[idx] = val
            result = evaluate_params(BEST_P1, P2_test, BASELINE_P3)
            if result and result[0] > best_kappa:
                print(f"  NEW BEST: b{idx}={val:.4f} → κ={result[0]:.6f}, c={result[1]:.4f}")
                best_kappa = result[0]
                best_results.append({
                    "kappa": result[0],
                    "c": result[1],
                    "P2_tilde": P2_test,
                    "P3_tilde": BASELINE_P3.copy(),
                    "change": f"P2[{idx}]={val:.4f}"
                })

    # P3 sweeps
    for idx in range(3):
        print(f"\nSweeping P3[{idx}] (c{idx})...")
        low, high = p3_ranges[idx]
        for val in np.linspace(low, high, 21):
            P3_test = BASELINE_P3.copy()
            P3_test[idx] = val
            result = evaluate_params(BEST_P1, BASELINE_P2, P3_test)
            if result and result[0] > best_kappa:
                print(f"  NEW BEST: c{idx}={val:.4f} → κ={result[0]:.6f}, c={result[1]:.4f}")
                best_kappa = result[0]
                best_results.append({
                    "kappa": result[0],
                    "c": result[1],
                    "P2_tilde": BASELINE_P2.copy(),
                    "P3_tilde": P3_test,
                    "change": f"P3[{idx}]={val:.4f}"
                })

    # Stage 2: Joint P2, P3 grid search (coarse)
    print("\n\nStage 2: Joint P2/P3 grid search (b0, c0 only)")

    for b0 in np.linspace(p2_ranges[0][0], p2_ranges[0][1], 11):
        for c0 in np.linspace(p3_ranges[0][0], p3_ranges[0][1], 11):
            P2_test = BASELINE_P2.copy()
            P3_test = BASELINE_P3.copy()
            P2_test[0] = b0
            P3_test[0] = c0
            result = evaluate_params(BEST_P1, P2_test, P3_test)
            if result and result[0] > best_kappa:
                print(f"  NEW BEST: b0={b0:.4f}, c0={c0:.4f} → κ={result[0]:.6f}")
                best_kappa = result[0]
                best_results.append({
                    "kappa": result[0],
                    "c": result[1],
                    "P2_tilde": P2_test,
                    "P3_tilde": P3_test,
                    "change": f"b0={b0:.4f}, c0={c0:.4f}"
                })

    print(f"\nPhase 1 complete. Best κ = {best_kappa:.6f}")
    return best_results

# =============================================================================
# PHASE 2: R OPTIMIZATION
# =============================================================================

def optimize_R():
    """Optimize R parameter with best P1/P2/P3."""
    print("\n")
    print("=" * 70)
    print("PHASE 2: R Parameter Optimization")
    print("=" * 70)
    print("NOTE: These results use non-standard R values")
    print()

    # Use best known polynomials
    P1 = BEST_P1
    P2 = BASELINE_P2
    P3 = BASELINE_P3

    # Baseline at R=1.3036
    baseline = evaluate_params(P1, P2, P3, R=1.3036)
    if baseline:
        print(f"Baseline (R=1.3036): κ={baseline[0]:.6f}, c={baseline[1]:.6f}")

    r_results = []

    # Search R from 1.0 to 1.6
    print("\nSweeping R from 1.0 to 1.6...")

    for R in np.linspace(1.0, 1.6, 31):
        result = evaluate_params(P1, P2, P3, R=R)
        if result:
            if result[0] > (r_results[-1]["kappa"] if r_results else 0):
                print(f"  R={R:.4f}: κ={result[0]:.6f}, c={result[1]:.6f}")
            r_results.append({
                "R": R,
                "kappa": result[0],
                "c": result[1],
                "P1_tilde": P1,
                "P2_tilde": P2,
                "P3_tilde": P3,
            })

    # Find best R
    if r_results:
        best = max(r_results, key=lambda x: x["kappa"])
        print(f"\nBest R = {best['R']:.4f}: κ = {best['kappa']:.6f}, c = {best['c']:.6f}")

    return r_results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("OVERNIGHT OPTIMIZATION: P2, P3, and R")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "P1_fixed": BEST_P1,
            "Q_basis": Q_BASIS,
            "theta": THETA,
            "n_quad": N_QUAD,
        },
        "p2p3_optimization": [],
        "R_optimization": [],
    }

    # Phase 1: P2/P3
    p2p3_results = optimize_p2p3()
    all_results["p2p3_optimization"] = p2p3_results

    # Save intermediate
    save_results(all_results, OUTPUT_FILE)

    # Phase 2: R
    r_results = optimize_R()
    all_results["R_optimization"] = r_results

    # Final save
    all_results["completed"] = datetime.now().isoformat()
    save_results(all_results, OUTPUT_FILE)

    # Summary
    print("\n")
    print("=" * 70)
    print("OVERNIGHT OPTIMIZATION COMPLETE")
    print("=" * 70)

    if p2p3_results:
        best_p2p3 = max(p2p3_results, key=lambda x: x["kappa"])
        print(f"Best P2/P3 result: κ={best_p2p3['kappa']:.6f}")
        print(f"  Change: {best_p2p3['change']}")

    if r_results:
        best_r = max(r_results, key=lambda x: x["kappa"])
        print(f"\nBest R result: R={best_r['R']:.4f}, κ={best_r['kappa']:.6f}")

    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
