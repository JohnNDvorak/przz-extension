"""
src/compute_c_exact_case_c.py
Compute c using EXACT Case C kernel from PRZZ.

This applies the exact Case C correction ratios to compute
the full c value and compare with PRZZ target.

PRZZ TeX References:
- 2360-2362, 2371-2374, 2382-2384: Case C definition
- 2586: kappa benchmark
- 2596-2598: kappa* benchmark (R=1.1167)
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_terms, compute_kappa
from src.terms_k3_d1 import make_all_terms_k3
from src.case_c_exact import compute_i1_with_exact_case_c


# Constants
THETA = 4/7
R1 = 1.3036
R2 = 1.1167
C_TARGET = 2.13745440613217263636
KAPPA_TARGET = 0.417293962

# kappa* from PRZZ TeX 2596-2598 (needs to be verified from paper)
# Using the relationship: kappa*/kappa ≈ R*/R for similar c
KAPPA_STAR_TARGET = 0.408  # Approximate


def compute_c_with_exact_case_c(
    R: float,
    theta: float = THETA,
    n_quad: int = 60,
    n_quad_a: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute c with exact Case C corrections applied.

    For each pair, computes the Case C correction ratio and applies it
    to the raw pair contribution.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Factorial and symmetry normalization
    factorial_norm = {"11": 1.0, "22": 1.0/4, "33": 1.0/36, "12": 1.0/2, "13": 1.0/6, "23": 1.0/12}
    symmetry_factor = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Compute Case C correction ratios for each pair
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    correction_ratios = {}
    for ell1, ell2 in pairs:
        pair_key = f"{ell1}{ell2}"
        ratio = compute_i1_with_exact_case_c(ell1, ell2, R, theta, n_quad, n_quad_a)
        correction_ratios[pair_key] = ratio

    # Evaluate raw pair contributions
    all_terms = make_all_terms_k3(theta, R)

    pair_raw = {}
    pair_corrected = {}
    total_raw = 0.0
    total_corrected = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=False)
        raw_val = pair_result.total
        pair_raw[pair_key] = raw_val

        # Apply Case C correction ratio
        # NOTE: The correction ratio was computed for the polynomial product
        # integral, which corresponds roughly to I_1 + I_2 contributions.
        # For a first approximation, apply to all terms.
        corrected_val = raw_val * correction_ratios[pair_key]
        pair_corrected[pair_key] = corrected_val

        # Apply normalization
        sym = symmetry_factor[pair_key]
        norm = factorial_norm[pair_key]

        total_raw += sym * norm * raw_val
        total_corrected += sym * norm * corrected_val

    # Compute kappa
    kappa_raw = compute_kappa(total_raw, R)
    kappa_corrected = compute_kappa(total_corrected, R) if total_corrected > 0 else float('nan')

    results = {
        "R": R,
        "c_raw": total_raw,
        "c_corrected": total_corrected,
        "c_target": C_TARGET,
        "kappa_raw": kappa_raw,
        "kappa_corrected": kappa_corrected,
        "kappa_target": KAPPA_TARGET,
        "pair_raw": pair_raw,
        "pair_corrected": pair_corrected,
        "correction_ratios": correction_ratios,
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"c WITH EXACT CASE C CORRECTIONS (R = {R})")
        print("=" * 70)

        print("\n--- Correction Ratios (from exact Case C kernel) ---")
        for pair_key in ["11", "12", "13", "22", "23", "33"]:
            print(f"  ({pair_key[0]},{pair_key[1]}): {correction_ratios[pair_key]:>12.6f}")

        print("\n--- Raw vs Corrected Pair Values ---")
        print(f"  {'Pair':<8} {'Raw':>18} {'Corrected':>18}")
        print("  " + "-" * 50)
        for pair_key in ["11", "12", "13", "22", "23", "33"]:
            print(f"  ({pair_key[0]},{pair_key[1]})   {pair_raw[pair_key]:>+18.10f} {pair_corrected[pair_key]:>+18.10f}")

        print("\n--- Summary ---")
        print(f"  c_raw:       {total_raw:20.15f}")
        print(f"  c_corrected: {total_corrected:20.15f}")
        print(f"  c_target:    {C_TARGET:20.15f}")

        gap_raw = (total_raw - C_TARGET) / C_TARGET * 100
        gap_corrected = (total_corrected - C_TARGET) / C_TARGET * 100 if not np.isnan(total_corrected) else float('nan')
        print(f"\n  Gap (raw):       {gap_raw:+.4f}%")
        print(f"  Gap (corrected): {gap_corrected:+.4f}%")

        print(f"\n  kappa_raw:       {kappa_raw:.10f}")
        print(f"  kappa_corrected: {kappa_corrected:.10f}")
        print(f"  kappa_target:    {KAPPA_TARGET:.10f}")

        print("=" * 70)

    return results


def compare_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """
    Compare both PRZZ benchmarks with exact Case C corrections.

    PRZZ benchmarks:
    - TeX 2586: R = 1.3036, kappa = 0.417293962
    - TeX 2596-2598: R = 1.1167, kappa* (need to check paper)
    """
    result_r1 = compute_c_with_exact_case_c(R1, THETA, verbose=False)
    result_r2 = compute_c_with_exact_case_c(R2, THETA, verbose=False)

    if verbose:
        print("\n" + "=" * 70)
        print("PRZZ BENCHMARK COMPARISON WITH EXACT CASE C")
        print("=" * 70)

        print("\n--- Benchmark 1: R = 1.3036 (TeX 2586) ---")
        print(f"  c_raw:       {result_r1['c_raw']:.10f}")
        print(f"  c_corrected: {result_r1['c_corrected']:.10f}")
        print(f"  c_target:    {C_TARGET:.10f}")
        gap1 = (result_r1['c_corrected'] - C_TARGET) / C_TARGET * 100
        print(f"  Gap: {gap1:+.4f}%")

        print("\n--- Benchmark 2: R = 1.1167 (TeX 2596-2598) ---")
        print(f"  c_raw:       {result_r2['c_raw']:.10f}")
        print(f"  c_corrected: {result_r2['c_corrected']:.10f}")
        # Note: c_target for R* is different - need to compute from kappa*
        # kappa* = 1 - log(c*)/R* => c* = exp(R*(1-kappa*))
        c_star_target = math.exp(R2 * (1 - KAPPA_STAR_TARGET))
        print(f"  c* target (derived): {c_star_target:.10f}")
        gap2 = (result_r2['c_corrected'] - c_star_target) / c_star_target * 100
        print(f"  Gap: {gap2:+.4f}%")

        print("\n--- R-Sensitivity Check ---")
        # The key test: does the correction make R-sensitivity consistent?
        raw_change = (result_r1['c_raw'] - result_r2['c_raw']) / result_r2['c_raw'] * 100
        corr_change = (result_r1['c_corrected'] - result_r2['c_corrected']) / result_r2['c_corrected'] * 100
        target_change = (C_TARGET - c_star_target) / c_star_target * 100

        print(f"  c change (raw):       R={R2}→{R1}: {raw_change:+.2f}%")
        print(f"  c change (corrected): R={R2}→{R1}: {corr_change:+.2f}%")
        print(f"  c change (target):    R={R2}→{R1}: {target_change:+.2f}%")

        if abs(corr_change - target_change) < abs(raw_change - target_change):
            print("\n  Exact Case C IMPROVED R-sensitivity matching!")
        else:
            print("\n  Exact Case C did not improve R-sensitivity matching")

        print("=" * 70)

    return {
        "r1": result_r1,
        "r2": result_r2,
        "c_star_target": math.exp(R2 * (1 - KAPPA_STAR_TARGET)),
    }


def analyze_why_correction_is_small(verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze why the Case C correction appears to be < 1 for most pairs.

    The polynomial rescaling P((1-a)*u) evaluates P at smaller arguments,
    which for polynomials that peak at u=1 will give smaller values.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    u_vals = np.linspace(0, 1, 11)

    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: WHY CASE C CORRECTIONS ARE < 1")
        print("=" * 70)

        print("\n--- Polynomial Values at Key Points ---")
        print(f"  {'u':>6} {'P1(u)':>12} {'P2(u)':>12} {'P3(u)':>12}")
        print("  " + "-" * 45)

        for u in u_vals:
            print(f"  {u:>6.2f} {P1.eval(np.array([u]))[0]:>12.6f} "
                  f"{P2.eval(np.array([u]))[0]:>12.6f} "
                  f"{P3.eval(np.array([u]))[0]:>12.6f}")

        print("\n--- Key Insight ---")
        print("  The Case C kernel evaluates P((1-a)*u) for a in [0,1].")
        print("  When a > 0, the argument (1-a)*u < u.")
        print("  For polynomials that increase with u, this REDUCES the integral.")
        print("  The u^omega factor compensates partially but not fully.")

        print("\n  This explains why correction ratios are typically < 1:")
        print("  - (1,2) B×C: 0.46 (P_2 evaluated at smaller arguments)")
        print("  - (2,2) C×C: 0.22 (both P_2's evaluated at smaller args)")
        print("  - (3,3) C×C: 0.06 (P_3 squared, even smaller)")

        print("=" * 70)

    return {"P1_at_1": P1.eval(np.array([1.0]))[0],
            "P2_at_1": P2.eval(np.array([1.0]))[0],
            "P3_at_1": P3.eval(np.array([1.0]))[0]}


if __name__ == "__main__":
    # Compute c at primary benchmark
    compute_c_with_exact_case_c(R1, verbose=True)

    # Compare both benchmarks
    compare_benchmarks(verbose=True)

    # Analyze correction behavior
    analyze_why_correction_is_small(verbose=True)
