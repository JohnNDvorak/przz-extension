#!/usr/bin/env python3
"""
Simple ratio breakdown using existing evaluate infrastructure.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from terms_k3_d1 import make_all_terms_k3_v2
from evaluate import evaluate_terms

def compute_c_by_pair(benchmark="kappa", n_quad=80):
    """Compute c with per-pair breakdown"""

    if benchmark == "kappa":
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        R = 1.3036
        theta = 4.0/7.0
        label = "κ"
    else:  # kappa_star
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
        R = 1.1167
        theta = 4.0/7.0
        label = "κ*"

    # Build polynomial dict
    poly_dict = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Get all terms
    all_terms_dict = make_all_terms_k3_v2(theta, R)

    print(f"\n{'=' * 60}")
    print(f"{label} BREAKDOWN BY PAIR (R={R}, n={n_quad})")
    print(f"{'=' * 60}\n")

    pair_results = {}
    total_c = 0.0

    for pair_key in ['11', '12', '13', '22', '23', '33']:
        terms = all_terms_dict[pair_key]
        result = evaluate_terms(terms, poly_dict, n_quad, return_breakdown=True, R=R, theta=theta)

        pair_c = result.total
        pair_results[pair_key] = pair_c
        total_c += pair_c

        print(f"Pair ({pair_key[0]},{pair_key[1]}): c = {pair_c:.8f}")

    print(f"\nTotal c: {total_c:.8f}")

    kappa_val = 1 - np.log(total_c) / R
    print(f"{label}: {kappa_val:.9f}")
    print()

    return pair_results, total_c, kappa_val

def main():
    print("=" * 80)
    print("κ vs κ* PER-PAIR RATIO ANALYSIS")
    print("=" * 80)

    # Compute both
    print("\n" + "="*80)
    pairs_k, total_k, kappa_k = compute_c_by_pair("kappa", n_quad=80)

    print("\n" + "="*80)
    pairs_s, total_s, kappa_s = compute_c_by_pair("kappa_star", n_quad=80)

    # Ratio analysis
    print("\n" + "=" * 80)
    print("RATIO ANALYSIS")
    print("=" * 80)
    print()

    print(f"{'Pair':<8} {'κ value':<15} {'κ* value':<15} {'Ratio κ/κ*':<15}")
    print("-" * 60)

    for pair_key in ['11', '12', '13', '22', '23', '33']:
        val_k = pairs_k[pair_key]
        val_s = pairs_s[pair_key]
        ratio = val_k / val_s if abs(val_s) > 1e-15 else float('inf')

        print(f"({pair_key[0]},{pair_key[1]})     {val_k:>14.8f}  {val_s:>14.8f}  {ratio:>14.6f}")

    print("-" * 60)
    total_ratio = total_k / total_s
    print(f"{'TOTAL':<8} {total_k:>14.8f}  {total_s:>14.8f}  {total_ratio:>14.6f}")
    print()

    # Target comparison
    c_target_k = 2.13745440613217263636
    c_target_s = 1.9379524124677437
    target_ratio = c_target_k / c_target_s

    print(f"Computed total c ratio: {total_ratio:.6f}")
    print(f"Target c ratio:         {target_ratio:.6f}")
    print(f"Relative error:         {100*(total_ratio - target_ratio)/target_ratio:+.2f}%")
    print()

    # Target individual c values
    print(f"Computed κ  c: {total_k:.10f}  (target: {c_target_k:.10f})")
    print(f"Computed κ* c: {total_s:.10f}  (target: {c_target_s:.10f})")
    print()

    error_k = (total_k - c_target_k) / c_target_k
    error_s = (total_s - c_target_s) / c_target_s

    print(f"κ  error: {100*error_k:+.2f}%")
    print(f"κ* error: {100*error_s:+.2f}%")
    print()

    # KEY INSIGHT
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("The per-pair ratios range from 0.43 to 2.83 (from u-integral analysis).")
    print("But the TOTAL ratio should match the target ratio ~1.103.")
    print()
    if abs(total_ratio - target_ratio) / target_ratio < 0.01:
        print("✓ TOTAL ratio matches target (< 1% error)")
        print("  → Individual pair variations are compensating correctly")
        print("  → This is EXPECTED behavior, not a bug")
    else:
        print("✗ TOTAL ratio DIFFERS from target (> 1% error)")
        print("  → Systematic error in the computation")
        print(f"  → Computed: {total_ratio:.6f}")
        print(f"  → Target:   {target_ratio:.6f}")
    print()

if __name__ == "__main__":
    main()
