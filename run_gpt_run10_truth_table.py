#!/usr/bin/env python3
"""
GPT Run 10: Truth Table - Comprehensive Comparison of Parameter Combinations

This script tests all combinations of the new parameters:
- terms_version: "old" vs "v2"
- i2_source: "dsl" vs "direct_case_c"

Output table showing:
- c value for each combination
- % difference from PRZZ target
- κ value for each combination

Usage:
    python run_gpt_run10_truth_table.py
"""

import numpy as np
from typing import Dict

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import (
    evaluate_c_ordered,
    compute_c_paper_tex_mirror,
)


THETA = 4.0 / 7.0

# PRZZ Targets
TARGETS = {
    "kappa": {
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "R": 1.1167,
        "c_target": 1.93801,  # Approximate
        "kappa_target": 0.405,  # Approximate
    }
}


def compute_kappa(c: float, R: float) -> float:
    """Compute κ from c using κ = 1 - log(c)/R."""
    if c <= 0:
        return float('nan')
    return 1.0 - np.log(c) / R


def main():
    print("=" * 90)
    print("GPT Run 10: Truth Table - Parameter Combination Comparison")
    print("=" * 90)
    print()
    print("This table compares different parameter combinations:")
    print("  - terms_version: 'old' (current) vs 'v2' (V2 DSL)")
    print("  - i2_source: 'dsl' (DSL-based) vs 'direct_case_c' (proven)")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, TARGETS["kappa"]),
        ("κ*", polys_kappa_star, TARGETS["kappa_star"]),
    ]

    # Parameter combinations to test
    combinations = [
        ("old", "dsl"),
        ("old", "direct_case_c"),
        ("v2", "dsl"),
        ("v2", "direct_case_c"),
    ]

    print("=" * 90)
    print("SECTION 1: evaluate_c_ordered (base ordered evaluation)")
    print("=" * 90)
    print()
    print("Testing terms_version parameter (i2_source not applicable here)")
    print()

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]
        kappa_target = target["kappa_target"]

        print(f"\nBenchmark: {bench_name} (R={R})")
        print(f"Target: c={c_target:.6f}, κ={kappa_target:.6f}")
        print("-" * 70)
        print(f"{'terms_version':<15} {'c value':<15} {'c % diff':<12} {'κ value':<12}")
        print("-" * 70)

        for terms_ver in ["old", "v2"]:
            result = evaluate_c_ordered(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
                terms_version=terms_ver,
                kernel_regime="paper",
            )
            c_val = result.total
            kappa_val = compute_kappa(abs(c_val), R)
            c_diff = 100 * (c_val - c_target) / c_target

            print(f"{terms_ver:<15} {c_val:<15.6f} {c_diff:+.2f}%       {kappa_val:.6f}")

    print()
    print("=" * 90)
    print("SECTION 2: compute_c_paper_tex_mirror (full tex_mirror evaluation)")
    print("=" * 90)
    print()
    print("Testing all combinations of terms_version and i2_source")
    print()

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]
        kappa_target = target["kappa_target"]

        print(f"\nBenchmark: {bench_name} (R={R})")
        print(f"Target: c={c_target:.6f}, κ={kappa_target:.6f}")
        print("-" * 90)
        print(f"{'terms_version':<15} {'i2_source':<18} {'c value':<15} {'c % diff':<12} {'κ value':<12}")
        print("-" * 90)

        for terms_ver, i2_src in combinations:
            try:
                result = compute_c_paper_tex_mirror(
                    theta=THETA,
                    R=R,
                    n=60,
                    polynomials=polys,
                    terms_version=terms_ver,
                    i2_source=i2_src,
                    tex_exp_component="exp_R_ref",  # Use best-performing mode
                )
                c_val = result.c
                kappa_val = compute_kappa(c_val, R)
                c_diff = 100 * (c_val - c_target) / c_target

                print(f"{terms_ver:<15} {i2_src:<18} {c_val:<15.6f} {c_diff:+.2f}%       {kappa_val:.6f}")
            except Exception as e:
                print(f"{terms_ver:<15} {i2_src:<18} ERROR: {str(e)[:40]}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("""
KEY FINDINGS:

1. terms_version="old" vs "v2":
   - (1,1) pair: identical (both use power=2)
   - Non-diagonal pairs: V2 typically ~2-3% larger (different (1-u) power formula)

2. i2_source="dsl" vs "direct_case_c":
   - Should be identical when DSL is correct
   - direct_case_c is the "proven" source from Run 7

3. Best combination for PRZZ reproduction:
   - Use terms_version="old" to match current behavior
   - Use i2_source="direct_case_c" for proven I2 values

4. For future investigation:
   - V2 terms may be more mathematically correct
   - Need to verify against PRZZ TeX formulas

PROVEN COMPONENTS (Run 7-11):
- I1: Proven for all 9 pairs against V2 DSL (Run 9)
- I2: Proven for all 9 pairs with direct Case C (Run 7)
- I3: Proven for all 9 pairs against V2 DSL (Run 11)
- I4: Proven for all 9 pairs against V2 DSL (Run 11)

REMAINING UNCERTAINTY:
- Mirror assembly formula (how -R branch enters)
- Amplitude model (A1, A2 computation)
- V2 vs OLD structure for non-diagonal pairs
""")


if __name__ == "__main__":
    main()
