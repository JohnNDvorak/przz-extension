#!/usr/bin/env python3
"""
Phase 37 Verification: Match Phase 35 Methodology

Phase 35 computed the ratio c_derived/c_empirical for different microcases.
This script verifies that our frozen-Q experiment matches Phase 35 findings.

Phase 35 results (for reference):
| Microcase      | Ratio  | Gap from 1.01361 |
|----------------|--------|------------------|
| P=Q=1          | 1.00853| -0.50%           |
| P=real, Q=1    | 1.01406| +0.05%           |
| P=1, Q=real    | 1.00920| -0.43%           |
| P=Q=real       | 1.01233| -0.13%           |

This script computes the same ratios using our frozen-Q infrastructure.

Created: 2025-12-26 (Phase 37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials, Polynomial
from src.evaluator.decomposition import compute_decomposition


def main():
    print("=" * 70)
    print("PHASE 37 VERIFICATION: MATCHING PHASE 35 METHODOLOGY")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()

    # Create Q=1 polynomial
    Q_one = Polynomial(np.array([1.0]))

    # Create P=1 polynomials (degree 0, constant 1)
    P_one = Polynomial(np.array([1.0]))

    theta = 4 / 7
    R = 1.3036
    K = 3
    n_quad = 60

    corr_beta = 1 + theta / (2 * K * (2 * K + 1))

    print(f"Parameters: θ={theta:.6f}, R={R}, K={K}")
    print(f"corr_beta = 1 + θ/42 = {corr_beta:.8f}")
    print()

    print("PHASE 35 REFERENCE RESULTS:")
    print("-" * 70)
    print("| Microcase      | Ratio  | Gap from Beta |")
    print("|----------------|--------|---------------|")
    print("| P=Q=1          | 1.00853| -0.50%        |")
    print("| P=real, Q=1    | 1.01406| +0.05%        |")
    print("| P=1, Q=real    | 1.00920| -0.43%        |")
    print("| P=Q=real       | 1.01233| -0.13%        |")
    print()

    # Define microcases
    microcases = [
        ("P=Q=1", {"P1": P_one, "P2": P_one, "P3": P_one, "Q": Q_one}),
        ("P=real, Q=1", {"P1": P1, "P2": P2, "P3": P3, "Q": Q_one}),
        ("P=1, Q=real", {"P1": P_one, "P2": P_one, "P3": P_one, "Q": Q}),
        ("P=Q=real", {"P1": P1, "P2": P2, "P3": P3, "Q": Q}),
    ]

    print("COMPUTING WITH DECOMPOSITION EVALUATOR:")
    print("-" * 70)
    print(f"{'Microcase':<15} | {'c_empirical':<12} | {'c_derived':<12} | {'ratio':<10} | {'gap':<10}")
    print("-" * 70)

    for name, polys in microcases:
        # Compute with empirical formula (m = exp(R) + 5)
        decomp_emp = compute_decomposition(
            theta=theta, R=R, K=K, polynomials=polys,
            kernel_regime="paper", n_quad=n_quad,
            mirror_formula="empirical",
        )
        c_empirical = decomp_emp.total

        # Compute with derived formula (m = [1+θ/42]×[exp(R)+5])
        decomp_der = compute_decomposition(
            theta=theta, R=R, K=K, polynomials=polys,
            kernel_regime="paper", n_quad=n_quad,
            mirror_formula="derived",
        )
        c_derived = decomp_der.total

        ratio = c_derived / c_empirical if abs(c_empirical) > 1e-15 else float('inf')
        gap = (ratio - corr_beta) / corr_beta * 100

        print(f"{name:<15} | {c_empirical:+.6f}   | {c_derived:+.6f}   | {ratio:.6f}  | {gap:+.4f}%")

    print()
    print("INTERPRETATION")
    print("-" * 70)
    print("  Compare the computed ratios to Phase 35 reference:")
    print("  - P=Q=1:       Should be ~1.00853 (-0.50%)")
    print("  - P=real, Q=1: Should be ~1.01406 (+0.05%) ← Q=1 baseline")
    print("  - P=1, Q=real: Should be ~1.00920 (-0.43%) ← Q effect isolated")
    print("  - P=Q=real:    Should be ~1.01233 (-0.13%) ← Full production case")
    print()
    print("  The Q effect is: (P=real,Q=real) - (P=real,Q=1)")
    print("                 = -0.13% - (+0.05%) = -0.18%")
    print()
    print("  This -0.18% is the Q deviation we need to explain in Phase 38.")


if __name__ == "__main__":
    main()
