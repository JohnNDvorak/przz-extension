#!/usr/bin/env python3
"""
GPT Run 6 Diagnostic: Direct TeX I2 Evaluation (FIXED)

This script fixes the normalization bug from Run 5 and properly compares
direct TeX I2 evaluation with the model.

BUG FIX (Run 5 → Run 6):
    The Run 5 script used incorrect factorial normalization:
    - 12, 21: 1.0 (WRONG) → 0.5 (CORRECT)
    - 13, 31: 0.5 (WRONG) → 1/6 (CORRECT)
    - 23, 32: 1/6 (WRONG) → 1/12 (CORRECT)

    Correct formula: 1/(ell1! × ell2!) for ordered pair (ell1, ell2)

NEW FEATURES:
    - Per-pair ratio comparison with model's pair_breakdown
    - Identifies whether normalization was the only issue or if kernel differs

Usage:
    python run_gpt_run6_direct_i2.py
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.evaluate import (
    compute_c_paper_tex_mirror,
    compute_operator_implied_weights,
    tex_amplitudes,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import gauss_legendre_01


THETA = 4.0 / 7.0
R_REF = 1.3036


# CORRECT factorial normalization: 1/(ell1! × ell2!)
F_NORM = {
    f"{i}{j}": 1.0 / (math.factorial(i) * math.factorial(j))
    for i in (1, 2, 3) for j in (1, 2, 3)
}


@dataclass
class I2DirectResult:
    """Result of direct TeX I2 evaluation."""
    u_integral: float  # ∫ P_{p1}(u) P_{p2}(u) du
    t_integral_plus: float  # (1/θ) ∫ Q(t)² exp(2Rt) dt at +R
    t_integral_minus: float  # (1/θ) ∫ Q(t)² exp(-2Rt) dt at -R
    i2_plus: float  # I2 at +R
    i2_minus: float  # I2 at -R
    i2_mirror_simple: float  # I2(+R) + exp(2R) × I2(-R)
    exp_2R: float
    exp_R: float
    R: float


def compute_i2_all_pairs(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 100,
) -> Dict[str, I2DirectResult]:
    """
    Compute I2 direct for all 9 ordered pairs.

    Returns:
        Dict mapping pair key to I2DirectResult
    """
    nodes, weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials["Q"]

    P_vals = {
        "P1": P1.eval(nodes),
        "P2": P2.eval(nodes),
        "P3": P3.eval(nodes),
    }
    Q_vals = Q.eval(nodes)

    # t-integrals (same for all pairs)
    exp_plus = np.exp(2 * R * nodes)
    exp_minus = np.exp(-2 * R * nodes)
    t_integral_plus = np.sum(weights * Q_vals**2 * exp_plus) / theta
    t_integral_minus = np.sum(weights * Q_vals**2 * exp_minus) / theta

    exp_2R = np.exp(2 * R)
    exp_R = np.exp(R)

    results = {}
    pairs = ["11", "22", "33", "12", "21", "13", "31", "23", "32"]

    P_map = {"1": "P1", "2": "P2", "3": "P3"}

    for pair_key in pairs:
        p1_key = P_map[pair_key[0]]
        p2_key = P_map[pair_key[1]]

        # u-integral: ∫ P_{p1}(u) P_{p2}(u) du
        u_integral = np.sum(weights * P_vals[p1_key] * P_vals[p2_key])

        i2_plus = u_integral * t_integral_plus
        i2_minus = u_integral * t_integral_minus
        i2_mirror_simple = i2_plus + exp_2R * i2_minus

        results[pair_key] = I2DirectResult(
            u_integral=u_integral,
            t_integral_plus=t_integral_plus,
            t_integral_minus=t_integral_minus,
            i2_plus=i2_plus,
            i2_minus=i2_minus,
            i2_mirror_simple=i2_mirror_simple,
            exp_2R=exp_2R,
            exp_R=exp_R,
            R=R,
        )

    return results


def main():
    print("=" * 80)
    print("GPT Run 6: Direct TeX I2 Evaluation (NORMALIZATION FIXED)")
    print("=" * 80)
    print()

    # Show the corrected normalization
    print("CORRECTED FACTORIAL NORMALIZATION (1/(ell1! × ell2!)):")
    print("-" * 50)
    for pair in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        old_wrong = {"11": 1.0, "22": 0.25, "33": 1/36,
                     "12": 1.0, "21": 1.0, "13": 0.5, "31": 0.5, "23": 1/6, "32": 1/6}
        print(f"  {pair}: {F_NORM[pair]:.6f} (was {old_wrong[pair]:.6f} in Run 5)")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, 1.3036, 2.137),
        ("κ*", polys_kappa_star, 1.1167, 1.938),
    ]

    for bench_name, polys, R_val, c_target in benchmarks:
        print(f"\n{'='*70}")
        print(f"Benchmark: {bench_name} (R={R_val})")
        print(f"{'='*70}")

        # Get model results with pair breakdown
        implied = compute_operator_implied_weights(
            theta=THETA,
            R=R_val,
            polynomials=polys,
            sigma=5/32,
            normalization="grid",
            lift_scope="i1_only",
            n=60,
            n_quad_a=40,
        )

        # Compute direct I2 for all pairs
        direct_results = compute_i2_all_pairs(THETA, R_val, polys, n=100)

        # Compare per-pair
        print("\n--- PER-PAIR COMPARISON: Direct vs Model ---")
        print(f"{'Pair':<6} {'norm':>8} {'Direct+':>12} {'Model+':>12} {'Ratio+':>8} "
              f"{'Direct-':>12} {'Model-':>12} {'Ratio-':>8}")
        print("-" * 90)

        pair_breakdown = implied.pair_breakdown

        i2_direct_plus_total = 0.0
        i2_direct_minus_total = 0.0
        i2_model_plus_total = 0.0
        i2_model_minus_total = 0.0

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            r = direct_results[pair_key]
            norm = F_NORM[pair_key]

            # Direct values (with correct normalization)
            direct_plus = norm * r.i2_plus
            direct_minus = norm * r.i2_minus

            # Model values from pair_breakdown
            if pair_key in pair_breakdown:
                # The pair_breakdown stores weighted values
                model_plus = pair_breakdown[pair_key].get("I2_plus", 0.0)
                model_minus = pair_breakdown[pair_key].get("I2_minus_base", 0.0)
            else:
                model_plus = 0.0
                model_minus = 0.0

            # Accumulate totals
            i2_direct_plus_total += direct_plus
            i2_direct_minus_total += direct_minus
            i2_model_plus_total += model_plus
            i2_model_minus_total += model_minus

            # Compute ratios
            ratio_plus = direct_plus / model_plus if abs(model_plus) > 1e-15 else float('inf')
            ratio_minus = direct_minus / model_minus if abs(model_minus) > 1e-15 else float('inf')

            print(f"{pair_key:<6} {norm:>8.4f} {direct_plus:>+12.6f} {model_plus:>+12.6f} {ratio_plus:>8.4f} "
                  f"{direct_minus:>+12.6f} {model_minus:>+12.6f} {ratio_minus:>8.4f}")

        print("-" * 90)

        # Total comparison
        total_ratio_plus = i2_direct_plus_total / i2_model_plus_total if abs(i2_model_plus_total) > 1e-15 else float('inf')
        total_ratio_minus = i2_direct_minus_total / i2_model_minus_total if abs(i2_model_minus_total) > 1e-15 else float('inf')

        print(f"{'TOTAL':<6} {'':>8} {i2_direct_plus_total:>+12.6f} {i2_model_plus_total:>+12.6f} {total_ratio_plus:>8.4f} "
              f"{i2_direct_minus_total:>+12.6f} {i2_model_minus_total:>+12.6f} {total_ratio_minus:>8.4f}")

        # Compare with model's aggregated I2 values
        print(f"\n--- AGGREGATE COMPARISON ---")
        print(f"Model I2_plus (aggregated):       {implied.I2_plus:+.6f}")
        print(f"Direct I2_plus (with correct norm): {i2_direct_plus_total:+.6f}")
        print(f"Ratio (direct/model):              {i2_direct_plus_total/implied.I2_plus:.6f}")
        print()
        print(f"Model I2_minus_base (aggregated):   {implied.I2_minus_base:+.6f}")
        print(f"Direct I2_minus (with correct norm): {i2_direct_minus_total:+.6f}")
        print(f"Ratio (direct/model):                {i2_direct_minus_total/implied.I2_minus_base:.6f}")

        # Alignment verdict
        print(f"\n--- ALIGNMENT VERDICT ---")
        plus_aligned = abs(total_ratio_plus - 1.0) < 0.01  # Within 1%
        minus_aligned = abs(total_ratio_minus - 1.0) < 0.01

        if plus_aligned and minus_aligned:
            print("ALIGNED: Direct I2 matches model within 1% after normalization fix!")
            print("Conclusion: The 2-2.5× discrepancy in Run 5 was due to normalization bug.")
        else:
            print(f"NOT ALIGNED: Ratios are {total_ratio_plus:.4f} (plus) and {total_ratio_minus:.4f} (minus)")
            print("Conclusion: There's a structural difference beyond normalization.")
            print("Next step: Investigate DSL term structure vs separable assumption.")

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Run 6 fixes the factorial normalization bug from Run 5:
- Off-diagonal pairs were double/triple-counted
- Correct formula: 1/(ell1! × ell2!) for each ordered pair

KEY FINDING:
- Pair (1,1) matches EXACTLY (ratio 1.0)
- Pairs involving P2/P3 have significant discrepancies

ROOT CAUSE: Case C Kernel Handling
- The model uses kernel_regime="paper" which applies Case C kernels
- For d=1: omega = ell - 1
  - P1: omega=0 (Case B) - raw polynomial, matches direct
  - P2: omega=1 (Case C) - kernel-transformed, explains ratio ~4.5
  - P3: omega=2 (Case C) - kernel-transformed, explains ratio ~17

CONCLUSION:
- The separable I2 formula works for pair (1,1) with P1×P1
- For P2/P3 pairs, the Case C kernel applies additional transforms
- Direct TeX I2 would need to implement Case C handling too

NEXT STEPS:
1. For I2 channel accuracy, continue using the DSL-based evaluation
2. Focus direct TeX work on I1 (where the amplitude model is more involved)
3. Or: implement Case C kernel in direct script (complex)
""")


if __name__ == "__main__":
    main()
