#!/usr/bin/env python3
"""
GPT Run 5 Diagnostic: Direct TeX I2 Evaluation

This script evaluates I2 directly from TeX formulas and compares with the
amplitude model to identify what the model is compensating for.

Key insight from GPT:
- The exp(2Rt) factor is INSIDE the integral, not a standalone multiplier
- The amplitude model uses A = exp(R) + K-1 + ε as a surrogate
- Direct TeX evaluation should compute the mirror contribution directly

TeX formula structure (lines 1502-1548):
    I₂ = (T·Φ̂(0)/θ) × ∫∫ Q(t)² exp(2Rt) P₁(u)P₂(u) dt du

Separability:
    I₂ = [∫ P₁(u)P₂(u) du] × [(1/θ) ∫ Q(t)² exp(2Rt) dt]

Mirror contribution (from TeX):
    I₂_mirror = I₂(+R) + exp(2R) × I₂(-R)  -- if exp(2R) is factored out
    OR
    I₂_mirror_direct = (1/θ) × ∫ Q(t)² × [exp(2Rt) + exp(-2Rt)] dt × ∫ P₁P₂ du

This script computes both and compares with the model.

Usage:
    python run_gpt_run5_direct_i2.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.evaluate import (
    compute_c_paper_tex_mirror,
    evaluate_I2_separable,
    tex_amplitudes,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import gauss_legendre_01


THETA = 4.0 / 7.0
R_REF = 1.3036


@dataclass
class I2DirectResult:
    """Result of direct TeX I2 evaluation."""
    # Separable integrals
    u_integral: float  # ∫ P₁(u)P₂(u) du
    t_integral_plus: float  # (1/θ) ∫ Q(t)² exp(2Rt) dt at +R
    t_integral_minus: float  # (1/θ) ∫ Q(t)² exp(2Rt) dt at -R

    # I2 values
    i2_plus: float  # I2 at +R
    i2_minus: float  # I2 at -R

    # Direct TeX mirror (different formulations)
    i2_mirror_simple: float  # I2(+R) + exp(2R) × I2(-R)
    i2_mirror_combined: float  # (1/θ) × ∫ Q² × [exp(2Rt) + exp(2R)·exp(-2Rt)] dt × ∫ P₁P₂

    # For comparison with model
    exp_2R: float
    exp_R: float
    R: float


def compute_i2_direct_tex(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 100,
) -> I2DirectResult:
    """
    Compute I2 contribution directly from TeX formula (no amplitude model).

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with P1, P2, Q
        n: Quadrature points

    Returns:
        I2DirectResult with all components
    """
    nodes, weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    Q = polynomials["Q"]

    # Evaluate polynomials at nodes
    P1_vals = P1.eval(nodes)
    P2_vals = P2.eval(nodes)
    Q_vals = Q.eval(nodes)

    # u-integral: ∫ P₁(u)P₂(u) du (for (1,2) pair, this is P1×P2, not P1²)
    # For the diagonal pairs this would be P₁² etc.
    u_integral = np.sum(weights * P1_vals * P2_vals)

    # t-integral at +R: (1/θ) ∫ Q(t)² exp(2Rt) dt
    exp_plus = np.exp(2 * R * nodes)
    t_integral_plus = np.sum(weights * Q_vals**2 * exp_plus) / theta

    # t-integral at -R: (1/θ) ∫ Q(t)² exp(-2Rt) dt
    exp_minus = np.exp(-2 * R * nodes)
    t_integral_minus = np.sum(weights * Q_vals**2 * exp_minus) / theta

    # I2 values
    i2_plus = u_integral * t_integral_plus
    i2_minus = u_integral * t_integral_minus

    # Direct mirror formulations
    exp_2R = np.exp(2 * R)
    exp_R = np.exp(R)

    # Simple: I2(+R) + exp(2R) × I2(-R)
    # This assumes the mirror factor is exp(2R), factored outside
    i2_mirror_simple = i2_plus + exp_2R * i2_minus

    # Combined integral: (1/θ) ∫ Q² × [exp(2Rt) + exp(2R)·exp(-2Rt)] dt × u_integral
    combined_exp = exp_plus + exp_2R * exp_minus
    t_integral_combined = np.sum(weights * Q_vals**2 * combined_exp) / theta
    i2_mirror_combined = u_integral * t_integral_combined

    return I2DirectResult(
        u_integral=u_integral,
        t_integral_plus=t_integral_plus,
        t_integral_minus=t_integral_minus,
        i2_plus=i2_plus,
        i2_minus=i2_minus,
        i2_mirror_simple=i2_mirror_simple,
        i2_mirror_combined=i2_mirror_combined,
        exp_2R=exp_2R,
        exp_R=exp_R,
        R=R,
    )


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

    # Combined t-integral
    combined_exp = exp_plus + exp_2R * exp_minus
    t_integral_combined = np.sum(weights * Q_vals**2 * combined_exp) / theta

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
        i2_mirror_combined = u_integral * t_integral_combined

        results[pair_key] = I2DirectResult(
            u_integral=u_integral,
            t_integral_plus=t_integral_plus,
            t_integral_minus=t_integral_minus,
            i2_plus=i2_plus,
            i2_minus=i2_minus,
            i2_mirror_simple=i2_mirror_simple,
            i2_mirror_combined=i2_mirror_combined,
            exp_2R=exp_2R,
            exp_R=exp_R,
            R=R,
        )

    return results


def main():
    print("=" * 80)
    print("GPT Run 5: Direct TeX I2 Evaluation")
    print("=" * 80)
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

        # Get model results
        model_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_val,
            n=60,
            polynomials=polys,
            tex_exp_component="exp_R",
            n_quad_a=40,
        )

        # Compute direct I2 for all pairs
        direct_results = compute_i2_all_pairs(THETA, R_val, polys, n=100)

        # Aggregate direct I2
        i2_direct_plus_total = 0.0
        i2_direct_minus_total = 0.0
        i2_direct_mirror_simple = 0.0
        i2_direct_mirror_combined = 0.0

        # Factorial normalization for ordered-pair evaluation:
        #   f(pq) = 1 / (ℓ_p! · ℓ_q!)
        # (No triangle folding / no symmetry factor.)
        f_norm = {
            "11": 1.0,
            "22": 0.25,
            "33": 1.0 / 36.0,
            "12": 0.5,
            "21": 0.5,
            "13": 1.0 / 6.0,
            "31": 1.0 / 6.0,
            "23": 1.0 / 12.0,
            "32": 1.0 / 12.0,
        }

        print("\nPer-pair direct TeX I2:")
        print(f"{'Pair':<6} {'u_int':>10} {'I2(+R)':>12} {'I2(-R)':>12} {'mirror':>12} {'weighted':>12}")
        print("-" * 70)

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            r = direct_results[pair_key]
            norm = f_norm[pair_key]

            i2_direct_plus_total += norm * r.i2_plus
            i2_direct_minus_total += norm * r.i2_minus
            i2_direct_mirror_simple += norm * r.i2_mirror_simple
            i2_direct_mirror_combined += norm * r.i2_mirror_combined

            print(f"{pair_key:<6} {r.u_integral:>10.6f} {r.i2_plus:>12.6f} {r.i2_minus:>12.6f} "
                  f"{r.i2_mirror_simple:>12.6f} {norm * r.i2_mirror_simple:>12.6f}")

        print("-" * 70)
        print(f"{'TOTAL':<6} {'':<10} {i2_direct_plus_total:>12.6f} {i2_direct_minus_total:>12.6f} "
              f"{i2_direct_mirror_simple:>12.6f}")

        # Compare with model
        print(f"\n--- Comparison with Model ---")
        print()

        # Model I2 values from tex_mirror result
        i2_model_plus = model_result.I2_plus
        i2_model_minus_base = model_result.I2_minus_base
        m2_model = model_result.m2
        A2_model = model_result.A2

        print(f"Model I2_plus:       {i2_model_plus:+.6f}")
        print(f"Model I2_minus_base: {i2_model_minus_base:+.6f}")
        print(f"Model m2:            {m2_model:.4f} (= A2 × m2_implied)")
        print(f"Model A2:            {A2_model:.4f}")
        print()
        print(f"Direct I2_plus:      {i2_direct_plus_total:+.6f}")
        print(f"Direct I2_minus:     {i2_direct_minus_total:+.6f}")
        print()

        # Model mirror contribution
        model_mirror_contrib = i2_model_plus + m2_model * i2_model_minus_base
        print(f"Model mirror:        I2(+) + m2×I2_base(-) = {model_mirror_contrib:+.6f}")

        # Direct TeX mirror contributions
        exp_2R = np.exp(2 * R_val)
        direct_simple = i2_direct_plus_total + exp_2R * i2_direct_minus_total
        print(f"Direct (exp(2R)):    I2(+) + exp(2R)×I2(-) = {direct_simple:+.6f}")
        print(f"                     where exp(2R) = {exp_2R:.4f}")
        print()

        # What multiplier does direct imply?
        if abs(i2_direct_minus_total) > 1e-10:
            implied_m2_direct = (model_mirror_contrib - i2_direct_plus_total) / i2_direct_minus_total
            print(f"Direct implies m2 = {implied_m2_direct:.4f} (to match model mirror)")

        # Gap analysis
        print(f"\n--- Gap Analysis ---")
        delta_mirror = model_mirror_contrib - direct_simple
        print(f"Model mirror - Direct: {delta_mirror:+.6f} ({100*delta_mirror/model_mirror_contrib:+.2f}%)")

        # What's the ratio?
        if abs(direct_simple) > 1e-10:
            ratio = model_mirror_contrib / direct_simple
            print(f"Ratio (model/direct): {ratio:.6f}")

        # Comparison at different exp values
        print(f"\n--- Direct I2 Mirror with Different Factors ---")
        for factor_name, factor_val in [
            ("exp(2R)", np.exp(2 * R_val)),
            ("exp(R)", np.exp(R_val)),
            ("Model m2", m2_model),
            ("A2", A2_model),
            ("A2 × m2_implied", m2_model),
        ]:
            mirror_val = i2_direct_plus_total + factor_val * i2_direct_minus_total
            gap = mirror_val - model_mirror_contrib
            print(f"  {factor_name:<15} = {factor_val:>8.4f} → mirror = {mirror_val:+.6f} (gap: {gap:+.6f})")

    # Cross-benchmark comparison
    print("\n")
    print("=" * 80)
    print("CROSS-BENCHMARK ANALYSIS")
    print("=" * 80)

    # What exp(2Rt) integral ratio tells us
    print("\nThe t-integral (1/θ)∫Q²exp(2Rt)dt at different R:")
    for bench_name, polys, R_val, _ in benchmarks:
        nodes, weights = gauss_legendre_01(100)
        Q = polys["Q"]
        Q_vals = Q.eval(nodes)

        exp_plus = np.exp(2 * R_val * nodes)
        t_int_plus = np.sum(weights * Q_vals**2 * exp_plus) / THETA

        exp_minus = np.exp(-2 * R_val * nodes)
        t_int_minus = np.sum(weights * Q_vals**2 * exp_minus) / THETA

        print(f"  {bench_name}: R={R_val}")
        print(f"    t_integral(+R) = {t_int_plus:.6f}")
        print(f"    t_integral(-R) = {t_int_minus:.6f}")
        print(f"    ratio (+R)/(-R) = {t_int_plus/t_int_minus:.4f}")
        print(f"    exp(4R) = {np.exp(4*R_val):.4f} (expected ratio if Q=1)")

    print("\n" + "=" * 80)
    print("KEY FINDING")
    print("=" * 80)
    print("""
The direct TeX evaluation shows that:

1. The t-integral includes exp(2Rt) INSIDE the integral
2. The simple mirror formula I2(+R) + exp(2R)×I2(-R) doesn't match model
3. The discrepancy reveals what the amplitude model is compensating for

The amplitude model (A2 = exp(R) + 4 + ε) appears to be a SURROGATE
that combines:
- The effect of exp(2Rt) inside the integral
- The Q² weighting effect
- Polynomial normalization factors

To move toward direct TeX evaluation:
1. Keep the t-integral with exp(2Rt) inside (already correct)
2. The mirror recombination needs the correct TeX prescription
3. Check TeX lines 1502-1548 for the exact recombination formula
""")


if __name__ == "__main__":
    main()
