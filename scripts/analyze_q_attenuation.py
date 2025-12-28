#!/usr/bin/env python3
"""
Analyze Q Polynomial Attenuation Pattern

The Q=1 hypothesis failed because Q fundamentally changes the c ratio:
- With Q=1: c_κ/c_κ* ≈ 0.85 (κ* is larger!)
- With Q=real: c_κ/c_κ* ≈ 1.10 (κ is larger)

This suggests Q has VERY different effects on the two benchmarks.
Let's analyze the Q attenuation factors for I1 and I2 separately.

Created: 2025-12-27 (Phase 45 investigation)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star, Polynomial
from src.evaluator.g_functional import compute_I1_I2_totals
from src.evaluator.g_first_principles import compute_S34


def analyze_q_effects():
    """Analyze how Q affects I1 and I2 differently for each benchmark."""

    theta = 4 / 7
    n_quad = 60

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Q=1 polynomial
    Q_unity = Polynomial(np.array([1.0]))

    R_kappa = 1.3036
    R_kappa_star = 1.1167

    print("=" * 80)
    print("Q ATTENUATION ANALYSIS")
    print("=" * 80)
    print()

    results = {}

    for name, polys, R in [("κ", polys_kappa, R_kappa), ("κ*", polys_kappa_star, R_kappa_star)]:
        print(f"\n{'='*40}")
        print(f"Benchmark: {name} (R={R})")
        print(f"{'='*40}")

        # With real Q
        polys_real = polys
        I1_plus_real, I2_plus_real = compute_I1_I2_totals(R, theta, polys_real, n_quad)
        I1_minus_real, I2_minus_real = compute_I1_I2_totals(-R, theta, polys_real, n_quad)

        # With Q=1
        polys_q1 = {"P1": polys["P1"], "P2": polys["P2"], "P3": polys["P3"], "Q": Q_unity}
        I1_plus_q1, I2_plus_q1 = compute_I1_I2_totals(R, theta, polys_q1, n_quad)
        I1_minus_q1, I2_minus_q1 = compute_I1_I2_totals(-R, theta, polys_q1, n_quad)

        # Compute attenuation factors (Q=real / Q=1)
        att_I1_plus = I1_plus_real / I1_plus_q1 if abs(I1_plus_q1) > 1e-15 else float('nan')
        att_I2_plus = I2_plus_real / I2_plus_q1 if abs(I2_plus_q1) > 1e-15 else float('nan')
        att_I1_minus = I1_minus_real / I1_minus_q1 if abs(I1_minus_q1) > 1e-15 else float('nan')
        att_I2_minus = I2_minus_real / I2_minus_q1 if abs(I2_minus_q1) > 1e-15 else float('nan')

        print(f"\nI1 values:")
        print(f"  I1_plus:  Q=1: {I1_plus_q1:+.6f}  Q=real: {I1_plus_real:+.6f}  ratio: {att_I1_plus:.4f}")
        print(f"  I1_minus: Q=1: {I1_minus_q1:+.6f}  Q=real: {I1_minus_real:+.6f}  ratio: {att_I1_minus:.4f}")

        print(f"\nI2 values:")
        print(f"  I2_plus:  Q=1: {I2_plus_q1:+.6f}  Q=real: {I2_plus_real:+.6f}  ratio: {att_I2_plus:.4f}")
        print(f"  I2_minus: Q=1: {I2_minus_q1:+.6f}  Q=real: {I2_minus_real:+.6f}  ratio: {att_I2_minus:.4f}")

        results[name] = {
            "att_I1_plus": att_I1_plus,
            "att_I2_plus": att_I2_plus,
            "att_I1_minus": att_I1_minus,
            "att_I2_minus": att_I2_minus,
            "I1_minus_real": I1_minus_real,
            "I2_minus_real": I2_minus_real,
            "I1_minus_q1": I1_minus_q1,
            "I2_minus_q1": I2_minus_q1,
        }

    # Compare attenuation ratios
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("\n| Component | κ attenuation | κ* attenuation | Ratio κ*/κ |")
    print("|-----------|---------------|----------------|------------|")
    print(f"| I1_plus   | {results['κ']['att_I1_plus']:.4f}        | {results['κ*']['att_I1_plus']:.4f}          | {results['κ*']['att_I1_plus']/results['κ']['att_I1_plus']:.4f}     |")
    print(f"| I2_plus   | {results['κ']['att_I2_plus']:.4f}        | {results['κ*']['att_I2_plus']:.4f}          | {results['κ*']['att_I2_plus']/results['κ']['att_I2_plus']:.4f}     |")
    print(f"| I1_minus  | {results['κ']['att_I1_minus']:.4f}        | {results['κ*']['att_I1_minus']:.4f}          | {results['κ*']['att_I1_minus']/results['κ']['att_I1_minus']:.4f}     |")
    print(f"| I2_minus  | {results['κ']['att_I2_minus']:.4f}        | {results['κ*']['att_I2_minus']:.4f}          | {results['κ*']['att_I2_minus']/results['κ']['att_I2_minus']:.4f}     |")

    # The key insight: differential attenuation
    print("\n" + "=" * 80)
    print("KEY INSIGHT: Differential Q Attenuation")
    print("=" * 80)

    ratio_I1_I2_kappa = results['κ']['att_I1_minus'] / results['κ']['att_I2_minus']
    ratio_I1_I2_kappa_star = results['κ*']['att_I1_minus'] / results['κ*']['att_I2_minus']

    print(f"\nFor mirror term (I_minus):")
    print(f"  κ:  att_I1/att_I2 = {results['κ']['att_I1_minus']:.4f} / {results['κ']['att_I2_minus']:.4f} = {ratio_I1_I2_kappa:.4f}")
    print(f"  κ*: att_I1/att_I2 = {results['κ*']['att_I1_minus']:.4f} / {results['κ*']['att_I2_minus']:.4f} = {ratio_I1_I2_kappa_star:.4f}")

    print(f"\nQ attenuates I1 and I2 DIFFERENTLY:")
    print(f"  - I2 is attenuated MORE than I1 (att_I2 << att_I1)")
    print(f"  - This ratio differs between κ and κ*")

    # Try to derive g correction from attenuation difference
    print("\n" + "=" * 80)
    print("DERIVATION ATTEMPT: g from Q attenuation")
    print("=" * 80)

    # The idea: g_I1 and g_I2 compensate for differential Q attenuation
    # If Q=1 would give g_I1 = g_I2 = g_baseline, then
    # g_I1(Q=real) = g_baseline × (1 / att_I1) ?
    # g_I2(Q=real) = g_baseline × (1 / att_I2) ?

    theta = 4/7
    K = 3
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # This is wrong because the attenuation already happened in the I values
    # What we need is the RELATIVE effect on g

    # Let's think about it differently:
    # c = I1_plus + g_I1 * base * I1_minus + I2_plus + g_I2 * base * I2_minus + S34

    # With Q=1:
    # c(Q=1) = I1_plus(Q=1) + g * base * I1_minus(Q=1) + I2_plus(Q=1) + g * base * I2_minus(Q=1) + S34(Q=1)

    # With Q=real:
    # c(Q=real) = I1_plus(Q=real) + g * base * I1_minus(Q=real) + I2_plus(Q=real) + g * base * I2_minus(Q=real) + S34(Q=real)

    # The Q attenuation affects the I values directly.
    # The g correction is SEPARATE from Q attenuation.

    # Maybe the g correction comes from the difference between how Q enters I1 vs I2?

    print("\nQ ENTRY MECHANISM:")
    print("  I1: Q enters via Q(Arg_α)Q(Arg_β) where Arg depends on (x,y,t)")
    print("      d²/dxdy picks out Q' terms, not just Q")
    print("  I2: Q enters via Q(t)² directly")
    print("      No derivative transformation")
    print()
    print("The attenuation att = I(Q=real)/I(Q=1) measures the NET Q effect.")
    print("But the G CORRECTION might relate to the DERIVATIVE structure of Q.")

    # Compute Q polynomial info
    print("\n" + "=" * 80)
    print("Q POLYNOMIAL ANALYSIS")
    print("=" * 80)

    Q_kappa = polys_kappa["Q"]
    Q_kappa_star = polys_kappa_star["Q"]

    print(f"\nκ Q polynomial:")
    print(f"  Q(0) = {Q_kappa.eval(np.array([0.0]))[0]:.6f}")
    print(f"  Q(1) = {Q_kappa.eval(np.array([1.0]))[0]:.6f}")
    print(f"  Q'(0) = {Q_kappa.eval_deriv(np.array([0.0]), 1)[0]:.6f}")

    print(f"\nκ* Q polynomial:")
    print(f"  Q(0) = {Q_kappa_star.eval(np.array([0.0]))[0]:.6f}")
    print(f"  Q(1) = {Q_kappa_star.eval(np.array([1.0]))[0]:.6f}")
    print(f"  Q'(0) = {Q_kappa_star.eval_deriv(np.array([0.0]), 1)[0]:.6f}")

    # Compute ∫Q(t)²dt and ∫Q'(t)²dt
    from src.quadrature import gauss_legendre_nodes_weights

    nodes, weights = gauss_legendre_nodes_weights(60)
    t_vals = (nodes + 1) / 2  # Map [-1,1] to [0,1]
    w_vals = weights / 2

    for name, Q in [("κ", Q_kappa), ("κ*", Q_kappa_star)]:
        Q_vals = Q.eval(t_vals)
        Qp_vals = Q.eval_deriv(t_vals, 1)

        int_Q2 = np.sum(w_vals * Q_vals**2)
        int_Qp2 = np.sum(w_vals * Qp_vals**2)
        int_Q_Qp = np.sum(w_vals * Q_vals * Qp_vals)

        print(f"\n{name}:")
        print(f"  ∫₀¹ Q(t)² dt = {int_Q2:.6f}")
        print(f"  ∫₀¹ Q'(t)² dt = {int_Qp2:.6f}")
        print(f"  ∫₀¹ Q(t)Q'(t) dt = {int_Q_Qp:.6f}")
        print(f"  Ratio ∫Q'²/∫Q² = {int_Qp2/int_Q2:.4f}")


if __name__ == "__main__":
    analyze_q_effects()
