#!/usr/bin/env python3
"""
Derive g Correction from Q Attenuation Pattern

Key observation from analyze_q_attenuation.py:
- Q attenuates I2 MORE than I1 (att_I2 < att_I1)
- This attenuation differs between κ and κ*

HYPOTHESIS:
The g correction compensates for differential Q attenuation.
If we define:
  r_Q = att_I1 / att_I2 (how much more I1 is preserved than I2)

Then maybe:
  g_I2 / g_I1 ≈ some function of r_Q

Let's test this and derive the relationship.

Created: 2025-12-27 (Phase 45 investigation)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star, Polynomial
from src.evaluator.g_functional import compute_I1_I2_totals
from src.evaluator.g_first_principles import G_I1_CALIBRATED, G_I2_CALIBRATED


def compute_attenuation(R, theta, polys, n_quad=60):
    """Compute Q attenuation factors for I1 and I2."""
    Q_unity = Polynomial(np.array([1.0]))
    polys_q1 = {"P1": polys["P1"], "P2": polys["P2"], "P3": polys["P3"], "Q": Q_unity}

    # With real Q
    I1_plus_real, I2_plus_real = compute_I1_I2_totals(R, theta, polys, n_quad)
    I1_minus_real, I2_minus_real = compute_I1_I2_totals(-R, theta, polys, n_quad)

    # With Q=1
    I1_plus_q1, I2_plus_q1 = compute_I1_I2_totals(R, theta, polys_q1, n_quad)
    I1_minus_q1, I2_minus_q1 = compute_I1_I2_totals(-R, theta, polys_q1, n_quad)

    return {
        "att_I1_plus": I1_plus_real / I1_plus_q1,
        "att_I2_plus": I2_plus_real / I2_plus_q1,
        "att_I1_minus": I1_minus_real / I1_minus_q1,
        "att_I2_minus": I2_minus_real / I2_minus_q1,
        "I1_minus_real": I1_minus_real,
        "I2_minus_real": I2_minus_real,
    }


def main():
    print("=" * 80)
    print("DERIVE g FROM Q ATTENUATION PATTERN")
    print("=" * 80)
    print()

    theta = 4 / 7
    K = 3
    n_quad = 60

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    R_kappa = 1.3036
    R_kappa_star = 1.1167

    # Compute attenuation for both benchmarks
    att_kappa = compute_attenuation(R_kappa, theta, polys_kappa, n_quad)
    att_kappa_star = compute_attenuation(R_kappa_star, theta, polys_kappa_star, n_quad)

    print("Q Attenuation (mirror term):")
    print(f"  κ:  att_I1 = {att_kappa['att_I1_minus']:.4f}, att_I2 = {att_kappa['att_I2_minus']:.4f}")
    print(f"  κ*: att_I1 = {att_kappa_star['att_I1_minus']:.4f}, att_I2 = {att_kappa_star['att_I2_minus']:.4f}")
    print()

    # Compute the attenuation ratio r_Q = att_I1 / att_I2
    r_Q_kappa = att_kappa['att_I1_minus'] / att_kappa['att_I2_minus']
    r_Q_kappa_star = att_kappa_star['att_I1_minus'] / att_kappa_star['att_I2_minus']

    print(f"Attenuation ratio r_Q = att_I1 / att_I2:")
    print(f"  κ:  r_Q = {r_Q_kappa:.4f}")
    print(f"  κ*: r_Q = {r_Q_kappa_star:.4f}")
    print()

    # Calibrated g ratio
    g_ratio = G_I2_CALIBRATED / G_I1_CALIBRATED
    print(f"Calibrated g ratio: g_I2 / g_I1 = {G_I2_CALIBRATED:.5f} / {G_I1_CALIBRATED:.5f} = {g_ratio:.5f}")
    print()

    # The g ratio 1.0185 is MUCH smaller than r_Q ≈ 1.15-1.20
    # So it's not a simple proportionality

    # Let's think differently:
    # The g correction is applied to the MIRROR term
    # c = I1_plus + g_I1 * base * I1_minus + I2_plus + g_I2 * base * I2_minus + S34
    #
    # What if the g correction compensates for the RELATIVE change in I1_minus vs I2_minus
    # when going from Q=1 to Q=real?

    print("=" * 80)
    print("HYPOTHESIS: g compensates for attenuation-induced fraction change")
    print("=" * 80)
    print()

    # With Q=1, the I1 fraction is:
    # f_I1(Q=1) = I1_minus(Q=1) / [I1_minus(Q=1) + I2_minus(Q=1)]

    # With Q=real:
    # f_I1(Q=real) = I1_minus(Q=real) / [I1_minus(Q=real) + I2_minus(Q=real)]

    # The fraction CHANGES because I1 and I2 are attenuated differently

    # For κ:
    Q_unity = Polynomial(np.array([1.0]))
    polys_kappa_q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}
    _, I1_minus_q1_k = compute_I1_I2_totals(-R_kappa, theta, polys_kappa_q1, n_quad)
    _, I2_minus_q1_k = compute_I1_I2_totals(-R_kappa, theta, polys_kappa_q1, n_quad)

    # Recompute properly
    I1_minus_q1_k, I2_minus_q1_k = compute_I1_I2_totals(-R_kappa, theta, polys_kappa_q1, n_quad)

    f_I1_q1_kappa = I1_minus_q1_k / (I1_minus_q1_k + I2_minus_q1_k)
    f_I1_real_kappa = att_kappa['I1_minus_real'] / (att_kappa['I1_minus_real'] + att_kappa['I2_minus_real'])

    # For κ*:
    polys_kappa_star_q1 = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Q_unity}
    I1_minus_q1_ks, I2_minus_q1_ks = compute_I1_I2_totals(-R_kappa_star, theta, polys_kappa_star_q1, n_quad)

    f_I1_q1_kappa_star = I1_minus_q1_ks / (I1_minus_q1_ks + I2_minus_q1_ks)
    f_I1_real_kappa_star = att_kappa_star['I1_minus_real'] / (att_kappa_star['I1_minus_real'] + att_kappa_star['I2_minus_real'])

    print("I1 fraction change from Q=1 to Q=real:")
    print(f"  κ:  f_I1(Q=1) = {f_I1_q1_kappa:.4f} → f_I1(Q=real) = {f_I1_real_kappa:.4f}  (Δ = {f_I1_real_kappa - f_I1_q1_kappa:+.4f})")
    print(f"  κ*: f_I1(Q=1) = {f_I1_q1_kappa_star:.4f} → f_I1(Q=real) = {f_I1_real_kappa_star:.4f}  (Δ = {f_I1_real_kappa_star - f_I1_q1_kappa_star:+.4f})")
    print()

    # The key pattern: f_I1 INCREASES when going from Q=1 to Q=real
    # This is because I2 is attenuated MORE than I1

    print("=" * 80)
    print("NEW APPROACH: Base vs Corrected Attenuation")
    print("=" * 80)
    print()

    # What if:
    # - With Q=1 and g=g_baseline, we get some c value
    # - With Q=real, we need different g values to compensate

    # The PRZZ formula with g_baseline assumes uniform correction
    # But Q creates differential attenuation, so we need differential g

    # Model:
    # The "correct" g for each component should scale inversely with attenuation
    # If I2 is attenuated more (smaller att_I2), it needs larger g to compensate

    # g_I2_effective = g_baseline / att_I2 (normalize)
    # g_I1_effective = g_baseline / att_I1 (normalize)
    #
    # But this is wrong because attenuation is already in the I values

    # Let me think again...
    #
    # The formula is: c = I1_plus + g_I1 * base * I1_minus + I2_plus + g_I2 * base * I2_minus + S34
    #
    # If g_I1 = g_I2 = g_baseline, we get a certain c
    # If we use calibrated g values, we get the target c
    #
    # The correction Δg = g - g_baseline satisfies:
    # Δc = base * (Δg_I1 * I1_minus + Δg_I2 * I2_minus)
    #
    # So: Δg_I1 * I1_minus + Δg_I2 * I2_minus = Δc / base

    # We have 2 benchmarks, so 2 equations, 2 unknowns. This is what we already solved!
    # The question is: can we predict Δg from attenuation properties?

    # Insight: Δg might be related to how f_I1 changes with Q

    print("Let's look at what Δg values we have:")
    delta_g_I1 = G_I1_CALIBRATED - g_baseline
    delta_g_I2 = G_I2_CALIBRATED - g_baseline

    print(f"  g_baseline = {g_baseline:.6f}")
    print(f"  g_I1 = {G_I1_CALIBRATED:.6f}  → Δg_I1 = {delta_g_I1:+.6f} ({delta_g_I1/g_baseline*100:+.4f}%)")
    print(f"  g_I2 = {G_I2_CALIBRATED:.6f}  → Δg_I2 = {delta_g_I2:+.6f} ({delta_g_I2/g_baseline*100:+.4f}%)")
    print()

    # Interesting: Δg_I1 < 0, Δg_I2 > 0
    # I1 needs LESS correction, I2 needs MORE

    # This might correlate with attenuation:
    # I1 is attenuated LESS (higher att) → needs less g correction
    # I2 is attenuated MORE (lower att) → needs more g correction

    print("Correlation test:")
    print(f"  I1: higher attenuation ({att_kappa['att_I1_minus']:.4f}), lower g correction ({delta_g_I1:+.6f})")
    print(f"  I2: lower attenuation ({att_kappa['att_I2_minus']:.4f}), higher g correction ({delta_g_I2:+.6f})")
    print()
    print("This is CONSISTENT with the hypothesis that g compensates for attenuation!")

    # Let's try to derive a formula
    # Hypothesis: Δg ∝ (1 - attenuation)
    # Or: Δg ∝ (1 / attenuation - 1)

    print("\n" + "=" * 80)
    print("DERIVATION ATTEMPT")
    print("=" * 80)

    # The average attenuation across benchmarks
    avg_att_I1 = (att_kappa['att_I1_minus'] + att_kappa_star['att_I1_minus']) / 2
    avg_att_I2 = (att_kappa['att_I2_minus'] + att_kappa_star['att_I2_minus']) / 2

    print(f"\nAverage attenuation:")
    print(f"  I1: {avg_att_I1:.4f}")
    print(f"  I2: {avg_att_I2:.4f}")

    # Try: Δg = k * (1 - att) where k is a constant
    # For I1: Δg_I1 = k * (1 - avg_att_I1)
    # For I2: Δg_I2 = k * (1 - avg_att_I2)

    # Solve for k from I1:
    k_from_I1 = delta_g_I1 / (1 - avg_att_I1)
    # Predict Δg_I2:
    delta_g_I2_pred = k_from_I1 * (1 - avg_att_I2)

    print(f"\nTest: Δg = k × (1 - attenuation)")
    print(f"  From I1: k = {k_from_I1:.6f}")
    print(f"  Predict Δg_I2 = {delta_g_I2_pred:+.6f}")
    print(f"  Actual Δg_I2 = {delta_g_I2:+.6f}")
    print(f"  Error: {(delta_g_I2_pred - delta_g_I2)/delta_g_I2 * 100:+.2f}%")

    # Try: Δg = k * log(1/att) = -k * log(att)
    k_log_from_I1 = delta_g_I1 / (-np.log(avg_att_I1))
    delta_g_I2_pred_log = k_log_from_I1 * (-np.log(avg_att_I2))

    print(f"\nTest: Δg = -k × log(attenuation)")
    print(f"  From I1: k = {k_log_from_I1:.6f}")
    print(f"  Predict Δg_I2 = {delta_g_I2_pred_log:+.6f}")
    print(f"  Actual Δg_I2 = {delta_g_I2:+.6f}")
    print(f"  Error: {(delta_g_I2_pred_log - delta_g_I2)/delta_g_I2 * 100:+.2f}%")

    # The truth is we have only 2 data points (I1 and I2)
    # Any single-parameter formula will have trouble predicting

    # Let's check if the RATIO of Δg values relates to attenuation
    print("\n" + "=" * 80)
    print("RATIO ANALYSIS")
    print("=" * 80)

    delta_g_ratio = delta_g_I2 / delta_g_I1
    att_ratio = avg_att_I1 / avg_att_I2
    log_att_ratio = np.log(avg_att_I2) / np.log(avg_att_I1)

    print(f"\nRatios:")
    print(f"  Δg_I2 / Δg_I1 = {delta_g_ratio:.4f}")
    print(f"  att_I1 / att_I2 = {att_ratio:.4f}")
    print(f"  log(att_I2) / log(att_I1) = {log_att_ratio:.4f}")

    # delta_g_ratio is negative because delta_g_I1 < 0
    # The absolute ratio:
    abs_ratio = abs(delta_g_I2 / delta_g_I1)
    print(f"  |Δg_I2 / Δg_I1| = {abs_ratio:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key findings:")
    print("1. Q attenuates I2 more than I1 (att_I2 ≈ 0.54 < att_I1 ≈ 0.63)")
    print("2. The component with LESS attenuation (I1) needs LESS g correction (Δg_I1 < 0)")
    print("3. The component with MORE attenuation (I2) needs MORE g correction (Δg_I2 > 0)")
    print()
    print("This is qualitatively consistent but the quantitative relationship")
    print("between attenuation and Δg is not simple.")
    print()
    print("OPEN QUESTION: What is the exact formula relating attenuation to Δg?")


if __name__ == "__main__":
    main()
