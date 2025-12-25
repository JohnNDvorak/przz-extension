"""
run_qshift_diagnostic.py
Diagnostic to test the Q-operator shift hypothesis.

GPT Analysis (2025-12-19)
-------------------------
The mirror term in PRZZ is INSIDE the Q-differential operator:

    I_d = Q(-∂_α/logT) Q(-∂_β/logT) [ I_{1,d}(α,β) + T^{-α-β} I_{1,d}(-β,-α) ]

Since D_α(T^{-α-β}) = T^{-α-β}, we get:

    Q(D_α) [T^{-α-β} F] = T^{-α-β} Q(1 + D_α) F

The mirror contribution uses Q(1+D), not Q(D)!

This diagnostic tests whether Q-argument shifting (a0 → a0 + 1) can
replace the empirical exp(R)+5 multiplier.
"""

from __future__ import annotations

import math
from typing import Dict, List

from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import make_all_terms_k3_v2
from src.mirror_transform import (
    transform_terms_exp_factors,
    transform_terms_q_shift,
)

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Benchmarks
KAPPA_R = 1.3036
KAPPA_TARGET = 2.137

KAPPA_STAR_R = 1.1167
KAPPA_STAR_TARGET = 1.938

# Factorial normalization
FACTORIAL_NORM: Dict[str, float] = {
    "11": 1.0,
    "22": 1.0 / 4.0,
    "33": 1.0 / 36.0,
    "12": 1.0 / 2.0,
    "13": 1.0 / 6.0,
    "23": 1.0 / 12.0,
}

SYMMETRY_FACTOR: Dict[str, float] = {
    "11": 1.0, "22": 1.0, "33": 1.0,
    "12": 2.0, "13": 2.0, "23": 2.0
}


def evaluate_terms_total(
    terms: List,
    polynomials: Dict,
    R: float,
    pair_key: str,
) -> Dict[str, float]:
    """Evaluate a list of terms and return I1..I4 totals."""
    result = {"I1": 0.0, "I2": 0.0, "I3": 0.0, "I4": 0.0, "total": 0.0}

    for term in terms:
        val = evaluate_term(
            term, polynomials, N_QUAD, R=R, theta=THETA, n_quad_a=N_QUAD_A
        ).value

        # Determine which I-term this is
        name = term.name
        if name.startswith("I1"):
            result["I1"] += val
        elif name.startswith("I2"):
            result["I2"] += val
        elif name.startswith("I3"):
            result["I3"] += val
        elif name.startswith("I4"):
            result["I4"] += val

        result["total"] += val

    return result


def run_benchmark(name: str, R: float, c_target: float, polynomials: Dict):
    """Run Q-shift diagnostic for one benchmark."""
    print()
    print("=" * 78)
    print(f"Q-SHIFT DIAGNOSTIC: {name} (R={R})")
    print("=" * 78)

    # Build terms
    all_terms = make_all_terms_k3_v2(THETA, R)

    # Accumulate totals
    direct_I12 = 0.0
    direct_I34 = 0.0

    qshift_I12 = 0.0
    qshift_I34 = 0.0

    expflip_I12 = 0.0
    expflip_I34 = 0.0

    print()
    print("Per-pair comparison:")
    print(f"{'Pair':>6}  {'direct I12':>14}  {'qshift I12':>14}  {'expflip I12':>14}")

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]

        # Direct evaluation
        r_direct = evaluate_terms_total(terms, polynomials, R, pair_key)

        # Q-shifted evaluation (the GPT hypothesis)
        terms_qshift = transform_terms_q_shift(terms, shift=1.0)
        r_qshift = evaluate_terms_total(terms_qshift, polynomials, R, pair_key)

        # Exp-sign-flipped evaluation (previous approach)
        terms_expflip = transform_terms_exp_factors(terms, scale_multiplier=-1.0)
        r_expflip = evaluate_terms_total(terms_expflip, polynomials, R, pair_key)

        # Normalize
        norm = FACTORIAL_NORM[pair_key] * SYMMETRY_FACTOR[pair_key]

        direct_I12 += norm * (r_direct["I1"] + r_direct["I2"])
        direct_I34 += norm * (r_direct["I3"] + r_direct["I4"])

        qshift_I12 += norm * (r_qshift["I1"] + r_qshift["I2"])
        qshift_I34 += norm * (r_qshift["I3"] + r_qshift["I4"])

        expflip_I12 += norm * (r_expflip["I1"] + r_expflip["I2"])
        expflip_I34 += norm * (r_expflip["I3"] + r_expflip["I4"])

        print(f"  {pair_key}:  {norm * (r_direct['I1'] + r_direct['I2']):+14.8f}  "
              f"{norm * (r_qshift['I1'] + r_qshift['I2']):+14.8f}  "
              f"{norm * (r_expflip['I1'] + r_expflip['I2']):+14.8f}")

    direct_c = direct_I12 + direct_I34
    qshift_c = qshift_I12 + qshift_I34
    expflip_c = expflip_I12 + expflip_I34

    print()
    print("Totals (normalized):")
    print(f"  direct I12:     {direct_I12:+14.8f}")
    print(f"  direct I34:     {direct_I34:+14.8f}")
    print(f"  direct c:       {direct_c:+14.8f}")
    print()
    print(f"  qshift I12:     {qshift_I12:+14.8f}")
    print(f"  qshift I34:     {qshift_I34:+14.8f}")
    print(f"  qshift c:       {qshift_c:+14.8f}")
    print()
    print(f"  expflip I12:    {expflip_I12:+14.8f}")
    print(f"  expflip I34:    {expflip_I34:+14.8f}")
    print(f"  expflip c:      {expflip_c:+14.8f}")

    # Recombination tests
    print()
    print("Recombination Analysis:")
    print(f"  c_target: {c_target:+14.8f}")
    print()

    exp_2R = math.exp(2.0 * R)
    exp_R_plus_5 = math.exp(R) + 5.0

    # Model 1: direct + qshift (no multiplier needed?)
    c_model_1 = direct_c + qshift_c
    print(f"Model 1: direct + qshift (no multiplier)")
    print(f"  c = {direct_c:.6f} + {qshift_c:.6f} = {c_model_1:+14.8f}")
    print(f"  gap: {(c_model_1 - c_target) / c_target * 100:+.2f}%")
    print()

    # Model 2: direct + exp(2R) * qshift
    c_model_2 = direct_c + exp_2R * qshift_c
    print(f"Model 2: direct + exp(2R)*qshift")
    print(f"  c = {direct_c:.6f} + {exp_2R:.4f} * {qshift_c:.6f} = {c_model_2:+14.8f}")
    print(f"  gap: {(c_model_2 - c_target) / c_target * 100:+.2f}%")
    print()

    # Model 3: direct I34 only + (direct + exp(2R)*qshift) for I12
    # This tests: maybe only I12 gets the Q-shift mirror
    c_model_3 = direct_I34 + direct_I12 + exp_2R * qshift_I12
    print(f"Model 3: I34(direct) + I12(direct + exp(2R)*qshift)")
    print(f"  c = {direct_I34:.6f} + {direct_I12:.6f} + {exp_2R:.4f}*{qshift_I12:.6f}")
    print(f"  c = {c_model_3:+14.8f}")
    print(f"  gap: {(c_model_3 - c_target) / c_target * 100:+.2f}%")
    print()

    # Model 4: Compare qshift vs expflip
    # What multiplier on qshift_I12 would hit target?
    if qshift_I12 != 0:
        m_qshift = (c_target - direct_c) / qshift_I12
    else:
        m_qshift = float("inf")
    if expflip_I12 != 0:
        m_expflip = (c_target - direct_c) / expflip_I12
    else:
        m_expflip = float("inf")

    print(f"Multiplier needed to hit target (from I12 mirror only):")
    print(f"  Using qshift:  m = {m_qshift:+14.8f}")
    print(f"  Using expflip: m = {m_expflip:+14.8f}")
    print(f"  exp(R)+5:      m = {exp_R_plus_5:+14.8f}")
    print(f"  exp(2R):       m = {exp_2R:+14.8f}")

    return {
        "direct_c": direct_c,
        "qshift_c": qshift_c,
        "expflip_c": expflip_c,
        "direct_I12": direct_I12,
        "qshift_I12": qshift_I12,
        "expflip_I12": expflip_I12,
        "m_qshift": m_qshift,
        "m_expflip": m_expflip,
    }


def main():
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Run diagnostics
    result_k = run_benchmark("κ", R=KAPPA_R, c_target=KAPPA_TARGET, polynomials=polys_k)
    result_s = run_benchmark("κ*", R=KAPPA_STAR_R, c_target=KAPPA_STAR_TARGET, polynomials=polys_s)

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    print("Multiplier comparison (m_needed for I12 mirror only):")
    print(f"  κ  qshift:  {result_k['m_qshift']:.6f}")
    print(f"  κ* qshift:  {result_s['m_qshift']:.6f}")
    print(f"  Difference: {abs(result_k['m_qshift'] - result_s['m_qshift']):.6f}")
    print()
    print(f"  κ  expflip: {result_k['m_expflip']:.6f}")
    print(f"  κ* expflip: {result_s['m_expflip']:.6f}")
    print(f"  Difference: {abs(result_k['m_expflip'] - result_s['m_expflip']):.6f}")
    print()

    # Check if qshift produces more consistent multipliers than expflip
    qshift_consistency = abs(result_k['m_qshift'] - result_s['m_qshift'])
    expflip_consistency = abs(result_k['m_expflip'] - result_s['m_expflip'])

    print("Consistency analysis:")
    print(f"  qshift m-variation:  {qshift_consistency:.6f}")
    print(f"  expflip m-variation: {expflip_consistency:.6f}")

    if qshift_consistency < expflip_consistency:
        print("  → qshift produces more consistent multipliers across benchmarks")
    else:
        print("  → expflip produces more consistent multipliers across benchmarks")


if __name__ == "__main__":
    main()
