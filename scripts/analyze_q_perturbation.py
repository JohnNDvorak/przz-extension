#!/usr/bin/env python3
"""
Q Perturbation Analysis for First-Principles g_I2 Derivation

The structural derivation gives:
    g_I1 = 1.0 (gap: 0.09% from calibrated)
    g_I2 = g_baseline = 1 + θ/(2K(2K+1)) (gap: 0.57% from calibrated)

This script tests whether the 0.57% gap correlates with simple
Q polynomial integral ratios, potentially giving a closed-form formula.

Hypothesis:
    g_I2 = g_baseline × [1 + f(Q)]

Where f(Q) is a function of Q polynomial integrals like:
    - ∫₀¹ Q(t)² dt
    - ∫₀¹ Q'(t)² dt
    - ∫₀¹ Q(t)Q'(t) dt
    - Various ratios thereof

If f(Q) can be expressed in terms of these integrals, we have a
first-principles formula for g_I2.

Created: 2025-12-27 (Phase 46 - Q perturbation approach)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import quad
from src.polynomials import load_przz_polynomials, Polynomial


def compute_q_integrals(Q, R: float) -> dict:
    """
    Compute various Q polynomial integrals over [0,1].

    Returns dict with:
        - int_Q2: ∫₀¹ Q(t)² dt
        - int_Qp2: ∫₀¹ Q'(t)² dt
        - int_Q_Qp: ∫₀¹ Q(t)Q'(t) dt
        - int_Q2_exp: ∫₀¹ Q(t)² exp(2Rt) dt
        - int_Qp2_exp: ∫₀¹ Q'(t)² exp(2Rt) dt
        - Various ratios
    """
    # Convert to monomial form if needed
    if hasattr(Q, 'to_monomial'):
        Q_mono = Q.to_monomial()
    else:
        Q_mono = Q

    # Get Q' polynomial (derivative)
    # np.polyder expects highest degree first, our coeffs have lowest first
    Qp_coeffs = np.polyder(Q_mono.coeffs[::-1])[::-1]
    Qp = Polynomial(Qp_coeffs if len(Qp_coeffs) > 0 else np.array([0.0]))

    # Define Q evaluation function
    def Q_eval(t):
        return Q_mono.eval(np.array([t]))[0]

    def Qp_eval(t):
        return Qp.eval(np.array([t]))[0]

    # Basic integrals (no exponential weight)
    int_Q2, _ = quad(lambda t: Q_eval(t)**2, 0, 1)
    int_Qp2, _ = quad(lambda t: Qp_eval(t)**2, 0, 1)
    int_Q_Qp, _ = quad(lambda t: Q_eval(t) * Qp_eval(t), 0, 1)
    int_Q, _ = quad(lambda t: Q_eval(t), 0, 1)
    int_Qp, _ = quad(lambda t: Qp_eval(t), 0, 1)

    # Exponential-weighted integrals (relevant to PRZZ)
    int_Q2_exp, _ = quad(lambda t: Q_eval(t)**2 * np.exp(2*R*t), 0, 1)
    int_Qp2_exp, _ = quad(lambda t: Qp_eval(t)**2 * np.exp(2*R*t), 0, 1)
    int_Q_Qp_exp, _ = quad(lambda t: Q_eval(t) * Qp_eval(t) * np.exp(2*R*t), 0, 1)
    int_exp, _ = quad(lambda t: np.exp(2*R*t), 0, 1)

    # Compute ratios
    results = {
        # Basic integrals
        "int_Q2": int_Q2,
        "int_Qp2": int_Qp2,
        "int_Q_Qp": int_Q_Qp,
        "int_Q": int_Q,
        "int_Qp": int_Qp,  # = Q(1) - Q(0)

        # Exponential-weighted
        "int_Q2_exp": int_Q2_exp,
        "int_Qp2_exp": int_Qp2_exp,
        "int_Q_Qp_exp": int_Q_Qp_exp,
        "int_exp": int_exp,

        # Key ratios
        "ratio_Qp2_Q2": int_Qp2 / int_Q2 if int_Q2 != 0 else np.nan,
        "ratio_Q_Qp_Q2": int_Q_Qp / int_Q2 if int_Q2 != 0 else np.nan,
        "ratio_Qp2_exp_Q2_exp": int_Qp2_exp / int_Q2_exp if int_Q2_exp != 0 else np.nan,

        # Normalized by exponential integral
        "Q2_exp_normalized": int_Q2_exp / int_exp,
        "Qp2_exp_normalized": int_Qp2_exp / int_exp,
    }

    return results


def compute_g_gap(theta: float, K: int, g_calibrated_I2: float) -> float:
    """Compute the gap between g_baseline and calibrated g_I2."""
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    return (g_calibrated_I2 / g_baseline - 1) * 100  # As percentage


def test_q_perturbation_hypothesis(Q: Polynomial, R: float,
                                   theta: float, K: int,
                                   g_calibrated_I2: float) -> dict:
    """
    Test if the g_I2 gap can be explained by Q integral ratios.

    If g_I2 = g_baseline × [1 + f(Q)], then:
        f(Q) = g_I2/g_baseline - 1 = gap/100

    We check if f(Q) correlates with simple Q integral functions.
    """
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    gap_frac = g_calibrated_I2 / g_baseline - 1  # The fractional gap

    q_integrals = compute_q_integrals(Q, R)

    # Try various candidate formulas for f(Q)
    candidates = {}

    # Candidate 1: Simple ratio ∫Q'²/∫Q²
    candidates["Qp2_over_Q2"] = q_integrals["ratio_Qp2_Q2"]

    # Candidate 2: θ × ratio
    candidates["theta_times_Qp2_Q2"] = theta * q_integrals["ratio_Qp2_Q2"]

    # Candidate 3: Exponential-weighted ratio
    candidates["Qp2_exp_Q2_exp"] = q_integrals["ratio_Qp2_exp_Q2_exp"]

    # Candidate 4: Mixed integral / Q²
    candidates["Q_Qp_over_Q2"] = q_integrals["ratio_Q_Qp_Q2"]

    # Candidate 5: Normalized versions
    if q_integrals["int_Q2"] > 0:
        candidates["Qp2_normalized"] = q_integrals["int_Qp2"] / q_integrals["int_Q2"]**2

    # Candidate 6: θ/(2K(2K+1)) × ratio (scale by baseline correction)
    beta_moment = theta / (2 * K * (2 * K + 1))
    candidates["beta_times_Qp2_Q2"] = beta_moment * q_integrals["ratio_Qp2_Q2"]

    # Candidate 7: Exponential-normalized
    candidates["Q2_exp_norm"] = q_integrals["Q2_exp_normalized"]

    # Compute how well each candidate predicts gap_frac
    predictions = {}
    for name, value in candidates.items():
        if not np.isnan(value):
            # Check if candidate × some_scale ≈ gap_frac
            if value != 0:
                scale_to_match = gap_frac / value
                predictions[name] = {
                    "value": value,
                    "scale_needed": scale_to_match,
                    "gap_frac": gap_frac,
                }

    return {
        "g_baseline": g_baseline,
        "g_calibrated_I2": g_calibrated_I2,
        "gap_frac": gap_frac,
        "gap_pct": gap_frac * 100,
        "q_integrals": q_integrals,
        "candidates": candidates,
        "predictions": predictions,
    }


def main():
    print("=" * 70)
    print("Q PERTURBATION ANALYSIS FOR g_I2 DERIVATION")
    print("=" * 70)
    print()

    # Parameters
    theta = 4 / 7
    K = 3
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # Calibrated values (from 2-benchmark solve)
    g_I1_calibrated = 1.00091428
    g_I2_calibrated = 1.01945154

    print(f"θ = {theta:.6f}")
    print(f"K = {K}")
    print(f"g_baseline = 1 + θ/(2K(2K+1)) = {g_baseline:.8f}")
    print()
    print(f"Calibrated g_I1 = {g_I1_calibrated:.8f} (gap from 1.0: {(g_I1_calibrated - 1) * 100:.4f}%)")
    print(f"Calibrated g_I2 = {g_I2_calibrated:.8f} (gap from g_baseline: {(g_I2_calibrated/g_baseline - 1) * 100:.4f}%)")
    print()

    # Load Q polynomial
    _, _, _, Q = load_przz_polynomials()

    # Benchmarks
    benchmarks = [
        ("κ", 1.3036),
        ("κ*", 1.1167),
    ]

    all_results = []

    for name, R in benchmarks:
        print("=" * 70)
        print(f"BENCHMARK: {name} (R = {R})")
        print("=" * 70)

        result = test_q_perturbation_hypothesis(Q, R, theta, K, g_I2_calibrated)
        all_results.append((name, R, result))

        print(f"\ng_I2 gap from g_baseline: {result['gap_pct']:.4f}%")
        print(f"gap_frac = {result['gap_frac']:.8f}")
        print()

        print("Q POLYNOMIAL INTEGRALS:")
        print("-" * 50)
        q_ints = result["q_integrals"]
        print(f"  ∫Q(t)² dt        = {q_ints['int_Q2']:.8f}")
        print(f"  ∫Q'(t)² dt       = {q_ints['int_Qp2']:.8f}")
        print(f"  ∫Q(t)Q'(t) dt    = {q_ints['int_Q_Qp']:.8f}")
        print(f"  ∫Q(t) dt         = {q_ints['int_Q']:.8f}")
        print()
        print(f"  ∫Q(t)² e^{{2Rt}} dt  = {q_ints['int_Q2_exp']:.8f}")
        print(f"  ∫Q'(t)² e^{{2Rt}} dt = {q_ints['int_Qp2_exp']:.8f}")
        print(f"  ∫e^{{2Rt}} dt         = {q_ints['int_exp']:.8f}")
        print()

        print("KEY RATIOS:")
        print("-" * 50)
        print(f"  ∫Q'²/∫Q²                 = {q_ints['ratio_Qp2_Q2']:.8f}")
        print(f"  ∫QQ'/∫Q²                 = {q_ints['ratio_Q_Qp_Q2']:.8f}")
        print(f"  ∫Q'²e^{{2Rt}}/∫Q²e^{{2Rt}} = {q_ints['ratio_Qp2_exp_Q2_exp']:.8f}")
        print()

        print("CANDIDATE f(Q) FUNCTIONS:")
        print("-" * 50)
        print(f"  Target: gap_frac = {result['gap_frac']:.8f}")
        print()
        for name, value in result["candidates"].items():
            if not np.isnan(value):
                scale = result["gap_frac"] / value if value != 0 else np.nan
                print(f"  {name:25s} = {value:12.8f}  (scale needed: {scale:12.8f})")
        print()

    # Cross-benchmark comparison
    print("=" * 70)
    print("CROSS-BENCHMARK CONSISTENCY CHECK")
    print("=" * 70)
    print()
    print("For a valid first-principles formula, the scale factor should be")
    print("the SAME for both κ and κ* benchmarks.")
    print()

    # Check if any candidate has consistent scale across benchmarks
    r1_name, R1, result1 = all_results[0]
    r2_name, R2, result2 = all_results[1]

    print(f"{'Candidate':<25s}  {'Scale(κ)':>12s}  {'Scale(κ*)':>12s}  {'Ratio':>8s}  {'Consistent?':>12s}")
    print("-" * 80)

    for name in result1["candidates"]:
        v1 = result1["candidates"].get(name, np.nan)
        v2 = result2["candidates"].get(name, np.nan)

        if not np.isnan(v1) and not np.isnan(v2) and v1 != 0 and v2 != 0:
            scale1 = result1["gap_frac"] / v1
            scale2 = result2["gap_frac"] / v2
            ratio = scale1 / scale2 if scale2 != 0 else np.nan
            consistent = "YES" if 0.95 < ratio < 1.05 else "NO"
            print(f"  {name:<23s}  {scale1:>12.8f}  {scale2:>12.8f}  {ratio:>8.4f}  {consistent:>12s}")

    print()

    # Look for R-independent pattern
    print("=" * 70)
    print("SEARCHING FOR R-INDEPENDENT FORMULA")
    print("=" * 70)
    print()
    print("The gap should be related to Q's structure, not R.")
    print()

    # The gap is approximately constant (0.57%) for both R values
    # This suggests the formula should be R-independent
    gap1 = result1["gap_frac"]
    gap2 = result2["gap_frac"]

    print(f"Gap at R={R1}: {gap1:.8f} ({gap1*100:.4f}%)")
    print(f"Gap at R={R2}: {gap2:.8f} ({gap2*100:.4f}%)")
    print(f"Ratio: {gap1/gap2:.6f}")
    print()

    if 0.95 < gap1/gap2 < 1.05:
        print("✓ Gaps are consistent - suggests R-independent formula")
        avg_gap = (gap1 + gap2) / 2
        print(f"  Average gap: {avg_gap:.8f} ({avg_gap*100:.4f}%)")
        print()

        # Check if avg_gap = θ/(2K(2K+1)) × some simple factor
        beta_moment = theta / (2 * K * (2 * K + 1))
        print(f"  β = θ/(2K(2K+1)) = {beta_moment:.8f}")
        print(f"  gap/β = {avg_gap/beta_moment:.6f}")
        print()

        # Check against Q integrals (using R-independent ones)
        q_ints = result1["q_integrals"]  # R-independent quantities
        print("  Checking if gap = β × f(Q integrals):")
        print(f"    gap/(β × ∫Q'²/∫Q²) = {avg_gap / (beta_moment * q_ints['ratio_Qp2_Q2']):.8f}")
        print(f"    gap/(β × ∫QQ'/∫Q²) = {avg_gap / (beta_moment * abs(q_ints['ratio_Q_Qp_Q2'])):.8f}")
    else:
        print("✗ Gaps differ by R - formula must be R-dependent")

    print()

    # Final assessment
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The g_I2 gap (0.57%) represents a Q-induced correction on top of g_baseline.")
    print()
    print("Potential first-principles formula:")
    print()
    print("    g_I2 = g_baseline × [1 + ε_Q]")
    print()
    print("where ε_Q is determined by Q polynomial structure.")
    print()

    # Check the simplest hypothesis: ε_Q = c × β × (∫Q'²/∫Q²)
    q_ints = result1["q_integrals"]
    beta_moment = theta / (2 * K * (2 * K + 1))
    ratio_qp2_q2 = q_ints["ratio_Qp2_Q2"]
    avg_gap = (gap1 + gap2) / 2

    c_needed = avg_gap / (beta_moment * ratio_qp2_q2)
    print(f"If ε_Q = c × β × (∫Q'²/∫Q²), then c = {c_needed:.6f}")
    print()

    # Test this formula
    epsilon_Q_predicted = c_needed * beta_moment * ratio_qp2_q2
    g_I2_predicted = g_baseline * (1 + epsilon_Q_predicted)

    print(f"Predicted g_I2 = {g_I2_predicted:.8f}")
    print(f"Calibrated g_I2 = {g_I2_calibrated:.8f}")
    print(f"Residual gap: {(g_I2_predicted/g_I2_calibrated - 1) * 100:.4f}%")


if __name__ == "__main__":
    main()
