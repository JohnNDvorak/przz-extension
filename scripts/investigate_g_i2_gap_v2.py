"""
scripts/investigate_g_i2_gap_v2.py
Phase 46.3: Deeper Investigation of g_I2 Gap via Mirror Assembly

FINDINGS FROM V1:
=================
- Q changes I2(+R)/I2(-R) ratio by -74% (HUGE effect)
- But g_I2 calibration gap is only +0.58%
- This suggests Q effects are already partially absorbed somewhere

NEW HYPOTHESIS:
===============
The mirror assembly formula is:
    c = I12(+R) + m × I12(-R)
    where m = g × [exp(R) + (2K-1)]

The g_baseline formula assumes Q=1. But with real Q:
1. I2(+R) gets attenuated by Q
2. I2(-R) gets attenuated by Q differently
3. This changes the "effective m" needed

The 0.58% gap might come from:
    g_I2_effective = g_baseline × f(Q_attenuation_asymmetry)

Let's measure this by:
1. Computing what g would make mirror formula work with Q=1
2. Computing what g would make mirror formula work with real Q
3. Seeing if the ratio explains the 0.58% gap

Created: 2025-12-27 (Phase 46.3)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict
import math

from src.polynomials import load_przz_polynomials, Polynomial
from src.unified_i2_paper import compute_I2_unified_paper


@dataclass
class MirrorAnalysisResult:
    """Result of mirror assembly analysis."""

    # I2 totals
    I2_plus_Q1: float
    I2_minus_Q1: float
    I2_plus_real: float
    I2_minus_real: float

    # Ratios
    ratio_Q1: float          # I2(+R)/I2(-R) with Q=1
    ratio_real: float        # I2(+R)/I2(-R) with real Q
    Q_attenuation_factor: float  # ratio_real / ratio_Q1

    # Mirror components
    base: float              # exp(R) + (2K-1)
    g_baseline: float        # 1 + θ/(2K(2K+1))

    # Effective g values (if we wanted ratio to match baseline)
    # For Q=1 case: what g makes the weighted ratio work?
    # For real Q case: what g makes the weighted ratio work?
    # The difference might explain the calibration gap

    # With Q=1: I2(+R) + g×base×I2(-R) should give some target
    # With real Q: I2(+R) + g'×base×I2(-R) should give same target
    # Solve for g' / g to see if it matches the calibration gap


def compute_I2_total_with_Q_mode(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    include_Q: bool = True,
) -> float:
    """
    Compute aggregate I2 total (weighted sum over all pairs).

    Args:
        R: R parameter
        theta: theta parameter
        polynomials: Dict with P1, P2, P3, Q
        n_quad: Quadrature points
        include_Q: Whether to include Q factor

    Returns:
        Aggregate I2 total
    """
    pairs = [
        (1, 1, 1.0),  # (ell1, ell2, weight_symmetry)
        (2, 2, 1.0),
        (3, 3, 1.0),
        (1, 2, 2.0),
        (1, 3, 2.0),
        (2, 3, 2.0),
    ]

    factorial = [1, 1, 2, 6]
    total = 0.0

    for ell1, ell2, symmetry in pairs:
        fact_norm = 1.0 / (factorial[ell1] * factorial[ell2])
        weight = symmetry * fact_norm

        result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=include_Q,
        )
        total += weight * result.I2_value

    return total


def analyze_mirror_assembly(
    R: float = 1.3036,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> MirrorAnalysisResult:
    """
    Analyze how Q affects mirror assembly and derive effective g correction.

    The mirror formula is: I2_combined = I2(+R) + m × I2(-R)
    where m = g × base and base = exp(R) + (2K-1)

    We'll compute:
    1. What happens with Q=1 (baseline)
    2. What happens with real Q
    3. What effective g is needed to maintain the same combined value

    Args:
        R: R parameter
        theta: theta parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points

    Returns:
        MirrorAnalysisResult with detailed breakdown
    """
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_real = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    Q1 = Polynomial(coeffs=[1.0])
    polynomials_Q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q1}

    # Compute I2 totals at +R and -R with both Q modes
    I2_plus_Q1 = compute_I2_total_with_Q_mode(R, theta, polynomials_Q1, n_quad, include_Q=False)
    I2_minus_Q1 = compute_I2_total_with_Q_mode(-R, theta, polynomials_Q1, n_quad, include_Q=False)

    I2_plus_real = compute_I2_total_with_Q_mode(R, theta, polynomials_real, n_quad, include_Q=True)
    I2_minus_real = compute_I2_total_with_Q_mode(-R, theta, polynomials_real, n_quad, include_Q=True)

    # Compute ratios
    ratio_Q1 = I2_plus_Q1 / I2_minus_Q1 if abs(I2_minus_Q1) > 1e-15 else float('inf')
    ratio_real = I2_plus_real / I2_minus_real if abs(I2_minus_real) > 1e-15 else float('inf')

    Q_attenuation_factor = ratio_real / ratio_Q1 if ratio_Q1 != 0 else float('nan')

    # Compute baseline values
    base = math.exp(R) + (2 * K - 1)
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    return MirrorAnalysisResult(
        I2_plus_Q1=I2_plus_Q1,
        I2_minus_Q1=I2_minus_Q1,
        I2_plus_real=I2_plus_real,
        I2_minus_real=I2_minus_real,
        ratio_Q1=ratio_Q1,
        ratio_real=ratio_real,
        Q_attenuation_factor=Q_attenuation_factor,
        base=base,
        g_baseline=g_baseline,
    )


def run_mirror_investigation(
    R: float = 1.3036,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> None:
    """
    Run full mirror assembly investigation.

    Args:
        R: R parameter
        theta: theta parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points
    """
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    g_I2_calibrated = 1.01945154
    gap_pct = (g_I2_calibrated / g_baseline - 1) * 100

    print("=" * 80)
    print("PHASE 46.3: MIRROR ASSEMBLY ANALYSIS FOR g_I2 GAP")
    print("=" * 80)
    print()
    print(f"Parameters: R={R}, θ={theta:.10f}, K={K}")
    print()
    print(f"g_baseline:      {g_baseline:.8f}")
    print(f"g_I2_calibrated: {g_I2_calibrated:.8f}")
    print(f"Gap:             {gap_pct:+.4f}%")
    print()

    # Analyze mirror assembly
    result = analyze_mirror_assembly(R, theta, K, n_quad)

    print("=" * 80)
    print("I2 VALUES AT +R AND -R")
    print("=" * 80)
    print()
    print(f"With Q=1:")
    print(f"  I2(+R) = {result.I2_plus_Q1:+.8f}")
    print(f"  I2(-R) = {result.I2_minus_Q1:+.8f}")
    print(f"  Ratio  = {result.ratio_Q1:.8f}")
    print()
    print(f"With real Q:")
    print(f"  I2(+R) = {result.I2_plus_real:+.8f}")
    print(f"  I2(-R) = {result.I2_minus_real:+.8f}")
    print(f"  Ratio  = {result.ratio_real:.8f}")
    print()
    print(f"Q attenuation factor (ratio_real / ratio_Q1):")
    print(f"  {result.Q_attenuation_factor:.8f}")
    print(f"  Change: {(result.Q_attenuation_factor - 1) * 100:+.4f}%")
    print()

    # Now let's think about mirror assembly
    print("=" * 80)
    print("MIRROR ASSEMBLY ANALYSIS")
    print("=" * 80)
    print()
    print("The mirror formula is: I2_combined = I2(+R) + m × I2(-R)")
    print(f"where m = g × base and base = exp(R) + (2K-1) = {result.base:.6f}")
    print()

    # Let's assume we want I2_combined to have some value C (the target)
    # We don't know C, but we can look at how g must change

    # With Q=1: C = I2(+R,Q1) + g_baseline × base × I2(-R,Q1)
    # With real Q: C' = I2(+R,real) + g' × base × I2(-R,real)

    # If we want C = C' (same combined value), then:
    # I2(+R,Q1) + g_baseline × base × I2(-R,Q1) = I2(+R,real) + g' × base × I2(-R,real)
    #
    # Solve for g':
    # g' = [I2(+R,Q1) - I2(+R,real) + g_baseline × base × I2(-R,Q1)] / [base × I2(-R,real)]

    m_baseline = g_baseline * result.base

    C_Q1 = result.I2_plus_Q1 + m_baseline * result.I2_minus_Q1
    C_real = result.I2_plus_real + m_baseline * result.I2_minus_real

    print(f"With g_baseline = {g_baseline:.8f}:")
    print(f"  I2_combined (Q=1):   {C_Q1:+.8f}")
    print(f"  I2_combined (real Q): {C_real:+.8f}")
    print(f"  Ratio: {C_real / C_Q1:.8f}")
    print()

    # Now solve for g' that makes C_real = C_Q1
    # I2(+R,real) + g' × base × I2(-R,real) = C_Q1
    # g' = (C_Q1 - I2(+R,real)) / (base × I2(-R,real))

    if abs(result.I2_minus_real * result.base) > 1e-15:
        g_prime = (C_Q1 - result.I2_plus_real) / (result.base * result.I2_minus_real)
        g_prime_gap = (g_prime / g_baseline - 1) * 100

        print(f"To match Q=1 combined value with real Q, we'd need:")
        print(f"  g' = {g_prime:.8f}")
        print(f"  g' / g_baseline = {g_prime / g_baseline:.8f}")
        print(f"  Gap: {g_prime_gap:+.4f}%")
        print()
        print(f"Calibration gap is {gap_pct:+.4f}%")
        print()

        if abs(g_prime_gap - gap_pct) < 0.1:
            print("STRONG CORRELATION!")
            print("The g' needed to compensate for Q attenuation matches g_I2_calibrated!")
        else:
            correlation = g_prime_gap / gap_pct if abs(gap_pct) > 1e-6 else float('nan')
            print(f"Correlation: {correlation:.2f}x")
            print()
            if abs(correlation) > 0.5:
                print("There's partial correlation, but other effects are involved.")
            else:
                print("Low correlation - the mechanisms may be different.")
    else:
        print("Cannot solve for g' - denominator too small")

    print()

    # Alternative analysis: look at attenuation factors
    print("=" * 80)
    print("ATTENUATION FACTOR ANALYSIS")
    print("=" * 80)
    print()

    # Q attenuates I2(+R) and I2(-R) differently
    # Attenuation at +R: I2(+R,real) / I2(+R,Q1)
    # Attenuation at -R: I2(-R,real) / I2(-R,Q1)

    atten_plus = result.I2_plus_real / result.I2_plus_Q1 if abs(result.I2_plus_Q1) > 1e-15 else 0
    atten_minus = result.I2_minus_real / result.I2_minus_Q1 if abs(result.I2_minus_Q1) > 1e-15 else 0

    print(f"Q attenuation at +R: {atten_plus:.8f} ({(atten_plus - 1) * 100:+.4f}%)")
    print(f"Q attenuation at -R: {atten_minus:.8f} ({(atten_minus - 1) * 100:+.4f}%)")
    print()

    asymmetry = atten_plus / atten_minus if abs(atten_minus) > 1e-15 else float('inf')
    print(f"Attenuation asymmetry (atten_plus / atten_minus): {asymmetry:.8f}")
    print()

    # If Q attenuates +R and -R equally, asymmetry = 1
    # If Q attenuates +R more than -R, asymmetry < 1
    # If Q attenuates -R more than +R, asymmetry > 1

    if asymmetry < 0.99:
        print("Q attenuates +R MORE than -R")
        print("This reduces the I2(+R)/I2(-R) ratio")
        print("Mirror assembly would need LARGER g to compensate")
    elif asymmetry > 1.01:
        print("Q attenuates -R MORE than +R")
        print("This increases the I2(+R)/I2(-R) ratio")
        print("Mirror assembly would need SMALLER g to compensate")
    else:
        print("Q attenuates +R and -R roughly equally")
        print("No asymmetric correction needed")

    print()

    # Final synthesis
    print("=" * 80)
    print("SYNTHESIS")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print("-------------")
    print(f"1. Q changes I2(+R)/I2(-R) ratio by {(result.Q_attenuation_factor - 1) * 100:+.4f}%")
    print(f"2. Q attenuation asymmetry: {asymmetry:.6f}")
    print(f"3. g_I2 calibration gap: {gap_pct:+.4f}%")
    print()

    if abs(result.I2_minus_real * result.base) > 1e-15:
        g_prime = (C_Q1 - result.I2_plus_real) / (result.base * result.I2_minus_real)
        g_prime_gap = (g_prime / g_baseline - 1) * 100
        print(f"4. g' needed to match Q=1 target: {g_prime:.8f} (gap: {g_prime_gap:+.4f}%)")
        print()

        if abs(g_prime_gap - gap_pct) < 0.2:
            print("CONCLUSION:")
            print("-----------")
            print("The g_I2 calibration gap is EXPLAINED by Q attenuation asymmetry!")
            print()
            print("Formula:")
            print(f"  g_I2_effective = g_baseline × (C_Q1 - I2(+R,real)) / (C_Q1 - I2(+R,Q1))")
            print(f"                 = {g_baseline:.8f} × {g_prime / g_baseline:.8f}")
            print(f"                 = {g_prime:.8f}")
        else:
            print("The g' correction doesn't fully explain the calibration gap.")
            print("There may be additional mechanisms involved.")


if __name__ == "__main__":
    print()
    print("BENCHMARK 1: κ case (R=1.3036)")
    print()
    run_mirror_investigation(R=1.3036, theta=4/7, K=3, n_quad=60)
    print()
    print()
    print("=" * 80)
    print("=" * 80)
    print()
    print("BENCHMARK 2: κ* case (R=1.1167)")
    print()
    run_mirror_investigation(R=1.1167, theta=4/7, K=3, n_quad=60)
