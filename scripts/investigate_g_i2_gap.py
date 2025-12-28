"""
scripts/investigate_g_i2_gap.py
Phase 46.3: Investigate g_I2 Calibration Gap

QUESTION:
=========
Why does g_I2 need to be 1.01945 (calibrated) instead of 1.0136 (g_baseline)?

CONTEXT:
========
- g_baseline = 1 + θ/(2K(2K+1)) = 1.0136 (first-principles from Beta moment)
- g_I2_calibrated = 1.01945154 (from 2-benchmark solve)
- Gap: 0.58% higher than theoretical baseline
- Hypothesis: Q polynomial creates differential attenuation at +R vs -R

EXPERIMENT DESIGN:
==================
We'll measure I2(-R)/I2(+R) ratio with three Q modes:
1. Q=1 (baseline case)
2. Q=frozen (Q(t)² reweighting only)
3. Q=real (full PRZZ Q polynomial)

If Q changes the ratio, it creates a Q-dependent correction:
    g_I2_effective = g_baseline × f(Q)

We'll also look at per-pair patterns to see if certain pairs are more sensitive
to Q effects.

Created: 2025-12-27 (Phase 46.3)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.polynomials import load_przz_polynomials, Polynomial
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode
from src.unified_i2_paper import compute_I2_unified_paper


@dataclass
class I2RatioResult:
    """Result of I2(+R) vs I2(-R) ratio measurement."""

    pair_key: str
    ell1: int
    ell2: int

    # I2 values at +R
    I2_plus_Q1: float      # Q=1
    I2_plus_frozen: float  # Q(t)² frozen
    I2_plus_real: float    # Full Q

    # I2 values at -R
    I2_minus_Q1: float
    I2_minus_frozen: float
    I2_minus_real: float

    @property
    def ratio_Q1(self) -> float:
        """I2(+R)/I2(-R) with Q=1."""
        if abs(self.I2_minus_Q1) < 1e-15:
            return float('inf')
        return self.I2_plus_Q1 / self.I2_minus_Q1

    @property
    def ratio_frozen(self) -> float:
        """I2(+R)/I2(-R) with frozen Q."""
        if abs(self.I2_minus_frozen) < 1e-15:
            return float('inf')
        return self.I2_plus_frozen / self.I2_minus_frozen

    @property
    def ratio_real(self) -> float:
        """I2(+R)/I2(-R) with real Q."""
        if abs(self.I2_minus_real) < 1e-15:
            return float('inf')
        return self.I2_plus_real / self.I2_minus_real

    @property
    def Q_frozen_effect_on_ratio(self) -> float:
        """How much does frozen Q change the ratio vs Q=1?"""
        return self.ratio_frozen / self.ratio_Q1 - 1.0

    @property
    def Q_real_effect_on_ratio(self) -> float:
        """How much does real Q change the ratio vs Q=1?"""
        return self.ratio_real / self.ratio_Q1 - 1.0

    @property
    def Q_derivative_effect_on_ratio(self) -> float:
        """Effect from Q derivatives (real - frozen)."""
        return self.ratio_real / self.ratio_frozen - 1.0


def compute_I2_with_Q_mode(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials_Q_mode: Dict,
    n_quad: int = 60,
    include_Q: bool = True,
) -> float:
    """
    Compute I2 with specified Q polynomial.

    Args:
        R: R parameter
        theta: theta parameter
        ell1, ell2: Pair indices
        polynomials_Q_mode: Dict with P1, P2, P3, Q (Q can be constant or real)
        n_quad: Quadrature points
        include_Q: Whether to include Q factor

    Returns:
        I2 value
    """
    result = compute_I2_unified_paper(
        R, theta, ell1, ell2, polynomials_Q_mode,
        n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        include_Q=include_Q,
    )
    return result.I2_value


def measure_I2_ratio_for_pair(
    ell1: int,
    ell2: int,
    R: float,
    theta: float,
    polynomials_real: Dict,
    polynomials_Q1: Dict,
    n_quad: int = 60,
) -> I2RatioResult:
    """
    Measure I2(+R)/I2(-R) ratio with different Q modes.

    Args:
        ell1, ell2: Pair indices
        R: R parameter (we'll use +R and -R)
        theta: theta parameter
        polynomials_real: Polynomials with real PRZZ Q
        polynomials_Q1: Polynomials with Q=1
        n_quad: Quadrature points

    Returns:
        I2RatioResult with ratios under different Q modes
    """
    # For frozen Q, we use Q=1 since I2 uses Q(t)² already (no x,y dependence)
    # So frozen and Q=1 should be identical for I2

    # At +R with different Q modes
    I2_plus_Q1 = compute_I2_with_Q_mode(
        R, theta, ell1, ell2, polynomials_Q1, n_quad, include_Q=False
    )
    I2_plus_frozen = I2_plus_Q1  # I2 already uses Q(t)², so frozen = Q1
    I2_plus_real = compute_I2_with_Q_mode(
        R, theta, ell1, ell2, polynomials_real, n_quad, include_Q=True
    )

    # At -R with different Q modes
    I2_minus_Q1 = compute_I2_with_Q_mode(
        -R, theta, ell1, ell2, polynomials_Q1, n_quad, include_Q=False
    )
    I2_minus_frozen = I2_minus_Q1  # Same as above
    I2_minus_real = compute_I2_with_Q_mode(
        -R, theta, ell1, ell2, polynomials_real, n_quad, include_Q=True
    )

    pair_key = f"({ell1},{ell2})"

    return I2RatioResult(
        pair_key=pair_key,
        ell1=ell1,
        ell2=ell2,
        I2_plus_Q1=I2_plus_Q1,
        I2_plus_frozen=I2_plus_frozen,
        I2_plus_real=I2_plus_real,
        I2_minus_Q1=I2_minus_Q1,
        I2_minus_frozen=I2_minus_frozen,
        I2_minus_real=I2_minus_real,
    )


def measure_aggregate_I2_totals(
    R: float,
    theta: float,
    polynomials_real: Dict,
    polynomials_Q1: Dict,
    n_quad: int = 60,
) -> Dict:
    """
    Measure aggregate I2 totals at +R and -R with different Q modes.

    This computes the weighted sum over all pairs to see the overall effect.

    Returns:
        Dict with aggregate I2(+R), I2(-R), and ratios
    """
    pairs = [
        ("11", 1, 1, 1.0),  # (pair_key, ell1, ell2, weight)
        ("22", 2, 2, 1.0),
        ("33", 3, 3, 1.0),
        ("12", 1, 2, 2.0),
        ("13", 1, 3, 2.0),
        ("23", 2, 3, 2.0),
    ]

    # Factorial normalization
    factorial = [1, 1, 2, 6]

    I2_plus_Q1_total = 0.0
    I2_plus_real_total = 0.0
    I2_minus_Q1_total = 0.0
    I2_minus_real_total = 0.0

    for pair_key, ell1, ell2, symmetry in pairs:
        fact_norm = 1.0 / (factorial[ell1] * factorial[ell2])
        weight = symmetry * fact_norm

        # +R
        I2_plus_Q1 = compute_I2_with_Q_mode(
            R, theta, ell1, ell2, polynomials_Q1, n_quad, include_Q=False
        )
        I2_plus_real = compute_I2_with_Q_mode(
            R, theta, ell1, ell2, polynomials_real, n_quad, include_Q=True
        )

        # -R
        I2_minus_Q1 = compute_I2_with_Q_mode(
            -R, theta, ell1, ell2, polynomials_Q1, n_quad, include_Q=False
        )
        I2_minus_real = compute_I2_with_Q_mode(
            -R, theta, ell1, ell2, polynomials_real, n_quad, include_Q=True
        )

        I2_plus_Q1_total += weight * I2_plus_Q1
        I2_plus_real_total += weight * I2_plus_real
        I2_minus_Q1_total += weight * I2_minus_Q1
        I2_minus_real_total += weight * I2_minus_real

    ratio_Q1 = I2_plus_Q1_total / I2_minus_Q1_total if abs(I2_minus_Q1_total) > 1e-15 else float('inf')
    ratio_real = I2_plus_real_total / I2_minus_real_total if abs(I2_minus_real_total) > 1e-15 else float('inf')

    Q_effect_on_ratio = ratio_real / ratio_Q1 - 1.0 if ratio_Q1 != 0 else float('nan')

    return {
        "I2_plus_Q1": I2_plus_Q1_total,
        "I2_plus_real": I2_plus_real_total,
        "I2_minus_Q1": I2_minus_Q1_total,
        "I2_minus_real": I2_minus_real_total,
        "ratio_Q1": ratio_Q1,
        "ratio_real": ratio_real,
        "Q_effect_on_ratio_pct": Q_effect_on_ratio * 100,
    }


def run_g_i2_gap_investigation(
    R: float = 1.3036,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> None:
    """
    Run the full investigation of g_I2 calibration gap.

    Args:
        R: R parameter
        theta: theta parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points
    """
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_real = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Create Q=1 version
    Q1 = Polynomial(coeffs=[1.0])
    polynomials_Q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q1}

    # Compute theoretical baseline
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    g_I2_calibrated = 1.01945154  # From correction_policy.py
    gap_pct = (g_I2_calibrated / g_baseline - 1) * 100

    print("=" * 80)
    print("PHASE 46.3: INVESTIGATING g_I2 CALIBRATION GAP")
    print("=" * 80)
    print()
    print(f"Parameters: R={R}, θ={theta:.10f}, K={K}")
    print()
    print(f"g_baseline (first-principles):  {g_baseline:.8f}")
    print(f"g_I2_calibrated (2-benchmark):  {g_I2_calibrated:.8f}")
    print(f"Gap:                             {gap_pct:+.4f}%")
    print()
    print("HYPOTHESIS:")
    print("-----------")
    print("The Q polynomial creates differential attenuation at +R vs -R,")
    print("requiring a Q-dependent correction on top of g_baseline.")
    print()
    print("We'll measure I2(-R)/I2(+R) ratio with Q=1 vs real Q to see if")
    print("Q changes the ratio in a way that explains the 0.58% gap.")
    print()

    # Measure per-pair ratios
    print("=" * 80)
    print("PER-PAIR I2 RATIO ANALYSIS")
    print("=" * 80)
    print()

    pairs = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]
    per_pair_results = []

    for ell1, ell2 in pairs:
        result = measure_I2_ratio_for_pair(
            ell1, ell2, R, theta, polynomials_real, polynomials_Q1, n_quad
        )
        per_pair_results.append(result)

    # Print header
    print(f"{'Pair':<6} | {'I2(+R,Q1)':<12} | {'I2(-R,Q1)':<12} | {'Ratio(Q1)':<11} | {'I2(+R,Q)':<12} | {'I2(-R,Q)':<12} | {'Ratio(Q)':<11} | {'Q effect':<10}")
    print("-" * 115)

    for r in per_pair_results:
        Q_effect_pct = r.Q_real_effect_on_ratio * 100
        print(f"{r.pair_key:<6} | {r.I2_plus_Q1:+11.6f} | {r.I2_minus_Q1:+11.6f} | {r.ratio_Q1:10.6f} | "
              f"{r.I2_plus_real:+11.6f} | {r.I2_minus_real:+11.6f} | {r.ratio_real:10.6f} | {Q_effect_pct:+9.4f}%")

    print()

    # Measure aggregate totals
    print("=" * 80)
    print("AGGREGATE I2 TOTALS (WEIGHTED SUM OVER ALL PAIRS)")
    print("=" * 80)
    print()

    agg = measure_aggregate_I2_totals(R, theta, polynomials_real, polynomials_Q1, n_quad)

    print(f"  I2(+R) with Q=1:         {agg['I2_plus_Q1']:+.8f}")
    print(f"  I2(+R) with real Q:      {agg['I2_plus_real']:+.8f}")
    print()
    print(f"  I2(-R) with Q=1:         {agg['I2_minus_Q1']:+.8f}")
    print(f"  I2(-R) with real Q:      {agg['I2_minus_real']:+.8f}")
    print()
    print(f"  Ratio I2(+R)/I2(-R) with Q=1:   {agg['ratio_Q1']:.8f}")
    print(f"  Ratio I2(+R)/I2(-R) with Q:     {agg['ratio_real']:.8f}")
    print()
    print(f"  Q effect on ratio:              {agg['Q_effect_on_ratio_pct']:+.4f}%")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # The mirror formula is: c = I12(+R) + m × I12(-R)
    # where m = g × base and base = exp(R) + (2K-1)
    #
    # If Q changes the I2(+R)/I2(-R) ratio, it effectively changes the
    # "required m" to achieve the correct c.
    #
    # For g_baseline, we expect: m_baseline = g_baseline × base
    # For calibrated g_I2, we need: m_calibrated = g_I2_calibrated × base
    #
    # The question is: does the Q-induced ratio change explain the gap?

    Q_ratio_effect = agg['Q_effect_on_ratio_pct']

    print(f"1. Q polynomial changes I2(+R)/I2(-R) ratio by {Q_ratio_effect:+.4f}%")
    print()
    print(f"2. g_I2 calibration gap is {gap_pct:+.4f}%")
    print()

    if abs(Q_ratio_effect) < 0.01:
        print("INTERPRETATION:")
        print("---------------")
        print("The Q effect on I2 ratio is negligible (<0.01%).")
        print()
        print("This suggests the g_I2 calibration gap is NOT from Q changing")
        print("the I2(+R)/I2(-R) ratio. The gap likely comes from:")
        print()
        print("  a) Different mechanism (e.g., Q interacts with mirror assembly)")
        print("  b) I1 contamination (if I1/I2 separation isn't perfect)")
        print("  c) Missing higher-order correction in g_baseline derivation")
    else:
        correlation = abs(Q_ratio_effect / gap_pct)
        print("INTERPRETATION:")
        print("---------------")
        print(f"The Q effect on ratio ({Q_ratio_effect:+.4f}%) has magnitude {correlation:.2f}x")
        print(f"the calibration gap ({gap_pct:+.4f}%).")
        print()
        if 0.8 <= correlation <= 1.2:
            print("STRONG CORRELATION! The Q-induced ratio change explains the gap.")
            print()
            print("Proposed formula:")
            print(f"  g_I2_effective = g_baseline × (1 + Q_ratio_correction)")
            print(f"  where Q_ratio_correction ≈ {Q_ratio_effect/100:.6f}")
        elif correlation > 1.2:
            print("Q effect is LARGER than the calibration gap.")
            print("This suggests Q changes the ratio, but there may be other")
            print("compensating effects.")
        else:
            print("Q effect is present but doesn't fully explain the gap.")
            print("There may be additional mechanisms at play.")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Check if the Q effect scales with R (test at R=1.1167 for κ*)")
    print("2. Look for Q×mirror interaction in the formula assembly")
    print("3. Analyze whether g_I2 gap has polynomial-degree dependence")
    print()


if __name__ == "__main__":
    run_g_i2_gap_investigation()
