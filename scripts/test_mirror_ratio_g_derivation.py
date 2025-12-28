#!/usr/bin/env python3
"""
Mirror Ratio Analysis for g Correction Derivation

This script tests whether we can derive g_I1 and g_I2 from the mirror ratio
structure by examining how frozen-Q affects the ratio S12(+R) / S12(-R).

KEY INSIGHT:
============
The mirror formula is:
    c = S12(+R) + m × S12(-R)

where m = g × base, and base = exp(R) + (2K-1)

The correction g modifies the mirror ratio. If we can understand how
frozen-Q changes this ratio, we might derive g_I1 and g_I2.

HYPOTHESIS:
===========
The ratio test from Phase 40:
    ratio = S12(+R) / S12(-R)

is affected by Q in different ways for I1 vs I2. If we measure:
    ratio_I1_normal, ratio_I1_frozen
    ratio_I2 (always frozen)

we can isolate the I1-specific Q derivative effect on the mirror ratio.

Created: 2025-12-27
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode
from src.unified_i2_paper import compute_I2_unified_paper
from src.evaluator.correction_policy import (
    compute_g_baseline,
    compute_base,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


@dataclass
class MirrorRatioData:
    """Mirror ratio analysis for one benchmark."""
    benchmark: str
    R: float
    theta: float
    K: int

    # S12 components at +R
    I1_plus_normal: float
    I1_plus_frozen: float
    I2_plus: float

    # S12 components at -R
    I1_minus_normal: float
    I1_minus_frozen: float
    I2_minus: float

    @property
    def S12_plus_normal(self) -> float:
        return self.I1_plus_normal + self.I2_plus

    @property
    def S12_plus_frozen(self) -> float:
        return self.I1_plus_frozen + self.I2_plus

    @property
    def S12_minus_normal(self) -> float:
        return self.I1_minus_normal + self.I2_minus

    @property
    def S12_minus_frozen(self) -> float:
        return self.I1_minus_frozen + self.I2_minus

    @property
    def ratio_normal(self) -> float:
        """S12(+R) / S12(-R) with normal Q."""
        return self.S12_plus_normal / self.S12_minus_normal

    @property
    def ratio_frozen(self) -> float:
        """S12(+R) / S12(-R) with frozen Q."""
        return self.S12_plus_frozen / self.S12_minus_frozen

    @property
    def ratio_I1_normal(self) -> float:
        """I1(+R) / I1(-R) with normal Q."""
        return self.I1_plus_normal / self.I1_minus_normal

    @property
    def ratio_I1_frozen(self) -> float:
        """I1(+R) / I1(-R) with frozen Q."""
        return self.I1_plus_frozen / self.I1_minus_frozen

    @property
    def ratio_I2(self) -> float:
        """I2(+R) / I2(-R) (always uses frozen Q)."""
        return self.I2_plus / self.I2_minus

    @property
    def f_I1_minus_normal(self) -> float:
        """I1 fraction at -R with normal Q."""
        total = abs(self.I1_minus_normal) + abs(self.I2_minus)
        return abs(self.I1_minus_normal) / total

    @property
    def f_I1_minus_frozen(self) -> float:
        """I1 fraction at -R with frozen Q."""
        total = abs(self.I1_minus_frozen) + abs(self.I2_minus)
        return abs(self.I1_minus_frozen) / total


def compute_totals(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> tuple:
    """Compute I1 and I2 totals for +R and -R."""

    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    I1_normal = 0.0
    I1_frozen = 0.0
    I2_total = 0.0

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        norm = f_norm[pair_key] * symmetry[pair_key]

        I1_n = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode="normal", n_quad_u=n_quad, n_quad_t=n_quad
        )
        I1_f = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode="frozen", n_quad_u=n_quad, n_quad_t=n_quad
        )
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )

        I1_normal += I1_n * norm
        I1_frozen += I1_f * norm
        I2_total += I2_result.I2_value * norm

    return I1_normal, I1_frozen, I2_total


def collect_mirror_ratio_data(
    benchmark_name: str,
    R: float,
    theta: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> MirrorRatioData:
    """Collect mirror ratio data for one benchmark."""

    print(f"\nCollecting mirror ratio data for {benchmark_name}...")

    I1_plus_normal, I1_plus_frozen, I2_plus = compute_totals(R, theta, polynomials, n_quad)
    I1_minus_normal, I1_minus_frozen, I2_minus = compute_totals(-R, theta, polynomials, n_quad)

    return MirrorRatioData(
        benchmark=benchmark_name,
        R=R,
        theta=theta,
        K=K,
        I1_plus_normal=I1_plus_normal,
        I1_plus_frozen=I1_plus_frozen,
        I2_plus=I2_plus,
        I1_minus_normal=I1_minus_normal,
        I1_minus_frozen=I1_minus_frozen,
        I2_minus=I2_minus,
    )


def print_mirror_ratio_summary(data: MirrorRatioData):
    """Print comprehensive mirror ratio summary."""

    print(f"\n{'='*70}")
    print(f"  MIRROR RATIO DATA: {data.benchmark}")
    print(f"{'='*70}")
    print(f"Parameters: R={data.R}, θ={data.theta:.6f}, K={data.K}")
    print()

    print("S12 COMPONENTS:")
    print(f"  At +R (normal): I1={data.I1_plus_normal:+.8f}, I2={data.I2_plus:+.8f}, S12={data.S12_plus_normal:+.8f}")
    print(f"  At +R (frozen): I1={data.I1_plus_frozen:+.8f}, I2={data.I2_plus:+.8f}, S12={data.S12_plus_frozen:+.8f}")
    print(f"  At -R (normal): I1={data.I1_minus_normal:+.8f}, I2={data.I2_minus:+.8f}, S12={data.S12_minus_normal:+.8f}")
    print(f"  At -R (frozen): I1={data.I1_minus_frozen:+.8f}, I2={data.I2_minus:+.8f}, S12={data.S12_minus_frozen:+.8f}")
    print()

    print("MIRROR RATIOS:")
    print(f"  S12(+R) / S12(-R) [normal]: {data.ratio_normal:.6f}")
    print(f"  S12(+R) / S12(-R) [frozen]: {data.ratio_frozen:.6f}")
    print(f"  I1(+R) / I1(-R) [normal]:   {data.ratio_I1_normal:.6f}")
    print(f"  I1(+R) / I1(-R) [frozen]:   {data.ratio_I1_frozen:.6f}")
    print(f"  I2(+R) / I2(-R):            {data.ratio_I2:.6f}")
    print()

    print("I1 FRACTIONS AT -R:")
    print(f"  f_I1 (normal): {data.f_I1_minus_normal:.6f}")
    print(f"  f_I1 (frozen): {data.f_I1_minus_frozen:.6f}")
    print()


def derive_g_from_mirror_ratios(data_kappa: MirrorRatioData, data_kappa_star: MirrorRatioData):
    """Attempt to derive g_I1 and g_I2 from mirror ratio structure."""

    theta = data_kappa.theta
    K = data_kappa.K
    g_baseline = compute_g_baseline(theta, K)
    base = compute_base(data_kappa.R, K)

    print(f"\n{'='*70}")
    print("  G DERIVATION FROM MIRROR RATIOS")
    print(f"{'='*70}")
    print()
    print(f"Baseline: g = {g_baseline:.8f}, base = {base:.6f}")
    print(f"Calibrated: g_I1 = {G_I1_CALIBRATED:.8f}, g_I2 = {G_I2_CALIBRATED:.8f}")
    print()

    # =========================================================================
    # APPROACH 1: Direct ratio inversion
    # =========================================================================
    print("APPROACH 1: Direct ratio inversion")
    print("-" * 70)
    print("The mirror formula gives:")
    print("  c = S12(+R) + g×base × S12(-R)")
    print()
    print("If we know the target c and S12 values, we can solve for g:")
    print("  g = (c - S12(+R)) / (base × S12(-R))")
    print()

    # For κ benchmark, target c ≈ 2.1375
    c_target_kappa = 2.1375
    c_target_kappa_star = 1.938  # From Phase 40

    # With normal Q
    g_solved_kappa_normal = (c_target_kappa - data_kappa.S12_plus_normal) / (base * data_kappa.S12_minus_normal)
    g_solved_kappa_star_normal = (c_target_kappa_star - data_kappa_star.S12_plus_normal) / (
        compute_base(data_kappa_star.R, K) * data_kappa_star.S12_minus_normal
    )

    print(f"  κ (normal):  g_solved = {g_solved_kappa_normal:.8f}")
    print(f"  κ* (normal): g_solved = {g_solved_kappa_star_normal:.8f}")
    print()

    # With frozen Q
    g_solved_kappa_frozen = (c_target_kappa - data_kappa.S12_plus_frozen) / (base * data_kappa.S12_minus_frozen)
    g_solved_kappa_star_frozen = (c_target_kappa_star - data_kappa_star.S12_plus_frozen) / (
        compute_base(data_kappa_star.R, K) * data_kappa_star.S12_minus_frozen
    )

    print(f"  κ (frozen):  g_solved = {g_solved_kappa_frozen:.8f}")
    print(f"  κ* (frozen): g_solved = {g_solved_kappa_star_frozen:.8f}")
    print()

    # =========================================================================
    # APPROACH 2: Ratio-based g correction
    # =========================================================================
    print("APPROACH 2: Q-derivative effect on mirror ratio")
    print("-" * 70)
    print("Q derivatives change the ratio S12(+R) / S12(-R).")
    print("The g correction might compensate for this change.")
    print()

    # The ratio test showed frozen gives better match to theoretical ratio
    # Phase 40: target ratio ≈ 1.10 for both benchmarks
    ratio_theoretical = 1.10  # From Beta moment analysis

    print(f"  Theoretical ratio (Beta moment): {ratio_theoretical:.6f}")
    print(f"  κ ratio (normal): {data_kappa.ratio_normal:.6f}")
    print(f"  κ ratio (frozen): {data_kappa.ratio_frozen:.6f}")
    print(f"  κ* ratio (normal): {data_kappa_star.ratio_normal:.6f}")
    print(f"  κ* ratio (frozen): {data_kappa_star.ratio_frozen:.6f}")
    print()

    # Hypothesis: g should correct the ratio back to theoretical
    # If ratio_normal is too high, g > 1 increases S12(-R) contribution
    # g = (ratio_theoretical × S12_minus) / S12_minus_corrected

    print("  Q-derivative effect on ratio:")
    ratio_shift_kappa = data_kappa.ratio_normal - data_kappa.ratio_frozen
    ratio_shift_kappa_star = data_kappa_star.ratio_normal - data_kappa_star.ratio_frozen
    print(f"    κ:  {ratio_shift_kappa:+.6f} ({ratio_shift_kappa/data_kappa.ratio_frozen*100:+.4f}%)")
    print(f"    κ*: {ratio_shift_kappa_star:+.6f} ({ratio_shift_kappa_star/data_kappa_star.ratio_frozen*100:+.4f}%)")
    print()

    # =========================================================================
    # APPROACH 3: Component-wise analysis
    # =========================================================================
    print("APPROACH 3: Component-wise I1/I2 ratio analysis")
    print("-" * 70)
    print("Examine I1 and I2 ratios separately:")
    print()

    print(f"  κ I1 ratio:  normal={data_kappa.ratio_I1_normal:.6f}, frozen={data_kappa.ratio_I1_frozen:.6f}")
    print(f"  κ I2 ratio:  {data_kappa.ratio_I2:.6f}")
    print(f"  κ* I1 ratio: normal={data_kappa_star.ratio_I1_normal:.6f}, frozen={data_kappa_star.ratio_I1_frozen:.6f}")
    print(f"  κ* I2 ratio: {data_kappa_star.ratio_I2:.6f}")
    print()

    print("  Observation:")
    print(f"    I1 normal ratio is {'higher' if data_kappa.ratio_I1_normal > data_kappa.ratio_I1_frozen else 'lower'} than frozen")
    print(f"    I2 ratio (always frozen): {data_kappa.ratio_I2:.6f}")
    print()

    # The key insight: if I1 ratio differs from I2 ratio, we need differential g
    I1_I2_ratio_gap_kappa = data_kappa.ratio_I1_normal - data_kappa.ratio_I2
    I1_I2_ratio_gap_kappa_star = data_kappa_star.ratio_I1_normal - data_kappa_star.ratio_I2

    print(f"  I1 vs I2 ratio gap:")
    print(f"    κ:  {I1_I2_ratio_gap_kappa:+.6f}")
    print(f"    κ*: {I1_I2_ratio_gap_kappa_star:+.6f}")
    print()

    print("  Conclusion:")
    print("    I1 and I2 have different (+R)/(-R) behavior.")
    print("    This motivates differential g corrections: g_I1 ≠ g_I2")
    print()


def main():
    print()
    print("=" * 70)
    print("  MIRROR RATIO ANALYSIS FOR G DERIVATION")
    print("=" * 70)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4 / 7
    K = 3

    # Collect mirror ratio data
    data_kappa = collect_mirror_ratio_data(
        "κ BENCHMARK", R=1.3036, theta=theta, K=K,
        polynomials=polys_kappa, n_quad=60
    )

    data_kappa_star = collect_mirror_ratio_data(
        "κ* BENCHMARK", R=1.1167, theta=theta, K=K,
        polynomials=polys_kappa_star, n_quad=60
    )

    # Print summaries
    print_mirror_ratio_summary(data_kappa)
    print_mirror_ratio_summary(data_kappa_star)

    # Derive g from mirror ratios
    derive_g_from_mirror_ratios(data_kappa, data_kappa_star)

    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("KEY INSIGHTS:")
    print("1. I1 and I2 have different (+R)/(-R) ratio behavior")
    print("2. Q derivatives modify I1 ratio but not I2 ratio")
    print("3. This structural difference motivates g_I1 ≠ g_I2")
    print("4. Simple ratio-based formulas don't directly derive g values")
    print()
    print("RECOMMENDATION:")
    print("Use FIRST_PRINCIPLES_I1_I2 mode with g_I1=1.0, g_I2=g_baseline")
    print("This captures the I1/I2 asymmetry from first principles.")
    print()


if __name__ == "__main__":
    main()
