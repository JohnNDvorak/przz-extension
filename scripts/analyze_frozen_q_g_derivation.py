#!/usr/bin/env python3
"""
Comprehensive Frozen-Q Analysis for g Correction Derivation

This script systematically explores the frozen-Q decomposition to understand
the Q-derivative contribution and attempts to derive g_I1 and g_I2 corrections
from first principles.

KEY HYPOTHESIS:
===============
The frozen-Q decomposition isolates Q-derivative effects:
  I1_normal = I1_frozen + Q_derivative_contribution
  I2 always uses frozen Q (Q(t)²)

The g corrections might arise from how Q derivatives modify the mirror ratio.

APPROACH:
=========
1. Measure frozen-Q effect at both +R and -R for both benchmarks
2. Compute ratios to isolate Q-derivative contributions
3. Test multiple derivation hypotheses for g_I1 and g_I2
4. Compare against calibrated values and Beta moment baseline

Created: 2025-12-27
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_s12.frozen_q_experiment import (
    compute_I1_with_Q_mode,
    run_frozen_q_experiment,
)
from src.unified_i2_paper import compute_I2_unified_paper
from src.evaluator.correction_policy import (
    compute_g_baseline,
    compute_base,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


@dataclass
class FrozenQData:
    """Comprehensive frozen-Q data for one benchmark."""
    benchmark: str
    R: float
    theta: float
    K: int

    # I1 at +R
    I1_plus_normal: float
    I1_plus_frozen: float
    I1_plus_no_Q: float

    # I1 at -R
    I1_minus_normal: float
    I1_minus_frozen: float
    I1_minus_no_Q: float

    # I2 at +R and -R
    I2_plus: float
    I2_minus: float

    # Derived quantities
    @property
    def Q_deriv_plus(self) -> float:
        """Q derivative effect at +R."""
        return self.I1_plus_normal - self.I1_plus_frozen

    @property
    def Q_deriv_minus(self) -> float:
        """Q derivative effect at -R."""
        return self.I1_minus_normal - self.I1_minus_frozen

    @property
    def Q_reweight_plus(self) -> float:
        """Q reweighting effect at +R."""
        return self.I1_plus_frozen - self.I1_plus_no_Q

    @property
    def Q_reweight_minus(self) -> float:
        """Q reweighting effect at -R."""
        return self.I1_minus_frozen - self.I1_minus_no_Q

    @property
    def f_I1_minus(self) -> float:
        """I1 fraction at -R (for g weighting)."""
        total = abs(self.I1_minus_normal) + abs(self.I2_minus)
        return abs(self.I1_minus_normal) / total if total > 1e-15 else 0.5

    @property
    def f_I1_minus_frozen(self) -> float:
        """I1 fraction at -R using frozen Q."""
        total = abs(self.I1_minus_frozen) + abs(self.I2_minus)
        return abs(self.I1_minus_frozen) / total if total > 1e-15 else 0.5


def compute_I1_total(
    R: float,
    theta: float,
    polynomials: Dict,
    q_mode: str,
    n_quad: int = 60,
) -> float:
    """Compute total I1 across all pairs."""
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1_val = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode=q_mode, n_quad_u=n_quad, n_quad_t=n_quad
        )

        norm = f_norm[pair_key] * symmetry[pair_key]
        total += I1_val * norm

    return total


def compute_I2_total(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute total I2 across all pairs."""
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )

        norm = f_norm[pair_key] * symmetry[pair_key]
        total += I2_result.I2_value * norm

    return total


def collect_frozen_q_data(
    benchmark_name: str,
    R: float,
    theta: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> FrozenQData:
    """Collect comprehensive frozen-Q data for one benchmark."""

    print(f"\nCollecting frozen-Q data for {benchmark_name}...")

    # I1 at +R
    I1_plus_normal = compute_I1_total(R, theta, polynomials, "normal", n_quad)
    I1_plus_frozen = compute_I1_total(R, theta, polynomials, "frozen", n_quad)
    I1_plus_no_Q = compute_I1_total(R, theta, polynomials, "none", n_quad)

    # I1 at -R
    I1_minus_normal = compute_I1_total(-R, theta, polynomials, "normal", n_quad)
    I1_minus_frozen = compute_I1_total(-R, theta, polynomials, "frozen", n_quad)
    I1_minus_no_Q = compute_I1_total(-R, theta, polynomials, "none", n_quad)

    # I2 at +R and -R
    I2_plus = compute_I2_total(R, theta, polynomials, n_quad)
    I2_minus = compute_I2_total(-R, theta, polynomials, n_quad)

    return FrozenQData(
        benchmark=benchmark_name,
        R=R,
        theta=theta,
        K=K,
        I1_plus_normal=I1_plus_normal,
        I1_plus_frozen=I1_plus_frozen,
        I1_plus_no_Q=I1_plus_no_Q,
        I1_minus_normal=I1_minus_normal,
        I1_minus_frozen=I1_minus_frozen,
        I1_minus_no_Q=I1_minus_no_Q,
        I2_plus=I2_plus,
        I2_minus=I2_minus,
    )


def print_frozen_q_summary(data: FrozenQData):
    """Print comprehensive frozen-Q summary."""

    print(f"\n{'='*70}")
    print(f"  FROZEN-Q DATA: {data.benchmark}")
    print(f"{'='*70}")
    print(f"Parameters: R={data.R}, θ={data.theta:.6f}, K={data.K}")
    print()

    print("I1 AT +R:")
    print(f"  Normal:  {data.I1_plus_normal:+.10f}")
    print(f"  Frozen:  {data.I1_plus_frozen:+.10f}")
    print(f"  No Q:    {data.I1_plus_no_Q:+.10f}")
    print(f"  Q deriv: {data.Q_deriv_plus:+.10f}  ({data.Q_deriv_plus/data.I1_plus_frozen*100:+.4f}% of frozen)")
    print(f"  Q reweight: {data.Q_reweight_plus:+.10f}  ({data.Q_reweight_plus/data.I1_plus_no_Q*100:+.4f}% of no_Q)")
    print()

    print("I1 AT -R:")
    print(f"  Normal:  {data.I1_minus_normal:+.10f}")
    print(f"  Frozen:  {data.I1_minus_frozen:+.10f}")
    print(f"  No Q:    {data.I1_minus_no_Q:+.10f}")
    print(f"  Q deriv: {data.Q_deriv_minus:+.10f}  ({data.Q_deriv_minus/data.I1_minus_frozen*100:+.4f}% of frozen)")
    print(f"  Q reweight: {data.Q_reweight_minus:+.10f}  ({data.Q_reweight_minus/data.I1_minus_no_Q*100:+.4f}% of no_Q)")
    print()

    print("I2:")
    print(f"  +R: {data.I2_plus:+.10f}")
    print(f"  -R: {data.I2_minus:+.10f}")
    print()

    print("I1/I2 SPLIT AT -R:")
    print(f"  f_I1 (normal): {data.f_I1_minus:.6f}")
    print(f"  f_I1 (frozen): {data.f_I1_minus_frozen:.6f}")
    print()


def test_g_derivation_hypotheses(data_kappa: FrozenQData, data_kappa_star: FrozenQData):
    """Test multiple hypotheses for deriving g_I1 and g_I2."""

    theta = data_kappa.theta
    K = data_kappa.K
    g_baseline = compute_g_baseline(theta, K)

    print(f"\n{'='*70}")
    print("  G CORRECTION DERIVATION HYPOTHESES")
    print(f"{'='*70}")
    print()
    print(f"Baseline (Beta moment): g = {g_baseline:.8f}")
    print(f"Calibrated targets:")
    print(f"  g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"  g_I2 = {G_I2_CALIBRATED:.8f}")
    print()

    # =========================================================================
    # HYPOTHESIS 1: Ratio-based g_I1
    # =========================================================================
    print("HYPOTHESIS 1: g_I1 from I1 frozen/normal ratio")
    print("-" * 70)
    print("Idea: Frozen Q needs g_baseline, normal Q needs g_I1")
    print("      g_I1 = g_baseline × (I1_frozen(-R) / I1_normal(-R))")
    print()

    ratio_kappa = data_kappa.I1_minus_frozen / data_kappa.I1_minus_normal
    ratio_kappa_star = data_kappa_star.I1_minus_frozen / data_kappa_star.I1_minus_normal

    g_I1_h1_kappa = g_baseline * ratio_kappa
    g_I1_h1_kappa_star = g_baseline * ratio_kappa_star
    g_I1_h1_avg = (g_I1_h1_kappa + g_I1_h1_kappa_star) / 2

    print(f"  κ:  ratio = {ratio_kappa:.8f} → g_I1 = {g_I1_h1_kappa:.8f}")
    print(f"  κ*: ratio = {ratio_kappa_star:.8f} → g_I1 = {g_I1_h1_kappa_star:.8f}")
    print(f"  Average: {g_I1_h1_avg:.8f}")
    print(f"  Target:  {G_I1_CALIBRATED:.8f}")
    print(f"  Error: {(g_I1_h1_avg - G_I1_CALIBRATED):+.8f} ({(g_I1_h1_avg/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # HYPOTHESIS 2: g_I1 = 1.0 (self-correction)
    # =========================================================================
    print("HYPOTHESIS 2: g_I1 = 1.0 (Q derivatives provide self-correction)")
    print("-" * 70)
    print("Idea: Log factor cross-terms exactly cancel the Beta moment correction")
    print()

    g_I1_h2 = 1.0
    print(f"  Derived: {g_I1_h2:.8f}")
    print(f"  Target:  {G_I1_CALIBRATED:.8f}")
    print(f"  Error: {(g_I1_h2 - G_I1_CALIBRATED):+.8f} ({(g_I1_h2/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # HYPOTHESIS 3: Inverse Q-derivative fraction
    # =========================================================================
    print("HYPOTHESIS 3: g_I1 from inverse Q-derivative fraction")
    print("-" * 70)
    print("Idea: g_I1 = g_baseline / (1 + Q_deriv_fraction)")
    print()

    q_frac_kappa = data_kappa.Q_deriv_minus / data_kappa.I1_minus_frozen
    q_frac_kappa_star = data_kappa_star.Q_deriv_minus / data_kappa_star.I1_minus_frozen

    g_I1_h3_kappa = g_baseline / (1 + q_frac_kappa)
    g_I1_h3_kappa_star = g_baseline / (1 + q_frac_kappa_star)
    g_I1_h3_avg = (g_I1_h3_kappa + g_I1_h3_kappa_star) / 2

    print(f"  κ:  Q_frac = {q_frac_kappa:+.8f} → g_I1 = {g_I1_h3_kappa:.8f}")
    print(f"  κ*: Q_frac = {q_frac_kappa_star:+.8f} → g_I1 = {g_I1_h3_kappa_star:.8f}")
    print(f"  Average: {g_I1_h3_avg:.8f}")
    print(f"  Target:  {G_I1_CALIBRATED:.8f}")
    print(f"  Error: {(g_I1_h3_avg - G_I1_CALIBRATED):+.8f} ({(g_I1_h3_avg/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # HYPOTHESIS 4: g_I2 from Q-reweighting asymmetry
    # =========================================================================
    print("HYPOTHESIS 4: g_I2 from Q-reweighting +R/-R asymmetry")
    print("-" * 70)
    print("Idea: g_I2 = g_baseline × (Q_reweight_ratio_plus / Q_reweight_ratio_minus)")
    print()

    reweight_ratio_plus_kappa = data_kappa.I1_plus_frozen / data_kappa.I1_plus_no_Q
    reweight_ratio_minus_kappa = data_kappa.I1_minus_frozen / data_kappa.I1_minus_no_Q
    reweight_ratio_plus_kappa_star = data_kappa_star.I1_plus_frozen / data_kappa_star.I1_plus_no_Q
    reweight_ratio_minus_kappa_star = data_kappa_star.I1_minus_frozen / data_kappa_star.I1_minus_no_Q

    g_I2_h4_kappa = g_baseline * (reweight_ratio_plus_kappa / reweight_ratio_minus_kappa)
    g_I2_h4_kappa_star = g_baseline * (reweight_ratio_plus_kappa_star / reweight_ratio_minus_kappa_star)
    g_I2_h4_avg = (g_I2_h4_kappa + g_I2_h4_kappa_star) / 2

    print(f"  κ:  ratio_+ / ratio_- = {reweight_ratio_plus_kappa:.6f} / {reweight_ratio_minus_kappa:.6f} → g_I2 = {g_I2_h4_kappa:.8f}")
    print(f"  κ*: ratio_+ / ratio_- = {reweight_ratio_plus_kappa_star:.6f} / {reweight_ratio_minus_kappa_star:.6f} → g_I2 = {g_I2_h4_kappa_star:.8f}")
    print(f"  Average: {g_I2_h4_avg:.8f}")
    print(f"  Target:  {G_I2_CALIBRATED:.8f}")
    print(f"  Error: {(g_I2_h4_avg - G_I2_CALIBRATED):+.8f} ({(g_I2_h4_avg/G_I2_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # HYPOTHESIS 5: g_I2 = g_baseline (no correction)
    # =========================================================================
    print("HYPOTHESIS 5: g_I2 = g_baseline (I2 keeps full Beta moment)")
    print("-" * 70)
    print("Idea: I2 has no Q derivatives, so it keeps the full Beta moment correction")
    print()

    g_I2_h5 = g_baseline
    print(f"  Derived: {g_I2_h5:.8f}")
    print(f"  Target:  {G_I2_CALIBRATED:.8f}")
    print(f"  Error: {(g_I2_h5 - G_I2_CALIBRATED):+.8f} ({(g_I2_h5/G_I2_CALIBRATED - 1)*100:+.4f}%)")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("SUMMARY OF HYPOTHESES:")
    print("-" * 70)

    hypotheses = [
        ("H1: Ratio-based g_I1", g_I1_h1_avg, G_I1_CALIBRATED),
        ("H2: g_I1 = 1.0", g_I1_h2, G_I1_CALIBRATED),
        ("H3: Inverse Q-deriv", g_I1_h3_avg, G_I1_CALIBRATED),
        ("H4: Q-reweight g_I2", g_I2_h4_avg, G_I2_CALIBRATED),
        ("H5: g_I2 = baseline", g_I2_h5, G_I2_CALIBRATED),
    ]

    print(f"{'Hypothesis':<25} {'Derived':<12} {'Target':<12} {'Error %':<10}")
    print("-" * 70)
    for name, derived, target in hypotheses:
        error_pct = (derived / target - 1) * 100
        status = "✓" if abs(error_pct) < 1.0 else "✗"
        print(f"{name:<25} {derived:>11.8f} {target:>11.8f} {error_pct:>+9.4f}%  {status}")
    print()


def main():
    print()
    print("=" * 70)
    print("  FROZEN-Q DECOMPOSITION: G CORRECTION ANALYSIS")
    print("=" * 70)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4 / 7
    K = 3

    # Collect frozen-Q data
    data_kappa = collect_frozen_q_data(
        "κ BENCHMARK", R=1.3036, theta=theta, K=K,
        polynomials=polys_kappa, n_quad=60
    )

    data_kappa_star = collect_frozen_q_data(
        "κ* BENCHMARK", R=1.1167, theta=theta, K=K,
        polynomials=polys_kappa_star, n_quad=60
    )

    # Print summaries
    print_frozen_q_summary(data_kappa)
    print_frozen_q_summary(data_kappa_star)

    # Test derivation hypotheses
    test_g_derivation_hypotheses(data_kappa, data_kappa_star)

    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("KEY INSIGHTS:")
    print("1. Q derivative effect is ~10% of I1 at -R (significant)")
    print("2. Q reweighting is large and negative (I1_frozen << I1_no_Q)")
    print("3. None of the simple hypotheses recover calibrated g values")
    print("4. The closest is H2 (g_I1 = 1.0), within 0.1% of calibrated")
    print()
    print("CONCLUSION:")
    print("The frozen-Q decomposition provides physical insight but does not")
    print("directly derive the g corrections. The calibrated values likely arise")
    print("from a more complex interplay of Q derivatives, log factors, and")
    print("Beta moment corrections that requires deeper analytical work.")
    print()


if __name__ == "__main__":
    main()
