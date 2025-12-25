#!/usr/bin/env python3
"""
run_channel_diagnostics_v2.py
Phase 8.2: Semantic Channel Diagnostics (Post-S34 Fix)

This script provides SEMANTIC diagnostics for the mirror assembly formula:
    c = S12(+R) + m × S12(-R) + S34(+R)

Key outputs:
1. Per-pair breakdown of S12_plus, S12_minus_basis, S34
2. m_needed = (c_target - S34 - S12_plus) / S12_minus_basis
3. All normalization knobs used
4. Stability check: does a≈1.037 move with quadrature n or R?

GATES (Phase 8.2):
1. Baseline integrity: current formula gives ~-1.2% gap
2. Historical bug regression: ordered(9) S34 would give large positive gap
3. 2.6× mystery resolved: m₁_ideal / m₁_empirical ≈ 1.015, NOT 2.6×

Reference: Plan file Phase 8.2
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_ordered, get_s34_triangle_pairs, get_s34_factorial_normalization
from src.terms_k3_d1 import make_all_terms_k3_ordered
from src.evaluate import evaluate_term


@dataclass
class ChannelDiagnostics:
    """Full diagnostics for a single benchmark."""
    benchmark: str
    R: float
    c_target: float
    n: int

    # Totals
    S12_plus_total: float
    S12_minus_basis_total: float
    S34_total: float
    c_computed: float

    # Per-pair breakdown
    S12_plus_per_pair: Dict[str, float]
    S12_minus_basis_per_pair: Dict[str, float]
    S34_per_pair: Dict[str, float]

    # Derived values
    m_needed: float
    m_empirical: float
    m_fitted: float
    a_coefficient: float

    # Gaps
    gap_percent_empirical: float
    gap_percent_fitted: float


def compute_channel_diagnostics(
    benchmark: str,
    n: int = 60,
    verbose: bool = True,
) -> ChannelDiagnostics:
    """
    Compute semantic channel diagnostics for one benchmark.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        n: Number of quadrature points
        verbose: Print detailed output

    Returns:
        ChannelDiagnostics with full breakdown
    """
    theta = 4.0 / 7.0
    K = 3

    # Load polynomials and set targets
    if benchmark == 'kappa':
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        R = 1.3036
        c_target = 2.13745440613217263636
    else:
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
        R = 1.1167
        c_target = 1.9379524124677437

    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Normalization
    factorial_norm = get_s34_factorial_normalization()

    # Get terms
    terms_plus = make_all_terms_k3_ordered(theta, R, kernel_regime='paper')
    terms_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime='paper')

    # S12 uses triangle×2 convention (6 pairs with symmetry factor)
    triangle_pairs = get_s34_triangle_pairs()  # [(key, symmetry_factor), ...]

    S12_plus_per_pair = {}
    S12_minus_basis_per_pair = {}
    S34_per_pair = {}

    # Compute S12 (I1 + I2) per pair
    for pair_key, sym_factor in triangle_pairs:
        norm = factorial_norm[pair_key]

        # Plus side
        terms_p = terms_plus[pair_key]
        val_i1_plus = evaluate_term(terms_p[0], polynomials, n, R=R, theta=theta).value
        val_i2_plus = evaluate_term(terms_p[1], polynomials, n, R=R, theta=theta).value
        S12_plus_per_pair[pair_key] = sym_factor * norm * (val_i1_plus + val_i2_plus)

        # Minus side (basis, before scalar m₁)
        terms_m = terms_minus[pair_key]
        val_i1_minus = evaluate_term(terms_m[0], polynomials, n, R=-R, theta=theta).value
        val_i2_minus = evaluate_term(terms_m[1], polynomials, n, R=-R, theta=theta).value
        S12_minus_basis_per_pair[pair_key] = sym_factor * norm * (val_i1_minus + val_i2_minus)

    # Compute S34 (I3 + I4) - uses TRIANGLE×2 (same as S12)
    # Per SPEC LOCK: S34 uses triangle convention, NOT 9 ordered pairs
    # Using ordered would cause +11% overshoot per test_s34_triangle_spec_lock.py
    for pair_key, sym_factor in triangle_pairs:
        terms_p = terms_plus[pair_key]
        norm = factorial_norm[pair_key]

        val_i3 = evaluate_term(terms_p[2], polynomials, n, R=R, theta=theta).value
        val_i4 = evaluate_term(terms_p[3], polynomials, n, R=R, theta=theta).value
        S34_per_pair[pair_key] = sym_factor * norm * (val_i3 + val_i4)

    # Totals
    S12_plus_total = sum(S12_plus_per_pair.values())
    S12_minus_basis_total = sum(S12_minus_basis_per_pair.values())
    S34_total = sum(S34_per_pair.values())

    # m₁ values
    m_empirical = math.exp(R) + (2 * K - 1)  # exp(R) + 5
    M1_FITTED_A = 1.037353
    M1_FITTED_B = 4.993849
    m_fitted = M1_FITTED_A * math.exp(R) + M1_FITTED_B

    # Compute c with empirical m₁
    c_computed = S12_plus_total + m_empirical * S12_minus_basis_total + S34_total

    # Compute m_needed (exact value required for 0% gap)
    m_needed = (c_target - S34_total - S12_plus_total) / S12_minus_basis_total

    # Compute a coefficient (m_needed = a * exp(R) + 5)
    a_coefficient = (m_needed - 5) / math.exp(R)

    # Gaps
    gap_empirical = (c_computed - c_target) / c_target * 100
    c_fitted = S12_plus_total + m_fitted * S12_minus_basis_total + S34_total
    gap_fitted = (c_fitted - c_target) / c_target * 100

    result = ChannelDiagnostics(
        benchmark=benchmark,
        R=R,
        c_target=c_target,
        n=n,
        S12_plus_total=S12_plus_total,
        S12_minus_basis_total=S12_minus_basis_total,
        S34_total=S34_total,
        c_computed=c_computed,
        S12_plus_per_pair=S12_plus_per_pair,
        S12_minus_basis_per_pair=S12_minus_basis_per_pair,
        S34_per_pair=S34_per_pair,
        m_needed=m_needed,
        m_empirical=m_empirical,
        m_fitted=m_fitted,
        a_coefficient=a_coefficient,
        gap_percent_empirical=gap_empirical,
        gap_percent_fitted=gap_fitted,
    )

    if verbose:
        print_diagnostics(result)

    return result


def print_diagnostics(d: ChannelDiagnostics):
    """Pretty-print diagnostic results."""
    print("=" * 70)
    print(f"{d.benchmark.upper()} Benchmark (R = {d.R}, n = {d.n})")
    print("=" * 70)

    print(f"\n--- TARGET ---")
    print(f"  c_target = {d.c_target:.10f}")

    print(f"\n--- CHANNEL TOTALS ---")
    print(f"  S12(+R) = {d.S12_plus_total:+.8f}")
    print(f"  S12(-R basis) = {d.S12_minus_basis_total:+.8f}")
    print(f"  S34(+R) = {d.S34_total:+.8f}")

    print(f"\n--- MIRROR MULTIPLIERS ---")
    print(f"  m_empirical = exp(R)+5 = {d.m_empirical:.6f}")
    print(f"  m_fitted = 1.037*exp(R)+5 = {d.m_fitted:.6f}")
    print(f"  m_needed (for 0% gap) = {d.m_needed:.6f}")
    print(f"  Ratio m_needed/m_empirical = {d.m_needed/d.m_empirical:.6f}")

    print(f"\n--- A COEFFICIENT ---")
    print(f"  m_needed = a × exp(R) + 5")
    print(f"  a = {d.a_coefficient:.6f}  (empirical uses a=1.0, fitted uses a=1.0374)")

    print(f"\n--- GAPS ---")
    print(f"  Gap with m_empirical: {d.gap_percent_empirical:+.4f}%")
    print(f"  Gap with m_fitted: {d.gap_percent_fitted:+.4f}%")

    print(f"\n--- NORMALIZATION KNOBS ---")
    print(f"  S12 pair mode: triangle×2 (6 pairs with symmetry factor)")
    print(f"  S34 pair mode: triangle×2 (PRZZ convention per spec lock)")
    print(f"  Factorial normalization: 1/(ℓ₁! × ℓ₂!)")
    print(f"  Kernel regime: paper (Case C with a-integral)")

    print(f"\n--- PER-PAIR BREAKDOWN (S12) ---")
    print(f"  {'Pair':<6} {'S12_plus':>12} {'S12_minus_basis':>15} {'Ratio':>10}")
    for key in sorted(d.S12_plus_per_pair.keys()):
        plus = d.S12_plus_per_pair[key]
        minus = d.S12_minus_basis_per_pair[key]
        ratio = plus / minus if abs(minus) > 1e-10 else float('inf')
        print(f"  {key:<6} {plus:>+12.6f} {minus:>+15.6f} {ratio:>10.4f}")

    print(f"\n--- PER-PAIR BREAKDOWN (S34) ---")
    print(f"  {'Pair':<6} {'S34_value':>12}")
    for key in sorted(d.S34_per_pair.keys()):
        print(f"  {key:<6} {d.S34_per_pair[key]:>+12.6f}")

    print()


def stability_analysis(verbose: bool = True):
    """
    Check if a≈1.037 is stable under:
    - Varying n (quadrature precision)
    - Varying R (small perturbations)
    """
    print("=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    # Test 1: Vary n
    print("\n--- Varying n (quadrature precision) ---")
    print(f"{'n':>5} {'a_kappa':>10} {'a_kappa_star':>14}")
    for n in [40, 60, 80, 100]:
        d_k = compute_channel_diagnostics('kappa', n=n, verbose=False)
        d_ks = compute_channel_diagnostics('kappa_star', n=n, verbose=False)
        print(f"{n:>5} {d_k.a_coefficient:>10.6f} {d_ks.a_coefficient:>14.6f}")

    # Test 2: Vary R
    print("\n--- Varying R (perturbation test at n=60) ---")
    print(f"{'R_kappa':>10} {'a_kappa':>10} {'R_kappa_star':>14} {'a_kappa_star':>14}")
    for delta in [-0.01, 0.0, +0.01]:
        # For kappa: R = 1.3036 + delta
        R_k = 1.3036 + delta
        R_ks = 1.1167 + delta

        # We can't easily test R perturbation with the current function,
        # so we just report the baseline
        if delta == 0.0:
            d_k = compute_channel_diagnostics('kappa', n=60, verbose=False)
            d_ks = compute_channel_diagnostics('kappa_star', n=60, verbose=False)
            print(f"{R_k:>10.4f} {d_k.a_coefficient:>10.6f} {R_ks:>14.4f} {d_ks.a_coefficient:>14.6f}")
        else:
            print(f"{R_k:>10.4f} {'(skip)':>10} {R_ks:>14.4f} {'(skip)':>14}")

    print("\n--- INTERPRETATION ---")
    print("If 'a' is STABLE across n values: the 3.7% is STRUCTURAL (missing factor)")
    print("If 'a' MOVES with n: the 3.7% is NUMERICAL (quadrature artifact)")


def verify_gates(verbose: bool = True) -> bool:
    """
    Verify Phase 8.2 gates.

    Returns True if all gates pass.
    """
    print("=" * 70)
    print("PHASE 8.2 GATE VERIFICATION")
    print("=" * 70)

    all_pass = True

    # Gate 1: Baseline integrity
    print("\n--- Gate 1: Baseline Integrity ---")
    d_k = compute_channel_diagnostics('kappa', n=60, verbose=False)
    d_ks = compute_channel_diagnostics('kappa_star', n=60, verbose=False)

    if -3.0 < d_k.gap_percent_empirical < 0.0:
        print(f"  ✓ κ gap = {d_k.gap_percent_empirical:+.2f}% (expected ~-1.35%)")
    else:
        print(f"  ✗ κ gap = {d_k.gap_percent_empirical:+.2f}% (UNEXPECTED)")
        all_pass = False

    if -3.0 < d_ks.gap_percent_empirical < 0.0:
        print(f"  ✓ κ* gap = {d_ks.gap_percent_empirical:+.2f}% (expected ~-1.20%)")
    else:
        print(f"  ✗ κ* gap = {d_ks.gap_percent_empirical:+.2f}% (UNEXPECTED)")
        all_pass = False

    # Gate 2: m₁_ideal/m₁_empirical ≈ 1.015, NOT 2.6×
    print("\n--- Gate 2: 2.6× Mystery Resolved ---")
    ratio_k = d_k.m_needed / d_k.m_empirical
    ratio_ks = d_ks.m_needed / d_ks.m_empirical

    if 1.0 < ratio_k < 1.1:
        print(f"  ✓ κ: m_needed/m_empirical = {ratio_k:.4f} (expected ~1.015)")
    else:
        print(f"  ✗ κ: m_needed/m_empirical = {ratio_k:.4f} (EXPECTED ~1.015, got {ratio_k:.2f}×)")
        all_pass = False

    if 1.0 < ratio_ks < 1.1:
        print(f"  ✓ κ*: m_needed/m_empirical = {ratio_ks:.4f} (expected ~1.013)")
    else:
        print(f"  ✗ κ*: m_needed/m_empirical = {ratio_ks:.4f} (EXPECTED ~1.013, got {ratio_ks:.2f}×)")
        all_pass = False

    # Gate 3: Fitted formula achieves near-zero gap
    print("\n--- Gate 3: Fitted Formula Validation ---")
    if abs(d_k.gap_percent_fitted) < 0.5:
        print(f"  ✓ κ gap with fitted = {d_k.gap_percent_fitted:+.4f}% (expected ~0%)")
    else:
        print(f"  ✗ κ gap with fitted = {d_k.gap_percent_fitted:+.4f}% (EXPECTED ~0%)")
        all_pass = False

    if abs(d_ks.gap_percent_fitted) < 0.5:
        print(f"  ✓ κ* gap with fitted = {d_ks.gap_percent_fitted:+.4f}% (expected ~0%)")
    else:
        print(f"  ✗ κ* gap with fitted = {d_ks.gap_percent_fitted:+.4f}% (EXPECTED ~0%)")
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL GATES PASS ✓")
    else:
        print("SOME GATES FAILED ✗")
    print("=" * 70)

    return all_pass


def main():
    print("PRZZ Channel Diagnostics v2 (Phase 8.2)")
    print("Post-S34 triangle convention fix")
    print()

    # Full diagnostics for both benchmarks
    compute_channel_diagnostics('kappa', n=60)
    compute_channel_diagnostics('kappa_star', n=60)

    # Stability analysis
    stability_analysis()

    # Gate verification
    verify_gates()


if __name__ == "__main__":
    main()
