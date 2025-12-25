"""
tests/test_phase15_all_modes.py
Phase 15B: Test +5 Gate with All Four Laurent Modes

PURPOSE:
========
Compare B/A ratio (target = 5 for K=3) across all Laurent factor modes:
1. RAW_LOGDERIV: Laurent approximation (1/R + γ)²
2. POLE_CANCELLED: Limit as α→0: +1 constant
3. ACTUAL_LOGDERIV: Actual numerical (ζ'/ζ)(1-R)²
4. FULL_G_PRODUCT: G(-R)² = (ζ'/ζ²)² [includes 1/ζ factor]

HYPOTHESIS:
===========
Phase 15A found:
- Laurent approximation has 17-22% error vs actual (ζ'/ζ)(1-R)²
- G² (full G-product) is 19-35x larger - too big to be correct

If ACTUAL_LOGDERIV reduces the δ gap, it confirms the Laurent approximation
is the source of the 5% gap in the +5 gate.
"""

import pytest
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def run_plus5_gate_all_modes(benchmark: str = "kappa"):
    """Run +5 gate test with all four Laurent modes."""
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R

    results = {}
    for mode in LaurentMode:
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0,
            R=R,
            polys=polys,
            K=3,
            laurent_mode=mode,
        )
        results[mode.value] = {
            'B_over_A': result['B_over_A'],
            'A': result['exp_coefficient'],
            'B': result['constant_offset'],
            'delta': result['delta'],
        }

    return R, results


def print_comparison_table():
    """Print comparison table for all modes and both benchmarks."""
    print("=" * 80)
    print("PHASE 15B: +5 GATE TEST - ALL LAURENT MODES")
    print("=" * 80)
    print()
    print(f"{'Mode':<20} {'κ B/A':<12} {'κ δ%':<10} {'κ* B/A':<12} {'κ* δ%':<10}")
    print("-" * 80)

    R_k, results_k = run_plus5_gate_all_modes("kappa")
    R_ks, results_ks = run_plus5_gate_all_modes("kappa_star")

    target = 5.0

    for mode in LaurentMode:
        mode_name = mode.value

        ba_k = results_k[mode_name]['B_over_A']
        ba_ks = results_ks[mode_name]['B_over_A']

        delta_k = (ba_k - target) / target * 100
        delta_ks = (ba_ks - target) / target * 100

        print(f"{mode_name:<20} {ba_k:<12.6f} {delta_k:+8.2f}%  {ba_ks:<12.6f} {delta_ks:+8.2f}%")

    print("-" * 80)
    print(f"Target B/A = {target} (= 2K-1 for K=3)")
    print()

    # Print detailed breakdown for best mode
    print("\nDETAILED BREAKDOWN (ACTUAL_LOGDERIV vs RAW_LOGDERIV):")
    print("-" * 60)
    for benchmark, (R, results) in [("κ", (R_k, results_k)), ("κ*", (R_ks, results_ks))]:
        raw = results['raw_logderiv']
        actual = results['actual_logderiv']

        print(f"\n{benchmark} (R={R}):")
        print(f"  RAW_LOGDERIV:    B/A = {raw['B_over_A']:.6f}, δ = {(raw['B_over_A']-5)/5*100:+.2f}%")
        print(f"  ACTUAL_LOGDERIV: B/A = {actual['B_over_A']:.6f}, δ = {(actual['B_over_A']-5)/5*100:+.2f}%")

        improvement = (raw['B_over_A'] - target) - (actual['B_over_A'] - target)
        print(f"  Change in gap: {improvement:+.6f} (positive = improvement)")

    print()
    print("=" * 80)

    return results_k, results_ks


def test_phase15b_all_modes():
    """Run +5 gate with all modes and verify results are reasonable."""
    results_k, results_ks = print_comparison_table()

    # All modes should give some result (not NaN or inf)
    for mode in LaurentMode:
        mode_name = mode.value
        assert abs(results_k[mode_name]['B_over_A']) < 100, f"κ {mode_name} B/A is unreasonable"
        assert abs(results_ks[mode_name]['B_over_A']) < 100, f"κ* {mode_name} B/A is unreasonable"


def test_raw_vs_actual_comparison():
    """Compare RAW_LOGDERIV with ACTUAL_LOGDERIV to quantify the difference."""
    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        raw_result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.RAW_LOGDERIV
        )

        actual_result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        # The J12 factor should be larger for ACTUAL_LOGDERIV
        # because actual (ζ'/ζ)² > Laurent (1/R + γ)²
        # This should increase the magnitude of j12 contributions

        print(f"\n{benchmark} (R={R}):")
        print(f"  RAW j12(+R): {raw_result['i12_plus_pieces']['j12']:.6f}")
        print(f"  ACTUAL j12(+R): {actual_result['i12_plus_pieces']['j12']:.6f}")
        print(f"  Ratio: {actual_result['i12_plus_pieces']['j12'] / raw_result['i12_plus_pieces']['j12']:.4f}")


if __name__ == "__main__":
    print_comparison_table()
    print("\n" + "=" * 80)
    print("J12 COMPARISON:")
    print("=" * 80)
    test_raw_vs_actual_comparison()
