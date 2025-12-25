"""
src/ratios/microcase_plus7_signature_k4.py
Phase 15E: K=4 Microcase Gate (+7 Signature)

PURPOSE:
========
Create the +7 gate (analog of +5 for K=4) to detect error amplification.

For K=4, the mirror assembly formula is:
    c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
    where m = exp(R) + (2K-1) = exp(R) + 7

So the target B/A should be approximately 7 (not 5 like K=3).

CRITICAL CHECK:
==============
If the gap at K=4 is significantly larger than K=3, the current approach
is fundamentally flawed and will blow up at higher K.

Conversely, if ACTUAL_LOGDERIV keeps the gap small at K=4, we have
confidence the fix works across all K.
"""

from __future__ import annotations
from typing import Dict
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def compute_plus7_signature(
    R: float,
    theta: float = 4.0/7.0,
    *,
    polys=None,
    laurent_mode: LaurentMode = LaurentMode.ACTUAL_LOGDERIV,
) -> Dict:
    """
    Compute K=4 analog of +5 gate: target B/A = 7.

    This uses the same mirror assembly formula but with K=4.

    NOTE: We don't have K=4 polynomials, so we reuse K=3 polynomials
    as a proxy. The key check is whether the structural formula
    behaves correctly at K=4.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        polys: K=3 polynomials (reused as proxy)
        laurent_mode: Laurent factor mode

    Returns:
        Dictionary with K=4 analysis
    """
    # Compute with K=4
    result = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=4,  # Key difference from K=3
        laurent_mode=laurent_mode,
    )

    target = 7  # 2K-1 for K=4
    delta = (result['B_over_A'] - target) / target * 100

    return {
        'K': 4,
        'target_constant': target,
        'B_over_A': result['B_over_A'],
        'delta_percent': delta,
        'A': result['exp_coefficient'],
        'B': result['constant_offset'],
        'D': result['D'],
        'laurent_mode': laurent_mode.value,
        'R': R,
    }


def compare_k3_k4_gates(benchmark: str = "kappa") -> Dict:
    """
    Compare +5 gate (K=3) and +7 gate (K=4) for a benchmark.

    This tests whether the gap amplifies with K.

    Args:
        benchmark: "kappa" or "kappa_star"

    Returns:
        Dictionary with comparison results
    """
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R

    results = {}

    for mode in LaurentMode:
        # K=3 result
        k3_result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3, laurent_mode=mode
        )
        k3_target = 5
        k3_delta = (k3_result['B_over_A'] - k3_target) / k3_target * 100

        # K=4 result
        k4_result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=4, laurent_mode=mode
        )
        k4_target = 7
        k4_delta = (k4_result['B_over_A'] - k4_target) / k4_target * 100

        # Amplification factor
        if abs(k3_delta) > 0.01:
            amplification = k4_delta / k3_delta
        else:
            amplification = float('nan')

        results[mode.value] = {
            'K3_B_over_A': k3_result['B_over_A'],
            'K3_delta_pct': k3_delta,
            'K4_B_over_A': k4_result['B_over_A'],
            'K4_delta_pct': k4_delta,
            'amplification': amplification,
        }

    return {
        'benchmark': benchmark,
        'R': R,
        'results': results,
    }


def print_k4_comparison():
    """Print K=3 vs K=4 comparison for both benchmarks."""
    print("=" * 90)
    print("PHASE 15E: K=4 MICROCASE GATE (+7 SIGNATURE)")
    print("=" * 90)
    print()
    print("Testing whether error gap amplifies from K=3 (+5 gate) to K=4 (+7 gate).")
    print("Amplification > 1 would indicate the fix is incomplete.")
    print()

    for benchmark in ["kappa", "kappa_star"]:
        comparison = compare_k3_k4_gates(benchmark)
        R = comparison['R']

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 80)
        print(f"{'Mode':<20} {'K=3 B/A':<12} {'K=3 δ%':<10} {'K=4 B/A':<12} {'K=4 δ%':<10} {'Amplify':<10}")
        print("-" * 80)

        for mode, data in comparison['results'].items():
            amp_str = f"{data['amplification']:.2f}" if data['amplification'] == data['amplification'] else "N/A"
            print(f"{mode:<20} {data['K3_B_over_A']:<12.4f} {data['K3_delta_pct']:+8.2f}% "
                  f"{data['K4_B_over_A']:<12.4f} {data['K4_delta_pct']:+8.2f}% {amp_str:<10}")

        print("-" * 80)
        print(f"K=3 target: 5 (=2×3-1),  K=4 target: 7 (=2×4-1)")

    print()
    print("=" * 90)
    print("\nINTERPRETATION:")
    print("- Amplification < 1: Gap shrinks from K=3 to K=4 (GOOD)")
    print("- Amplification ≈ 1: Gap stays constant (OK)")
    print("- Amplification > 1: Gap grows with K (PROBLEM)")
    print()


if __name__ == "__main__":
    print_k4_comparison()
