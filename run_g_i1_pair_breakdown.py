"""
run_g_i1_pair_breakdown.py
Deep dive into per-pair contributions to the log factor split.

This script examines each (ℓ₁, ℓ₂) pair individually to understand:
- Which pairs dominate the cross-ratio deviation
- How the weighting factors affect the aggregate
- Whether certain pairs have anomalous behavior

Created: 2025-12-27
"""
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.unified_s12.logfactor_split import (
    split_logfactor_for_pair,
    compute_aggregate_correction_k3,
)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_section(text: str):
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def analyze_pair_breakdown(name: str, R: float, polynomials: dict, theta: float = 4/7, K: int = 3):
    """
    Analyze per-pair contributions to log factor split.
    """
    print_header(f"{name} BENCHMARK - PER-PAIR BREAKDOWN (R = {R:.4f})")

    # Pair configuration for K=3
    pairs = ["11", "22", "33", "12", "13", "23"]

    # Weights from compute_aggregate_correction_k3
    factorial_norm = {
        "11": 1.0,       # 1/(1!×1!) = 1
        "22": 0.25,      # 1/(2!×2!) = 1/4
        "33": 1.0/36.0,  # 1/(3!×3!) = 1/36
        "12": 0.5,       # 1/(1!×2!) = 1/2
        "13": 1.0/6.0,   # 1/(1!×3!) = 1/6
        "23": 1.0/12.0,  # 1/(2!×3!) = 1/12
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    print("\nPair weights (factorial × symmetry):")
    for pair in pairs:
        weight = factorial_norm[pair] * symmetry_factor[pair]
        print(f"  {pair}: {weight:.6f}")

    # Compute splits for each pair
    pair_data = []
    for pair in pairs:
        split = split_logfactor_for_pair(pair, theta, R, K, polynomials, n_quad=60)
        weight = factorial_norm[pair] * symmetry_factor[pair]

        pair_data.append({
            "pair": pair,
            "weight": weight,
            "main": split.main_coeff,
            "cross_x": split.cross_from_x_term,
            "cross_y": split.cross_from_y_term,
            "cross_total": split.cross_from_x_term + split.cross_from_y_term,
            "total": split.total_coeff,
            "correction": split.correction_factor,
            "predicted": split.predicted_correction,
            "gap_pct": split.correction_gap,
        })

    # Print per-pair results
    print_section("PER-PAIR RESULTS")

    beta = 1 / (2 * K * (2 * K + 1))
    g_baseline = 1 + theta * beta

    print(f"\nTheoretical: Beta(2,2K) = {beta:.8f}, g_baseline = {g_baseline:.8f}\n")

    print(f"{'Pair':<6} {'Weight':<10} {'Main':<12} {'Cross':<12} {'C/M':<10} {'Correction':<12} {'Gap %':<8}")
    print("-" * 80)

    for p in pair_data:
        cross_ratio = p['cross_total'] / p['main'] if abs(p['main']) > 1e-15 else 0.0
        print(f"{p['pair']:<6} {p['weight']:<10.6f} {p['main']:<12.6e} {p['cross_total']:<12.6e} "
              f"{cross_ratio:<10.6f} {p['correction']:<12.8f} {p['gap_pct']:<8.2f}")

    # Compute weighted aggregates
    print_section("WEIGHTED AGGREGATES")

    total_main = sum(p['weight'] * p['main'] for p in pair_data)
    total_cross = sum(p['weight'] * p['cross_total'] for p in pair_data)
    total_weighted = total_main + total_cross

    print(f"\nWeighted sums:")
    print(f"  Total Main (M) = {total_main:.10f}")
    print(f"  Total Cross (C) = {total_cross:.10f}")
    print(f"  Total (M + C) = {total_weighted:.10f}")

    aggregate_cross_ratio = total_cross / total_main if abs(total_main) > 1e-15 else 0.0
    aggregate_correction = total_weighted / total_main if abs(total_main) > 1e-15 else 0.0

    print(f"\nAggregate ratios:")
    print(f"  C/M = {aggregate_cross_ratio:.8f}")
    print(f"  Beta(2,2K) = {beta:.8f}")
    print(f"  Ratio: (C/M) / Beta = {aggregate_cross_ratio / beta:.4f}x")

    print(f"\nAggregate correction:")
    print(f"  (M + C) / M = {aggregate_correction:.8f}")
    print(f"  g_baseline = {g_baseline:.8f}")
    print(f"  Gap: {(aggregate_correction / g_baseline - 1) * 100:+.2f}%")

    print(f"\nDerived g_I1:")
    g_I1_derived = g_baseline / aggregate_correction if abs(aggregate_correction) > 1e-15 else float('nan')
    print(f"  g_I1 = g_baseline / (M+C)/M = {g_I1_derived:.8f}")
    print(f"  Gap from 1.0: {(g_I1_derived - 1.0) * 100:+.4f}%")

    # Contribution analysis
    print_section("CONTRIBUTION ANALYSIS")

    print("\nWhich pairs dominate Main (M)?")
    main_contribs = [(p['pair'], p['weight'] * p['main']) for p in pair_data]
    main_contribs_sorted = sorted(main_contribs, key=lambda x: abs(x[1]), reverse=True)
    for pair, contrib in main_contribs_sorted:
        pct = 100 * contrib / total_main if abs(total_main) > 1e-15 else 0.0
        print(f"  {pair}: {contrib:+.6e} ({pct:+6.2f}%)")

    print("\nWhich pairs dominate Cross (C)?")
    cross_contribs = [(p['pair'], p['weight'] * p['cross_total']) for p in pair_data]
    cross_contribs_sorted = sorted(cross_contribs, key=lambda x: abs(x[1]), reverse=True)
    for pair, contrib in cross_contribs_sorted:
        pct = 100 * contrib / total_cross if abs(total_cross) > 1e-15 else 0.0
        print(f"  {pair}: {contrib:+.6e} ({pct:+6.2f}%)")

    # Per-pair cross ratios
    print_section("PER-PAIR CROSS RATIOS")

    print(f"\n{'Pair':<6} {'C/M (unweighted)':<18} {'Gap from Beta':<15}")
    print("-" * 50)
    for p in pair_data:
        cross_ratio = p['cross_total'] / p['main'] if abs(p['main']) > 1e-15 else 0.0
        gap_from_beta = (cross_ratio / beta - 1) * 100
        print(f"{p['pair']:<6} {cross_ratio:<18.8f} {gap_from_beta:+14.2f}%")

    print(f"\nAggregate C/M = {aggregate_cross_ratio:.8f} ({(aggregate_cross_ratio / beta - 1) * 100:+.2f}% from Beta)")

    return {
        "name": name,
        "R": R,
        "pair_data": pair_data,
        "total_main": total_main,
        "total_cross": total_cross,
        "aggregate_cross_ratio": aggregate_cross_ratio,
        "aggregate_correction": aggregate_correction,
        "g_I1_derived": g_I1_derived,
    }


def compare_benchmarks(kappa_result: dict, kappa_star_result: dict):
    """Compare the two benchmarks to identify systematic vs R-dependent effects."""
    print_header("CROSS-BENCHMARK COMPARISON")

    print("\nAggregate cross ratios:")
    print(f"  κ:  C/M = {kappa_result['aggregate_cross_ratio']:.8f}")
    print(f"  κ*: C/M = {kappa_star_result['aggregate_cross_ratio']:.8f}")
    print(f"  Ratio (κ*/κ): {kappa_star_result['aggregate_cross_ratio'] / kappa_result['aggregate_cross_ratio']:.4f}x")

    print("\nAggregate corrections:")
    print(f"  κ:  (M+C)/M = {kappa_result['aggregate_correction']:.8f}")
    print(f"  κ*: (M+C)/M = {kappa_star_result['aggregate_correction']:.8f}")

    print("\nDerived g_I1:")
    print(f"  κ:  g_I1 = {kappa_result['g_I1_derived']:.8f}")
    print(f"  κ*: g_I1 = {kappa_star_result['g_I1_derived']:.8f}")
    print(f"  Difference: {abs(kappa_result['g_I1_derived'] - kappa_star_result['g_I1_derived']) * 1e6:.1f} ppm")

    # Per-pair comparison
    print_section("PER-PAIR CROSS RATIO COMPARISON")

    beta = 1 / 42  # Beta(2, 6) for K=3

    print(f"\n{'Pair':<6} {'κ C/M':<12} {'κ* C/M':<12} {'Ratio':<10} {'κ gap%':<10} {'κ* gap%':<10}")
    print("-" * 70)

    for kappa_p, kappa_star_p in zip(kappa_result['pair_data'], kappa_star_result['pair_data']):
        assert kappa_p['pair'] == kappa_star_p['pair']
        pair = kappa_p['pair']

        kappa_cm = kappa_p['cross_total'] / kappa_p['main'] if abs(kappa_p['main']) > 1e-15 else 0.0
        kappa_star_cm = kappa_star_p['cross_total'] / kappa_star_p['main'] if abs(kappa_star_p['main']) > 1e-15 else 0.0
        ratio = kappa_star_cm / kappa_cm if abs(kappa_cm) > 1e-15 else 0.0

        kappa_gap = (kappa_cm / beta - 1) * 100
        kappa_star_gap = (kappa_star_cm / beta - 1) * 100

        print(f"{pair:<6} {kappa_cm:<12.6f} {kappa_star_cm:<12.6f} {ratio:<10.4f} "
              f"{kappa_gap:+9.2f}% {kappa_star_gap:+9.2f}%")

    print("\nKEY OBSERVATIONS:")
    print("1. If ratios are ~1, the pair has similar C/M across benchmarks (systematic)")
    print("2. If ratios vary, the pair has R-dependent or polynomial-structure-dependent C/M")
    print("3. Large gaps from Beta indicate deviation from theoretical prediction")


def main():
    """Main diagnostic routine."""
    theta = 4 / 7
    K = 3

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1_star, P2_star, P3_star, Q_star = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1_star, "P2": P2_star, "P3": P3_star, "Q": Q_star}

    # Analyze both benchmarks
    kappa_result = analyze_pair_breakdown("κ", 1.3036, kappa_polys, theta, K)
    kappa_star_result = analyze_pair_breakdown("κ*", 1.1167, kappa_star_polys, theta, K)

    # Compare
    compare_benchmarks(kappa_result, kappa_star_result)

    print_header("FINAL SUMMARY")
    print("\nThe 0.09% calibration gap arises from:")
    print("1. Cross ratio C/M significantly exceeds Beta(2,2K) = 0.02381")
    print(f"   - κ:  C/M = {kappa_result['aggregate_cross_ratio']:.6f} (2x Beta)")
    print(f"   - κ*: C/M = {kappa_star_result['aggregate_cross_ratio']:.6f} (5x Beta)")
    print("\n2. This excess internal correction drives g_I1 BELOW 1.0:")
    print(f"   - κ:  g_I1 = {kappa_result['g_I1_derived']:.6f} (3.2% below 1.0)")
    print(f"   - κ*: g_I1 = {kappa_star_result['g_I1_derived']:.6f} (10% below 1.0)")
    print("\n3. The gap is NOT systematic - it's strongly R-dependent:")
    print(f"   - Difference: {abs(kappa_result['g_I1_derived'] - kappa_star_result['g_I1_derived']) * 100:.1f}%")
    print("\n4. The calibrated value g_I1 = 1.00091 is ABOVE 1.0, but derived is BELOW 1.0")
    print("   → This suggests a SIGN ERROR or MISSING TERM in the derivation!")


if __name__ == "__main__":
    main()
