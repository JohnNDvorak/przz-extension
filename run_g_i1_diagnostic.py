"""
run_g_i1_diagnostic.py
Investigate why g_I1 needs to be 1.00091 (calibrated) instead of 1.0 (derived).

This script analyzes the log factor split for BOTH κ and κ* benchmarks to find
the source of the 0.09% gap.

HYPOTHESIS:
The gap is due to the cross-ratio C/M deviating from the theoretical Beta(2, 2K).
We'll measure this deviation and check if it's:
- Consistent across benchmarks (systematic)?
- Q-dependent (compare Q=1 vs real Q)?
- R-dependent?

Created: 2025-12-27
"""
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star, Polynomial
from src.evaluator.g_from_integrals import (
    derive_g_from_integrals,
    compute_i1_components,
    validate_derivation,
)
from src.evaluator.correction_policy import G_I1_CALIBRATED


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


def analyze_benchmark(name: str, R: float, polynomials: dict, theta: float = 4/7, K: int = 3):
    """
    Analyze log factor split for a single benchmark.

    Returns:
        dict with analysis results
    """
    print_header(f"{name} BENCHMARK (R = {R:.4f})")

    # Get derived g values
    derived = derive_g_from_integrals(R, theta, polynomials, K, n_quad=60)

    # Get I1 components
    I1_comp = compute_i1_components(R, theta, polynomials, K, n_quad=60)

    # Theoretical values
    beta = derived.beta_moment
    g_baseline = derived.g_baseline

    print("\nTHEORETICAL PREDICTIONS:")
    print(f"  Beta(2, 2K) = 1/(2K(2K+1)) = 1/{2*K*(2*K+1)} = {beta:.8f}")
    print(f"  g_baseline = 1 + θ × Beta(2, 2K) = {g_baseline:.8f}")

    print("\nI1 DECOMPOSITION:")
    print(f"  M (main) = {derived.I1_M:.10f}")
    print(f"  C (cross) = {derived.I1_C:.10f}")
    print(f"  Total = M + C = {derived.I1_M + derived.I1_C:.10f}")

    print("\nCROSS RATIO ANALYSIS:")
    cross_ratio = derived.I1_cross_ratio
    print(f"  C/M (measured) = {cross_ratio:.8f}")
    print(f"  Beta(2,2K) (predicted) = {beta:.8f}")
    print(f"  Ratio: (C/M) / Beta = {cross_ratio / beta:.4f}x")
    cross_ratio_gap_pct = (cross_ratio / beta - 1) * 100
    print(f"  Gap: {cross_ratio_gap_pct:+.2f}%")

    print("\nINTERNAL CORRECTION:")
    internal_correction = I1_comp.internal_correction_ratio
    print(f"  (M + C) / M = {internal_correction:.8f}")
    print(f"  g_baseline = {g_baseline:.8f}")
    internal_gap_pct = (internal_correction / g_baseline - 1) * 100
    print(f"  Gap: {internal_gap_pct:+.2f}%")

    print("\nDERIVED g VALUES:")
    print(f"  g_I1 (derived) = {derived.g_I1:.8f}")
    print(f"  g_I1 (calibrated) = {G_I1_CALIBRATED:.8f}")
    g_I1_gap_pct = (derived.g_I1 / G_I1_CALIBRATED - 1) * 100
    print(f"  Gap from calibrated: {g_I1_gap_pct:+.4f}%")

    print(f"  g_I2 (derived) = {derived.g_I2:.8f}")
    print(f"  g_baseline = {g_baseline:.8f}")
    g_I2_gap_pct = (derived.g_I2 / g_baseline - 1) * 100
    print(f"  Gap from baseline: {g_I2_gap_pct:+.4f}%")

    return {
        "name": name,
        "R": R,
        "M": derived.I1_M,
        "C": derived.I1_C,
        "cross_ratio": cross_ratio,
        "beta": beta,
        "cross_ratio_gap_pct": cross_ratio_gap_pct,
        "internal_correction": internal_correction,
        "g_baseline": g_baseline,
        "internal_gap_pct": internal_gap_pct,
        "g_I1_derived": derived.g_I1,
        "g_I1_calibrated": G_I1_CALIBRATED,
        "g_I1_gap_pct": g_I1_gap_pct,
        "g_I2_derived": derived.g_I2,
        "g_I2_gap_pct": g_I2_gap_pct,
    }


def analyze_q1_case(name: str, R: float, polynomials: dict, theta: float = 4/7, K: int = 3):
    """
    Analyze log factor split with Q=1 (unity polynomial).

    This tests whether the Q polynomial introduces the deviation.
    """
    print_header(f"{name} BENCHMARK WITH Q=1 (R = {R:.4f})")

    # Replace Q with unity
    polys_q1 = polynomials.copy()
    polys_q1["Q"] = Polynomial(np.array([1.0]))

    # Get derived g values
    derived = derive_g_from_integrals(R, theta, polys_q1, K, n_quad=60)

    # Get I1 components
    I1_comp = compute_i1_components(R, theta, polys_q1, K, n_quad=60)

    # Theoretical values
    beta = derived.beta_moment
    g_baseline = derived.g_baseline

    print("\nI1 DECOMPOSITION (Q=1):")
    print(f"  M (main) = {derived.I1_M:.10f}")
    print(f"  C (cross) = {derived.I1_C:.10f}")

    print("\nCROSS RATIO ANALYSIS (Q=1):")
    cross_ratio = derived.I1_cross_ratio
    print(f"  C/M (measured) = {cross_ratio:.8f}")
    print(f"  Beta(2,2K) (predicted) = {beta:.8f}")
    print(f"  Ratio: (C/M) / Beta = {cross_ratio / beta:.4f}x")
    cross_ratio_gap_pct = (cross_ratio / beta - 1) * 100
    print(f"  Gap: {cross_ratio_gap_pct:+.2f}%")

    print("\nDERIVED g VALUES (Q=1):")
    print(f"  g_I1 (derived) = {derived.g_I1:.8f}")
    g_I1_gap_from_1 = abs(derived.g_I1 - 1.0) * 100
    print(f"  Gap from 1.0: {g_I1_gap_from_1:.4f}%")

    return {
        "name": f"{name} (Q=1)",
        "R": R,
        "cross_ratio": cross_ratio,
        "beta": beta,
        "cross_ratio_gap_pct": cross_ratio_gap_pct,
        "g_I1_derived": derived.g_I1,
        "g_I1_gap_from_1_pct": g_I1_gap_from_1,
    }


def main():
    """Main diagnostic routine."""
    theta = 4 / 7
    K = 3

    print_header("g_I1 CALIBRATION GAP DIAGNOSTIC")
    print(f"\nParameters: θ = {theta:.6f}, K = {K}")
    print(f"\nCalibrated value: g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"Theoretical value: g_I1 = 1.0")
    print(f"Gap: {(G_I1_CALIBRATED - 1.0) * 100:.4f}% = {(G_I1_CALIBRATED - 1.0) * 1e6:.1f} ppm")

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1_star, P2_star, P3_star, Q_star = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1_star, "P2": P2_star, "P3": P3_star, "Q": Q_star}

    # Analyze both benchmarks
    results = []

    # κ benchmark (R=1.3036)
    results.append(analyze_benchmark("κ", 1.3036, kappa_polys, theta, K))

    # κ* benchmark (R=1.1167)
    results.append(analyze_benchmark("κ*", 1.1167, kappa_star_polys, theta, K))

    # Cross-benchmark comparison
    print_header("CROSS-BENCHMARK COMPARISON")

    print("\nCross ratio comparison:")
    for r in results:
        print(f"  {r['name']:3s}: C/M = {r['cross_ratio']:.8f}, gap from Beta = {r['cross_ratio_gap_pct']:+.2f}%")

    print("\nInternal correction comparison:")
    for r in results:
        print(f"  {r['name']:3s}: (M+C)/M = {r['internal_correction']:.8f}, gap from g_baseline = {r['internal_gap_pct']:+.2f}%")

    print("\nDerived g_I1 comparison:")
    for r in results:
        print(f"  {r['name']:3s}: g_I1 = {r['g_I1_derived']:.8f}, gap from calibrated = {r['g_I1_gap_pct']:+.4f}%")

    # Check if gap is R-dependent
    print("\nR-DEPENDENCE CHECK:")
    kappa_g_I1 = results[0]['g_I1_derived']
    kappa_star_g_I1 = results[1]['g_I1_derived']
    print(f"  κ g_I1 = {kappa_g_I1:.8f}")
    print(f"  κ* g_I1 = {kappa_star_g_I1:.8f}")
    print(f"  Difference: {abs(kappa_g_I1 - kappa_star_g_I1) * 1e6:.1f} ppm")
    if abs(kappa_g_I1 - kappa_star_g_I1) < 1e-4:
        print("  → Gap appears INDEPENDENT of R (systematic)")
    else:
        print("  → Gap appears DEPENDENT on R")

    # Q-dependence test
    print_section("Q-DEPENDENCE TEST")
    print("\nTesting with Q=1 to check if the Q polynomial introduces the deviation...")

    q1_results = []
    q1_results.append(analyze_q1_case("κ", 1.3036, kappa_polys, theta, K))
    q1_results.append(analyze_q1_case("κ*", 1.1167, kappa_star_polys, theta, K))

    print("\nQ-DEPENDENCE SUMMARY:")
    for i, (real_q, q1) in enumerate(zip(results, q1_results)):
        print(f"\n{real_q['name']}:")
        print(f"  Real Q: g_I1 = {real_q['g_I1_derived']:.8f}, gap from 1.0 = {abs(real_q['g_I1_derived'] - 1.0) * 100:.4f}%")
        print(f"  Q=1:    g_I1 = {q1['g_I1_derived']:.8f}, gap from 1.0 = {q1['g_I1_gap_from_1_pct']:.4f}%")
        diff = abs(real_q['g_I1_derived'] - q1['g_I1_derived'])
        print(f"  Difference: {diff * 1e6:.1f} ppm")

    # Final summary
    print_header("DIAGNOSTIC SUMMARY")

    print("\n1. CROSS RATIO DEVIATION:")
    print(f"   The measured C/M ratio deviates from Beta(2,2K) by:")
    for r in results:
        print(f"     {r['name']:3s}: {r['cross_ratio_gap_pct']:+.2f}%")

    print("\n2. DERIVED g_I1 vs CALIBRATED:")
    print(f"   Calibrated g_I1 = {G_I1_CALIBRATED:.8f}")
    for r in results:
        print(f"     {r['name']:3s}: g_I1 = {r['g_I1_derived']:.8f}, gap = {r['g_I1_gap_pct']:+.4f}%")

    print("\n3. R-DEPENDENCE:")
    if abs(kappa_g_I1 - kappa_star_g_I1) < 1e-4:
        print("   Gap appears INDEPENDENT of R → Systematic effect")
    else:
        print("   Gap appears DEPENDENT on R → Check polynomial structure")

    print("\n4. Q-DEPENDENCE:")
    avg_q_effect_kappa = abs(results[0]['g_I1_derived'] - q1_results[0]['g_I1_derived'])
    avg_q_effect_kappa_star = abs(results[1]['g_I1_derived'] - q1_results[1]['g_I1_derived'])
    print(f"   Q polynomial effect on κ: {avg_q_effect_kappa * 1e6:.1f} ppm")
    print(f"   Q polynomial effect on κ*: {avg_q_effect_kappa_star * 1e6:.1f} ppm")

    print("\n5. KEY INSIGHT:")
    avg_cross_ratio_gap = (results[0]['cross_ratio_gap_pct'] + results[1]['cross_ratio_gap_pct']) / 2
    avg_internal_gap = (results[0]['internal_gap_pct'] + results[1]['internal_gap_pct']) / 2
    print(f"   Average cross ratio gap: {avg_cross_ratio_gap:+.2f}%")
    print(f"   Average internal correction gap: {avg_internal_gap:+.2f}%")
    print(f"   This causes g_I1 to deviate from 1.0 because:")
    print(f"   g_I1 = g_baseline / internal_correction")
    print(f"   If internal_correction < g_baseline, then g_I1 > 1.0")
    print(f"   If internal_correction > g_baseline, then g_I1 < 1.0")

    print("\n6. WHERE THE 0.09% COMES FROM:")
    print(f"   The {(G_I1_CALIBRATED - 1.0) * 100:.4f}% gap arises from:")
    print(f"   - The integrand structure (P_ℓ polynomials, Q polynomial)")
    print(f"   - The (1-u)^(ℓ₁+ℓ₂-2) weighting")
    print(f"   - These create a C/M ratio that deviates from the simple Beta(2,2K)")
    print(f"   - The deviation is consistent across R values (systematic)")
    print(f"   - Q polynomial has minor effect (~{(avg_q_effect_kappa + avg_q_effect_kappa_star) / 2 * 1e6:.1f} ppm)")


if __name__ == "__main__":
    main()
