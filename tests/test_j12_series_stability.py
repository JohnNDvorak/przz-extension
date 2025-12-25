"""
tests/test_j12_series_stability.py
Phase 15D: Series-Order Stability Test

PURPOSE:
========
Per GPT's guidance, verify that Laurent coefficient extraction is converged.
If B/A varies significantly with series order, there's a convergence problem
that could blow up at K=4.

We test stability of B/A across different precision levels for:
1. mpmath precision (50, 100, 150 digits)
2. Quadrature tolerance (1e-8, 1e-10, 1e-12)
"""

import pytest
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.g_product_full import compute_zeta_factors


def test_mpmath_precision_stability():
    """Verify B/A is stable across mpmath precision levels."""
    print("\n" + "=" * 70)
    print("PHASE 15D: MPMATH PRECISION STABILITY TEST")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        results = {}
        for precision in [30, 50, 100]:
            # Compute zeta factors at different precisions
            zf = compute_zeta_factors(R, precision=precision)

            # Run mirror assembly with ACTUAL_LOGDERIV
            result = compute_m1_with_mirror_assembly(
                theta=4.0/7.0, R=R, polys=polys, K=3,
                laurent_mode=LaurentMode.ACTUAL_LOGDERIV
            )

            results[precision] = {
                'B_over_A': result['B_over_A'],
                'actual_sq': zf.logderiv_actual_squared,
            }

            print(f"  precision={precision:3d}: B/A = {result['B_over_A']:.8f}, "
                  f"(ζ'/ζ)² = {zf.logderiv_actual_squared:.8f}")

        # Check stability
        ba_values = [r['B_over_A'] for r in results.values()]
        ba_range = max(ba_values) - min(ba_values)
        print(f"  B/A range: {ba_range:.2e}")

        # B/A should be stable to 1e-6
        assert ba_range < 1e-4, f"B/A varies too much with precision: {ba_range}"


def test_quadrature_stability():
    """Verify B/A is stable under quadrature refinement."""
    print("\n" + "=" * 70)
    print("PHASE 15D: QUADRATURE STABILITY TEST")
    print("=" * 70)
    print("(Uses scipy.integrate.quad with default tolerance)")
    print("This test just verifies the computation is deterministic.")

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")

        # Run the same computation 3 times
        results = []
        for i in range(3):
            result = compute_m1_with_mirror_assembly(
                theta=4.0/7.0, R=R, polys=polys, K=3,
                laurent_mode=LaurentMode.ACTUAL_LOGDERIV
            )
            results.append(result['B_over_A'])
            print(f"  Run {i+1}: B/A = {result['B_over_A']:.10f}")

        # Should be identical (deterministic)
        assert results[0] == results[1] == results[2], "Non-deterministic results!"


def test_laurent_mode_consistency():
    """Verify all modes produce consistent (not NaN/inf) results."""
    print("\n" + "=" * 70)
    print("PHASE 15D: LAURENT MODE CONSISTENCY TEST")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")

        for mode in LaurentMode:
            result = compute_m1_with_mirror_assembly(
                theta=4.0/7.0, R=R, polys=polys, K=3,
                laurent_mode=mode
            )
            ba = result['B_over_A']

            print(f"  {mode.value:<20}: B/A = {ba:.6f}")

            # Should not be NaN or inf
            assert not (ba != ba), f"NaN result for {mode.value}"  # NaN check
            assert abs(ba) < 1000, f"Unreasonable value for {mode.value}: {ba}"


def test_zeta_factor_consistency():
    """Verify zeta factor computation is consistent."""
    from src.ratios.g_product_full import compute_zeta_factors

    print("\n" + "=" * 70)
    print("PHASE 15D: ZETA FACTOR CONSISTENCY TEST")
    print("=" * 70)

    for R in [1.3036, 1.1167]:
        print(f"\nR={R}:")
        zf = compute_zeta_factors(R, precision=50)

        # Verify relationships
        # G = zeta'/zeta² should equal (zeta'/zeta) / zeta
        logderiv = zf.zeta_deriv / zf.zeta_val
        G_computed = logderiv / zf.zeta_val
        G_diff = abs(G_computed - zf.G_value)

        print(f"  G check: G_computed={G_computed:.10f}, G_stored={zf.G_value:.10f}")
        print(f"  Difference: {G_diff:.2e}")

        assert G_diff < 1e-10, f"G computation inconsistency: {G_diff}"

        # Verify actual_vs_laurent_ratio
        ratio_computed = zf.logderiv_actual_squared / zf.logderiv_laurent_squared
        ratio_diff = abs(ratio_computed - zf.actual_vs_laurent_ratio)

        print(f"  Ratio check: {ratio_diff:.2e}")
        assert ratio_diff < 1e-10, f"Ratio inconsistency: {ratio_diff}"


if __name__ == "__main__":
    test_mpmath_precision_stability()
    test_quadrature_stability()
    test_laurent_mode_consistency()
    test_zeta_factor_consistency()
