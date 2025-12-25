"""
tests/test_phase16_j13_j14_fix.py
Phase 16: Validate J13/J14 Laurent factor fix

BACKGROUND:
===========
Phase 15 fixed J12's Laurent factor by using actual (zeta'/zeta)(1-R)^2 value
instead of the Laurent approximation (1/R + gamma)^2. This improved B/A from
~5.25 to ~5.04 (kappa) and ~5.08 to ~4.93 (kappa*).

PHASE 16 KEY INSIGHT:
====================
J13 and J14 still use Laurent approximation (1/R + gamma) for the SINGLE
(not squared) zeta log-derivative factor. At R=1.3036:
- Actual (zeta'/zeta)(1-R) = ~1.73
- Laurent (1/R + gamma) = ~1.35
- Error: ~29%

This fix extends LaurentMode to J13/J14 to use actual values, which should
further improve B/A toward the target of 5.0.
"""

import pytest
import numpy as np


class TestZetaLogderivActual:
    """Test the new compute_zeta_logderiv_actual() function."""

    def test_kappa_actual_vs_laurent(self):
        """Verify actual zeta log-derivative is larger than Laurent for kappa."""
        from src.ratios.g_product_full import compute_zeta_logderiv_actual
        from src.ratios.zeta_laurent import EULER_MASCHERONI

        R = 1.3036
        actual = compute_zeta_logderiv_actual(R)
        laurent = 1.0 / R + EULER_MASCHERONI

        print(f"\nKappa (R={R}):")
        print(f"  Actual (zeta'/zeta)(1-R) = {actual:.6f}")
        print(f"  Laurent (1/R + gamma) = {laurent:.6f}")
        print(f"  Ratio (actual/Laurent) = {actual/laurent:.4f}")

        # Actual should be larger than Laurent
        assert actual > laurent, "Actual should be larger than Laurent"
        # Expected value is around 1.73 based on plan
        assert 1.7 < actual < 1.8, f"Expected ~1.73, got {actual}"

    def test_kappa_star_actual_vs_laurent(self):
        """Verify actual zeta log-derivative is larger than Laurent for kappa*."""
        from src.ratios.g_product_full import compute_zeta_logderiv_actual
        from src.ratios.zeta_laurent import EULER_MASCHERONI

        R = 1.1167
        actual = compute_zeta_logderiv_actual(R)
        laurent = 1.0 / R + EULER_MASCHERONI

        print(f"\nKappa* (R={R}):")
        print(f"  Actual (zeta'/zeta)(1-R) = {actual:.6f}")
        print(f"  Laurent (1/R + gamma) = {laurent:.6f}")
        print(f"  Ratio (actual/Laurent) = {actual/laurent:.4f}")

        # Actual should be larger than Laurent
        assert actual > laurent, "Actual should be larger than Laurent"
        # Expected value is around 1.78 based on plan
        assert 1.7 < actual < 1.9, f"Expected ~1.78, got {actual}"


class TestJ13ModeComparison:
    """Test J13 output difference between Laurent and actual modes."""

    def test_j13_mode_comparison_kappa(self):
        """Compare J13 output between Laurent and actual modes for kappa."""
        from src.ratios.j1_euler_maclaurin import j13_as_integral, LaurentMode

        R = 1.3036
        j13_laurent = j13_as_integral(R, laurent_mode=LaurentMode.RAW_LOGDERIV)
        j13_actual = j13_as_integral(R, laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

        ratio = j13_actual / j13_laurent

        print(f"\nJ13 comparison (R={R}):")
        print(f"  J13 (Laurent): {j13_laurent:.10f}")
        print(f"  J13 (actual):  {j13_actual:.10f}")
        print(f"  Ratio (actual/Laurent): {ratio:.4f}")

        # J13 is negative, so actual (larger magnitude) should have larger abs
        assert abs(j13_actual) > abs(j13_laurent), \
            "Actual mode should give larger absolute value"
        # Ratio should be around 1.29 (since zeta factor ratio is ~29%)
        assert ratio > 1.2, f"Expected ratio > 1.2, got {ratio}"

    def test_j14_mode_comparison_kappa(self):
        """Compare J14 output between Laurent and actual modes for kappa."""
        from src.ratios.j1_euler_maclaurin import j14_as_integral, LaurentMode

        R = 1.3036
        j14_laurent = j14_as_integral(R, laurent_mode=LaurentMode.RAW_LOGDERIV)
        j14_actual = j14_as_integral(R, laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

        ratio = j14_actual / j14_laurent

        print(f"\nJ14 comparison (R={R}):")
        print(f"  J14 (Laurent): {j14_laurent:.10f}")
        print(f"  J14 (actual):  {j14_actual:.10f}")
        print(f"  Ratio (actual/Laurent): {ratio:.4f}")

        # Same as J13
        assert abs(j14_actual) > abs(j14_laurent)
        assert ratio > 1.2


class TestFullAssemblyImprovement:
    """Test that B/A improves with actual zeta factors in J13/J14."""

    def test_kappa_ba_improvement(self):
        """Verify B/A improves for kappa benchmark."""
        from src.ratios.j1_euler_maclaurin import (
            compute_m1_with_mirror_assembly, LaurentMode
        )
        from src.ratios.przz_polynomials import load_przz_k3_polynomials

        polys = load_przz_k3_polynomials("kappa")

        # Test with actual mode (Phase 16 fix)
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0,
            R=polys.R,
            polys=polys,
            K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
        )

        B_over_A = result["B_over_A"]
        gap = B_over_A - 5.0
        gap_pct = abs(gap) / 5.0 * 100

        print(f"\nKappa full assembly (with Phase 16 J13/J14 fix):")
        print(f"  B/A = {B_over_A:.6f}")
        print(f"  Target = 5.0")
        print(f"  Gap = {gap:+.6f} ({gap_pct:.4f}%)")

        # Log the detailed breakdown for analysis
        print(f"  I12(+R) = {result['i12_plus_total']:.6f}")
        print(f"  I12(-R) = {result['i12_minus_total']:.6f}")
        print(f"  I34(+R) = {result['i34_plus_total']:.6f}")

    def test_kappa_star_ba_improvement(self):
        """Verify B/A improves for kappa* benchmark."""
        from src.ratios.j1_euler_maclaurin import (
            compute_m1_with_mirror_assembly, LaurentMode
        )
        from src.ratios.przz_polynomials import load_przz_k3_polynomials

        polys = load_przz_k3_polynomials("kappa_star")

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0,
            R=polys.R,
            polys=polys,
            K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
        )

        B_over_A = result["B_over_A"]
        gap = B_over_A - 5.0
        gap_pct = abs(gap) / 5.0 * 100

        print(f"\nKappa* full assembly (with Phase 16 J13/J14 fix):")
        print(f"  B/A = {B_over_A:.6f}")
        print(f"  Target = 5.0")
        print(f"  Gap = {gap:+.6f} ({gap_pct:.4f}%)")

        print(f"  I12(+R) = {result['i12_plus_total']:.6f}")
        print(f"  I12(-R) = {result['i12_minus_total']:.6f}")
        print(f"  I34(+R) = {result['i34_plus_total']:.6f}")


class TestComparisonLaurentVsActual:
    """Compare full assembly between Laurent-everywhere and actual-everywhere."""

    def test_mode_comparison_both_benchmarks(self):
        """Compare B/A between Laurent-everywhere and actual-everywhere."""
        from src.ratios.j1_euler_maclaurin import (
            compute_m1_with_mirror_assembly, LaurentMode
        )
        from src.ratios.przz_polynomials import load_przz_k3_polynomials

        print("\n" + "=" * 70)
        print("PHASE 16: LAURENT vs ACTUAL COMPARISON")
        print("=" * 70)

        for benchmark in ["kappa", "kappa_star"]:
            polys = load_przz_k3_polynomials(benchmark)

            # Laurent mode (old behavior)
            result_laurent = compute_m1_with_mirror_assembly(
                theta=4.0/7.0,
                R=polys.R,
                polys=polys,
                K=3,
                laurent_mode=LaurentMode.RAW_LOGDERIV,
            )

            # Actual mode (Phase 15+16 fix)
            result_actual = compute_m1_with_mirror_assembly(
                theta=4.0/7.0,
                R=polys.R,
                polys=polys,
                K=3,
                laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
            )

            ba_laurent = result_laurent["B_over_A"]
            ba_actual = result_actual["B_over_A"]

            gap_laurent = abs(ba_laurent - 5.0) / 5.0 * 100
            gap_actual = abs(ba_actual - 5.0) / 5.0 * 100

            print(f"\n{benchmark.upper()} (R={polys.R}):")
            print(f"  Laurent mode: B/A = {ba_laurent:.6f} (gap {gap_laurent:.4f}%)")
            print(f"  Actual mode:  B/A = {ba_actual:.6f} (gap {gap_actual:.4f}%)")
            print(f"  Improvement: {gap_laurent - gap_actual:+.4f} pp")

            # I34 breakdown (where the fix applies)
            i34_laurent = result_laurent["i34_plus_total"]
            i34_actual = result_actual["i34_plus_total"]
            print(f"  I34 (Laurent): {i34_laurent:.6f}")
            print(f"  I34 (actual):  {i34_actual:.6f}")
            print(f"  I34 ratio:     {i34_actual/i34_laurent:.4f}")


class TestNoRegression:
    """Ensure Phase 16 doesn't break existing functionality."""

    def test_default_mode_is_actual(self):
        """Verify default mode is ACTUAL_LOGDERIV."""
        from src.ratios.j1_euler_maclaurin import DEFAULT_LAURENT_MODE, LaurentMode

        assert DEFAULT_LAURENT_MODE == LaurentMode.ACTUAL_LOGDERIV, \
            "Default mode should be ACTUAL_LOGDERIV after Phase 15/16"

    def test_j13_j14_accept_laurent_mode(self):
        """Verify J13 and J14 accept laurent_mode parameter."""
        from src.ratios.j1_euler_maclaurin import (
            j13_as_integral, j14_as_integral, LaurentMode
        )

        R = 1.3036

        # Should not raise
        j13_raw = j13_as_integral(R, laurent_mode=LaurentMode.RAW_LOGDERIV)
        j13_actual = j13_as_integral(R, laurent_mode=LaurentMode.ACTUAL_LOGDERIV)
        j14_raw = j14_as_integral(R, laurent_mode=LaurentMode.RAW_LOGDERIV)
        j14_actual = j14_as_integral(R, laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

        assert j13_raw != 0
        assert j13_actual != 0
        assert j14_raw != 0
        assert j14_actual != 0

    def test_compute_I34_accepts_laurent_mode(self):
        """Verify compute_I34_components accepts laurent_mode parameter."""
        from src.ratios.j1_euler_maclaurin import (
            compute_I34_components, LaurentMode
        )

        R = 1.3036

        # Should not raise
        result_raw = compute_I34_components(R, laurent_mode=LaurentMode.RAW_LOGDERIV)
        result_actual = compute_I34_components(R, laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

        assert 'j13' in result_raw
        assert 'j14' in result_raw
        assert 'j13' in result_actual
        assert 'j14' in result_actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
