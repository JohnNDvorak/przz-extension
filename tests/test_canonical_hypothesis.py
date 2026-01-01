#!/usr/bin/env python3
"""
tests/test_canonical_hypothesis.py
Phase 51: Test the canonical hypothesis and document its failure

HYPOTHESIS (DISPROVED):
The exp(2Rt) factor inside the compute_I1/I2_unified_paper() kernels
encodes the combined (direct + mirror) contribution, making the external
scalar m unnecessary.

FINDING:
The hypothesis is WRONG. The exp(2Rt) factor is a structural element
of the kernel, NOT a replacement for the mirror term.

KEY EVIDENCE:
1. Without m: c = 0.197, κ = 2.24 (WRONG)
2. With m: c = 2.137, κ = 0.4173 (CORRECT)
3. Mirror contribution (m × S12_minus) = 90.8% of c

CONCLUSION:
The calibrated scalar m = exp(R) + (2K-1) is ESSENTIAL for reproducing
PRZZ's κ = 0.417293962. The canonical hypothesis was incorrect.

Created: 2025-12-28 (Phase 51)
"""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_evaluator import (
    compute_przz_baseline_przz_canonical,
    compute_integrals_przz_canonical,
    PRZZCanonicalResult,
)
from src.kappa_engine import KappaEngine


class TestCanonicalHypothesisFails:
    """Tests demonstrating that the canonical hypothesis is wrong."""

    def test_canonical_gives_10x_collapse(self):
        """Canonical mode (no m) gives 10x collapse in c."""
        result = compute_przz_baseline_przz_canonical(n_quad=80)

        # Canonical gives c ≈ 0.2 (should be ~2.14)
        assert result.c < 0.3, f"Expected c < 0.3, got {result.c}"

        # This is about 10x too small
        c_target = 2.13745440613217263636
        ratio = result.c / c_target
        assert ratio < 0.15, f"Expected c ratio < 0.15, got {ratio}"

    def test_canonical_kappa_is_wrong(self):
        """Canonical mode gives κ ≈ 2.2, not 0.417."""
        result = compute_przz_baseline_przz_canonical(n_quad=80)

        # κ should be around 0.417, but canonical gives ~2.2
        assert result.kappa > 2.0, f"Expected κ > 2.0, got {result.kappa}"

        kappa_target = 0.417293962
        gap_pct = (result.kappa / kappa_target - 1) * 100
        assert gap_pct > 400, f"Expected gap > 400%, got {gap_pct}%"

    def test_scalar_mode_works(self):
        """Scalar mode (with m) correctly reproduces PRZZ."""
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        result = engine.compute_kappa()

        # Scalar mode should match PRZZ target
        kappa_target = 0.417293962
        gap_pct = abs(result.kappa / kappa_target - 1) * 100
        assert gap_pct < 0.1, f"Expected gap < 0.1%, got {gap_pct}%"


class TestMirrorContributionDominates:
    """Tests showing that the mirror contribution is ~90% of c."""

    def test_mirror_is_90_percent_of_c(self):
        """The mirror term m × S12_minus is ~90% of c."""
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        result = engine.compute_kappa()

        mirror_contribution = result.corrections.m * result.integrals.S12_minus
        mirror_fraction = mirror_contribution / result.c

        # Mirror should be ~90% of c
        assert 0.85 < mirror_fraction < 0.95, f"Expected mirror fraction 0.85-0.95, got {mirror_fraction}"

    def test_without_mirror_c_is_10x_smaller(self):
        """Without mirror, c is about 10x smaller."""
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        result = engine.compute_kappa()

        c_no_mirror = result.integrals.S12_plus + result.integrals.S34_plus
        ratio = c_no_mirror / result.c

        # Without mirror, c should be ~10% of full c
        assert 0.05 < ratio < 0.15, f"Expected ratio 0.05-0.15, got {ratio}"


class TestDifferenceQuotientLimitDoesNotWork:
    """Tests showing that the DQ scalar limit doesn't reproduce PRZZ."""

    def test_dq_limit_gives_wrong_kappa(self):
        """Using m = (exp(2R)-1)/(2R) gives wrong κ."""
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        result = engine.compute_kappa()

        # Difference quotient scalar limit
        R = 1.3036
        dq_limit = (math.exp(2 * R) - 1) / (2 * R)  # ≈ 4.82

        # Compute c with DQ limit instead of calibrated m
        c_dq = (
            result.integrals.S12_plus
            + dq_limit * result.integrals.S12_minus
            + result.integrals.S34_plus
        )
        kappa_dq = 1 - math.log(c_dq) / R

        # DQ limit gives κ ≈ 0.82, not 0.42
        kappa_target = 0.417293962
        gap_pct = (kappa_dq / kappa_target - 1) * 100
        assert gap_pct > 90, f"Expected DQ κ gap > 90%, got {gap_pct}%"

    def test_calibrated_m_is_almost_2x_dq_limit(self):
        """The calibrated m = 8.81 is almost 2x the DQ limit = 4.82."""
        R = 1.3036
        K = 3

        dq_limit = (math.exp(2 * R) - 1) / (2 * R)  # ≈ 4.82
        calibrated_base = math.exp(R) + (2 * K - 1)  # ≈ 8.68

        ratio = calibrated_base / dq_limit
        # Ratio should be around 1.8
        assert 1.7 < ratio < 1.9, f"Expected ratio 1.7-1.9, got {ratio}"


class TestIntegralComponentsMatch:
    """Tests showing that canonical I values match scalar +R values."""

    def test_i1_plus_matches_canonical(self):
        """Canonical I1_total matches scalar I1_plus."""
        canonical = compute_przz_baseline_przz_canonical(n_quad=80)
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        scalar = engine.compute_kappa()

        rel_diff = abs(canonical.I1_total - scalar.integrals.I1_plus) / abs(scalar.integrals.I1_plus)
        assert rel_diff < 1e-6, f"I1 mismatch: {rel_diff}"

    def test_i2_plus_matches_canonical(self):
        """Canonical I2_total matches scalar I2_plus."""
        canonical = compute_przz_baseline_przz_canonical(n_quad=80)
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        scalar = engine.compute_kappa()

        rel_diff = abs(canonical.I2_total - scalar.integrals.I2_plus) / abs(scalar.integrals.I2_plus)
        assert rel_diff < 1e-6, f"I2 mismatch: {rel_diff}"

    def test_s34_matches_canonical(self):
        """Canonical (I3 + I4) matches scalar S34_plus."""
        canonical = compute_przz_baseline_przz_canonical(n_quad=80)
        engine = KappaEngine.from_przz_kappa(n_quad=80)
        scalar = engine.compute_kappa()

        canonical_s34 = canonical.I3_total + canonical.I4_total
        rel_diff = abs(canonical_s34 - scalar.integrals.S34_plus) / abs(scalar.integrals.S34_plus)
        assert rel_diff < 1e-6, f"S34 mismatch: {rel_diff}"


class TestMirrorScalarGapAnalysis:
    """Tests documenting the 1.8× gap between DQ limit and production m.

    See docs/MIRROR_SCALAR_GAP_ANALYSIS.md for full analysis.

    KEY FINDING:
    - DQ scalar limit = (exp(2R)-1)/(2R) = 4.82
    - Production m = exp(R) + 5 = 8.68
    - Ratio = 1.80× (unexplained gap)

    The derivation chain from PRZZ to m = exp(R) + (2K-1) has a gap
    that needs resolution.
    """

    def test_dq_scalar_limit_value(self):
        """Document the DQ scalar limit = (exp(2R)-1)/(2R)."""
        R = 1.3036
        dq_limit = (math.exp(2 * R) - 1) / (2 * R)

        # Should be approximately 4.82
        assert 4.8 < dq_limit < 4.9, f"DQ limit = {dq_limit}"
        print(f"\n  DQ scalar limit = {dq_limit:.4f}")

    def test_production_m_value(self):
        """Document the production m = exp(R) + (2K-1)."""
        R = 1.3036
        K = 3

        production_m = math.exp(R) + (2 * K - 1)

        # Should be approximately 8.68
        assert 8.6 < production_m < 8.8, f"Production m = {production_m}"
        print(f"\n  Production m = exp(R) + 5 = {production_m:.4f}")

    def test_gap_ratio_is_1_8x(self):
        """Document the 1.8× gap between DQ limit and production m."""
        R = 1.3036
        K = 3

        dq_limit = (math.exp(2 * R) - 1) / (2 * R)
        production_m = math.exp(R) + (2 * K - 1)

        ratio = production_m / dq_limit

        # Gap ratio should be ~1.8
        assert 1.75 < ratio < 1.85, f"Gap ratio = {ratio}"
        print(f"\n  Gap ratio = m / DQ_limit = {ratio:.4f}")
        print(f"  This gap is UNEXPLAINED in current derivation")

    def test_przz_prefactor_is_exp_2r(self):
        """PRZZ has T^{-(α+β)} = exp(2R), not exp(R)."""
        R = 1.3036

        # At α = β = -R/L, T^{-(α+β)} = T^{2R/L} = exp(2R)
        przz_prefactor = math.exp(2 * R)
        exp_r = math.exp(R)

        print(f"\n  PRZZ prefactor exp(2R) = {przz_prefactor:.4f}")
        print(f"  Production uses exp(R) = {exp_r:.4f}")
        print(f"  Ratio = {przz_prefactor / exp_r:.4f} (should be explained)")

        # exp(2R) ≈ 13.6, exp(R) ≈ 3.68
        assert przz_prefactor > 13, f"exp(2R) = {przz_prefactor}"
        assert exp_r < 4, f"exp(R) = {exp_r}"

    def test_candidate_m_formulas(self):
        """Test various candidate m formulas against PRZZ target.

        NOTE: The production formula uses m = g_total × base, where:
        - base = exp(R) + 5 (partially derived, ~2.5% gap)
        - g_total = calibrated factor (~1.0) to close the gap

        This test documents the gaps for various BASE formulas (without g).
        """
        R = 1.3036
        K = 3

        engine = KappaEngine.from_przz_kappa(n_quad=80)
        result = engine.compute_kappa()

        kappa_target = 0.417293962

        # Various candidate m values (BASE formulas, no g correction)
        candidates = {
            "base: exp(R) + 5": math.exp(R) + 5,
            "DQ limit: (exp(2R)-1)/(2R)": (math.exp(2 * R) - 1) / (2 * R),
            "PRZZ prefactor: exp(2R)": math.exp(2 * R),
            "half PRZZ: exp(2R)/2": math.exp(2 * R) / 2,
            "simple: exp(R)": math.exp(R),
        }

        print("\n  κ values for different BASE m formulas (no g correction):")
        print("  " + "-" * 50)

        for name, m_val in candidates.items():
            c_test = (
                result.integrals.S12_plus
                + m_val * result.integrals.S12_minus
                + result.integrals.S34_plus
            )
            kappa_test = 1 - math.log(c_test) / R
            gap_pct = (kappa_test / kappa_target - 1) * 100
            print(f"  {name}:")
            print(f"    m = {m_val:.4f}, κ = {kappa_test:.4f}, gap = {gap_pct:+.1f}%")

        # Base formula (exp(R) + 5) gives ~2.5% gap
        # This is the "partially derived" formula
        base_m = math.exp(R) + 5
        c_base = (
            result.integrals.S12_plus
            + base_m * result.integrals.S12_minus
            + result.integrals.S34_plus
        )
        kappa_base = 1 - math.log(c_base) / R
        base_gap = abs(kappa_base / kappa_target - 1) * 100

        # Base formula gives ~2.5% gap (acceptable - improved by g factor)
        assert base_gap < 5, f"Base gap = {base_gap}% (expected < 5%)"

        # Production formula (with g correction) should give < 0.1% gap
        prod_gap = abs(result.kappa / kappa_target - 1) * 100
        print(f"\n  Production (with g correction):")
        print(f"    m = {result.corrections.m:.4f}, κ = {result.kappa:.4f}, gap = {prod_gap:+.3f}%")

        assert prod_gap < 0.1, f"Production gap = {prod_gap}%"


if __name__ == "__main__":
    print("=" * 70)
    print("CANONICAL HYPOTHESIS TEST")
    print("=" * 70)
    print()

    # Quick summary
    canonical = compute_przz_baseline_przz_canonical(n_quad=80)
    engine = KappaEngine.from_przz_kappa(n_quad=80)
    scalar = engine.compute_kappa()

    print("CANONICAL MODE (no m):")
    print(f"  c = {canonical.c:.4f}")
    print(f"  κ = {canonical.kappa:.4f}")
    print()

    print("SCALAR MODE (with m):")
    print(f"  c = {scalar.c:.4f}")
    print(f"  κ = {scalar.kappa:.4f}")
    print(f"  m = {scalar.corrections.m:.4f}")
    print()

    print("MIRROR CONTRIBUTION:")
    mirror = scalar.corrections.m * scalar.integrals.S12_minus
    print(f"  m × S12_minus = {mirror:.4f}")
    print(f"  Fraction of c = {mirror / scalar.c * 100:.1f}%")
    print()

    print("CONCLUSION: Canonical hypothesis DISPROVED")
    print("  - exp(2Rt) inside kernel is NOT combined mirror")
    print("  - Scalar m is ESSENTIAL (contributes ~91% of c)")
    print("  - Calibrated m = exp(R) + 5 is necessary")
