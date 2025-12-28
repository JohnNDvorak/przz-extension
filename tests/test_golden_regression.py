"""
tests/test_golden_regression.py
Regression tests against golden output snapshot.

This test ensures the production kappa_engine output doesn't drift
from the locked golden values. Any change to the core formulas or
integrals that causes drift will fail these tests.

Created: 2025-12-27 (Phase 46++)
"""

import pytest
import json
from pathlib import Path

from src.kappa_engine import (
    KappaEngine,
    compute_g_I1,
    compute_g_I2,
    compute_base,
)


# Load golden output
GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_kappa_production.json"


@pytest.fixture
def golden():
    """Load golden output snapshot."""
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestGoldenDerivedConstants:
    """Verify derived constants match golden values exactly."""

    def test_g_I1_matches_golden(self, golden):
        """g_I1 must match golden value exactly."""
        theta = golden["parameters"]["theta"]
        K = golden["parameters"]["K"]

        computed = compute_g_I1(theta, K)
        expected = golden["derived_constants"]["g_I1"]

        assert computed == pytest.approx(expected, rel=1e-12), \
            f"g_I1 drifted: {computed} vs golden {expected}"

    def test_g_I2_matches_golden(self, golden):
        """g_I2 must match golden value exactly."""
        theta = golden["parameters"]["theta"]
        K = golden["parameters"]["K"]

        computed = compute_g_I2(theta, K)
        expected = golden["derived_constants"]["g_I2"]

        assert computed == pytest.approx(expected, rel=1e-12), \
            f"g_I2 drifted: {computed} vs golden {expected}"


class TestGoldenKappaBenchmark:
    """Verify kappa benchmark matches golden values."""

    def test_kappa_matches_golden(self, golden):
        """kappa computed value must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_benchmark"]["kappa_computed"]

        # Allow small tolerance for quadrature variations
        assert result.kappa == pytest.approx(expected, rel=1e-8), \
            f"kappa drifted: {result.kappa} vs golden {expected}"

    def test_c_matches_golden(self, golden):
        """c computed value must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_benchmark"]["c_computed"]

        assert result.c == pytest.approx(expected, rel=1e-8), \
            f"c drifted: {result.c} vs golden {expected}"

    def test_S12_plus_matches_golden(self, golden):
        """S12(+R) must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_benchmark"]["S12_plus"]

        assert result.integrals.S12_plus == pytest.approx(expected, rel=1e-6), \
            f"S12_plus drifted: {result.integrals.S12_plus} vs golden {expected}"

    def test_m_matches_golden(self, golden):
        """Mirror multiplier m must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_benchmark"]["m"]

        assert result.corrections.m == pytest.approx(expected, rel=1e-8), \
            f"m drifted: {result.corrections.m} vs golden {expected}"


class TestGoldenKappaStarBenchmark:
    """Verify kappa* benchmark matches golden values."""

    def test_kappa_star_matches_golden(self, golden):
        """kappa* computed value must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa_star(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_star_benchmark"]["kappa_computed"]

        assert result.kappa == pytest.approx(expected, rel=1e-8), \
            f"kappa* drifted: {result.kappa} vs golden {expected}"

    def test_c_star_matches_golden(self, golden):
        """c* computed value must match golden."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa_star(n_quad=n_quad)
        result = engine.compute_kappa()

        expected = golden["kappa_star_benchmark"]["c_computed"]

        assert result.c == pytest.approx(expected, rel=1e-8), \
            f"c* drifted: {result.c} vs golden {expected}"


class TestGoldenTolerances:
    """Verify both benchmarks are within golden tolerances."""

    def test_kappa_within_tolerance(self, golden):
        """kappa must be within tolerance of target."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
        result = engine.compute_kappa()

        target = golden["kappa_benchmark"]["kappa_target"]
        tolerance = golden["tolerances"]["kappa_gap_pct"]

        gap_pct = abs(result.kappa / target - 1) * 100

        assert gap_pct < tolerance, \
            f"kappa gap {gap_pct:.6f}% exceeds tolerance {tolerance}%"

    def test_kappa_star_within_tolerance(self, golden):
        """kappa* must be within tolerance of target."""
        n_quad = golden["metadata"]["n_quad"]
        engine = KappaEngine.from_przz_kappa_star(n_quad=n_quad)
        result = engine.compute_kappa()

        target = golden["kappa_star_benchmark"]["kappa_target"]
        tolerance = golden["tolerances"]["kappa_gap_pct"]

        gap_pct = abs(result.kappa / target - 1) * 100

        assert gap_pct < tolerance, \
            f"kappa* gap {gap_pct:.6f}% exceeds tolerance {tolerance}%"
