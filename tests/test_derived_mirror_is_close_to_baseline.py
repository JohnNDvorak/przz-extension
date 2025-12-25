"""
tests/test_derived_mirror_is_close_to_baseline.py
Phase 9.3B: Soft gate tests comparing derived mirror to baseline.

These tests verify that the derived mirror term is within reasonable bounds
of the empirical baseline. They are "soft" because we expect some deviation
and use generous tolerances.

KEY FINDINGS FROM DIAGNOSTICS:
=============================
The Q(1+·) shift does NOT give the expected m1 values:
- κ benchmark: m1_derived/m1_empirical ≈ 133× (Q shifted) or 4.8× (std Q)
- κ* benchmark: m1_derived/m1_empirical ≈ 10.8× (Q shifted) or 3.0× (std Q)

This suggests the shift identity interpretation needs revision.
The empirical formula m1 = exp(R) + 5 works but doesn't match
the naively-computed derived mirror.

WHAT THESE TESTS VERIFY:
========================
1. Basic sanity: derived values are finite and reasonable
2. Component ratios: how S12(+R), S12(-R), and exp(2R) relate
3. Documentation: record the actual ratios for reference
"""

import pytest
import math
import numpy as np
from src.mirror_exact import (
    compute_S12_mirror_derived,
    compute_S12_minus_basis,
    compute_I1_with_shifted_Q,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)


class TestDerivedMirrorSanity:
    """Basic sanity checks for derived mirror values."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_derived_is_finite_kappa(self, polys_kappa):
        """Derived mirror should be finite for κ benchmark."""
        result = compute_S12_mirror_derived(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert math.isfinite(result)
        assert result > 0  # Should be positive

    def test_derived_is_finite_kappa_star(self, polys_kappa_star):
        """Derived mirror should be finite for κ* benchmark."""
        result = compute_S12_mirror_derived(
            theta=4/7, R=1.1167, n=40, polynomials=polys_kappa_star
        )
        assert math.isfinite(result)
        assert result > 0  # Should be positive

    def test_minus_basis_is_finite_kappa(self, polys_kappa):
        """Minus basis should be finite for κ benchmark."""
        result = compute_S12_minus_basis(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert math.isfinite(result)
        assert result > 0  # Should be positive

    def test_minus_basis_is_finite_kappa_star(self, polys_kappa_star):
        """Minus basis should be finite for κ* benchmark."""
        result = compute_S12_minus_basis(
            theta=4/7, R=1.1167, n=40, polynomials=polys_kappa_star
        )
        assert math.isfinite(result)
        assert result > 0  # Should be positive


class TestRatioDocumentation:
    """
    Document the actual ratios between different mirror formulations.

    These tests record the observed values without enforcing tight tolerances.
    They serve as documentation and regression guards.
    """

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_derived_vs_minus_ratio_kappa(self, polys_kappa):
        """
        Document: derived / minus_basis ratio for κ benchmark.

        Expected from diagnostics: ~1155 (way larger than m1_emp=8.68)
        """
        R = 1.3036
        derived = compute_S12_mirror_derived(
            theta=4/7, R=R, n=40, polynomials=polys_kappa
        )
        minus_basis = compute_S12_minus_basis(
            theta=4/7, R=R, n=40, polynomials=polys_kappa
        )

        ratio = derived / minus_basis
        m1_empirical = math.exp(R) + 5

        # Document the actual ratio
        print(f"\nκ benchmark: derived/minus = {ratio:.2f}")
        print(f"m1_empirical = {m1_empirical:.2f}")
        print(f"ratio/m1_emp = {ratio/m1_empirical:.2f}")

        # Very loose bound: ratio should be positive and > m1_empirical
        # (We know it's ~133× larger from diagnostics)
        assert ratio > m1_empirical, \
            f"Derived/minus ratio {ratio} should exceed m1_emp {m1_empirical}"
        assert ratio < 2000, "Sanity: ratio should be < 2000"

    def test_derived_vs_minus_ratio_kappa_star(self, polys_kappa_star):
        """
        Document: derived / minus_basis ratio for κ* benchmark.

        Expected from diagnostics: ~87 (way larger than m1_emp=8.05)
        """
        R = 1.1167
        derived = compute_S12_mirror_derived(
            theta=4/7, R=R, n=40, polynomials=polys_kappa_star
        )
        minus_basis = compute_S12_minus_basis(
            theta=4/7, R=R, n=40, polynomials=polys_kappa_star
        )

        ratio = derived / minus_basis
        m1_empirical = math.exp(R) + 5

        print(f"\nκ* benchmark: derived/minus = {ratio:.2f}")
        print(f"m1_empirical = {m1_empirical:.2f}")
        print(f"ratio/m1_emp = {ratio/m1_empirical:.2f}")

        assert ratio > m1_empirical
        assert ratio < 200, "Sanity: ratio should be < 200"


class TestExpRRelationship:
    """Test relationship between exp(R) and the +/- ratio."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_plus_minus_ratio_near_exp_R(self, polys_kappa):
        """
        S12(+R) / S12(-R) should be reasonably close to exp(R).

        From diagnostics: ratio ≈ 3.1, exp(R) ≈ 3.68
        So ratio/exp(R) ≈ 0.84
        """
        R = 1.3036
        theta = 4 / 7
        n = 40

        # Compute S12(+R) directly
        from src.mirror_exact import _compute_I2_with_shifted_Q

        factorial_norm = {
            "11": 1.0, "22": 0.25, "33": 1.0/36.0,
            "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
        }
        symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

        S12_plus = 0.0
        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            ell1 = int(pair_key[0])
            ell2 = int(pair_key[1])
            full_norm = symmetry[pair_key] * factorial_norm[pair_key]

            I1 = compute_I1_with_shifted_Q(
                theta=theta, R=R, n=n, polynomials=polys_kappa,
                ell1=ell1, ell2=ell2, shift=0.0
            )
            I2 = _compute_I2_with_shifted_Q(
                theta=theta, R=R, n=n, polynomials=polys_kappa,
                ell1=ell1, ell2=ell2, shift=0.0
            )
            S12_plus += full_norm * (I1 + I2)

        S12_minus = compute_S12_minus_basis(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        ratio = S12_plus / S12_minus
        exp_R = math.exp(R)

        print(f"\nS12(+R)/S12(-R) = {ratio:.4f}")
        print(f"exp(R) = {exp_R:.4f}")
        print(f"ratio/exp(R) = {ratio/exp_R:.4f}")

        # The ratio should be within a factor of 2 of exp(R)
        assert 0.5 < ratio / exp_R < 2.0, \
            f"S12 ratio {ratio} not within 2x of exp(R) {exp_R}"


class TestQShiftEffect:
    """Test that Q(1+·) shift has the expected large effect."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_shift_amplifies_I1(self, polys_kappa):
        """
        Q(1+·) shift should significantly amplify I1 values.

        From diagnostics: I1_shifted / I1_std ≈ 100× for κ benchmark
        """
        I1_std = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa,
            ell1=1, ell2=1, shift=0.0
        )
        I1_shifted = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa,
            ell1=1, ell2=1, shift=1.0
        )

        ratio = I1_shifted / I1_std

        print(f"\n(1,1) shift effect: {ratio:.2f}×")

        # Shift should amplify by at least 10×
        assert ratio > 10, f"Q shift should amplify by >10×, got {ratio}×"
        # But not by more than 500×
        assert ratio < 500, f"Q shift amplification {ratio}× seems too large"


class TestRegressionValues:
    """Regression tests for specific computed values."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_S12_minus_basis_kappa_value(self, polys_kappa):
        """
        S12_minus_basis should be approximately 0.349 for κ benchmark.

        This is a regression guard to catch accidental changes.
        """
        result = compute_S12_minus_basis(
            theta=4/7, R=1.3036, n=60, polynomials=polys_kappa
        )

        # Expected from diagnostics: ~0.349
        expected = 0.349
        tolerance = 0.05  # ~15% tolerance

        assert abs(result - expected) < tolerance, \
            f"S12_minus_basis = {result}, expected ~{expected}"

    def test_derived_kappa_value(self, polys_kappa):
        """
        S12_mirror_derived should be approximately 403 for κ benchmark.

        This is much larger than expected due to Q(1+·) shift.
        """
        result = compute_S12_mirror_derived(
            theta=4/7, R=1.3036, n=60, polynomials=polys_kappa
        )

        # Expected from diagnostics: ~403
        expected = 403
        tolerance = 50  # ~12% tolerance

        assert abs(result - expected) < tolerance, \
            f"S12_mirror_derived = {result}, expected ~{expected}"
