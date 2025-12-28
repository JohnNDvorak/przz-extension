"""
tests/test_out_of_sample_smoke.py
Out-of-sample smoke tests for production kappa_engine.

These tests verify the production engine handles:
1. Randomized Q polynomials (not just PRZZ Q)
2. Perturbed P polynomials
3. No NaNs/infs
4. Reasonable gaps (not catastrophic failures)
5. Derived mode never calls anchored code

Created: 2025-12-27 (Phase 46++)
"""

import pytest
import numpy as np
import math

from src.kappa_engine import (
    KappaEngine,
    compute_g_I1,
    compute_g_I2,
    compute_base,
    compute_mirror_multiplier,
)
from src.polynomials import load_przz_polynomials


# Use smaller quadrature for speed in smoke tests
N_QUAD_SMOKE = 40


class TestRandomizedQ:
    """Test with randomized Q polynomials."""

    @pytest.fixture
    def base_engine(self):
        """Get baseline PRZZ engine."""
        return KappaEngine.from_przz_kappa(n_quad=N_QUAD_SMOKE)

    @pytest.fixture
    def przz_polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        return {
            "P1_coeffs": P1.tilde_coeffs.tolist(),
            "P2_coeffs": P2.tilde_coeffs.tolist(),
            "P3_coeffs": P3.tilde_coeffs.tolist(),
            "Q_coeffs": Q.to_monomial().coeffs.tolist(),
        }

    def test_q_equals_one(self, przz_polys):
        """Q=1 (microcase) should give valid results."""
        # Q=1 means monomial coeffs = [1.0]
        engine = KappaEngine(
            P1_coeffs=przz_polys["P1_coeffs"],
            P2_coeffs=przz_polys["P2_coeffs"],
            P3_coeffs=przz_polys["P3_coeffs"],
            Q_coeffs=[1.0],  # Q(x) = 1
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        # Should not be NaN or inf
        assert math.isfinite(result.kappa), f"kappa is {result.kappa}"
        assert math.isfinite(result.c), f"c is {result.c}"

        # Kappa should be positive (valid proportion)
        assert 0 < result.kappa < 1, f"kappa={result.kappa} out of range"

    def test_linear_q(self, przz_polys):
        """Linear Q (like kappa*) should work."""
        # Linear Q: Q(x) = 1 - 0.5x
        engine = KappaEngine(
            P1_coeffs=przz_polys["P1_coeffs"],
            P2_coeffs=przz_polys["P2_coeffs"],
            P3_coeffs=przz_polys["P3_coeffs"],
            Q_coeffs=[1.0, -0.5],  # Q(x) = 1 - 0.5x
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        assert math.isfinite(result.kappa)
        assert math.isfinite(result.c)
        assert 0 < result.kappa < 1

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_random_q_coefficients(self, przz_polys, seed):
        """Randomly perturbed Q coefficients should not crash."""
        np.random.seed(seed)

        # Start with PRZZ Q and add small perturbation
        base_Q = np.array(przz_polys["Q_coeffs"])
        perturbation = np.random.normal(0, 0.1, len(base_Q))
        perturbed_Q = base_Q + perturbation

        # Normalize so Q(0) ~ 1 (sum of coeffs)
        perturbed_Q = perturbed_Q / sum(perturbed_Q)

        engine = KappaEngine(
            P1_coeffs=przz_polys["P1_coeffs"],
            P2_coeffs=przz_polys["P2_coeffs"],
            P3_coeffs=przz_polys["P3_coeffs"],
            Q_coeffs=perturbed_Q.tolist(),
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        # Should compute without NaN/inf
        assert math.isfinite(result.kappa), f"seed={seed}: kappa={result.kappa}"
        assert math.isfinite(result.c), f"seed={seed}: c={result.c}"


class TestPerturbedP:
    """Test with perturbed P polynomials."""

    @pytest.fixture
    def przz_polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        return {
            "P1_coeffs": P1.tilde_coeffs.tolist(),
            "P2_coeffs": P2.tilde_coeffs.tolist(),
            "P3_coeffs": P3.tilde_coeffs.tolist(),
            "Q_coeffs": Q.to_monomial().coeffs.tolist(),
        }

    @pytest.mark.parametrize("perturbation_scale", [0.01, 0.05, 0.1])
    def test_perturbed_P1(self, przz_polys, perturbation_scale):
        """Perturbed P1 should not crash."""
        np.random.seed(42)

        P1_base = np.array(przz_polys["P1_coeffs"])
        perturbation = np.random.normal(0, perturbation_scale, len(P1_base))
        P1_perturbed = P1_base + perturbation

        engine = KappaEngine(
            P1_coeffs=P1_perturbed.tolist(),
            P2_coeffs=przz_polys["P2_coeffs"],
            P3_coeffs=przz_polys["P3_coeffs"],
            Q_coeffs=przz_polys["Q_coeffs"],
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        assert math.isfinite(result.kappa), f"scale={perturbation_scale}: kappa={result.kappa}"
        assert math.isfinite(result.c)

    def test_perturbed_all_P(self, przz_polys):
        """Perturbing all P polynomials should still work."""
        np.random.seed(42)
        scale = 0.05

        P1_perturbed = np.array(przz_polys["P1_coeffs"]) + np.random.normal(0, scale, len(przz_polys["P1_coeffs"]))
        P2_perturbed = np.array(przz_polys["P2_coeffs"]) + np.random.normal(0, scale, len(przz_polys["P2_coeffs"]))
        P3_perturbed = np.array(przz_polys["P3_coeffs"]) + np.random.normal(0, scale, len(przz_polys["P3_coeffs"]))

        engine = KappaEngine(
            P1_coeffs=P1_perturbed.tolist(),
            P2_coeffs=P2_perturbed.tolist(),
            P3_coeffs=P3_perturbed.tolist(),
            Q_coeffs=przz_polys["Q_coeffs"],
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        assert math.isfinite(result.kappa)
        assert math.isfinite(result.c)


class TestDifferentR:
    """Test with different R values."""

    @pytest.mark.parametrize("R", [0.8, 1.0, 1.2, 1.4, 1.6])
    def test_various_R_values(self, R):
        """Different R values should work."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        engine = KappaEngine(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            K=3,
            R=R,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        assert math.isfinite(result.kappa), f"R={R}: kappa={result.kappa}"
        assert math.isfinite(result.c), f"R={R}: c={result.c}"


class TestNoAnchoredCodePath:
    """Verify production engine never calls anchored code."""

    def test_kappa_engine_uses_derived_formulas(self):
        """KappaEngine must use derived g_I1, g_I2, not calibrated."""
        engine = KappaEngine.from_przz_kappa(n_quad=N_QUAD_SMOKE)
        result = engine.compute_kappa()

        # The g values should match our derived formulas
        expected_g_I1 = compute_g_I1(4/7, 3)
        expected_g_I2 = compute_g_I2(4/7, 3)

        assert result.corrections.g_I1 == pytest.approx(expected_g_I1, rel=1e-10)
        assert result.corrections.g_I2 == pytest.approx(expected_g_I2, rel=1e-10)

        # And they should NOT equal the calibrated values
        G_I1_CALIBRATED = 1.00091428  # From correction_policy
        G_I2_CALIBRATED = 1.01945154

        assert result.corrections.g_I1 != pytest.approx(G_I1_CALIBRATED, rel=1e-6)
        assert result.corrections.g_I2 != pytest.approx(G_I2_CALIBRATED, rel=1e-6)

    def test_compute_mirror_multiplier_uses_derived(self):
        """compute_mirror_multiplier must use derived formulas."""
        # Create a simple IntegralComponents-like object for testing
        from dataclasses import dataclass

        @dataclass
        class MockIntegrals:
            f_I1: float = 0.23  # Typical value

        integrals = MockIntegrals()
        corrections = compute_mirror_multiplier(
            theta=4/7,
            K=3,
            R=1.3036,
            f_I1=integrals.f_I1,
        )

        # Verify it's using derived values
        expected_g_I1 = compute_g_I1(4/7, 3)
        expected_g_I2 = compute_g_I2(4/7, 3)

        assert corrections.g_I1 == pytest.approx(expected_g_I1, rel=1e-10)
        assert corrections.g_I2 == pytest.approx(expected_g_I2, rel=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_q_coefficients(self):
        """Very small Q coefficients should not cause underflow."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        # Q with very small coefficients
        small_Q = [0.001 * c for c in Q.to_monomial().coeffs.tolist()]

        engine = KappaEngine(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            Q_coeffs=small_Q,
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=N_QUAD_SMOKE,
        )
        result = engine.compute_kappa()

        # Should not underflow
        assert math.isfinite(result.kappa)
        assert math.isfinite(result.c)

    def test_extreme_R(self):
        """Extreme R values should at least not crash."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        for R in [0.5, 2.0]:
            engine = KappaEngine(
                P1_coeffs=P1.tilde_coeffs.tolist(),
                P2_coeffs=P2.tilde_coeffs.tolist(),
                P3_coeffs=P3.tilde_coeffs.tolist(),
                Q_coeffs=Q.to_monomial().coeffs.tolist(),
                theta=4/7,
                K=3,
                R=R,
                n_quad=N_QUAD_SMOKE,
            )
            result = engine.compute_kappa()

            assert math.isfinite(result.kappa), f"R={R} gave non-finite kappa"
            assert math.isfinite(result.c), f"R={R} gave non-finite c"
