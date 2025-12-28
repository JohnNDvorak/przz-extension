"""
tests/test_phase45_out_of_sample_bounds.py
Phase 45.2: Out-of-Sample Stability Tests

These tests verify that the anchored constants (g_I1, g_I2) behave sanely
on polynomial sets beyond the κ/κ* benchmarks used to calibrate them.

STABILITY GATES:
- Q=1 microcases: correction should be 0.0–0.5%
- Real Q polynomials: correction should be ±0.5%
- Random Q polynomials: correction should be < 5%

Created: 2025-12-27 (Phase 45.2)
"""

import pytest
import numpy as np

from src.polynomials import Polynomial, load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.evaluator.correction_policy import (
    compute_g_baseline,
    compute_g_anchored,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


def create_q_one_polynomial() -> Polynomial:
    """Create Q(x) = 1 (constant polynomial)."""
    return Polynomial(coeffs=np.array([1.0]))


def create_random_q_polynomial(seed: int = 42, degree: int = 3) -> Polynomial:
    """Create a random Q polynomial satisfying Q(0)=1, Q(1)=-1."""
    rng = np.random.RandomState(seed)
    interior = rng.uniform(-0.5, 0.5, size=degree)
    c0 = 1.0
    adjust = -2.0 - np.sum(interior[:-1])
    interior[-1] = adjust
    coeffs = np.concatenate([[c0], interior])
    return Polynomial(coeffs=coeffs)


def compute_f_I1(R: float, theta: float, polynomials: dict, n_quad: int = 60) -> float:
    """Compute the I1 fraction f_I1 = I1(-R) / (I1(-R) + I2(-R))."""
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)
    S12_minus = I1_minus + I2_minus
    if abs(S12_minus) < 1e-15:
        return 0.5
    return I1_minus / S12_minus


def compute_correction_pct(polynomials: dict, R: float, theta: float = 4/7, K: int = 3) -> float:
    """Compute the correction percentage for a polynomial set."""
    f_I1 = compute_f_I1(R, theta, polynomials)
    g_total = compute_g_anchored(f_I1)
    g_baseline = compute_g_baseline(theta, K)
    return (g_total / g_baseline - 1) * 100


def create_q_modified_polynomials(base_polynomials: dict, Q: Polynomial) -> dict:
    """Create a polynomial dict with a modified Q polynomial."""
    return {
        "P1": base_polynomials["P1"],
        "P2": base_polynomials["P2"],
        "P3": base_polynomials["P3"],
        "Q": Q,
    }


@pytest.fixture
def kappa_polynomials():
    """Load κ benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_star_polynomials():
    """Load κ* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestQ1Microcases:
    """Test that Q=1 microcases stay near baseline."""

    def test_kappa_polys_q1_r_kappa(self, kappa_polynomials):
        """κ polynomials with Q=1 at R=1.3036."""
        Q_one = create_q_one_polynomial()
        modified = create_q_modified_polynomials(kappa_polynomials, Q_one)
        correction = compute_correction_pct(modified, R=1.3036)

        # Q=1 should give correction near 0% (baseline is designed for this)
        assert abs(correction) < 0.5, f"Correction {correction:.4f}% exceeds 0.5% bound"

    def test_kappa_star_polys_q1_r_kappa_star(self, kappa_star_polynomials):
        """κ* polynomials with Q=1 at R=1.1167."""
        Q_one = create_q_one_polynomial()
        modified = create_q_modified_polynomials(kappa_star_polynomials, Q_one)
        correction = compute_correction_pct(modified, R=1.1167)

        assert abs(correction) < 0.5, f"Correction {correction:.4f}% exceeds 0.5% bound"

    @pytest.mark.parametrize("R", [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    def test_q1_r_sweep(self, kappa_polynomials, R):
        """Test Q=1 across a range of R values."""
        Q_one = create_q_one_polynomial()
        modified = create_q_modified_polynomials(kappa_polynomials, Q_one)
        correction = compute_correction_pct(modified, R=R)

        assert abs(correction) < 0.5, f"Correction {correction:.4f}% exceeds 0.5% bound at R={R}"


class TestRealQPolynomials:
    """Test that real Q polynomials (κ and κ*) stay within bounds."""

    def test_kappa_benchmark(self, kappa_polynomials):
        """κ benchmark should have correction within ±0.5%."""
        correction = compute_correction_pct(kappa_polynomials, R=1.3036)

        # This is the benchmark we calibrated to
        assert abs(correction) < 0.5, f"Correction {correction:.4f}% exceeds 0.5% bound"

    def test_kappa_star_benchmark(self, kappa_star_polynomials):
        """κ* benchmark should have correction within ±0.5%."""
        correction = compute_correction_pct(kappa_star_polynomials, R=1.1167)

        # This is the benchmark we calibrated to
        assert abs(correction) < 0.5, f"Correction {correction:.4f}% exceeds 0.5% bound"


class TestCrossMatchedR:
    """Test polynomials at non-native R values."""

    def test_kappa_at_kappa_star_r(self, kappa_polynomials):
        """κ polynomials at κ* R value."""
        correction = compute_correction_pct(kappa_polynomials, R=1.1167)

        assert abs(correction) < 1.0, f"Correction {correction:.4f}% exceeds 1.0% bound"

    def test_kappa_star_at_kappa_r(self, kappa_star_polynomials):
        """κ* polynomials at κ R value."""
        correction = compute_correction_pct(kappa_star_polynomials, R=1.3036)

        assert abs(correction) < 1.0, f"Correction {correction:.4f}% exceeds 1.0% bound"


class TestRandomQPolynomials:
    """Test that random Q polynomials don't cause blow-up."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1001, 2022, 3033])
    def test_random_q_stability(self, kappa_polynomials, seed):
        """Random Q should not cause correction blow-up."""
        Q_random = create_random_q_polynomial(seed=seed, degree=3)
        modified = create_q_modified_polynomials(kappa_polynomials, Q_random)
        correction = compute_correction_pct(modified, R=1.3036)

        # Random Q should stay within 5% (generous bound)
        assert abs(correction) < 5.0, f"Correction {correction:.4f}% exceeds 5.0% bound for seed={seed}"

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_random_q_degree_sweep(self, kappa_polynomials, degree):
        """Test random Q at different polynomial degrees."""
        Q_random = create_random_q_polynomial(seed=42, degree=degree)
        modified = create_q_modified_polynomials(kappa_polynomials, Q_random)
        correction = compute_correction_pct(modified, R=1.3036)

        assert abs(correction) < 5.0, f"Correction {correction:.4f}% exceeds 5.0% bound for degree={degree}"


class TestCorrectionStatistics:
    """Test overall correction statistics."""

    def test_correction_mean_is_small(self, kappa_polynomials, kappa_star_polynomials):
        """Mean correction across test suite should be small."""
        Q_one = create_q_one_polynomial()

        corrections = []

        # Q=1 cases
        for R in [1.0, 1.1, 1.2, 1.3, 1.4]:
            modified = create_q_modified_polynomials(kappa_polynomials, Q_one)
            corrections.append(compute_correction_pct(modified, R=R))

        # Real Q cases
        corrections.append(compute_correction_pct(kappa_polynomials, R=1.3036))
        corrections.append(compute_correction_pct(kappa_star_polynomials, R=1.1167))

        mean_correction = np.mean(np.abs(corrections))

        # Mean should be under 0.5%
        assert mean_correction < 0.5, f"Mean correction {mean_correction:.4f}% exceeds 0.5%"

    def test_correction_std_is_small(self, kappa_polynomials, kappa_star_polynomials):
        """Standard deviation of corrections should be small."""
        Q_one = create_q_one_polynomial()

        corrections = []

        # Q=1 cases
        for R in [1.0, 1.1, 1.2, 1.3, 1.4]:
            modified = create_q_modified_polynomials(kappa_polynomials, Q_one)
            corrections.append(compute_correction_pct(modified, R=R))

        # Real Q cases
        corrections.append(compute_correction_pct(kappa_polynomials, R=1.3036))
        corrections.append(compute_correction_pct(kappa_star_polynomials, R=1.1167))

        std_correction = np.std(corrections)

        # Std should be under 0.5%
        assert std_correction < 0.5, f"Correction std {std_correction:.4f}% exceeds 0.5%"
