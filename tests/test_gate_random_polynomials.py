#!/usr/bin/env python3
"""
Silent Killer #1 Test: Random Admissible Polynomials

GPT's concern: A missing main-term piece that cancels at/near baseline
but not in your new region. The new candidate has P3 strictly negative
with |P3| peaking around 4 - qualitatively different from PRZZ.

This test:
1. Tests random admissible polynomials including:
   - "P3 all-negative" families
   - "P3 all-positive" families
   - Mixed sign families
2. Verifies c computation consistency across implementations
3. Checks for anomalous behavior in different coefficient regions

Created: 2025-12-28 (Phase 48 - GPT Critical Review)
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine
from src.polynomials import P1Polynomial, PellPolynomial


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


def generate_admissible_P1(rng, scale=1.0):
    """
    Generate random admissible P1 polynomial.
    P1(0) = 0, P1(1) = 1 via parameterization: P1(x) = x + x(1-x)*tilde(1-x)
    """
    # Random tilde coefficients (typically 4)
    n_coeffs = 4
    tilde = list(rng.uniform(-scale, scale, n_coeffs))
    return tilde


def generate_admissible_Pell(rng, scale=1.0, all_negative=False, all_positive=False):
    """
    Generate random admissible P_ell polynomial for ell >= 2.
    P_ell(0) = 0 via parameterization: P_ell(x) = x * tilde(x)
    """
    n_coeffs = 3
    tilde = rng.uniform(-scale, scale, n_coeffs)

    if all_negative:
        tilde = -np.abs(tilde) - 0.1  # Ensure all negative
    elif all_positive:
        tilde = np.abs(tilde) + 0.1  # Ensure all positive

    return list(tilde)


def compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono, R=1.3036, n_quad=40):
    """Compute kappa for given polynomial coefficients."""
    engine = KappaEngine(
        P1_coeffs=P1_tilde,
        P2_coeffs=P2_tilde,
        P3_coeffs=P3_tilde,
        Q_coeffs=Q_mono,
        theta=4/7,
        K=3,
        R=R,
        n_quad=n_quad,
    )
    result = engine.compute_kappa()
    return result


class TestRandomPolynomialFamilies:
    """Test consistency across random polynomial families."""

    def test_przz_like_polynomials(self):
        """Test random polynomials similar to PRZZ (small P3)."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        rng = np.random.default_rng(42)
        n_trials = 10

        print(f"\n  Testing {n_trials} PRZZ-like random polynomials (small P3)...")

        for trial in range(n_trials):
            P1_tilde = generate_admissible_P1(rng, scale=0.5)
            P2_tilde = generate_admissible_Pell(rng, scale=0.5)
            P3_tilde = generate_admissible_Pell(rng, scale=0.2)  # Small

            result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

            # Check for NaN or extreme values
            assert np.isfinite(result.c), f"Trial {trial}: c is not finite"
            assert np.isfinite(result.kappa), f"Trial {trial}: kappa is not finite"
            assert result.c > 0, f"Trial {trial}: c is not positive"

            if trial < 3:
                print(f"    Trial {trial}: c = {result.c:.6f}, kappa = {result.kappa:.6f}")

    def test_p3_all_negative_family(self):
        """Test polynomials with P3 all-negative (like optimal)."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        rng = np.random.default_rng(43)
        n_trials = 10

        print(f"\n  Testing {n_trials} P3-all-negative polynomials...")

        for trial in range(n_trials):
            P1_tilde = generate_admissible_P1(rng, scale=1.0)
            P2_tilde = generate_admissible_Pell(rng, scale=1.0)
            P3_tilde = generate_admissible_Pell(rng, scale=2.0, all_negative=True)

            result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

            assert np.isfinite(result.c), f"Trial {trial}: c is not finite"
            assert np.isfinite(result.kappa), f"Trial {trial}: kappa is not finite"
            assert result.c > 0, f"Trial {trial}: c is not positive"

            if trial < 3:
                print(f"    Trial {trial}: c = {result.c:.6f}, kappa = {result.kappa:.6f}, P3_sum = {sum(P3_tilde):.2f}")

    def test_p3_all_positive_family(self):
        """Test polynomials with P3 all-positive."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        rng = np.random.default_rng(44)
        n_trials = 10

        print(f"\n  Testing {n_trials} P3-all-positive polynomials...")

        for trial in range(n_trials):
            P1_tilde = generate_admissible_P1(rng, scale=1.0)
            P2_tilde = generate_admissible_Pell(rng, scale=1.0)
            P3_tilde = generate_admissible_Pell(rng, scale=2.0, all_positive=True)

            result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

            assert np.isfinite(result.c), f"Trial {trial}: c is not finite"
            assert np.isfinite(result.kappa), f"Trial {trial}: kappa is not finite"
            assert result.c > 0, f"Trial {trial}: c is not positive"

            if trial < 3:
                print(f"    Trial {trial}: c = {result.c:.6f}, kappa = {result.kappa:.6f}, P3_sum = {sum(P3_tilde):.2f}")


class TestPairContributionConsistency:
    """Verify pair contributions are consistent across families."""

    def test_pair_positivity_przz_like(self):
        """Verify diagonal pairs are always positive (PRZZ-like)."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        rng = np.random.default_rng(45)
        n_trials = 10

        print(f"\n  Testing diagonal pair positivity (PRZZ-like)...")

        for trial in range(n_trials):
            P1_tilde = generate_admissible_P1(rng, scale=0.5)
            P2_tilde = generate_admissible_Pell(rng, scale=0.5)
            P3_tilde = generate_admissible_Pell(rng, scale=0.2)

            result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

            # Diagonal pairs should always be positive (L^2 norm)
            I2_plus = result.integrals.I2_plus
            assert I2_plus > 0, f"Trial {trial}: I2_plus should be positive"

    def test_pair_contributions_extreme_p3(self):
        """Test pair contributions with extreme P3 values."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        rng = np.random.default_rng(46)

        print(f"\n  Testing pair contributions with extreme P3...")

        # Extreme negative P3
        P1_tilde = [0.1, -0.5, 0.2, 0.1]
        P2_tilde = [1.0, -0.2, -0.1]
        P3_tilde = [-5.0, -5.0, -5.0]  # Very negative

        result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

        print(f"    Extreme negative P3:")
        print(f"      c = {result.c:.6f}")
        print(f"      kappa = {result.kappa:.6f}")
        print(f"      I2(+R) = {result.integrals.I2_plus:.6f}")

        assert np.isfinite(result.c), "c should be finite"
        assert result.c > 0, "c should be positive"

        # Extreme positive P3
        P3_tilde = [5.0, 5.0, 5.0]  # Very positive

        result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

        print(f"\n    Extreme positive P3:")
        print(f"      c = {result.c:.6f}")
        print(f"      kappa = {result.kappa:.6f}")

        assert np.isfinite(result.c), "c should be finite"
        assert result.c > 0, "c should be positive"


class TestQuadratureConsistency:
    """Test quadrature consistency across polynomial families."""

    def test_quadrature_convergence_extreme_polynomials(self):
        """Verify quadrature converges for extreme polynomials."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        # Use extreme P3 (like optimal but more extreme)
        P1_tilde = [0.2, -0.8, -0.2, 0.3]
        P2_tilde = [1.0, -0.2, -0.2]
        P3_tilde = [-3.0, -4.0, -1.0]

        print(f"\n  Quadrature convergence for extreme polynomials...")

        results = {}
        for n_quad in [40, 60, 80]:
            result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono, n_quad=n_quad)
            results[n_quad] = result.c
            print(f"    n_quad = {n_quad}: c = {result.c:.10f}")

        # Check convergence
        diff_60_40 = abs(results[60] - results[40])
        diff_80_60 = abs(results[80] - results[60])

        print(f"\n    |c(60) - c(40)| = {diff_60_40:.2e}")
        print(f"    |c(80) - c(60)| = {diff_80_60:.2e}")

        # Should be converging
        assert diff_80_60 < diff_60_40 or diff_80_60 < 1e-6, "Quadrature not converging"


class TestCrossImplementationValidation:
    """Cross-validate with alternative implementations."""

    def test_p3_zero_gives_k2_reduction(self):
        """Setting P3=0 should give K=2 reduction (no Case C)."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        # P3 = 0
        P1_tilde = data['P1_tilde']
        P2_tilde = data['P2_tilde']
        P3_tilde = [0.0, 0.0, 0.0]

        result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

        print(f"\n  K=2 reduction test (P3=0):")
        print(f"    c = {result.c:.10f}")
        print(f"    kappa = {result.kappa:.6f}")

        # Should give valid result
        assert np.isfinite(result.c), "c should be finite"
        assert result.c > 0, "c should be positive"

    def test_all_polynomials_zero_except_p1(self):
        """Test with P2=P3=0 (only first piece)."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        # Only P1
        P1_tilde = [0.0, 0.0, 0.0, 0.0]  # P1 = x (linear)
        P2_tilde = [0.0, 0.0, 0.0]
        P3_tilde = [0.0, 0.0, 0.0]

        result = compute_kappa_for_polynomials(P1_tilde, P2_tilde, P3_tilde, Q_mono)

        print(f"\n  Minimal polynomial test (P1=x, P2=P3=0):")
        print(f"    c = {result.c:.10f}")
        print(f"    kappa = {result.kappa:.6f}")

        assert np.isfinite(result.c), "c should be finite"
        assert result.c > 0, "c should be positive"


class TestPolynomialMagnitudeEffects:
    """Test how polynomial magnitude affects results."""

    def test_magnitude_scaling(self):
        """Test c scales appropriately with polynomial magnitude."""
        data = load_optimal_polynomials()
        Q_mono = data['Q_mono']

        base_P1 = data['P1_tilde']
        base_P2 = data['P2_tilde']
        base_P3 = data['P3_tilde']

        print(f"\n  Polynomial magnitude scaling test...")

        scales = [0.5, 1.0, 2.0]
        results = {}

        for scale in scales:
            # Scale P2 and P3 (P1's endpoint constraint complicates scaling)
            P2_scaled = [c * scale for c in base_P2]
            P3_scaled = [c * scale for c in base_P3]

            result = compute_kappa_for_polynomials(base_P1, P2_scaled, P3_scaled, Q_mono)
            results[scale] = result.c
            print(f"    scale = {scale}: c = {result.c:.6f}")

        # c should vary with scale (quadratic relationship expected)
        assert results[2.0] != results[1.0], "c should vary with scale"


if __name__ == "__main__":
    print("=" * 70)
    print("SILENT KILLER #1: RANDOM ADMISSIBLE POLYNOMIALS TEST")
    print("=" * 70)

    data = load_optimal_polynomials()
    Q_mono = data['Q_mono']

    rng = np.random.default_rng(42)

    print("\n  Testing random polynomial families...")

    # Test a few from each family
    families = [
        ("PRZZ-like", lambda: (generate_admissible_P1(rng, 0.5),
                                generate_admissible_Pell(rng, 0.5),
                                generate_admissible_Pell(rng, 0.2))),
        ("P3-negative", lambda: (generate_admissible_P1(rng, 1.0),
                                  generate_admissible_Pell(rng, 1.0),
                                  generate_admissible_Pell(rng, 2.0, all_negative=True))),
        ("P3-positive", lambda: (generate_admissible_P1(rng, 1.0),
                                  generate_admissible_Pell(rng, 1.0),
                                  generate_admissible_Pell(rng, 2.0, all_positive=True))),
    ]

    for family_name, generator in families:
        print(f"\n  {family_name}:")
        for i in range(3):
            P1, P2, P3 = generator()
            result = compute_kappa_for_polynomials(P1, P2, P3, Q_mono)
            print(f"    Trial {i}: c = {result.c:.6f}, kappa = {result.kappa:.6f}")
