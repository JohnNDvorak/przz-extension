#!/usr/bin/env python3
"""
Gate 48.3: θ-Robustness

GPT's critical requirement for adversarial verification:
- Show κ > 0.5 for θ = 4/7 - ε with nontrivial ε
- Kills "Silent Killer B" - result at exactly θ = 4/7 only

Created: 2025-12-28 (Phase 48 - Adversarial Verification)
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


class TestThetaRobustness:
    """Test κ > 0.5 robustness under θ perturbation."""

    def test_kappa_above_05_at_exact_theta(self):
        """Verify κ > 0.5 at exact θ = 4/7."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7
        K = 3

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=K,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        print(f"\n  θ = 4/7 = {theta:.10f}")
        print(f"  κ = {result.kappa:.10f}")

        assert result.kappa > 0.5, f"κ ≤ 0.5 at exact θ: {result.kappa}"

    def test_kappa_above_05_at_theta_minus_1e3(self):
        """Verify κ > 0.5 at θ = 4/7 - 1e-3."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7 - 1e-3
        K = 3

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=K,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        print(f"\n  θ = 4/7 - 1e-3 = {theta:.10f}")
        print(f"  κ = {result.kappa:.10f}")

        assert result.kappa > 0.5, f"κ ≤ 0.5 at θ = 4/7 - 1e-3: {result.kappa}"

    def test_kappa_above_05_at_theta_minus_1e2(self):
        """Verify κ > 0.5 at θ = 4/7 - 1e-2 (largest perturbation)."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7 - 1e-2
        K = 3

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=K,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        print(f"\n  θ = 4/7 - 1e-2 = {theta:.10f}")
        print(f"  κ = {result.kappa:.10f}")

        assert result.kappa > 0.5, f"κ ≤ 0.5 at θ = 4/7 - 1e-2: {result.kappa}"


class TestThetaSweep:
    """Full θ sweep test."""

    def test_theta_sweep_all_above_05(self):
        """Sweep θ = 4/7 - ε for multiple ε values."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta_exact = 4/7
        K = 3

        epsilon_values = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        print(f"\n  θ-Robustness Sweep:")
        print(f"  {'ε':>10} | {'θ':>12} | {'κ':>12} | Status")
        print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-------")

        all_pass = True
        for eps in epsilon_values:
            theta = theta_exact - eps

            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=K,
                R=R,
                n_quad=60,
            )
            result = engine.compute_kappa()

            passed = result.kappa > 0.5
            all_pass &= passed
            status = "PASS" if passed else "FAIL"

            eps_str = f"{eps:.0e}" if eps > 0 else "0"
            print(f"  {eps_str:>10} | {theta:.10f} | {result.kappa:.10f} | {status}")

        assert all_pass, "Some ε values failed κ > 0.5"


class TestSlopeAnalysis:
    """Analyze dκ/dθ slope."""

    def test_slope_is_finite(self):
        """Verify dκ/dθ is finite (not singular at θ = 4/7)."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta_exact = 4/7
        K = 3

        # Compute at θ and θ - δ
        delta = 1e-3

        kappa_values = []
        for theta in [theta_exact, theta_exact - delta]:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=K,
                R=R,
                n_quad=60,
            )
            result = engine.compute_kappa()
            kappa_values.append(result.kappa)

        slope = (kappa_values[1] - kappa_values[0]) / (-delta)

        print(f"\n  Slope Analysis:")
        print(f"    κ(θ = 4/7) = {kappa_values[0]:.10f}")
        print(f"    κ(θ = 4/7 - 1e-3) = {kappa_values[1]:.10f}")
        print(f"    dκ/dθ ≈ {slope:.6f}")

        # Slope should be finite and positive (κ decreases as θ decreases)
        assert np.isfinite(slope), f"Slope is not finite: {slope}"
        assert slope > 0, f"Slope should be positive (κ decreases with θ): {slope}"


class TestGate483Summary:
    """Comprehensive Gate 48.3 summary."""

    def test_full_gate483_summary(self):
        """Run full Gate 48.3 summary."""
        print("\n" + "=" * 70)
        print("GATE 48.3: θ-ROBUSTNESS (GPT Critical Review)")
        print("=" * 70)

        data = load_optimal_polynomials()

        R = 1.3036
        theta_exact = 4/7
        K = 3

        all_passed = True

        # Sweep all ε values
        epsilon_values = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        print(f"\n  θ = 4/7 - ε sweep:")
        print(f"  {'ε':>10} | {'θ':>12} | {'κ':>12} | Status")
        print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-------")

        for eps in epsilon_values:
            theta = theta_exact - eps

            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=K,
                R=R,
                n_quad=60,
            )
            result = engine.compute_kappa()

            passed = result.kappa > 0.5
            all_passed &= passed
            status = "PASS" if passed else "FAIL"

            eps_str = f"{eps:.0e}" if eps > 0 else "0"
            print(f"  {eps_str:>10} | {theta:.10f} | {result.kappa:.10f} | {status}")

        # Compute margin at largest ε
        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta_exact - 1e-2,
            K=K,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()
        margin = result.kappa - 0.5

        print(f"\n  Margin at ε = 1e-2: κ - 0.5 = {margin:.6f}")
        print(f"  Conclusion: κ > 0.5 is {'ROBUST' if all_passed else 'NOT ROBUST'} over θ range")

        print("\n" + "=" * 70)
        overall = "PASS" if all_passed else "FAIL"
        print(f"GATE 48.3 OVERALL: {overall}")
        print("=" * 70)

        assert all_passed, "Gate 48.3 failed"


if __name__ == "__main__":
    # Quick check
    data = load_optimal_polynomials()

    print("\n" + "=" * 70)
    print("GATE 48.3: θ-ROBUSTNESS - Quick Check")
    print("=" * 70)

    R = 1.3036
    theta_exact = 4/7
    K = 3

    for eps in [0.0, 1e-3, 1e-2]:
        theta = theta_exact - eps

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=K,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        eps_str = f"{eps:.0e}" if eps > 0 else "0"
        status = "✓" if result.kappa > 0.5 else "✗"
        print(f"  ε = {eps_str}: θ = {theta:.6f}, κ = {result.kappa:.6f} {status}")
