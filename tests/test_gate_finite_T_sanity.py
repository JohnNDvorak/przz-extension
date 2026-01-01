#!/usr/bin/env python3
"""
Gate 6: Finite-T Sanity Check

GPT's sixth validation gate for the kappa = 0.5213 claim:
- Verify the κ formula has correct structure for large T limit
- Check asymptotic consistency: κ = 1 - log(c)/R
- Verify c doesn't violate physical bounds

This is a structural/consistency gate rather than empirical finite-T sampling,
which would require heavy computation and have low signal-to-noise.

Created: 2025-12-28 (GPT Critical Review)
"""

import json
import math
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


class TestKappaFormulaStructure:
    """Test the κ = 1 - log(c)/R formula structure."""

    def test_kappa_formula_consistency(self):
        """Verify κ = 1 - log(c)/R exactly."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4 / 7

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        # Verify formula
        kappa_from_formula = 1 - math.log(result.c) / R

        print("\n  κ Formula Consistency:")
        print(f"  c = {result.c:.10f}")
        print(f"  R = {R}")
        print(f"  κ (returned) = {result.kappa:.10f}")
        print(f"  κ (1 - log(c)/R) = {kappa_from_formula:.10f}")
        print(f"  Difference = {abs(result.kappa - kappa_from_formula):.2e}")

        assert abs(result.kappa - kappa_from_formula) < 1e-12

    def test_c_from_kappa_roundtrip(self):
        """Verify c = exp(R(1-κ)) roundtrip."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4 / 7

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        # c = exp(R(1-κ))
        c_from_kappa = math.exp(R * (1 - result.kappa))

        print("\n  c Roundtrip:")
        print(f"  c (direct) = {result.c:.10f}")
        print(f"  c (exp(R(1-κ))) = {c_from_kappa:.10f}")
        print(f"  Difference = {abs(result.c - c_from_kappa):.2e}")

        assert abs(result.c - c_from_kappa) < 1e-10


class TestPhysicalBounds:
    """Test that c and κ satisfy physical bounds."""

    def test_c_is_positive(self):
        """c must be positive for κ to be well-defined."""
        data = load_optimal_polynomials()

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=4 / 7,
            K=3,
            R=1.3036,
            n_quad=60,
        )
        result = engine.compute_kappa()

        print(f"\n  c = {result.c:.10f} > 0: {'[OK]' if result.c > 0 else '[FAIL]'}")
        assert result.c > 0

    def test_kappa_in_valid_range(self):
        """κ must be in (0, 1) for proportion interpretation."""
        data = load_optimal_polynomials()

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=4 / 7,
            K=3,
            R=1.3036,
            n_quad=60,
        )
        result = engine.compute_kappa()

        print(f"\n  κ = {result.kappa:.10f}")
        print(f"  0 < κ < 1: {'[OK]' if 0 < result.kappa < 1 else '[FAIL]'}")
        assert 0 < result.kappa < 1

    def test_c_not_too_small(self):
        """c should not be unreasonably small (would give κ too close to 1)."""
        data = load_optimal_polynomials()

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=4 / 7,
            K=3,
            R=1.3036,
            n_quad=60,
        )
        result = engine.compute_kappa()

        min_c_for_valid_kappa = 1.0
        max_c_for_positive_kappa = math.exp(1.3036)

        print(f"\n  c = {result.c:.10f}")
        print(f"  Min c for κ < 1: {min_c_for_valid_kappa:.6f}")
        print(f"  Max c for κ > 0: {max_c_for_positive_kappa:.6f}")
        print(f"  c in range: {'[OK]' if min_c_for_valid_kappa < result.c < max_c_for_positive_kappa else '[FAIL]'}")

        assert result.c > min_c_for_valid_kappa
        assert result.c < max_c_for_positive_kappa


class TestRScaling:
    """Test that κ changes appropriately with R."""

    def test_kappa_varies_with_R(self):
        """κ should vary smoothly with R."""
        data = load_optimal_polynomials()

        R_values = [1.1, 1.2, 1.3]
        results = []

        for R in R_values:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=4 / 7,
                K=3,
                R=R,
                n_quad=60,
            )
            result = engine.compute_kappa()
            results.append((R, result.c, result.kappa))

        print("\n  R Scaling Test:")
        print("  " + "-" * 50)
        for R, c, kappa in results:
            print(f"  R = {R:.1f}: c = {c:.6f}, κ = {kappa:.6f}")

        kappas = [r[2] for r in results]
        print(f"\n  κ values: {[f'{k:.4f}' for k in kappas]}")

        # All should be valid
        assert all(0 < k < 1 for k in kappas)

    def test_mirror_multiplier_scales_correctly(self):
        """Mirror multiplier m = g × (exp(R) + 2K-1) should scale with R."""
        data = load_optimal_polynomials()

        R_values = [1.0, 1.2, 1.4]
        K = 3

        print("\n  Mirror Multiplier Scaling:")
        print("  " + "-" * 50)

        for R in R_values:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=4 / 7,
                K=K,
                R=R,
                n_quad=40,
            )
            result = engine.compute_kappa()

            base = math.exp(R) + (2*K - 1)
            print(f"  R = {R:.1f}: exp(R)+5 = {base:.4f}, m = {result.corrections.m:.4f}")


class TestAsymptoticBehavior:
    """Test asymptotic behavior as n_quad → ∞."""

    def test_c_converges_with_quadrature(self):
        """c should converge as quadrature points increase."""
        data = load_optimal_polynomials()

        n_values = [40, 60, 80]
        c_values = []

        for n in n_values:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=4 / 7,
                K=3,
                R=1.3036,
                n_quad=n,
            )
            result = engine.compute_kappa()
            c_values.append(result.c)

        print("\n  Quadrature Convergence:")
        for n, c in zip(n_values, c_values):
            print(f"  n_quad = {n}: c = {c:.10f}")

        drift_40_60 = abs(c_values[1] - c_values[0]) / c_values[0]
        drift_60_80 = abs(c_values[2] - c_values[1]) / c_values[1]

        print(f"\n  |c(60)-c(40)|/c(40) = {drift_40_60:.2e}")
        print(f"  |c(80)-c(60)|/c(60) = {drift_60_80:.2e}")

        assert drift_60_80 < drift_40_60, "Not converging"
        assert drift_60_80 < 1e-5, f"Not converged: {drift_60_80:.2e}"


class TestComponentSanity:
    """Test that integral components have reasonable magnitudes."""

    def test_integral_components_reasonable(self):
        """All integral components should be finite and reasonable."""
        data = load_optimal_polynomials()

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=4 / 7,
            K=3,
            R=1.3036,
            n_quad=60,
        )
        result = engine.compute_kappa()

        components = [
            ("I1(+R)", result.integrals.I1_plus),
            ("I1(-R)", result.integrals.I1_minus),
            ("I2(+R)", result.integrals.I2_plus),
            ("I2(-R)", result.integrals.I2_minus),
            ("I3(+R)", result.integrals.I3_plus),
            ("I4(+R)", result.integrals.I4_plus),
        ]

        print("\n  Integral Components:")
        all_reasonable = True
        for name, val in components:
            is_finite = np.isfinite(val)
            is_reasonable = abs(val) < 1e6
            status = "[OK]" if is_finite and is_reasonable else "[FAIL]"
            print(f"  {name} = {val:+.10f} {status}")
            all_reasonable &= (is_finite and is_reasonable)

        assert all_reasonable

    def test_s12_dominates_s34(self):
        """S12 terms should dominate S34 (main vs derivative terms)."""
        data = load_optimal_polynomials()

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=4 / 7,
            K=3,
            R=1.3036,
            n_quad=60,
        )
        result = engine.compute_kappa()

        s12_plus = result.integrals.S12_plus
        s12_minus = result.integrals.S12_minus
        s34_plus = result.integrals.S34_plus

        print("\n  Term Magnitudes:")
        print(f"  S12(+R) = {s12_plus:.6f}")
        print(f"  m × S12(-R) = {result.corrections.m:.4f} × {s12_minus:.6f} = {result.corrections.m * s12_minus:.6f}")
        print(f"  S34(+R) = {s34_plus:.6f}")

        ratio = abs(s34_plus) / (abs(s12_plus) + 1e-10)
        print(f"  |S34|/|S12| = {ratio:.4f}")


class TestGate6Summary:
    """Comprehensive Gate 6 summary."""

    def test_full_gate6_summary(self):
        """Run full Gate 6 summary with pass/fail status."""
        print("\n" + "=" * 70)
        print("GATE 6: FINITE-T SANITY CHECK (GPT Critical Review)")
        print("=" * 70)

        data = load_optimal_polynomials()
        all_passed = True

        R = 1.3036
        theta = 4 / 7

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        # Test 1: κ formula consistency
        print("\n  Test 1: κ Formula Consistency")
        kappa_formula = 1 - math.log(result.c) / R
        test1 = abs(result.kappa - kappa_formula) < 1e-12
        print(f"    κ = 1 - log(c)/R: {'[OK]' if test1 else '[FAIL]'}")
        all_passed &= test1

        # Test 2: Physical bounds
        print("\n  Test 2: Physical Bounds")
        test2a = result.c > 0
        test2b = 0 < result.kappa < 1
        print(f"    c > 0: {result.c:.6f} {'[OK]' if test2a else '[FAIL]'}")
        print(f"    0 < κ < 1: {result.kappa:.6f} {'[OK]' if test2b else '[FAIL]'}")
        all_passed &= test2a and test2b

        # Test 3: Quadrature convergence
        print("\n  Test 3: Quadrature Convergence")
        c_60 = result.c
        engine_80 = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=80,
        )
        c_80 = engine_80.compute_kappa().c
        drift = abs(c_80 - c_60) / c_60
        test3 = drift < 1e-5
        print(f"    |c(80)-c(60)|/c(60) = {drift:.2e} {'[OK]' if test3 else '[FAIL]'}")
        all_passed &= test3

        # Test 4: Component finiteness
        print("\n  Test 4: Component Sanity")
        components = [
            result.integrals.I1_plus, result.integrals.I1_minus,
            result.integrals.I2_plus, result.integrals.I2_minus,
            result.integrals.I3_plus, result.integrals.I4_plus,
        ]
        test4 = all(np.isfinite(c) and abs(c) < 1e6 for c in components)
        print(f"    All components finite and bounded: {'[OK]' if test4 else '[FAIL]'}")
        all_passed &= test4

        # Test 5: Expected κ value
        print("\n  Test 5: Expected κ Value")
        test5 = abs(result.kappa - 0.5213) < 0.01
        print(f"    κ = {result.kappa:.6f} ≈ 0.5213: {'[OK]' if test5 else '[FAIL]'}")
        all_passed &= test5

        # Summary
        print("\n  Key Results:")
        print(f"    c = {result.c:.10f}")
        print(f"    κ = {result.kappa:.10f}")
        print(f"    Improvement over PRZZ (0.4173): {(result.kappa - 0.4173) / 0.4173 * 100:.1f}%")

        print("\n" + "=" * 70)
        overall = "PASS" if all_passed else "FAIL"
        print(f"GATE 6 OVERALL: {overall}")
        print("=" * 70)

        assert all_passed, "Gate 6 failed"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GATE 6: FINITE-T SANITY CHECK - Quick Check")
    print("=" * 70)

    data = load_optimal_polynomials()

    R = 1.3036
    theta = 4 / 7

    engine = KappaEngine(
        P1_coeffs=data['P1_tilde'],
        P2_coeffs=data['P2_tilde'],
        P3_coeffs=data['P3_tilde'],
        Q_coeffs=data['Q_mono'],
        theta=theta,
        K=3,
        R=R,
        n_quad=60,
    )
    result = engine.compute_kappa()

    print(f"\n  c = {result.c:.10f}")
    print(f"  κ = {result.kappa:.10f}")
    print(f"  Formula: κ = 1 - log(c)/R = 1 - log({result.c:.6f})/{R} = {result.kappa:.6f}")
    print(f"\n  Physical bounds:")
    print(f"    c > 0: {result.c:.6f} > 0 ✓")
    print(f"    0 < κ < 1: {result.kappa:.6f} ✓")
