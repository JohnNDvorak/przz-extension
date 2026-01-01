#!/usr/bin/env python3
"""
GPT Admissibility Concerns - Comprehensive Verification

Addresses GPT's critical questions:
1. P_ell(1) = 0 constraint - INVESTIGATED, NOT REQUIRED BY PRZZ
2. Factored vs unfactored c formula - VERIFIED EQUIVALENT
3. Actual PRZZ admissibility constraints - ALL SATISFIED

Created: 2025-12-28 (GPT Critical Feedback Response)
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine, compute_g_I1, compute_g_I2, compute_base
from src.polynomials import P1Polynomial, PellPolynomial, Polynomial


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


def load_przz_baseline():
    """Load PRZZ baseline polynomials."""
    path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
    with open(path) as f:
        return json.load(f)


class TestPRZZConstraintsVerification:
    """Verify actual PRZZ constraints (not GPT's assumed constraints)."""

    def test_p1_at_0_is_zero(self):
        """P1(0) = 0 - REQUIRED by PRZZ."""
        data = load_optimal_polynomials()
        P1 = P1Polynomial(data['P1_tilde'])
        assert abs(P1.eval(np.array([0.0]))[0]) < 1e-10

    def test_p1_at_1_is_one(self):
        """P1(1) = 1 - REQUIRED by PRZZ."""
        data = load_optimal_polynomials()
        P1 = P1Polynomial(data['P1_tilde'])
        assert abs(P1.eval(np.array([1.0]))[0] - 1.0) < 1e-10

    def test_p2_at_0_is_zero(self):
        """P2(0) = 0 - REQUIRED by PRZZ."""
        data = load_optimal_polynomials()
        P2 = PellPolynomial(data['P2_tilde'])
        assert abs(P2.eval(np.array([0.0]))[0]) < 1e-10

    def test_p3_at_0_is_zero(self):
        """P3(0) = 0 - REQUIRED by PRZZ."""
        data = load_optimal_polynomials()
        P3 = PellPolynomial(data['P3_tilde'])
        assert abs(P3.eval(np.array([0.0]))[0]) < 1e-10

    def test_Q_at_0_is_one(self):
        """Q(0) = 1 - REQUIRED by PRZZ."""
        data = load_optimal_polynomials()
        Q = Polynomial(np.array(data['Q_mono']))
        assert abs(Q.eval(np.array([0.0]))[0] - 1.0) < 0.001


class TestPEll1NotRequired:
    """Demonstrate that P_ell(1) = 0 is NOT required by PRZZ."""

    def test_przz_baseline_p2_at_1_not_zero(self):
        """PRZZ's own P2(1) ≠ 0 - proves constraint not required."""
        # PRZZ P2(x) = 1.048274*x + 1.319912*x^2 - 0.940058*x^3
        P2_przz_at_1 = 1.048274 + 1.319912 - 0.940058
        
        print(f"\n  PRZZ P2(1) = {P2_przz_at_1:.6f} ≠ 0")
        assert abs(P2_przz_at_1) > 1.0, "PRZZ P2(1) should be non-zero"

    def test_przz_baseline_p3_at_1_not_zero(self):
        """PRZZ's own P3(1) ≠ 0 - proves constraint not required."""
        # PRZZ P3(x) = 0.522811*x - 0.686510*x^2 - 0.049923*x^3
        P3_przz_at_1 = 0.522811 - 0.686510 - 0.049923
        
        print(f"\n  PRZZ P3(1) = {P3_przz_at_1:.6f} ≠ 0")
        assert abs(P3_przz_at_1) > 0.1, "PRZZ P3(1) should be non-zero"


class TestFactoredVsUnfactored:
    """Test that factored and unfactored c formulas are equivalent."""

    def test_formulas_match(self):
        """Factored and unfactored c should match to machine precision."""
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
        
        # Unfactored computation
        g_I1 = compute_g_I1(theta, K)
        g_I2 = compute_g_I2(theta, K)
        base = compute_base(R, K)
        
        term1 = result.integrals.I1_plus + result.integrals.I2_plus
        term2 = (g_I1 * base * result.integrals.I1_minus + 
                 g_I2 * base * result.integrals.I2_minus)
        term3 = result.integrals.I3_plus + result.integrals.I4_plus
        
        c_unfactored = term1 + term2 + term3
        
        print(f"\n  c_factored = {result.c:.10f}")
        print(f"  c_unfactored = {c_unfactored:.10f}")
        print(f"  Difference = {abs(result.c - c_unfactored):.2e}")
        
        assert abs(result.c - c_unfactored) < 1e-14


class TestRandomPolynomialAdmissibility:
    """Test that our parameterization always produces admissible polynomials."""

    def test_random_p1_satisfies_constraints(self):
        """Any P1_tilde coefficients produce admissible P1."""
        np.random.seed(42)
        
        for i in range(10):
            tilde = list(np.random.randn(4) * 2)
            P1 = P1Polynomial(tilde)
            
            p1_at_0 = P1.eval(np.array([0.0]))[0]
            p1_at_1 = P1.eval(np.array([1.0]))[0]
            
            assert abs(p1_at_0) < 1e-10, f"Trial {i}: P1(0) = {p1_at_0}"
            assert abs(p1_at_1 - 1.0) < 1e-10, f"Trial {i}: P1(1) = {p1_at_1}"

    def test_random_pell_satisfies_constraints(self):
        """Any Pell_tilde coefficients produce P_ell(0) = 0."""
        np.random.seed(42)
        
        for i in range(10):
            tilde = list(np.random.randn(3) * 2)
            P = PellPolynomial(tilde)
            
            p_at_0 = P.eval(np.array([0.0]))[0]
            
            assert abs(p_at_0) < 1e-10, f"Trial {i}: P(0) = {p_at_0}"


class TestAdmissibilitySummary:
    """Comprehensive summary of admissibility verification."""

    def test_full_admissibility_summary(self):
        """Run full admissibility check."""
        print("\n" + "=" * 70)
        print("PRZZ ADMISSIBILITY VERIFICATION SUMMARY")
        print("=" * 70)
        
        data = load_optimal_polynomials()
        
        P1 = P1Polynomial(data['P1_tilde'])
        P2 = PellPolynomial(data['P2_tilde'])
        P3 = PellPolynomial(data['P3_tilde'])
        Q = Polynomial(np.array(data['Q_mono']))
        
        x0 = np.array([0.0])
        x1 = np.array([1.0])
        
        print("\n  ACTUAL PRZZ CONSTRAINTS (all must pass):")
        print("  " + "-" * 50)
        
        all_passed = True
        
        # P1(0) = 0
        val = P1.eval(x0)[0]
        passed = abs(val) < 1e-10
        all_passed &= passed
        print(f"  P1(0) = 0: {val:.2e} {'[PASS]' if passed else '[FAIL]'}")
        
        # P1(1) = 1
        val = P1.eval(x1)[0]
        passed = abs(val - 1.0) < 1e-10
        all_passed &= passed
        print(f"  P1(1) = 1: {val:.10f} {'[PASS]' if passed else '[FAIL]'}")
        
        # P2(0) = 0
        val = P2.eval(x0)[0]
        passed = abs(val) < 1e-10
        all_passed &= passed
        print(f"  P2(0) = 0: {val:.2e} {'[PASS]' if passed else '[FAIL]'}")
        
        # P3(0) = 0
        val = P3.eval(x0)[0]
        passed = abs(val) < 1e-10
        all_passed &= passed
        print(f"  P3(0) = 0: {val:.2e} {'[PASS]' if passed else '[FAIL]'}")
        
        # Q(0) = 1
        val = Q.eval(x0)[0]
        passed = abs(val - 1.0) < 0.001
        all_passed &= passed
        print(f"  Q(0) = 1: {val:.6f} {'[PASS]' if passed else '[FAIL]'}")
        
        print("\n  NOT REQUIRED BY PRZZ (informational only):")
        print("  " + "-" * 50)
        print(f"  P2(1) = {P2.eval(x1)[0]:.6f} (no constraint)")
        print(f"  P3(1) = {P3.eval(x1)[0]:.6f} (no constraint)")
        
        # PRZZ baseline comparison
        P2_przz_at_1 = 1.048274 + 1.319912 - 0.940058
        P3_przz_at_1 = 0.522811 - 0.686510 - 0.049923
        print(f"\n  PRZZ baseline P2(1) = {P2_przz_at_1:.6f} (also non-zero)")
        print(f"  PRZZ baseline P3(1) = {P3_przz_at_1:.6f} (also non-zero)")
        
        print("\n" + "=" * 70)
        status = "ALL PRZZ CONSTRAINTS SATISFIED" if all_passed else "CONSTRAINTS VIOLATED"
        print(f"RESULT: {status}")
        print("=" * 70)
        
        assert all_passed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPT ADMISSIBILITY CONCERNS - Quick Check")
    print("=" * 70)
    
    data = load_optimal_polynomials()
    
    P1 = P1Polynomial(data['P1_tilde'])
    P2 = PellPolynomial(data['P2_tilde'])
    P3 = PellPolynomial(data['P3_tilde'])
    
    x0 = np.array([0.0])
    x1 = np.array([1.0])
    
    print("\n  Optimized polynomial endpoints:")
    print(f"    P1(0) = {P1.eval(x0)[0]:.10f} (should be 0)")
    print(f"    P1(1) = {P1.eval(x1)[0]:.10f} (should be 1)")
    print(f"    P2(0) = {P2.eval(x0)[0]:.10f} (should be 0)")
    print(f"    P2(1) = {P2.eval(x1)[0]:.6f} (NO CONSTRAINT)")
    print(f"    P3(0) = {P3.eval(x0)[0]:.10f} (should be 0)")
    print(f"    P3(1) = {P3.eval(x1)[0]:.6f} (NO CONSTRAINT)")
    
    print("\n  PRZZ baseline (proof P_ell(1)=0 not required):")
    print(f"    PRZZ P2(1) = {1.048274 + 1.319912 - 0.940058:.6f}")
    print(f"    PRZZ P3(1) = {0.522811 - 0.686510 - 0.049923:.6f}")
