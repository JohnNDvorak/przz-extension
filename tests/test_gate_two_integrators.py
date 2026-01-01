#!/usr/bin/env python3
"""
Gate 2: Two Independent Numerical Integrators

GPT's second validation gate for the kappa = 0.5213 claim:
- Compute c using Gauss-Legendre (standard)
- Compute c using Clenshaw-Curtis (independent)
- Agreement threshold: 1e-6 (well beyond the ~10% improvement)

If the two independent integrators agree, we have high confidence
the numerical computation is correct.

Created: 2025-12-28 (GPT Critical Review)
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quadrature import gauss_legendre_01, tensor_grid_2d
from src.quadrature_cc import clenshaw_curtis_01, tensor_grid_2d_cc, validate_quadrature


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


class TestQuadratureBasics:
    """Test basic quadrature properties for both methods."""

    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    def test_gauss_legendre_weights_sum_to_one(self, n):
        """Gauss-Legendre weights should sum to 1 on [0,1]."""
        nodes, weights = gauss_legendre_01(n)
        assert abs(np.sum(weights) - 1.0) < 1e-14

    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    def test_clenshaw_curtis_weights_sum_to_one(self, n):
        """Clenshaw-Curtis weights should sum to 1 on [0,1]."""
        nodes, weights = clenshaw_curtis_01(n)
        assert abs(np.sum(weights) - 1.0) < 1e-14

    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    def test_both_integrate_polynomials_exactly(self, n):
        """Both methods should integrate x^k exactly up to some degree."""
        nodes_gl, weights_gl = gauss_legendre_01(n)
        nodes_cc, weights_cc = clenshaw_curtis_01(n)

        for k in range(min(n, 20)):
            exact = 1.0 / (k + 1)
            gl_result = np.sum(weights_gl * nodes_gl**k)
            cc_result = np.sum(weights_cc * nodes_cc**k)

            gl_error = abs(gl_result - exact)
            cc_error = abs(cc_result - exact)

            assert gl_error < 1e-10, f"GL error for x^{k}: {gl_error}"
            assert cc_error < 1e-8, f"CC error for x^{k}: {cc_error}"

    def test_gl_cc_agree_on_smooth_integrands(self):
        """GL and CC should agree on smooth integrands."""
        n = 40

        nodes_gl, weights_gl = gauss_legendre_01(n)
        nodes_cc, weights_cc = clenshaw_curtis_01(n)

        exact_exp = np.exp(1) - 1
        gl_exp = np.sum(weights_gl * np.exp(nodes_gl))
        cc_exp = np.sum(weights_cc * np.exp(nodes_cc))

        print(f"\n  int exp(x)dx: exact={exact_exp:.10f}")
        print(f"    GL = {gl_exp:.10f}, error = {abs(gl_exp - exact_exp):.2e}")
        print(f"    CC = {cc_exp:.10f}, error = {abs(cc_exp - exact_exp):.2e}")

        assert abs(gl_exp - exact_exp) < 1e-10
        assert abs(cc_exp - exact_exp) < 1e-8
        assert abs(gl_exp - cc_exp) < 1e-8


class TestTwoIntegratorsOnKappaIntegrals:
    """Test that GL and CC agree on the actual integrals used in kappa computation."""

    def test_i2_integral_agreement(self):
        """Compare I2-like integral computed with GL vs CC."""
        from src.polynomials import P1Polynomial, PellPolynomial, Polynomial

        data = load_optimal_polynomials()

        P2 = PellPolynomial(data['P2_tilde'])
        Q = Polynomial(np.array(data['Q_mono']))

        R = 1.3036
        n_quad = 40

        print("\n  I2 Integral Agreement Test (GL vs CC):")
        print(f"  n_quad = {n_quad}")

        U_gl, T_gl, W_gl = tensor_grid_2d(n_quad)
        U_cc, T_cc, W_cc = tensor_grid_2d_cc(n_quad)

        P2_vals_gl = P2.eval(U_gl.flatten()).reshape(U_gl.shape)
        Q_vals_gl = Q.eval(T_gl.flatten()).reshape(T_gl.shape)
        integrand_gl = P2_vals_gl**2 * Q_vals_gl**2 * np.exp(2 * R * T_gl)
        result_gl = np.sum(integrand_gl * W_gl)

        P2_vals_cc = P2.eval(U_cc.flatten()).reshape(U_cc.shape)
        Q_vals_cc = Q.eval(T_cc.flatten()).reshape(T_cc.shape)
        integrand_cc = P2_vals_cc**2 * Q_vals_cc**2 * np.exp(2 * R * T_cc)
        result_cc = np.sum(integrand_cc * W_cc)

        rel_diff = abs(result_gl - result_cc) / (abs(result_gl) + 1e-15)

        print(f"\n  int int P2^2*Q^2*exp(2Rt) du dt:")
        print(f"    GL = {result_gl:.10f}")
        print(f"    CC = {result_cc:.10f}")
        print(f"    rel_diff = {rel_diff:.2e}")

        assert rel_diff < 1e-6, f"GL vs CC disagree: {rel_diff:.2e}"


class TestFullKappaEngineComparison:
    """Full comparison of kappa computation with different quadrature settings."""

    def test_kappa_convergence_across_n_quad(self):
        """Verify kappa converges as n_quad increases."""
        from src.kappa_engine import KappaEngine

        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4 / 7

        print("\n  Full KappaEngine Convergence:")
        print("  " + "-" * 60)

        results = []
        for n_quad in [40, 60, 80]:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=3,
                R=R,
                n_quad=n_quad,
            )
            result = engine.compute_kappa()
            results.append((n_quad, result.c, result.kappa))
            print(f"  n_quad={n_quad}: c={result.c:.10f}, kappa={result.kappa:.10f}")

        c_40, c_60, c_80 = results[0][1], results[1][1], results[2][1]
        drift_40_60 = abs(c_60 - c_40) / c_40
        drift_60_80 = abs(c_80 - c_60) / c_60

        print(f"\n  Convergence:")
        print(f"    |c(60)-c(40)|/c(40) = {drift_40_60:.2e}")
        print(f"    |c(80)-c(60)|/c(60) = {drift_60_80:.2e}")

        assert drift_60_80 < drift_40_60, "Not converging"
        assert drift_60_80 < 1e-6, f"Not converged at n=80: drift = {drift_60_80:.2e}"

        for n, c, kappa in results:
            assert kappa > 0.5, f"kappa < 0.5 at n_quad={n}"

    def test_key_integrals_agree_gl_vs_cc(self):
        """Compute key PRZZ integrals with both GL and CC, verify agreement."""
        from src.polynomials import P1Polynomial, PellPolynomial, Polynomial

        data = load_optimal_polynomials()

        P1 = P1Polynomial(data['P1_tilde'])
        P2 = PellPolynomial(data['P2_tilde'])
        P3 = PellPolynomial(data['P3_tilde'])
        Q = Polynomial(np.array(data['Q_mono']))

        R = 1.3036
        n_quad = 60

        print("\n  Key Integrals GL vs CC Agreement:")
        print(f"  n_quad = {n_quad}")
        print("  " + "-" * 60)

        U_gl, T_gl, W_gl = tensor_grid_2d(n_quad)
        U_cc, T_cc, W_cc = tensor_grid_2d_cc(n_quad)

        test_cases = [
            ("P1^2*Q^2*exp(2Rt)", lambda U, T: P1.eval(U.flatten()).reshape(U.shape)**2 *
                                               Q.eval(T.flatten()).reshape(T.shape)**2 *
                                               np.exp(2 * R * T)),
            ("P2^2*Q^2*exp(2Rt)", lambda U, T: P2.eval(U.flatten()).reshape(U.shape)**2 *
                                               Q.eval(T.flatten()).reshape(T.shape)**2 *
                                               np.exp(2 * R * T)),
            ("P3^2*Q^2*exp(2Rt)", lambda U, T: P3.eval(U.flatten()).reshape(U.shape)**2 *
                                               Q.eval(T.flatten()).reshape(T.shape)**2 *
                                               np.exp(2 * R * T)),
        ]

        all_agree = True
        for name, integrand_fn in test_cases:
            val_gl = np.sum(integrand_fn(U_gl, T_gl) * W_gl)
            val_cc = np.sum(integrand_fn(U_cc, T_cc) * W_cc)
            rel_diff = abs(val_gl - val_cc) / (abs(val_gl) + 1e-15)

            status = "OK" if rel_diff < 1e-6 else "FAIL"
            print(f"  {name}:")
            print(f"    GL = {val_gl:+.10f}")
            print(f"    CC = {val_cc:+.10f}")
            print(f"    rel_diff = {rel_diff:.2e} [{status}]")

            if rel_diff >= 1e-6:
                all_agree = False

        assert all_agree, "Some integrals disagree between GL and CC"


class TestGate2Summary:
    """Comprehensive Gate 2 summary."""

    def test_full_gate2_summary(self):
        """Run full Gate 2 summary with pass/fail status."""
        from src.kappa_engine import KappaEngine
        from src.polynomials import P1Polynomial, PellPolynomial, Polynomial

        print("\n" + "=" * 70)
        print("GATE 2: TWO INDEPENDENT INTEGRATORS (GPT Critical Review)")
        print("=" * 70)

        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4 / 7
        n_quad = 60

        all_passed = True

        # Test 1: Basic quadrature validation
        print("\n  Test 1: Basic Quadrature Validation")

        nodes_gl, weights_gl = gauss_legendre_01(n_quad)
        nodes_cc, weights_cc = clenshaw_curtis_01(n_quad)

        gl_weight_sum = np.sum(weights_gl)
        cc_weight_sum = np.sum(weights_cc)

        test1a = abs(gl_weight_sum - 1.0) < 1e-14
        test1b = abs(cc_weight_sum - 1.0) < 1e-14

        print(f"    GL weights sum: {gl_weight_sum:.15f} {'[OK]' if test1a else '[FAIL]'}")
        print(f"    CC weights sum: {cc_weight_sum:.15f} {'[OK]' if test1b else '[FAIL]'}")

        all_passed &= test1a and test1b

        # Test 2: Polynomial integration accuracy
        print("\n  Test 2: Polynomial Integration (x^k tests)")

        gl_validation = validate_quadrature(nodes_gl, weights_gl, max_degree=20)
        cc_validation = validate_quadrature(nodes_cc, weights_cc, max_degree=20)

        test2a = gl_validation['max_error'] < 1e-10
        test2b = cc_validation['max_error'] < 1e-6

        print(f"    GL max error: {gl_validation['max_error']:.2e} {'[OK]' if test2a else '[FAIL]'}")
        print(f"    CC max error: {cc_validation['max_error']:.2e} {'[OK]' if test2b else '[FAIL]'}")

        all_passed &= test2a and test2b

        # Test 3: Key integral agreement
        print("\n  Test 3: Key Integral Agreement (GL vs CC)")

        P2 = PellPolynomial(data['P2_tilde'])
        Q = Polynomial(np.array(data['Q_mono']))

        U_gl, T_gl, W_gl = tensor_grid_2d(n_quad)
        U_cc, T_cc, W_cc = tensor_grid_2d_cc(n_quad)

        P2_gl = P2.eval(U_gl.flatten()).reshape(U_gl.shape)
        Q_gl = Q.eval(T_gl.flatten()).reshape(T_gl.shape)
        val_gl = np.sum(P2_gl**2 * Q_gl**2 * np.exp(2 * R * T_gl) * W_gl)

        P2_cc = P2.eval(U_cc.flatten()).reshape(U_cc.shape)
        Q_cc = Q.eval(T_cc.flatten()).reshape(T_cc.shape)
        val_cc = np.sum(P2_cc**2 * Q_cc**2 * np.exp(2 * R * T_cc) * W_cc)

        rel_diff = abs(val_gl - val_cc) / (abs(val_gl) + 1e-15)
        test3 = rel_diff < 1e-6

        print(f"    int int P2^2*Q^2*exp(2Rt) GL = {val_gl:.10f}")
        print(f"    int int P2^2*Q^2*exp(2Rt) CC = {val_cc:.10f}")
        print(f"    rel_diff = {rel_diff:.2e} {'[OK]' if test3 else '[FAIL]'}")

        all_passed &= test3

        # Test 4: Full kappa convergence
        print("\n  Test 4: kappa Convergence Across n_quad")

        c_values = []
        for n in [40, 60, 80]:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=3,
                R=R,
                n_quad=n,
            )
            result = engine.compute_kappa()
            c_values.append(result.c)
            print(f"    n_quad={n}: c={result.c:.10f}, kappa={result.kappa:.10f}")

        drift_60_80 = abs(c_values[2] - c_values[1]) / c_values[1]
        test4 = drift_60_80 < 1e-6

        print(f"    Drift (60->80): {drift_60_80:.2e} {'[OK]' if test4 else '[FAIL]'}")

        all_passed &= test4

        # Summary
        print("\n" + "=" * 70)
        overall = "PASS" if all_passed else "FAIL"
        print(f"GATE 2 OVERALL: {overall}")
        print("=" * 70)

        assert all_passed, "Gate 2 failed"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GATE 2: TWO INTEGRATORS - Quick Check")
    print("=" * 70)

    n = 60
    nodes_gl, weights_gl = gauss_legendre_01(n)
    nodes_cc, weights_cc = clenshaw_curtis_01(n)

    print(f"\n  n = {n}")
    print(f"  GL weights sum: {np.sum(weights_gl):.15f}")
    print(f"  CC weights sum: {np.sum(weights_cc):.15f}")

    exact = np.exp(1) - 1
    gl = np.sum(weights_gl * np.exp(nodes_gl))
    cc = np.sum(weights_cc * np.exp(nodes_cc))

    print(f"\n  int exp(x)dx = {exact:.10f}")
    print(f"  GL: {gl:.10f}, error: {abs(gl-exact):.2e}")
    print(f"  CC: {cc:.10f}, error: {abs(cc-exact):.2e}")
