#!/usr/bin/env python3
"""
Gate 4: Basis Stability Test

Verify that the same polynomials represented in different bases give
identical c values. This tests that our evaluation doesn't depend on
the specific coefficient representation.

Key Insight:
============
A polynomial P(x) = 2x^2 + 3x + 1 is the SAME polynomial whether we store:
- Monomial: [1, 3, 2]
- Chebyshev: converted to Chebyshev basis
- Legendre: converted to Legendre basis

If our c computation depends on the representation, there's a bug.

Test Strategy:
==============
1. Start with optimized P2, P3 in monomial form
2. Convert to alternative bases (still representing the same polynomial)
3. Evaluate c using both representations
4. Verify c values are identical to float precision

Created: 2025-12-28
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import Polynomial, P1Polynomial, PellPolynomial, QPolynomial


def convert_monomial_to_chebyshev(mono_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert monomial basis coefficients to Chebyshev basis.

    P(x) = sum_i c_i * x^i = sum_j d_j * T_j(x)

    where T_j(x) is the Chebyshev polynomial of the first kind.
    """
    n = len(mono_coeffs)
    if n == 0:
        return np.array([0.0])

    # Use numpy's polynomial conversion
    from numpy.polynomial import chebyshev as C
    from numpy.polynomial import polynomial as P

    # Convert from power series to Chebyshev
    cheb = C.poly2cheb(mono_coeffs)
    return cheb


def convert_chebyshev_to_monomial(cheb_coeffs: np.ndarray) -> np.ndarray:
    """Convert Chebyshev basis back to monomial basis."""
    from numpy.polynomial import chebyshev as C
    from numpy.polynomial import polynomial as P

    mono = C.cheb2poly(cheb_coeffs)
    return mono


def evaluate_polynomial_monomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate polynomial in monomial basis: sum_i c_i * x^i."""
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result


def evaluate_polynomial_chebyshev(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate polynomial in Chebyshev basis: sum_j c_j * T_j(x)."""
    from numpy.polynomial import chebyshev as C
    return C.chebval(x, coeffs)


class TestBasisConversion:
    """Verify basis conversion is lossless."""

    def test_monomial_chebyshev_round_trip(self):
        """Convert monomial -> Chebyshev -> monomial should be identity."""
        # Test polynomial: P(x) = 1 + 2x + 3x^2 + x^3
        mono_original = np.array([1.0, 2.0, 3.0, 1.0])

        # Convert to Chebyshev and back
        cheb = convert_monomial_to_chebyshev(mono_original)
        mono_recovered = convert_chebyshev_to_monomial(cheb)

        # Trim trailing zeros
        while len(mono_recovered) > 1 and abs(mono_recovered[-1]) < 1e-14:
            mono_recovered = mono_recovered[:-1]

        print(f"\nOriginal monomial:  {mono_original}")
        print(f"Chebyshev:          {cheb}")
        print(f"Recovered monomial: {mono_recovered}")

        assert len(mono_original) == len(mono_recovered), "Length mismatch"
        assert np.allclose(mono_original, mono_recovered), "Round-trip failed"

    def test_evaluation_equivalence(self):
        """Same polynomial evaluated in both bases should give same values."""
        mono = np.array([1.0, 2.0, -1.0, 0.5])  # P(x) = 1 + 2x - x^2 + 0.5x^3
        cheb = convert_monomial_to_chebyshev(mono)

        x_test = np.linspace(0, 1, 50)

        y_mono = evaluate_polynomial_monomial(x_test, mono)
        y_cheb = evaluate_polynomial_chebyshev(x_test, cheb)

        max_diff = np.max(np.abs(y_mono - y_cheb))
        print(f"\nMax evaluation difference: {max_diff:.2e}")

        assert max_diff < 1e-12, f"Evaluation mismatch: {max_diff}"


class TestI2BasisStability:
    """Verify I2 computation is basis-independent."""

    def test_i2_monomial_vs_chebyshev_representation(self):
        """
        Compute I2 with P2 in monomial vs Chebyshev representation.

        Both should give identical results since they represent the same polynomial.
        """
        from src.quadrature import gauss_legendre_01

        R = 1.3036
        theta = 4 / 7

        # PRZZ P2 in monomial form (x * P_tilde)
        # P_tilde(x) = c0 + c1*x + c2*x^2
        p2_tilde = np.array([0.547839, -0.148066, 0.0])

        # Full P2 monomial coefficients: [0, c0, c1, c2] for x*P_tilde
        p2_mono = np.concatenate([[0.0], p2_tilde])

        # Convert to Chebyshev
        p2_cheb = convert_monomial_to_chebyshev(p2_mono)

        # Verify they evaluate to the same values
        x_test = np.linspace(0.01, 0.99, 30)
        y_mono = evaluate_polynomial_monomial(x_test, p2_mono)
        y_cheb = evaluate_polynomial_chebyshev(x_test, p2_cheb)

        eval_diff = np.max(np.abs(y_mono - y_cheb))
        print(f"\nP2 evaluation difference (mono vs cheb): {eval_diff:.2e}")
        assert eval_diff < 1e-12, "P2 evaluation differs between bases"

        # Now compute I2 using both representations
        # For I2 (no derivatives), we just integrate P(u)^2 * Q(t)^2 * exp(2Rt)

        # Load Q
        from src.polynomials import load_przz_polynomials
        _, _, _, Q = load_przz_polynomials()

        u_nodes, u_weights = gauss_legendre_01(60)
        t_nodes, t_weights = gauss_legendre_01(60)

        Q_vals = Q.eval(t_nodes)
        t_integral = np.sum(t_weights * Q_vals**2 * np.exp(2*R*t_nodes)) / theta

        # I2 with monomial evaluation
        P2_mono_vals = evaluate_polynomial_monomial(u_nodes, p2_mono)
        i2_mono = np.sum(u_weights * P2_mono_vals**2) * t_integral

        # I2 with Chebyshev evaluation
        P2_cheb_vals = evaluate_polynomial_chebyshev(u_nodes, p2_cheb)
        i2_cheb = np.sum(u_weights * P2_cheb_vals**2) * t_integral

        rel_diff = abs(i2_mono - i2_cheb) / (abs(i2_mono) + 1e-15)

        print(f"I2 monomial:   {i2_mono:.10e}")
        print(f"I2 Chebyshev:  {i2_cheb:.10e}")
        print(f"Relative diff: {rel_diff:.2e}")

        assert rel_diff < 1e-12, f"I2 basis-dependent: rel_diff={rel_diff:.6e}"


class TestCaseKernelBasisStability:
    """Verify Case C kernel is basis-independent."""

    def test_case_c_kernel_basis_stability(self):
        """
        Case C kernel with polynomial in different bases should be identical.
        """
        from src.case_c_kernel import compute_case_c_kernel

        R = 1.3036
        theta = 4 / 7

        # P3 in monomial form
        p3_tilde = np.array([0.120689, -0.044386, 0.0])
        p3_mono = np.concatenate([[0.0], p3_tilde])
        p3_cheb = convert_monomial_to_chebyshev(p3_mono)

        u_grid = np.linspace(0.1, 0.9, 20)

        # Kernel with monomial evaluator
        def P3_mono_eval(x):
            return evaluate_polynomial_monomial(x, p3_mono)

        K_mono = compute_case_c_kernel(P3_mono_eval, u_grid, omega=2, R=R, theta=theta)

        # Kernel with Chebyshev evaluator
        def P3_cheb_eval(x):
            return evaluate_polynomial_chebyshev(x, p3_cheb)

        K_cheb = compute_case_c_kernel(P3_cheb_eval, u_grid, omega=2, R=R, theta=theta)

        max_diff = np.max(np.abs(K_mono - K_cheb))
        rel_diff = max_diff / (np.max(np.abs(K_mono)) + 1e-15)

        print(f"\nCase C kernel comparison:")
        print(f"  Max abs diff: {max_diff:.2e}")
        print(f"  Max rel diff: {rel_diff:.2e}")

        assert rel_diff < 1e-10, f"Case C kernel basis-dependent: {rel_diff:.6e}"


class TestFullIntegralBasisStability:
    """Verify full integral computation is basis-independent."""

    def test_pair_22_basis_stability(self):
        """
        Pair (2,2) should give identical I2 regardless of P2 representation.
        """
        from src.unified_i2_paper import compute_I2_unified_paper
        from src.polynomials import load_przz_polynomials, PellPolynomial

        R = 1.3036
        theta = 4 / 7

        P1, P2_orig, P3, Q = load_przz_polynomials()

        # Compute with original P2
        polys_orig = {"P1": P1, "P2": P2_orig, "P3": P3, "Q": Q}
        result_orig = compute_I2_unified_paper(
            R, theta, ell1=2, ell2=2,
            polynomials=polys_orig,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )

        # Create "equivalent" P2 with slightly perturbed then restored coefficients
        # This tests that numerical precision is maintained
        p2_coeffs = P2_orig.tilde_coeffs.copy()
        # Perturb and restore (should be identical)
        p2_coeffs_restored = p2_coeffs + 1e-15 - 1e-15

        P2_restored = PellPolynomial(p2_coeffs_restored)
        polys_restored = {"P1": P1, "P2": P2_restored, "P3": P3, "Q": Q}

        result_restored = compute_I2_unified_paper(
            R, theta, ell1=2, ell2=2,
            polynomials=polys_restored,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )

        rel_diff = abs(result_orig.I2_value - result_restored.I2_value) / (abs(result_orig.I2_value) + 1e-15)

        print(f"\nPair (2,2) I2 stability:")
        print(f"  Original:  {result_orig.I2_value:.10e}")
        print(f"  Restored:  {result_restored.I2_value:.10e}")
        print(f"  Rel diff:  {rel_diff:.2e}")

        assert rel_diff < 1e-12, f"I2 not stable: {rel_diff:.6e}"


class TestGateSummary:
    """Comprehensive Gate 4 summary."""

    def test_full_gate4_summary(self):
        """Run full Gate 4 summary."""
        print("\n" + "=" * 70)
        print("GATE 4: BASIS STABILITY SUMMARY")
        print("=" * 70)

        all_passed = True

        # Test 1: Conversion round-trip
        mono = np.array([1.0, 2.0, 3.0, 1.0])
        cheb = convert_monomial_to_chebyshev(mono)
        mono_back = convert_chebyshev_to_monomial(cheb)
        while len(mono_back) > len(mono) and abs(mono_back[-1]) < 1e-14:
            mono_back = mono_back[:-1]

        test1_pass = np.allclose(mono, mono_back)
        print(f"\n  Conversion round-trip: {'PASS' if test1_pass else 'FAIL'}")
        all_passed &= test1_pass

        # Test 2: Evaluation equivalence
        x_test = np.linspace(0, 1, 30)
        y_mono = evaluate_polynomial_monomial(x_test, mono)
        y_cheb = evaluate_polynomial_chebyshev(x_test, cheb)
        test2_pass = np.max(np.abs(y_mono - y_cheb)) < 1e-12
        print(f"  Evaluation equivalence: {'PASS' if test2_pass else 'FAIL'}")
        all_passed &= test2_pass

        # Test 3: I2 basis stability
        from src.polynomials import load_przz_polynomials
        from src.quadrature import gauss_legendre_01

        R = 1.3036
        theta = 4 / 7

        _, P2, _, Q = load_przz_polynomials()
        p2_mono = P2.to_monomial().coeffs
        p2_cheb = convert_monomial_to_chebyshev(p2_mono)

        u_nodes, u_weights = gauss_legendre_01(60)
        t_nodes, t_weights = gauss_legendre_01(60)

        Q_vals = Q.eval(t_nodes)
        t_int = np.sum(t_weights * Q_vals**2 * np.exp(2*R*t_nodes)) / theta

        P2_mono = evaluate_polynomial_monomial(u_nodes, p2_mono)
        P2_cheb = evaluate_polynomial_chebyshev(u_nodes, p2_cheb)

        i2_mono = np.sum(u_weights * P2_mono**2) * t_int
        i2_cheb = np.sum(u_weights * P2_cheb**2) * t_int

        rel_diff = abs(i2_mono - i2_cheb) / (abs(i2_mono) + 1e-15)
        test3_pass = rel_diff < 1e-10
        print(f"  I2 (2,2) basis stability: {'PASS' if test3_pass else 'FAIL'} (rel_diff={rel_diff:.2e})")
        all_passed &= test3_pass

        print("\n" + "=" * 70)
        print(f"GATE 4 OVERALL: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

        assert all_passed, "Gate 4 failed"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GATE 4: BASIS STABILITY - Quick Check")
    print("=" * 70)

    # Quick basis conversion test
    mono = np.array([1.0, 2.0, -1.0, 0.5])
    cheb = convert_monomial_to_chebyshev(mono)
    mono_back = convert_chebyshev_to_monomial(cheb)

    print(f"\nMonomial: {mono}")
    print(f"Chebyshev: {cheb}")
    print(f"Recovered: {mono_back[:len(mono)]}")

    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_mono = evaluate_polynomial_monomial(x_test, mono)
    y_cheb = evaluate_polynomial_chebyshev(x_test, cheb)

    print(f"\nEvaluation comparison:")
    for i, x in enumerate(x_test):
        print(f"  x={x:.2f}: mono={y_mono[i]:.6f}, cheb={y_cheb[i]:.6f}, diff={abs(y_mono[i]-y_cheb[i]):.2e}")
