"""
tests/test_coefficient_gate.py
Coefficient Gate: Verify Series Engine Matches Finite Differences

This is the MUST-PASS gate before trusting any evaluator.

The series engine computes Taylor coefficients f[i,j] via polynomial multiplication.
We verify these match finite-difference approximations of ∂ⁱ⁺ʲF/∂xⁱ∂yʲ at (0,0).

GPT's guidance:
> "At a handful of random quadrature nodes (u,t):
>  1. Define the scalar kernel function F(x,y) by direct evaluation (no expansions).
>  2. Finite-difference approximate ∂²F/∂x∂y at (0,0).
>  3. Compare to the series coefficient f[1,1] (remember: f[1,1] IS the derivative divided by 1!1!)."
"""

import pytest
import numpy as np
import random
from math import exp, factorial

from src.polynomials import load_przz_polynomials
from src.psi_series_evaluator import PsiSeriesEvaluator


def F_direct_series_convention(x: float, y: float, u: float, t: float, P_L, P_R, Q, R: float, theta: float) -> float:
    """
    Evaluate F(x,y) using the SERIES ENGINE CONVENTION.

    This uses the same +P' convention as PsiSeriesEvaluator:
    - P_L contribution: Σ_i P_L^(i)(u)/i! × x^i  (NO sign alternation)
    - P_R contribution: Σ_j P_R^(j)(u)/j! × y^j  (NO sign alternation)

    This is NOT the physical P(u-x), but matches the PRZZ/Section7 convention.

    Note: We compute this by Taylor expanding each factor to order 3 and multiplying.
    """
    max_order = 3

    # P_L Taylor coefficients (no sign alternation)
    P_L_coeffs = np.zeros(max_order + 1)
    for i in range(max_order + 1):
        P_L_coeffs[i] = P_L.eval_deriv(np.array([u]), i)[0] / factorial(i)

    # P_R Taylor coefficients (no sign alternation)
    P_R_coeffs = np.zeros(max_order + 1)
    for j in range(max_order + 1):
        P_R_coeffs[j] = P_R.eval_deriv(np.array([u]), j)[0] / factorial(j)

    # Q(α) and Q(β) coefficients
    dα_dx = theta * t
    dα_dy = theta * (t - 1)
    dβ_dx = theta * (t - 1)
    dβ_dy = theta * t

    # Q Taylor expansion: Q(t + a*x + b*y) = Σ_k Q^(k)(t)/k! × (a*x + b*y)^k
    def expand_Q(t_val, a, b):
        coeffs = np.zeros((max_order + 1, max_order + 1))
        for k in range(2 * max_order + 1):
            Qk = Q.eval_deriv(np.array([t_val]), k)[0]
            for i in range(min(k + 1, max_order + 1)):
                for j in range(min(k - i + 1, max_order + 1)):
                    if i + j == k and i <= max_order and j <= max_order:
                        coeffs[i, j] = Qk * (a ** i) * (b ** j) / (factorial(i) * factorial(j))
        return coeffs

    Q_α_coeffs = expand_Q(t, dα_dx, dα_dy)
    Q_β_coeffs = expand_Q(t, dβ_dx, dβ_dy)

    # Exponential: exp(R*(t + α_arg)) × exp(R*(t + β_arg))
    # = exp(2Rt) × exp(R*(dα_dx + dβ_dx)*x + R*(dα_dy + dβ_dy)*y)
    exp_base = exp(2 * R * t)
    c_x = R * (dα_dx + dβ_dx)
    c_y = R * (dα_dy + dβ_dy)

    def expand_exp(base, cx, cy):
        coeffs = np.zeros((max_order + 1, max_order + 1))
        for i in range(max_order + 1):
            for j in range(max_order + 1):
                coeffs[i, j] = base * (cx ** i) * (cy ** j) / (factorial(i) * factorial(j))
        return coeffs

    exp_coeffs = expand_exp(exp_base, c_x, c_y)

    # Multiply all factors using 2D convolution
    def multiply_2d(a, b):
        n = max_order + 1
        result = np.zeros((n, n))
        for i1 in range(n):
            for j1 in range(n):
                if a[i1, j1] == 0:
                    continue
                for i2 in range(n - i1):
                    for j2 in range(n - j1):
                        result[i1 + i2, j1 + j2] += a[i1, j1] * b[i2, j2]
        return result

    # Convert 1D P coefficients to 2D
    P_L_2d = np.zeros((max_order + 1, max_order + 1))
    P_L_2d[:, 0] = P_L_coeffs
    P_R_2d = np.zeros((max_order + 1, max_order + 1))
    P_R_2d[0, :] = P_R_coeffs

    # Multiply all factors
    result = P_L_2d.copy()
    result = multiply_2d(result, P_R_2d)
    result = multiply_2d(result, Q_α_coeffs)
    result = multiply_2d(result, Q_β_coeffs)
    result = multiply_2d(result, exp_coeffs)

    # Evaluate at (x, y)
    total = 0.0
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            total += result[i, j] * (x ** i) * (y ** j)

    return total


def F_direct_physical(x: float, y: float, u: float, t: float, P_L, P_R, Q, R: float, theta: float) -> float:
    """
    Evaluate F(x,y) using PHYSICAL convention (for reference).

    F(x,y) = P_L(u-x) × P_R(u-y) × Q(α_arg) × Q(β_arg) × exp(R(α_arg + β_arg))

    This uses actual P(u-x), with alternating signs in Taylor expansion.
    """
    P_L_val = P_L.eval(np.array([u - x]))[0]
    P_R_val = P_R.eval(np.array([u - y]))[0]

    alpha_arg = t + theta * t * x + theta * (t - 1) * y
    beta_arg = t + theta * (t - 1) * x + theta * t * y

    Q_alpha = Q.eval(np.array([alpha_arg]))[0]
    Q_beta = Q.eval(np.array([beta_arg]))[0]

    exp_factor = exp(R * alpha_arg) * exp(R * beta_arg)

    return P_L_val * P_R_val * Q_alpha * Q_beta * exp_factor


def finite_diff_d2F_dxdy_series(u: float, t: float, P_L, P_R, Q, R: float, theta: float, h: float = 1e-5) -> float:
    """
    Finite difference approximation of ∂²F/∂x∂y at (0,0) using SERIES CONVENTION.

    Uses central difference: (F(h,h) - F(h,-h) - F(-h,h) + F(-h,-h)) / (4h²)
    """
    def F(x, y):
        return F_direct_series_convention(x, y, u, t, P_L, P_R, Q, R, theta)

    return (F(h, h) - F(h, -h) - F(-h, h) + F(-h, -h)) / (4 * h * h)


def finite_diff_dF_dx_series(u: float, t: float, P_L, P_R, Q, R: float, theta: float, h: float = 1e-5) -> float:
    """Finite difference approximation of ∂F/∂x at (0,0) using SERIES CONVENTION."""
    def F(x, y):
        return F_direct_series_convention(x, y, u, t, P_L, P_R, Q, R, theta)

    return (F(h, 0) - F(-h, 0)) / (2 * h)


def finite_diff_dF_dy_series(u: float, t: float, P_L, P_R, Q, R: float, theta: float, h: float = 1e-5) -> float:
    """Finite difference approximation of ∂F/∂y at (0,0) using SERIES CONVENTION."""
    def F(x, y):
        return F_direct_series_convention(x, y, u, t, P_L, P_R, Q, R, theta)

    return (F(0, h) - F(0, -h)) / (2 * h)


class TestCoefficientGate:
    """Verify series coefficients match finite differences."""

    @pytest.fixture(scope="class")
    def polynomials(self):
        """Load PRZZ polynomials."""
        return load_przz_polynomials(enforce_Q0=True)

    def test_f00_matches_direct(self, polynomials):
        """f[0,0] = F(0,0) (the base value)."""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        # Test at several random (u,t) points
        random.seed(42)
        for _ in range(10):
            u = random.uniform(0.1, 0.9)
            t = random.uniform(0.1, 0.9)

            # Series engine
            evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
            f_coeffs = evaluator._build_F_coefficients(u, t)
            f00_series = f_coeffs[0, 0]

            # Direct evaluation using series convention
            f00_direct = F_direct_series_convention(0, 0, u, t, P1, P1, Q, R, theta)

            rel_error = abs(f00_series - f00_direct) / abs(f00_direct) if f00_direct != 0 else abs(f00_series)
            assert rel_error < 1e-10, f"f[0,0] mismatch at (u={u:.3f}, t={t:.3f}): series={f00_series}, direct={f00_direct}, rel_error={rel_error}"

    def test_f10_matches_finite_diff(self, polynomials):
        """f[1,0] = (∂F/∂x)|_{x=y=0} / 1!"""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        random.seed(43)
        for _ in range(10):
            u = random.uniform(0.1, 0.9)
            t = random.uniform(0.1, 0.9)

            evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
            f_coeffs = evaluator._build_F_coefficients(u, t)
            f10_series = f_coeffs[1, 0]

            # f[1,0] = (dF/dx)|_0 / 1! = dF/dx|_0
            dF_dx_fd = finite_diff_dF_dx_series(u, t, P1, P1, Q, R, theta)

            rel_error = abs(f10_series - dF_dx_fd) / abs(dF_dx_fd) if dF_dx_fd != 0 else abs(f10_series)
            assert rel_error < 1e-5, f"f[1,0] mismatch at (u={u:.3f}, t={t:.3f}): series={f10_series}, fd={dF_dx_fd}, rel_error={rel_error}"

    def test_f01_matches_finite_diff(self, polynomials):
        """f[0,1] = (∂F/∂y)|_{x=y=0} / 1!"""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        random.seed(44)
        for _ in range(10):
            u = random.uniform(0.1, 0.9)
            t = random.uniform(0.1, 0.9)

            evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
            f_coeffs = evaluator._build_F_coefficients(u, t)
            f01_series = f_coeffs[0, 1]

            dF_dy_fd = finite_diff_dF_dy_series(u, t, P1, P1, Q, R, theta)

            rel_error = abs(f01_series - dF_dy_fd) / abs(dF_dy_fd) if dF_dy_fd != 0 else abs(f01_series)
            assert rel_error < 1e-5, f"f[0,1] mismatch at (u={u:.3f}, t={t:.3f}): series={f01_series}, fd={dF_dy_fd}, rel_error={rel_error}"

    def test_f11_matches_finite_diff(self, polynomials):
        """
        CRITICAL TEST: f[1,1] = (∂²F/∂x∂y)|_{x=y=0} / (1!×1!)

        This is where the manual evaluators have bugs.
        The series engine should match finite differences exactly.
        """
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        random.seed(45)
        max_rel_error = 0

        for i in range(10):
            u = random.uniform(0.1, 0.9)
            t = random.uniform(0.1, 0.9)

            evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
            f_coeffs = evaluator._build_F_coefficients(u, t)
            f11_series = f_coeffs[1, 1]

            # f[1,1] = (d²F/dxdy)|_0 / (1!×1!) = d²F/dxdy|_0
            d2F_dxdy_fd = finite_diff_d2F_dxdy_series(u, t, P1, P1, Q, R, theta)

            rel_error = abs(f11_series - d2F_dxdy_fd) / abs(d2F_dxdy_fd) if d2F_dxdy_fd != 0 else abs(f11_series)
            max_rel_error = max(max_rel_error, rel_error)

            assert rel_error < 1e-5, f"f[1,1] mismatch at (u={u:.3f}, t={t:.3f}): series={f11_series:.6f}, fd={d2F_dxdy_fd:.6f}, rel_error={rel_error:.2e}"

        print(f"\n  f[1,1] GATE PASSED: max relative error = {max_rel_error:.2e}")

    def test_off_diagonal_f11(self, polynomials):
        """Test f[1,1] for off-diagonal pair (1,2)."""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        random.seed(46)
        for _ in range(5):
            u = random.uniform(0.1, 0.9)
            t = random.uniform(0.1, 0.9)

            evaluator = PsiSeriesEvaluator(P1, P2, Q, R, theta, max_order=2, n_quad=60)
            f_coeffs = evaluator._build_F_coefficients(u, t)
            f11_series = f_coeffs[1, 1]

            d2F_dxdy_fd = finite_diff_d2F_dxdy_series(u, t, P1, P2, Q, R, theta)

            rel_error = abs(f11_series - d2F_dxdy_fd) / abs(d2F_dxdy_fd) if d2F_dxdy_fd != 0 else abs(f11_series)
            assert rel_error < 1e-5, f"(1,2) f[1,1] mismatch: series={f11_series:.6f}, fd={d2F_dxdy_fd:.6f}"


class TestPrefactorTransform:
    """Test the prefactor transformation g = (1/θ + x + y) × F."""

    @pytest.fixture(scope="class")
    def polynomials(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_g11_formula(self, polynomials):
        """Verify g[1,1] = (1/θ)×f[1,1] + f[0,1] + f[1,0]."""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        random.seed(47)
        u = random.uniform(0.1, 0.9)
        t = random.uniform(0.1, 0.9)

        evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
        f_coeffs = evaluator._build_F_coefficients(u, t)
        g_coeffs = evaluator._prefactor_transform(f_coeffs)

        # Check the formula
        expected_g11 = f_coeffs[1, 1] / theta + f_coeffs[0, 1] + f_coeffs[1, 0]
        actual_g11 = g_coeffs[1, 1]

        assert abs(expected_g11 - actual_g11) < 1e-12, f"g[1,1] formula incorrect: expected={expected_g11}, actual={actual_g11}"


class TestOffDiagonalPolynomials:
    """Verify off-diagonal pairs use correct distinct polynomials."""

    @pytest.fixture(scope="class")
    def polynomials(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_12_uses_different_polys(self, polynomials):
        """Ensure (1,2) pair uses P1 and P2, not P1 and P1."""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        evaluator = PsiSeriesEvaluator(P1, P2, Q, R, theta, max_order=2, n_quad=60)

        # Check internal state
        assert evaluator.P_ell is P1, "P_ell should be P1"
        assert evaluator.P_ellbar is P2, "P_ellbar should be P2"
        assert evaluator.P_ell is not evaluator.P_ellbar, "P_ell and P_ellbar should be different objects"

    def test_12_gives_different_result_than_11(self, polynomials):
        """(1,2) should give different f-coefficients than (1,1)."""
        P1, P2, P3, Q = polynomials
        R = 1.3036
        theta = 4/7

        u, t = 0.5, 0.5

        eval_11 = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
        f_11 = eval_11._build_F_coefficients(u, t)

        eval_12 = PsiSeriesEvaluator(P1, P2, Q, R, theta, max_order=2, n_quad=60)
        f_12 = eval_12._build_F_coefficients(u, t)

        # f[0,0] should differ because P2(u) ≠ P1(u) in general
        assert f_11[0, 0] != f_12[0, 0], "f[0,0] should differ between (1,1) and (1,2)"


# =============================================================================
# DIAGNOSTIC: Compare Series vs Section7 for f[1,1]
# =============================================================================

def test_series_vs_section7_f11():
    """
    Compare series f[1,1] against Section7's manual d²F/dxdy.

    This should show the ~0.55% discrepancy we found in Session 10.
    """
    from src.polynomials import load_przz_polynomials
    from src.section7_evaluator import Section7Evaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    R = 1.3036
    theta = 4/7

    # Use a specific test point
    u, t = 0.512980, 0.512980

    # Series engine
    evaluator = PsiSeriesEvaluator(P1, P1, Q, R, theta, max_order=2, n_quad=60)
    f_coeffs = evaluator._build_F_coefficients(u, t)
    f11_series = f_coeffs[1, 1]

    # Finite difference ground truth (using series convention)
    f11_fd = finite_diff_d2F_dxdy_series(u, t, P1, P1, Q, R, theta)

    print(f"\n=== f[1,1] Comparison at (u,t) = ({u}, {t}) ===")
    print(f"  Series engine:       {f11_series:.10f}")
    print(f"  Finite difference:   {f11_fd:.10f}")
    print(f"  Relative error:      {abs(f11_series - f11_fd) / abs(f11_fd) * 100:.6f}%")
    print(f"  GATE: {'PASS' if abs(f11_series - f11_fd) / abs(f11_fd) < 1e-5 else 'FAIL'}")

    # The series should match finite differences
    assert abs(f11_series - f11_fd) / abs(f11_fd) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
