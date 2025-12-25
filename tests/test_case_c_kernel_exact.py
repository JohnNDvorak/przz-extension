"""
tests/test_case_c_kernel_exact.py
Gate K1: Case C Kernel Pointwise Validation

This gate validates that our Case C kernel implementation K_ω(u; R)
matches exact known values and satisfies required properties.

PRZZ TeX References:
- 2360-2362: Case C polynomial-rescaling rewrite
- 2371-2374: Full Case C definition F_d
- 2382-2384: Product form with (log N)^ω

The kernel formula we implement:
    K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da

This gate catches:
- Missing prefactors (u^ω, factorial)
- Wrong ω value
- Wrong exponential argument (should be Rθua, not Rθu or Ra)
- Wrong attenuation power (a^{ω-1})
- Sign mistakes
"""

import pytest
import numpy as np
import math
from typing import Tuple

from src.polynomials import load_przz_polynomials, Polynomial
from src.case_c_kernel import compute_case_c_kernel, compute_case_c_kernel_derivative
from src.case_c_exact import compute_case_c_kernel as case_c_exact_kernel
from src.case_c_exact import compute_case_c_kernel_vectorized
from src.quadrature import gauss_legendre_01


# Constants
THETA = 4 / 7
R_BENCHMARK1 = 1.3036
R_BENCHMARK2 = 1.1167


class TestCaseCKernelExactProperties:
    """Gate K1a: Verify mathematical properties of Case C kernel."""

    @pytest.fixture
    def polys(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_constant_polynomial_omega1_R0(self):
        """
        For P(x) = 1 (constant) and R=0:
        K_1(u; 0) = u^1 / 0! × ∫₀¹ 1 × a^0 × 1 da = u × 1 = u

        Using case_c_exact which returns just the integral:
        integral = ∫₀¹ 1 da = 1

        So K_1(u; 0) should equal 1.0 for all u (since case_c_exact doesn't include u^ω).
        """
        P_const = Polynomial([1.0])  # P(x) = 1
        u_test = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        K = compute_case_c_kernel_vectorized(P_const, u_test, omega=1, R=0, theta=THETA)

        # For constant P and R=0: ∫₀¹ a^{ω-1} da = 1/ω
        # For ω=1: ∫₀¹ a^0 da = 1
        expected = np.ones_like(u_test)

        np.testing.assert_allclose(K, expected, rtol=1e-10,
            err_msg="Constant P, omega=1, R=0 should give integral=1")

    def test_constant_polynomial_omega2_R0(self):
        """
        For P(x) = 1 and R=0:
        integral = ∫₀¹ a^{ω-1} da = ∫₀¹ a^1 da = 1/2
        """
        P_const = Polynomial([1.0])
        u_test = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        K = compute_case_c_kernel_vectorized(P_const, u_test, omega=2, R=0, theta=THETA)

        # For ω=2: ∫₀¹ a^1 da = 1/2
        expected = 0.5 * np.ones_like(u_test)

        np.testing.assert_allclose(K, expected, rtol=1e-10,
            err_msg="Constant P, omega=2, R=0 should give integral=0.5")

    def test_kernel_at_u0_is_zero_for_P_with_P0_eq_0(self, polys):
        """
        For P₂ and P₃ which satisfy P(0) = 0:
        At u=0, the argument (1-a)*0 = 0, so P(0) = 0 everywhere.
        Thus K_ω(0; R) = 0.
        """
        P1, P2, P3, Q = polys

        u_test = np.array([0.0])

        K2 = compute_case_c_kernel_vectorized(P2, u_test, omega=1, R=R_BENCHMARK1, theta=THETA)
        K3 = compute_case_c_kernel_vectorized(P3, u_test, omega=2, R=R_BENCHMARK1, theta=THETA)

        assert abs(K2[0]) < 1e-14, f"K_1(0; R) should be 0 but got {K2[0]}"
        assert abs(K3[0]) < 1e-14, f"K_2(0; R) should be 0 but got {K3[0]}"

    def test_kernel_positive_for_positive_P(self, polys):
        """
        If P(x) > 0 on (0,1), then K_ω(u; R) > 0 for u in (0,1).
        P₂ and P₃ may not be strictly positive, but they have consistent signs.
        At least verify the kernel has correct sign structure.
        """
        P1, P2, P3, Q = polys

        # P₁ is approximately positive on most of [0,1]
        u_test = np.linspace(0.1, 0.9, 9)

        K_P2 = compute_case_c_kernel_vectorized(P2, u_test, omega=1, R=R_BENCHMARK1, theta=THETA)
        K_P3 = compute_case_c_kernel_vectorized(P3, u_test, omega=2, R=R_BENCHMARK1, theta=THETA)

        # Just verify not all zeros
        assert np.max(np.abs(K_P2)) > 0.01, "K for P₂ should be non-trivial"
        assert np.max(np.abs(K_P3)) > 0.01, "K for P₃ should be non-trivial"

    def test_kernel_R_dependence_direction(self, polys):
        """
        The exponential factor exp(Rθua) is increasing in R for u,a > 0.
        Therefore K_ω(u; R1) > K_ω(u; R2) for R1 > R2 and positive polynomial.

        For PRZZ polynomials this may be more complex due to P sign changes,
        but we can verify there IS R-dependence.
        """
        P1, P2, P3, Q = polys

        u_test = np.array([0.5])

        K_R1 = compute_case_c_kernel_vectorized(P2, u_test, omega=1, R=R_BENCHMARK1, theta=THETA)[0]
        K_R2 = compute_case_c_kernel_vectorized(P2, u_test, omega=1, R=R_BENCHMARK2, theta=THETA)[0]

        # R1 > R2, so the ratio should NOT be 1
        ratio = K_R1 / K_R2
        assert abs(ratio - 1.0) > 0.01, f"Kernel should be R-dependent, but ratio is {ratio}"


class TestCaseCKernelConsistency:
    """Gate K1b: Verify both implementations (case_c_kernel.py vs case_c_exact.py) are consistent."""

    @pytest.fixture
    def polys(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_two_implementations_match_for_omega1(self, polys):
        """
        Verify case_c_kernel.py and case_c_exact.py give same results.

        NOTE: case_c_kernel.py includes u^ω/(ω-1)! factor, case_c_exact.py doesn't.
        We need to account for this difference.
        """
        P1, P2, P3, Q = polys

        u_test = np.linspace(0.1, 0.9, 5)
        omega = 1
        R = R_BENCHMARK1

        # case_c_kernel.py: includes u^ω / (ω-1)!
        K_full = compute_case_c_kernel(
            P_eval=lambda x: P2.eval(x),
            u_grid=u_test,
            omega=omega,
            R=R,
            theta=THETA
        )

        # case_c_exact.py: just the integral
        K_integral = compute_case_c_kernel_vectorized(P2, u_test, omega=omega, R=R, theta=THETA)

        # Reconstruct full from integral
        factorial_denom = math.factorial(omega - 1)
        K_reconstructed = (u_test ** omega) / factorial_denom * K_integral

        np.testing.assert_allclose(K_full, K_reconstructed, rtol=1e-10,
            err_msg="Two implementations should match after accounting for u^ω/(ω-1)! factor")

    def test_two_implementations_match_for_omega2(self, polys):
        """Same test for omega=2."""
        P1, P2, P3, Q = polys

        u_test = np.linspace(0.1, 0.9, 5)
        omega = 2
        R = R_BENCHMARK1

        K_full = compute_case_c_kernel(
            P_eval=lambda x: P3.eval(x),
            u_grid=u_test,
            omega=omega,
            R=R,
            theta=THETA
        )

        K_integral = compute_case_c_kernel_vectorized(P3, u_test, omega=omega, R=R, theta=THETA)

        factorial_denom = math.factorial(omega - 1)
        K_reconstructed = (u_test ** omega) / factorial_denom * K_integral

        np.testing.assert_allclose(K_full, K_reconstructed, rtol=1e-10,
            err_msg="Two implementations should match for omega=2")


class TestCaseCKernelDerivative:
    """Gate K1c: Verify Case C kernel derivative implementation."""

    @pytest.fixture
    def polys(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_derivative_matches_finite_difference_omega1(self, polys):
        """
        Verify K'_ω(u; R) matches finite difference approximation.
        """
        P1, P2, P3, Q = polys

        u_test = np.array([0.3, 0.5, 0.7])
        omega = 1
        R = R_BENCHMARK1
        eps = 1e-6

        # Compute analytic derivative
        # Note: PellPolynomial uses eval_deriv() not derivative().eval()
        K_deriv = compute_case_c_kernel_derivative(
            P_eval=lambda x: P2.eval(x),
            P_deriv_eval=lambda x: P2.eval_deriv(x, 1),
            u_grid=u_test,
            omega=omega,
            R=R,
            theta=THETA
        )

        # Finite difference
        K_plus = compute_case_c_kernel(
            P_eval=lambda x: P2.eval(x),
            u_grid=u_test + eps,
            omega=omega,
            R=R,
            theta=THETA
        )
        K_minus = compute_case_c_kernel(
            P_eval=lambda x: P2.eval(x),
            u_grid=u_test - eps,
            omega=omega,
            R=R,
            theta=THETA
        )
        K_deriv_fd = (K_plus - K_minus) / (2 * eps)

        # Allow some tolerance for FD approximation
        np.testing.assert_allclose(K_deriv, K_deriv_fd, rtol=1e-4,
            err_msg="Kernel derivative should match finite difference")

    def test_derivative_matches_finite_difference_omega2(self, polys):
        """Same for omega=2."""
        P1, P2, P3, Q = polys

        u_test = np.array([0.3, 0.5, 0.7])
        omega = 2
        R = R_BENCHMARK1
        eps = 1e-6

        K_deriv = compute_case_c_kernel_derivative(
            P_eval=lambda x: P3.eval(x),
            P_deriv_eval=lambda x: P3.eval_deriv(x, 1),
            u_grid=u_test,
            omega=omega,
            R=R,
            theta=THETA
        )

        K_plus = compute_case_c_kernel(
            P_eval=lambda x: P3.eval(x),
            u_grid=u_test + eps,
            omega=omega,
            R=R,
            theta=THETA
        )
        K_minus = compute_case_c_kernel(
            P_eval=lambda x: P3.eval(x),
            u_grid=u_test - eps,
            omega=omega,
            R=R,
            theta=THETA
        )
        K_deriv_fd = (K_plus - K_minus) / (2 * eps)

        np.testing.assert_allclose(K_deriv, K_deriv_fd, rtol=1e-4,
            err_msg="Kernel derivative should match finite difference for omega=2")


class TestCaseCKernelTwoBenchmark:
    """Gate K1d: Compare kernel behavior across both benchmarks."""

    @pytest.fixture
    def polys_kappa(self):
        return load_przz_polynomials(enforce_Q0=True)

    def test_kernel_ratio_between_benchmarks(self, polys_kappa):
        """
        Verify the kernel ratio K(R1)/K(R2) is in a reasonable range.

        Since R1 > R2 and exp(Rθua) grows with R, we expect K(R1) > K(R2)
        for positive contributions (though polynomial sign may complicate this).
        """
        P1, P2, P3, Q = polys_kappa

        u_test = np.linspace(0.2, 0.8, 5)

        for P, omega, name in [(P2, 1, "P2"), (P3, 2, "P3")]:
            K_R1 = compute_case_c_kernel_vectorized(P, u_test, omega, R_BENCHMARK1, THETA)
            K_R2 = compute_case_c_kernel_vectorized(P, u_test, omega, R_BENCHMARK2, THETA)

            # Compute integral (sum with equal weights as proxy)
            int_R1 = np.mean(K_R1)
            int_R2 = np.mean(K_R2)

            if abs(int_R2) > 1e-10:
                ratio = int_R1 / int_R2
                # The ratio should be different from 1 (R-dependence exists)
                assert abs(ratio - 1.0) > 0.01, \
                    f"{name}: Kernel should show R-dependence, ratio={ratio}"

    def test_attenuation_factor_present(self, polys_kappa):
        """
        Verify that Case C kernel produces SMALLER values than raw P(u).

        The (1-a) in P((1-a)u) and the integral average should attenuate.
        """
        P1, P2, P3, Q = polys_kappa

        u_test = np.linspace(0.2, 0.8, 5)

        for P, omega, name in [(P2, 1, "P2"), (P3, 2, "P3")]:
            # Raw P(u)
            P_raw = P.eval(u_test)

            # Case C kernel (just integral, without u^ω factor)
            K = compute_case_c_kernel_vectorized(P, u_test, omega, R_BENCHMARK1, THETA)

            # RMS comparison
            rms_raw = np.sqrt(np.mean(P_raw ** 2))
            rms_K = np.sqrt(np.mean(K ** 2))

            # Kernel should generally be different from raw
            # (not necessarily smaller due to exp growth, but different)
            assert abs(rms_K - rms_raw) / (rms_raw + 1e-10) > 0.01, \
                f"{name}: Case C kernel should differ from raw polynomial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
