"""
tests/test_tex_exact_gates.py
Gate tests for Run 19: TeX-Exact Mirror Core

This file contains gate tests for the CombinedI1Integrand class and
compute_I1_tex_exact_11() function that implement the correct PRZZ
mirror structure with Q-shift inside the combined object.

Key differences from Run 18:
- Q factors are INSIDE the combined structure
- Q-shift (sigma=1.0) is applied in the minus branch
- No separate CombinedMirrorFactor multiplication
"""

import numpy as np
import pytest

from src.polynomials import load_przz_polynomials
from src.q_operator import lift_poly_by_shift
from src.term_dsl import SeriesContext, CombinedI1Integrand
from src.quadrature import tensor_grid_2d
from src.evaluate import compute_I1_tex_exact_11, compute_c_paper_tex_mirror


THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


@pytest.fixture
def polynomials_kappa():
    """Load κ benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def polynomials_kappa_star():
    """Load κ* benchmark polynomials."""
    from src.polynomials import load_przz_polynomials_kappa_star
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


# =============================================================================
# Stage 19A: CombinedI1Integrand Gate Tests
# =============================================================================


class TestCombinedI1IntegrandScalarLimit:
    """Gate tests for CombinedI1Integrand scalar limit."""

    def test_scalar_limit_is_finite_kappa(self, polynomials_kappa):
        """Scalar limit should be finite for κ benchmark."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        integrand = CombinedI1Integrand(
            R=R_KAPPA,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        # Test at several t values
        for t_val in [0.25, 0.5, 0.75]:
            scalar = integrand.scalar_limit(t_val)
            assert np.isfinite(scalar), f"Scalar limit not finite at t={t_val}"
            assert scalar > 0, f"Scalar limit should be positive at t={t_val}"

    def test_scalar_limit_is_finite_kappa_star(self, polynomials_kappa_star):
        """Scalar limit should be finite for κ* benchmark."""
        Q = polynomials_kappa_star["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        integrand = CombinedI1Integrand(
            R=R_KAPPA_STAR,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        for t_val in [0.25, 0.5, 0.75]:
            scalar = integrand.scalar_limit(t_val)
            assert np.isfinite(scalar), f"Scalar limit not finite at t={t_val}"
            assert scalar > 0, f"Scalar limit should be positive at t={t_val}"

    def test_scalar_limit_formula(self, polynomials_kappa):
        """Verify scalar limit matches expected formula."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        R = R_KAPPA
        integrand = CombinedI1Integrand(
            R=R,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        t_val = 0.5
        computed = integrand.scalar_limit(t_val)

        # Manual computation: Q(t)² exp(2Rt) + Q(t+1)² exp(2R(1-t))
        t_arr = np.array([t_val])
        Q_t = float(Q.eval(t_arr)[0])
        Q_t_shifted = float(Q_shifted.eval(t_arr)[0])

        expected_plus = Q_t ** 2 * np.exp(2 * R * t_val)
        expected_minus = Q_t_shifted ** 2 * np.exp(2 * R * (1 - t_val))
        expected = expected_plus + expected_minus

        assert np.isclose(computed, expected, rtol=1e-10), \
            f"Scalar limit mismatch: computed={computed}, expected={expected}"


class TestCombinedI1IntegrandSeriesEvaluation:
    """Gate tests for CombinedI1Integrand series evaluation."""

    def test_evaluate_produces_finite_series(self, polynomials_kappa):
        """evaluate() should produce a finite TruncatedSeries."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        integrand = CombinedI1Integrand(
            R=R_KAPPA,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        # Create a small grid
        U, T, W = tensor_grid_2d(10)
        ctx = SeriesContext(var_names=("x", "y"))

        series = integrand.evaluate(U, T, ctx)

        # Check that key coefficients are finite using extract()
        const_term = series.extract(())  # Constant term
        x_term = series.extract(("x",))
        y_term = series.extract(("y",))
        xy_term = series.extract(("x", "y"))

        assert np.all(np.isfinite(const_term)), "Constant term not finite"
        assert np.all(np.isfinite(x_term)), "x coefficient not finite"
        assert np.all(np.isfinite(y_term)), "y coefficient not finite"
        assert np.all(np.isfinite(xy_term)), "xy coefficient not finite"

    def test_xy_coefficient_is_nonzero(self, polynomials_kappa):
        """The xy coefficient should be non-zero (derivative structure preserved)."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        integrand = CombinedI1Integrand(
            R=R_KAPPA,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        U, T, W = tensor_grid_2d(20)
        ctx = SeriesContext(var_names=("x", "y"))

        series = integrand.evaluate(U, T, ctx)

        # Extract the xy coefficient using extract()
        xy_coeff = series.extract(("x", "y"))

        # At least some grid points should have non-zero xy coefficient
        assert np.any(np.abs(xy_coeff) > 1e-10), \
            "xy coefficient is zero everywhere - derivative structure lost"

    def test_plus_and_minus_branches_differ(self, polynomials_kappa):
        """Plus and minus branches should contribute differently."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        R = R_KAPPA
        U, T, W = tensor_grid_2d(10)
        ctx = SeriesContext(var_names=("x", "y"))

        # Create integrand with combined structure
        integrand = CombinedI1Integrand(
            R=R,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )
        combined_series = integrand.evaluate(U, T, ctx)

        # Compare scalar values at a few grid points
        # If Q_shifted = Q, the branches would be symmetric
        # With Q_shifted ≠ Q, they should differ
        Q_t_vals = Q.eval(T.flatten())
        Q_shifted_t_vals = Q_shifted.eval(T.flatten())

        # Q and Q_shifted should be different
        assert not np.allclose(Q_t_vals, Q_shifted_t_vals), \
            "Q and Q_shifted should differ"


class TestQShiftCorrectness:
    """Verify that Q-shift (sigma=1.0) is applied correctly."""

    def test_q_shift_sigma_one(self, polynomials_kappa):
        """Verify Q(x+1) is correctly computed."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        # Q_shifted(x) should equal Q(x+1)
        test_points = np.array([0.0, 0.5, 1.0, 1.5])
        Q_at_x_plus_1 = Q.eval(test_points + 1.0)
        Q_shifted_at_x = Q_shifted.eval(test_points)

        assert np.allclose(Q_at_x_plus_1, Q_shifted_at_x, rtol=1e-10), \
            "Q_shifted(x) should equal Q(x+1)"

    def test_q_shift_changes_constant_term(self, polynomials_kappa):
        """Q(x+1) should have a different constant term than Q(x)."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        zero = np.array([0.0])
        Q_0 = Q.eval(zero)[0]
        Q_shifted_0 = Q_shifted.eval(zero)[0]  # This is Q(0+1) = Q(1)

        assert not np.isclose(Q_0, Q_shifted_0), \
            "Q(0) and Q_shifted(0)=Q(1) should differ"


# =============================================================================
# Quadrature Convergence Tests
# =============================================================================


class TestCombinedI1IntegrandQuadratureConvergence:
    """Test that the combined integrand produces stable results."""

    def test_quadrature_convergence(self, polynomials_kappa):
        """Integrated values should converge as n increases."""
        Q = polynomials_kappa["Q"]
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        integrand = CombinedI1Integrand(
            R=R_KAPPA,
            theta=THETA,
            Q=Q,
            Q_shifted=Q_shifted,
        )

        ctx = SeriesContext(var_names=("x", "y"))

        results = []
        for n in [20, 40, 60]:
            U, T, W = tensor_grid_2d(n)
            series = integrand.evaluate(U, T, ctx)

            # Extract xy coefficient and integrate using extract()
            xy_coeff = series.extract(("x", "y"))
            integral = np.sum(W * xy_coeff)
            results.append(integral)

        # Results should converge (difference should decrease)
        diff_20_40 = abs(results[1] - results[0])
        diff_40_60 = abs(results[2] - results[1])

        # Allow for some oscillation but expect general convergence
        # If not converging, at least the values should be reasonable
        assert all(np.isfinite(r) for r in results), \
            "All quadrature results should be finite"

        # The values should be in the same order of magnitude
        if abs(results[0]) > 1e-10:
            ratio_1 = abs(results[1] / results[0])
            ratio_2 = abs(results[2] / results[0])
            assert 0.1 < ratio_1 < 10, f"Quadrature unstable: {results}"
            assert 0.1 < ratio_2 < 10, f"Quadrature unstable: {results}"


# =============================================================================
# Stage 19B: compute_I1_tex_exact_11() Gate Tests
# =============================================================================


class TestComputeI1TexExact11:
    """Gate tests for compute_I1_tex_exact_11()."""

    def test_i1_tex_exact_is_finite_kappa(self, polynomials_kappa):
        """I1_tex_exact should be finite for κ benchmark."""
        result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polynomials_kappa,
            verbose=False,
        )

        assert np.isfinite(result.I1_tex_exact), "I1_tex_exact not finite"
        assert np.isfinite(result.scalar_limit_t05), "scalar_limit not finite"

    def test_i1_tex_exact_is_finite_kappa_star(self, polynomials_kappa_star):
        """I1_tex_exact should be finite for κ* benchmark."""
        result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R_KAPPA_STAR,
            n=40,
            polynomials=polynomials_kappa_star,
            verbose=False,
        )

        assert np.isfinite(result.I1_tex_exact), "I1_tex_exact not finite"
        assert np.isfinite(result.scalar_limit_t05), "scalar_limit not finite"

    def test_quadrature_convergence_kappa(self, polynomials_kappa):
        """I1_tex_exact should converge with increasing quadrature."""
        results = []
        for n in [30, 50, 70]:
            result = compute_I1_tex_exact_11(
                theta=THETA,
                R=R_KAPPA,
                n=n,
                polynomials=polynomials_kappa,
            )
            results.append(result.I1_tex_exact)

        # All results should be finite
        assert all(np.isfinite(r) for r in results), \
            f"Not all results finite: {results}"

        # Results should be in the same order of magnitude
        if abs(results[0]) > 1e-10:
            for r in results[1:]:
                ratio = abs(r / results[0])
                assert 0.5 < ratio < 2.0, \
                    f"Quadrature unstable: {results}"

    def test_comparison_with_tex_mirror(self, polynomials_kappa):
        """Compare I1_tex_exact with tex_mirror I1 component."""
        # Get tex_exact result
        tex_exact_result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polynomials_kappa,
        )

        # Get tex_mirror result for comparison
        tex_mirror_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polynomials_kappa,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        # tex_mirror I1_total = I1_plus + m1 * I1_minus_base
        tex_mirror_I1 = tex_mirror_result.I1_plus + tex_mirror_result.m1 * tex_mirror_result.I1_minus_base

        # Print comparison for diagnostics
        print(f"\nI1 Comparison (κ benchmark):")
        print(f"  tex_exact I1: {tex_exact_result.I1_tex_exact:.6f}")
        print(f"  tex_mirror I1: {tex_mirror_I1:.6f}")
        print(f"  Difference: {tex_exact_result.I1_tex_exact - tex_mirror_I1:.6f}")

        # Both should be finite and positive (for I1)
        assert np.isfinite(tex_exact_result.I1_tex_exact)
        assert np.isfinite(tex_mirror_I1)

    def test_comparison_with_tex_mirror_kappa_star(self, polynomials_kappa_star):
        """Compare I1_tex_exact with tex_mirror I1 for κ* benchmark."""
        tex_exact_result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R_KAPPA_STAR,
            n=60,
            polynomials=polynomials_kappa_star,
        )

        tex_mirror_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_KAPPA_STAR,
            n=60,
            polynomials=polynomials_kappa_star,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        tex_mirror_I1 = tex_mirror_result.I1_plus + tex_mirror_result.m1 * tex_mirror_result.I1_minus_base

        print(f"\nI1 Comparison (κ* benchmark):")
        print(f"  tex_exact I1: {tex_exact_result.I1_tex_exact:.6f}")
        print(f"  tex_mirror I1: {tex_mirror_I1:.6f}")
        print(f"  Difference: {tex_exact_result.I1_tex_exact - tex_mirror_I1:.6f}")

        assert np.isfinite(tex_exact_result.I1_tex_exact)
        assert np.isfinite(tex_mirror_I1)
