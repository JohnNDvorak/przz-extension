"""
tests/test_operator_post_identity_QQexp_coeffs.py
Coefficient-level validation of Q(Aα)Q(Aβ)exp(...) against ANALYTIC formulas.

This is the SKEPTIC-KILLER test file: validates that the series engine
produces the same coefficients as closed-form analytic computation.

GPT Guidance Phase 1, Step 2:
- Compute series coefficients {c00, cx, cy, cxy} via composition engine
- Compute same coefficients via analytic formulas (Q derivatives + nilpotent algebra)
- Assert they match to machine precision (1e-12)

Key insight: This tests the series engine independently of the I1 integration,
ensuring that any discrepancy is localized.
"""

import pytest
import numpy as np
from src.operator_post_identity import (
    apply_QQexp_post_identity_composition,
    compute_analytic_QQexp_coeffs,
)
from src.polynomials import load_przz_polynomials


# Tolerance for analytic comparison
# These should match to machine precision
ANALYTIC_TOL = 1e-12


class TestQQExpCoefficients:
    """
    Compare series engine coefficients against analytic computation.

    The series engine uses compose_polynomial_on_affine and compose_exp_on_affine.
    The analytic computation uses Q.eval_deriv() and explicit nilpotent algebra.

    These should match EXACTLY because both are computing the same thing
    through different code paths.
    """

    @pytest.fixture
    def Q_poly(self):
        """Load Q polynomial."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return Q

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("R", [1.3036, 1.1167])
    def test_coefficients_match_analytic(self, t, R, Q_poly, theta):
        """
        Series engine coefficients should EXACTLY match analytic computation.

        This is the key skeptic-killer test: proves the series engine is correct
        by comparing against an independent analytic derivation.
        """
        var_names = ("x", "y")

        # Compute via series engine
        core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
        c00 = core_series.extract(())
        cx = core_series.extract(("x",))
        cy = core_series.extract(("y",))
        cxy = core_series.extract(("x", "y"))

        # Compute analytically
        C00, Cx, Cy, Cxy = compute_analytic_QQexp_coeffs(Q_poly, t, theta, R)

        # Assert exact match
        assert abs(c00 - C00) < ANALYTIC_TOL, \
            f"c00 mismatch at t={t}, R={R}: series={c00:.12f}, analytic={C00:.12f}, diff={abs(c00-C00):.2e}"
        assert abs(cx - Cx) < ANALYTIC_TOL, \
            f"cx mismatch at t={t}, R={R}: series={cx:.12f}, analytic={Cx:.12f}, diff={abs(cx-Cx):.2e}"
        assert abs(cy - Cy) < ANALYTIC_TOL, \
            f"cy mismatch at t={t}, R={R}: series={cy:.12f}, analytic={Cy:.12f}, diff={abs(cy-Cy):.2e}"
        assert abs(cxy - Cxy) < ANALYTIC_TOL, \
            f"cxy mismatch at t={t}, R={R}: series={cxy:.12f}, analytic={Cxy:.12f}, diff={abs(cxy-Cxy):.2e}"

    def test_symmetry_cx_eq_cy(self, Q_poly, theta):
        """
        At t=0.5, the coefficients cx and cy should be equal by symmetry.

        This is because at t=0.5:
            ax_alpha = θ(t-1) = θ(-0.5) = -θ/2
            ay_alpha = θt = θ/2
            ax_beta = θt = θ/2
            ay_beta = θ(t-1) = -θ/2

        So Q(A_α) and Q(A_β) are swapped, but the product is symmetric.
        """
        t = 0.5
        R = 1.3036
        var_names = ("x", "y")

        core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
        cx = core_series.extract(("x",))
        cy = core_series.extract(("y",))

        assert abs(cx - cy) < ANALYTIC_TOL, \
            f"cx != cy at t=0.5: cx={cx:.12f}, cy={cy:.12f}, diff={abs(cx-cy):.2e}"

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_c00_positive(self, t, Q_poly, theta):
        """
        The constant coefficient c00 should always be positive.

        c00 = Q(t)² × exp(2Rt) > 0 for all t.
        """
        R = 1.3036
        var_names = ("x", "y")

        core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
        c00 = core_series.extract(())

        assert c00 > 0, f"c00 should be positive at t={t}: c00={c00}"

    def test_coefficients_vary_with_t(self, Q_poly, theta):
        """
        Coefficients should change as t varies.
        """
        R = 1.3036
        var_names = ("x", "y")

        t_values = [0.2, 0.5, 0.8]
        c00_values = []
        cxy_values = []

        for t in t_values:
            core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
            c00_values.append(core_series.extract(()))
            cxy_values.append(core_series.extract(("x", "y")))

        # All should be different
        assert len(set(f"{v:.8f}" for v in c00_values)) == 3, \
            f"c00 not varying with t: {c00_values}"
        assert len(set(f"{v:.8f}" for v in cxy_values)) == 3, \
            f"cxy not varying with t: {cxy_values}"

    def test_coefficients_vary_with_R(self, Q_poly, theta):
        """
        Coefficients should change as R varies.
        """
        t = 0.5
        var_names = ("x", "y")

        R_values = [1.0, 1.3036, 1.5]
        c00_values = []
        cxy_values = []

        for R in R_values:
            core_series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)
            c00_values.append(core_series.extract(()))
            cxy_values.append(core_series.extract(("x", "y")))

        # All should be different
        assert len(set(f"{v:.8f}" for v in c00_values)) == 3, \
            f"c00 not varying with R: {c00_values}"
        assert len(set(f"{v:.8f}" for v in cxy_values)) == 3, \
            f"cxy not varying with R: {cxy_values}"


class TestAnalyticFormulas:
    """
    Direct tests of the analytic formula correctness.
    """

    @pytest.fixture
    def Q_poly(self):
        """Load Q polynomial."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return Q

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_Q_derivatives_at_endpoints(self, Q_poly):
        """
        Q and its derivatives should be well-defined at t=0 and t=1.
        """
        for t in [0.0, 1.0]:
            t_arr = np.array([t])
            Q0 = Q_poly.eval(t_arr)[0]
            Q1 = Q_poly.eval_deriv(t_arr, 1)[0]
            Q2 = Q_poly.eval_deriv(t_arr, 2)[0]

            assert np.isfinite(Q0), f"Q({t}) not finite"
            assert np.isfinite(Q1), f"Q'({t}) not finite"
            assert np.isfinite(Q2), f"Q''({t}) not finite"

    def test_analytic_at_t0(self, Q_poly, theta):
        """
        At t=0: A_α = -θx, A_β = -θy (pure nilpotent).
        """
        t = 0.0
        R = 1.3036

        C00, Cx, Cy, Cxy = compute_analytic_QQexp_coeffs(Q_poly, t, theta, R)

        # At t=0: exp(2Rt) = exp(0) = 1
        # So c00 = Q(0)² = 1 (since Q(0) = 1 with enforce_Q0=True)
        assert abs(C00 - 1.0) < 1e-10, f"C00 at t=0 should be ~1: {C00}"

        # At t=0: b = R(2θ·0 - θ) = -Rθ
        # cx = c00·(-Rθ) + QQx·1 where QQx = 2Q(0)·Q'(0)·(-θ)
        # cy = c00·(-Rθ) + QQy·1 where QQy = 2Q(0)·Q'(0)·(-θ)
        # So cx = cy at t=0
        assert abs(Cx - Cy) < 1e-10, f"Cx != Cy at t=0: Cx={Cx}, Cy={Cy}"

    def test_analytic_at_t1(self, Q_poly, theta):
        """
        At t=1: A_α = 1 + θy, A_β = 1 + θx (shifted by 1).

        Note: Q(1) ≈ -0.019 is small, so C00 = Q(1)² × exp(2R) is small.
        """
        t = 1.0
        R = 1.3036

        C00, Cx, Cy, Cxy = compute_analytic_QQexp_coeffs(Q_poly, t, theta, R)

        # At t=1: Q(1) ≈ -0.019, so Q(1)² ≈ 0.00036
        # C00 = Q(1)² × exp(2R) ≈ 0.00036 × 13.56 ≈ 0.005
        Q_at_1 = Q_poly.eval(np.array([t]))[0]
        expected_C00_approx = Q_at_1 ** 2 * np.exp(2 * R * t)
        assert abs(C00 - expected_C00_approx) < 1e-10, \
            f"C00 mismatch at t=1: {C00} vs {expected_C00_approx}"

        # At t=1: b = R(2θ - θ) = Rθ > 0
        # All coefficients should be finite
        assert np.isfinite(C00), f"C00 at t=1 not finite"
        assert np.isfinite(Cx), f"Cx at t=1 not finite"
        assert np.isfinite(Cy), f"Cy at t=1 not finite"
        assert np.isfinite(Cxy), f"Cxy at t=1 not finite"


class TestEdgeCases:
    """
    Edge cases and boundary conditions.
    """

    @pytest.fixture
    def Q_poly(self):
        """Load Q polynomial."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return Q

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_very_small_R(self, Q_poly, theta):
        """
        At very small R, exp(2Rt) ≈ 1 and b ≈ 0.
        """
        R = 0.01
        t = 0.5

        C00, Cx, Cy, Cxy = compute_analytic_QQexp_coeffs(Q_poly, t, theta, R)

        # C00 should be close to Q(0.5)²
        Q_at_half = Q_poly.eval(np.array([t]))[0]
        expected_C00 = Q_at_half ** 2 * np.exp(2 * R * t)

        assert abs(C00 - expected_C00) < 1e-10, \
            f"C00 mismatch at small R: {C00} vs {expected_C00}"

    def test_large_R(self, Q_poly, theta):
        """
        At large R, the exp factor dominates.

        At t=0.5: exp(2Rt) = exp(5) ≈ 148
        C00 = Q(0.5)² × exp(5) ≈ 0.24 × 148 ≈ 35
        """
        R = 5.0
        t = 0.5

        C00, Cx, Cy, Cxy = compute_analytic_QQexp_coeffs(Q_poly, t, theta, R)

        # C00 should grow exponentially with R
        # Q(0.5)² ≈ 0.24, exp(2*5*0.5) = exp(5) ≈ 148
        # So C00 ≈ 35, not 100
        assert C00 > 10, f"C00 should be large at R=5: {C00}"
        assert np.isfinite(C00), "C00 should be finite"
        assert np.isfinite(Cxy), "Cxy should be finite"
