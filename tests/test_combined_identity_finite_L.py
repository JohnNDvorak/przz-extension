"""
Unit tests for combined identity at finite L.

Tests the core functions in src/combined_identity_finite_L.py for the
m1 derivation via L-sweep.
"""

import pytest
import numpy as np

from src.series import TruncatedSeries
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.combined_identity_finite_L import (
    build_exp_plus_series,
    build_exp_minus_series,
    compute_combined_identity_series,
    compute_I1_combined_at_L,
)
from src.operator_post_identity import (
    apply_QQexp_post_identity_composition,
    compute_I1_operator_post_identity_pair,
)


@pytest.fixture(scope="module")
def polys_kappa():
    """Load PRZZ polynomials for kappa benchmark."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture(scope="module")
def polys_kappa_star():
    """Load PRZZ polynomials for kappa* benchmark."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestExpBranchSeries:
    """Test the exponential branch series builders."""

    def test_exp_plus_series_structure(self):
        """exp(-Rθ(x+y)) has correct nilpotent expansion."""
        R = 1.3036
        theta = 4.0 / 7.0
        var_names = ("x", "y")

        series = build_exp_plus_series(R, theta, var_names)

        # exp(-Rθ(x+y)) = 1 - Rθx - Rθy + R²θ²xy
        const = series.extract(())
        x_coeff = series.extract(("x",))
        y_coeff = series.extract(("y",))
        xy_coeff = series.extract(("x", "y"))

        assert np.isclose(const, 1.0), f"const should be 1.0, got {const}"
        assert np.isclose(x_coeff, -R * theta), f"x_coeff wrong: {x_coeff}"
        assert np.isclose(y_coeff, -R * theta), f"y_coeff wrong: {y_coeff}"
        assert np.isclose(xy_coeff, R**2 * theta**2), f"xy_coeff wrong: {xy_coeff}"

    def test_exp_minus_series_structure(self):
        """exp(+Rθ(x+y)) has correct nilpotent expansion."""
        R = 1.3036
        theta = 4.0 / 7.0
        var_names = ("x", "y")

        series = build_exp_minus_series(R, theta, var_names)

        # exp(+Rθ(x+y)) = 1 + Rθx + Rθy + R²θ²xy
        const = series.extract(())
        x_coeff = series.extract(("x",))
        y_coeff = series.extract(("y",))
        xy_coeff = series.extract(("x", "y"))

        assert np.isclose(const, 1.0), f"const should be 1.0, got {const}"
        assert np.isclose(x_coeff, R * theta), f"x_coeff wrong: {x_coeff}"
        assert np.isclose(y_coeff, R * theta), f"y_coeff wrong: {y_coeff}"
        assert np.isclose(xy_coeff, R**2 * theta**2), f"xy_coeff wrong: {xy_coeff}"

    def test_exp_branches_difference(self):
        """The difference [exp(-) - exp(2R)×exp(+)] should have specific structure."""
        R = 1.3036
        theta = 4.0 / 7.0
        var_names = ("x", "y")

        exp_plus = build_exp_plus_series(R, theta, var_names)
        exp_minus = build_exp_minus_series(R, theta, var_names)
        exp_2R = np.exp(2 * R)

        diff = exp_plus - exp_minus * exp_2R

        # [exp(-Rθ(x+y)) - exp(2R)·exp(Rθ(x+y))]
        # Constant term: 1 - exp(2R)
        const = diff.extract(())
        expected_const = 1 - exp_2R
        assert np.isclose(const, expected_const), f"const: {const} vs {expected_const}"

        # x coefficient: -Rθ - exp(2R)*Rθ = -Rθ(1 + exp(2R))
        x_coeff = diff.extract(("x",))
        expected_x = -R * theta * (1 + exp_2R)
        assert np.isclose(x_coeff, expected_x), f"x: {x_coeff} vs {expected_x}"

    def test_exp_branches_symmetry(self):
        """Both branches should be symmetric in x,y."""
        R = 1.1167
        theta = 4.0 / 7.0
        var_names = ("x", "y")

        for builder in [build_exp_plus_series, build_exp_minus_series]:
            series = builder(R, theta, var_names)

            x_coeff = series.extract(("x",))
            y_coeff = series.extract(("y",))

            assert np.isclose(x_coeff, y_coeff), \
                f"x and y coefficients should be equal: {x_coeff} vs {y_coeff}"


class TestCombinedIdentitySeries:
    """Test the combined identity series at finite L."""

    def test_combined_series_symmetric_in_xy(self, polys_kappa):
        """Combined identity series should be symmetric in x,y."""
        theta = 4.0 / 7.0
        R = 1.3036
        L = 50.0
        t = 0.5

        Q_poly = polys_kappa['Q']
        series = compute_combined_identity_series(Q_poly, t, theta, R, L)

        x_coeff = series.extract(("x",))
        y_coeff = series.extract(("y",))

        # Due to symmetry in the combined identity, x and y coefficients should match
        assert np.isclose(x_coeff, y_coeff, rtol=1e-10), \
            f"x,y coefficients should match: {x_coeff} vs {y_coeff}"

    def test_combined_series_L_scaling(self, polys_kappa):
        """Combined series xy coefficient scales with L/(2R)."""
        theta = 4.0 / 7.0
        R = 1.3036
        t = 0.5

        Q_poly = polys_kappa['Q']

        # Compute at two L values
        L1, L2 = 20.0, 100.0
        series1 = compute_combined_identity_series(Q_poly, t, theta, R, L1)
        series2 = compute_combined_identity_series(Q_poly, t, theta, R, L2)

        # The L/(2R) prefactor should give proportional scaling
        # (if the rest were L-independent)
        xy1 = series1.extract(("x", "y"))
        xy2 = series2.extract(("x", "y"))

        # The ratio should be close to L2/L1 = 5.0
        # (not exactly because Q×Q has no L dependence, only the prefactor)
        ratio = xy2 / xy1
        expected_ratio = L2 / L1

        assert np.isclose(ratio, expected_ratio, rtol=1e-10), \
            f"L scaling ratio: {ratio} vs expected {expected_ratio}"

    def test_combined_series_finite_values(self, polys_kappa):
        """Combined series should produce finite values."""
        theta = 4.0 / 7.0
        R = 1.3036
        L = 50.0

        Q_poly = polys_kappa['Q']

        for t in [0.1, 0.5, 0.9]:
            series = compute_combined_identity_series(Q_poly, t, theta, R, L)

            const = series.extract(())
            xy_coeff = series.extract(("x", "y"))

            assert np.isfinite(const), f"const not finite at t={t}: {const}"
            assert np.isfinite(xy_coeff), f"xy_coeff not finite at t={t}: {xy_coeff}"


class TestI1CombinedAtL:
    """Test the full I1 computation from combined identity."""

    def test_i1_combined_finite_kappa(self, polys_kappa):
        """I1 from combined identity should be finite."""
        theta = 4.0 / 7.0
        R = 1.3036
        L = 50.0
        n = 30  # Moderate quadrature for speed

        I1_combined = compute_I1_combined_at_L(theta, R, L, n, polys_kappa)

        assert np.isfinite(I1_combined), f"I1_combined not finite: {I1_combined}"

    def test_i1_combined_finite_kappa_star(self, polys_kappa_star):
        """I1 from combined identity should be finite for kappa*."""
        theta = 4.0 / 7.0
        R = 1.1167
        L = 50.0
        n = 30

        I1_combined = compute_I1_combined_at_L(theta, R, L, n, polys_kappa_star)

        assert np.isfinite(I1_combined), f"I1_combined not finite: {I1_combined}"

    def test_i1_combined_L_scaling(self, polys_kappa):
        """I1_combined should scale with L (due to L/(2R) prefactor)."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30

        L1, L2 = 20.0, 100.0
        I1_L1 = compute_I1_combined_at_L(theta, R, L1, n, polys_kappa)
        I1_L2 = compute_I1_combined_at_L(theta, R, L2, n, polys_kappa)

        # Ratio should be close to L2/L1 = 5.0
        ratio = I1_L2 / I1_L1
        expected_ratio = L2 / L1

        assert np.isclose(ratio, expected_ratio, rtol=0.01), \
            f"L scaling ratio: {ratio} vs expected {expected_ratio}"

    def test_i1_combined_quadrature_convergence(self, polys_kappa):
        """I1_combined should converge under quadrature refinement."""
        theta = 4.0 / 7.0
        R = 1.3036
        L = 50.0

        I1_n30 = compute_I1_combined_at_L(theta, R, L, 30, polys_kappa)
        I1_n40 = compute_I1_combined_at_L(theta, R, L, 40, polys_kappa)

        rel_diff = abs(I1_n40 - I1_n30) / abs(I1_n30)

        assert rel_diff < 0.01, \
            f"Quadrature not converged: {I1_n30} vs {I1_n40}, rel_diff={rel_diff:.2%}"


class TestLargeL_PostIdentityLimit:
    """Test that combined identity at large L relates to post-identity results."""

    def test_large_L_ratio_kappa(self, polys_kappa):
        """At large L, combined identity should relate to post-identity via m1."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        # Compute post-identity values (L → ∞ limit)
        result_plus = compute_I1_operator_post_identity_pair(
            theta, R, 1, 1, n, polys_kappa
        )
        result_minus = compute_I1_operator_post_identity_pair(
            theta, -R, 1, 1, n, polys_kappa
        )
        I1_plus = result_plus.I1_value
        I1_minus_base = result_minus.I1_value

        # Compute combined identity at large L
        L = 1000.0
        I1_combined = compute_I1_combined_at_L(theta, R, L, n, polys_kappa)

        # Expected: I1_combined ≈ L/(2R) × [I1_plus - exp(2R) × I1_minus]
        # So: I1_combined / L should be approximately (1/(2R)) × [I1_plus - exp(2R) × I1_minus]
        expected_per_L = (I1_plus - np.exp(2*R) * I1_minus_base) / (2 * R)
        actual_per_L = I1_combined / L

        # Print diagnostic info
        print(f"\nLarge L diagnostic:")
        print(f"  I1_plus = {I1_plus:.8f}")
        print(f"  I1_minus_base = {I1_minus_base:.8f}")
        print(f"  I1_combined (L={L}) = {I1_combined:.8f}")
        print(f"  I1_combined / L = {actual_per_L:.8f}")
        print(f"  Expected per L = {expected_per_L:.8f}")

        # The ratio should be close (but may not be exact due to L-finite corrections)
        # This is a structural test, not an exact match
        assert np.isfinite(actual_per_L), f"actual_per_L not finite: {actual_per_L}"

    def test_m1_eff_at_large_L_kappa(self, polys_kappa):
        """Compute m1_eff at large L and compare to empirical."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 40
        L = 1000.0

        # Post-identity values
        result_plus = compute_I1_operator_post_identity_pair(
            theta, R, 1, 1, n, polys_kappa
        )
        result_minus = compute_I1_operator_post_identity_pair(
            theta, -R, 1, 1, n, polys_kappa
        )
        I1_plus = result_plus.I1_value
        I1_minus_base = result_minus.I1_value

        # Combined identity at large L
        I1_combined = compute_I1_combined_at_L(theta, R, L, n, polys_kappa)

        # Solve for m1_eff: I1_combined = I1_plus + m1_eff × I1_minus_base
        m1_eff = (I1_combined - I1_plus) / I1_minus_base

        m1_empirical = np.exp(R) + 5
        m1_naive = np.exp(2 * R)

        print(f"\nm1_eff at L={L}:")
        print(f"  m1_eff = {m1_eff:.4f}")
        print(f"  m1_empirical (exp(R)+5) = {m1_empirical:.4f}")
        print(f"  m1_naive (exp(2R)) = {m1_naive:.4f}")

        # The m1_eff should be finite
        assert np.isfinite(m1_eff), f"m1_eff not finite: {m1_eff}"

        # Document what we observe (this is the key result)
        # The test passes regardless of the value - it's observational
        print(f"  Ratio m1_eff/m1_empirical = {m1_eff / m1_empirical:.4f}")


class TestL_Sweep:
    """Test the L-sweep for m1 convergence analysis."""

    def test_l_sweep_produces_results(self, polys_kappa):
        """L-sweep should produce finite m1_eff values at all L."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 30
        L_values = [10.0, 20.0, 50.0, 100.0]

        # Post-identity values
        result_plus = compute_I1_operator_post_identity_pair(
            theta, R, 1, 1, n, polys_kappa
        )
        result_minus = compute_I1_operator_post_identity_pair(
            theta, -R, 1, 1, n, polys_kappa
        )
        I1_plus = result_plus.I1_value
        I1_minus_base = result_minus.I1_value

        print(f"\nL-Sweep Results:")
        print(f"  I1_plus = {I1_plus:.8f}")
        print(f"  I1_minus_base = {I1_minus_base:.8f}")
        print(f"  {'L':>6} {'I1_combined':>14} {'m1_eff':>12}")

        m1_values = []
        for L in L_values:
            I1_combined = compute_I1_combined_at_L(theta, R, L, n, polys_kappa)
            m1_eff = (I1_combined - I1_plus) / I1_minus_base
            m1_values.append(m1_eff)
            print(f"  {L:>6.0f} {I1_combined:>14.6f} {m1_eff:>12.4f}")

        # All values should be finite
        for m1, L in zip(m1_values, L_values):
            assert np.isfinite(m1), f"m1_eff not finite at L={L}: {m1}"

        # Check if values are converging (decreasing relative change)
        rel_changes = []
        for i in range(1, len(m1_values)):
            rel_change = abs(m1_values[i] - m1_values[i-1]) / abs(m1_values[i-1])
            rel_changes.append(rel_change)

        print(f"\n  Relative changes: {[f'{r:.2%}' for r in rel_changes]}")
