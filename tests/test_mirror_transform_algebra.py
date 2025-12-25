"""
tests/test_mirror_transform_algebra.py
Phase 9.1B: Tests for the Mirror Transform Algebra Harness

These tests validate the shift identity:
    Q(D_α)[T^{-s}F] = T^{-s} Q(1+D_α)[F]

This is the KEY algebraic property that underpins the TeX mirror term.

Test categories:
1. AffineOperatorAction tests - dataclass operations
2. Shift identity validation on toy kernels
3. Shift identity validation on PRZZ Q polynomial
4. T^{-(α+β)} weight tests
"""

import pytest
import math
import numpy as np
from src.mirror_transform_algebra import (
    AffineOperatorAction,
    MirrorTransform,
    get_direct_eigenvalues,
    get_shifted_eigenvalues,
    validate_shift_identity_analytic,
    validate_shift_identity_numerical,
    compute_full_mirror_contribution,
    implied_m1_from_derived,
)
from src.polynomials import load_przz_polynomials, Polynomial


class TestAffineOperatorAction:
    """Tests for the AffineOperatorAction dataclass."""

    def test_creation(self):
        """Can create an AffineOperatorAction."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        assert A.u0 == 0.5
        assert A.a_x == 0.2
        assert A.a_y == 0.3

    def test_apply_shift(self):
        """apply_shift adds delta to u0 only."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        A_shifted = A.apply_shift(1.0)
        assert A_shifted.u0 == 1.5
        assert A_shifted.a_x == 0.2
        assert A_shifted.a_y == 0.3

    def test_apply_shift_preserves_original(self):
        """apply_shift returns a new object, doesn't mutate."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        A_shifted = A.apply_shift(1.0)
        assert A.u0 == 0.5  # Original unchanged

    def test_negate(self):
        """negate returns -A."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=-0.3)
        A_neg = A.negate()
        assert A_neg.u0 == -0.5
        assert A_neg.a_x == -0.2
        assert A_neg.a_y == 0.3

    def test_swap_xy(self):
        """swap_xy swaps x and y coefficients."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        A_swapped = A.swap_xy()
        assert A_swapped.u0 == 0.5
        assert A_swapped.a_x == 0.3
        assert A_swapped.a_y == 0.2

    def test_evaluate(self):
        """evaluate computes u0 + a_x*x + a_y*y."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        result = A.evaluate(x=1.0, y=2.0)
        expected = 0.5 + 0.2 * 1.0 + 0.3 * 2.0
        assert abs(result - expected) < 1e-10

    def test_to_dict(self):
        """to_dict returns coefficients for composition."""
        A = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        d = A.to_dict()
        assert d == {"x": 0.2, "y": 0.3}


class TestMirrorTransform:
    """Tests for the MirrorTransform class."""

    def test_T_weight_formula(self):
        """T^{-(α+β)} = exp(-L(α+β))."""
        mt = MirrorTransform(L=1.0)
        alpha, beta = -0.5, -0.3
        expected = np.exp(-1.0 * (alpha + beta))
        result = mt.get_T_weight(alpha, beta)
        assert abs(result - expected) < 1e-10

    def test_T_weight_at_przz_point(self):
        """At α=β=-R/L, T^{-(α+β)} = exp(2R)."""
        mt = MirrorTransform(L=1.0)
        R = 1.3036
        result = mt.get_T_weight_at_przz_point(R)
        expected = np.exp(2 * R)
        assert abs(result - expected) < 1e-10

    def test_transform_eigenvalues(self):
        """transform_eigenvalues_for_mirror shifts both by +1."""
        mt = MirrorTransform()
        A_alpha = AffineOperatorAction(u0=0.5, a_x=0.2, a_y=0.3)
        A_beta = AffineOperatorAction(u0=0.4, a_x=0.1, a_y=0.25)

        A_alpha_m, A_beta_m = mt.transform_eigenvalues_for_mirror(A_alpha, A_beta)

        assert A_alpha_m.u0 == 1.5
        assert A_beta_m.u0 == 1.4
        # Linear coefficients unchanged
        assert A_alpha_m.a_x == 0.2
        assert A_beta_m.a_y == 0.25


class TestDirectEigenvalues:
    """Tests for get_direct_eigenvalues."""

    def test_eigenvalue_structure(self):
        """Eigenvalues should match PRZZ structure."""
        theta = 4 / 7
        t = 0.5

        A_alpha, A_beta = get_direct_eigenvalues(t, theta)

        # A_α = t + θ(t-1)·x + θt·y
        expected_u0 = t
        expected_ax = theta * (t - 1)
        expected_ay = theta * t

        assert abs(A_alpha.u0 - expected_u0) < 1e-10
        assert abs(A_alpha.a_x - expected_ax) < 1e-10
        assert abs(A_alpha.a_y - expected_ay) < 1e-10

        # A_β = t + θt·x + θ(t-1)·y (x,y swapped)
        assert abs(A_beta.u0 - expected_u0) < 1e-10
        assert abs(A_beta.a_x - expected_ay) < 1e-10  # swapped
        assert abs(A_beta.a_y - expected_ax) < 1e-10  # swapped

    def test_alpha_beta_are_swapped(self):
        """A_α and A_β have swapped x/y coefficients."""
        A_alpha, A_beta = get_direct_eigenvalues(t=0.7, theta=4/7)

        assert abs(A_alpha.a_x - A_beta.a_y) < 1e-10
        assert abs(A_alpha.a_y - A_beta.a_x) < 1e-10


class TestShiftedEigenvalues:
    """Tests for get_shifted_eigenvalues."""

    def test_shifted_by_1(self):
        """get_shifted_eigenvalues with shift=1 adds 1 to u0."""
        theta = 4 / 7
        t = 0.5

        A_alpha, A_beta = get_direct_eigenvalues(t, theta)
        A_alpha_s, A_beta_s = get_shifted_eigenvalues(t, theta, shift=1.0)

        assert abs(A_alpha_s.u0 - (A_alpha.u0 + 1.0)) < 1e-10
        assert abs(A_beta_s.u0 - (A_beta.u0 + 1.0)) < 1e-10
        # Linear coefficients unchanged
        assert abs(A_alpha_s.a_x - A_alpha.a_x) < 1e-10
        assert abs(A_alpha_s.a_y - A_alpha.a_y) < 1e-10


class TestShiftIdentityAnalytic:
    """
    Gate tests for the shift identity using toy kernels.

    The shift identity is:
        Q(D_α)[T^{-s}F] = T^{-s} Q(1+D_α)[F]

    For toy kernel F = exp(c₁α + c₂β), D_α has eigenvalue λ = -c₁/L.
    So Q(D_α)[F] = Q(λ)F, and the identity becomes:
        Q(λ) × T^{-s}F = T^{-s} × Q(1+λ)F

    Wait, that's not right. Let me reconsider...

    Actually for T^{-s}F where s = α+β:
        Q(D_α)[T^{-s}F] = T^{-s} Q(1+D_α)[F]

    The LHS applies to the product, the RHS uses shifted operator on F only.
    Both should give T^{-s} × Q(1+λ) × F.
    """

    def test_shift_identity_monomial_Q(self):
        """Q(z) = z should satisfy shift identity."""
        Q_coeffs = [0.0, 1.0]  # Q(z) = z
        result = validate_shift_identity_analytic(
            Q_coeffs, alpha=-0.5, beta=-0.3, x=0.0, y=0.0
        )
        assert result["passed"], f"Shift identity failed: error={result['error']}"

    def test_shift_identity_quadratic_Q(self):
        """Q(z) = 1 + z + z² should satisfy shift identity."""
        Q_coeffs = [1.0, 1.0, 1.0]  # Q(z) = 1 + z + z²
        result = validate_shift_identity_analytic(
            Q_coeffs, alpha=-0.5, beta=-0.3, x=0.0, y=0.0
        )
        assert result["passed"], f"Shift identity failed: error={result['error']}"

    def test_shift_identity_cubic_Q(self):
        """Q(z) = 1 - z + 2z² - 0.5z³ should satisfy shift identity."""
        Q_coeffs = [1.0, -1.0, 2.0, -0.5]
        result = validate_shift_identity_analytic(
            Q_coeffs, alpha=-0.5, beta=-0.3, x=0.0, y=0.0
        )
        assert result["passed"], f"Shift identity failed: error={result['error']}"

    def test_shift_has_effect(self):
        """Without shift, result should differ (unless Q is constant)."""
        Q_coeffs = [1.0, 1.0, 1.0]  # Q(z) = 1 + z + z²
        result = validate_shift_identity_analytic(
            Q_coeffs, alpha=-0.5, beta=-0.3, x=0.0, y=0.0
        )
        # The "wrong" value (without shift) should differ from correct
        assert abs(result["LHS"] - result["wrong_no_shift"]) > 1e-5, \
            "Shift should have measurable effect"


class TestShiftIdentityNumerical:
    """
    Tests for numerical validation of shift identity using series algebra.

    These tests use the actual PRZZ Q polynomial and composition machinery.
    """

    @pytest.fixture
    def przz_Q(self):
        """Load PRZZ Q polynomial."""
        _, _, _, Q = load_przz_polynomials()
        return Q

    def test_shift_equivalence(self, przz_Q):
        """
        Two methods of computing shifted Q should give same result:
        1. Shift eigenvalue, then compose Q on shifted eigenvalue
        2. Shift polynomial (Q → Q(1+·)), then compose on original eigenvalue

        Both should yield identical series coefficients.
        """
        result = validate_shift_identity_numerical(
            przz_Q, t=0.5, theta=4/7, R=1.3036
        )
        assert result["passed"], \
            f"Shift methods differ: error_xy={result['error_xy']}, error_const={result['error_const']}"

    def test_shift_equivalence_multiple_t(self, przz_Q):
        """Test shift equivalence at multiple t values."""
        for t in [0.25, 0.5, 0.75]:
            result = validate_shift_identity_numerical(
                przz_Q, t=t, theta=4/7, R=1.3036
            )
            assert result["passed"], \
                f"Shift failed at t={t}: error={result['error_xy']}"


class TestTWeightAtPRZZPoint:
    """Tests that T^{-(α+β)} = exp(2R) at the PRZZ evaluation point."""

    def test_kappa_benchmark(self):
        """T weight at κ benchmark (R=1.3036)."""
        R = 1.3036
        mt = MirrorTransform()
        result = mt.get_T_weight_at_przz_point(R)
        expected = np.exp(2 * R)
        assert abs(result - expected) < 1e-10
        # Sanity check: exp(2*1.3036) ≈ 13.56
        assert 13 < result < 14

    def test_kappa_star_benchmark(self):
        """T weight at κ* benchmark (R=1.1167)."""
        R = 1.1167
        mt = MirrorTransform()
        result = mt.get_T_weight_at_przz_point(R)
        expected = np.exp(2 * R)
        assert abs(result - expected) < 1e-10
        # Sanity check: exp(2*1.1167) ≈ 9.37
        assert 9 < result < 10


class TestDerivedMirrorHelpers:
    """Tests for derived mirror helper functions."""

    def test_compute_full_mirror_contribution(self):
        """Full mirror = exp(2R) × I_shifted."""
        I_shifted = 0.5
        R = 1.3036
        result = compute_full_mirror_contribution(I_shifted, R)
        expected = np.exp(2 * R) * I_shifted
        assert abs(result - expected) < 1e-10

    def test_implied_m1_from_derived(self):
        """implied_m1 = S12_mirror_derived / S12_minus_basis."""
        S12_mirror = 4.0
        S12_minus = 0.5
        m1 = implied_m1_from_derived(S12_mirror, S12_minus)
        assert abs(m1 - 8.0) < 1e-10

    def test_implied_m1_handles_zero(self):
        """implied_m1 returns inf for zero denominator."""
        m1 = implied_m1_from_derived(1.0, 0.0)
        assert m1 == float('inf')


class TestIntegrationWithExistingInfrastructure:
    """
    Integration tests with existing operator_post_identity infrastructure.

    These tests verify that the algebra harness is consistent with
    the existing eigenvalue extraction.
    """

    def test_matches_operator_post_identity_coeffs(self):
        """get_direct_eigenvalues matches get_A_alpha_affine_coeffs."""
        from src.operator_post_identity import (
            get_A_alpha_affine_coeffs,
            get_A_beta_affine_coeffs,
        )

        t = 0.6
        theta = 4 / 7

        # From algebra harness
        A_alpha, A_beta = get_direct_eigenvalues(t, theta)

        # From operator_post_identity
        u0_a, ax_a, ay_a = get_A_alpha_affine_coeffs(t, theta)
        u0_b, ax_b, ay_b = get_A_beta_affine_coeffs(t, theta)

        assert abs(A_alpha.u0 - u0_a) < 1e-10
        assert abs(A_alpha.a_x - ax_a) < 1e-10
        assert abs(A_alpha.a_y - ay_a) < 1e-10

        assert abs(A_beta.u0 - u0_b) < 1e-10
        assert abs(A_beta.a_x - ax_b) < 1e-10
        assert abs(A_beta.a_y - ay_b) < 1e-10
