"""
tests/test_mirror_term_derived.py
Phase 9.2C: Gate tests for derived mirror term implementation.

These tests verify:
1. Q≡1 gives pure kernel × exp(2R)
2. Derived mirror is computed correctly
3. Relationship between derived and DSL minus basis
"""

import pytest
import math
import numpy as np
from src.mirror_exact import (
    compute_I1_mirror_derived,
    compute_I2_mirror_derived,
    compute_S12_mirror_derived,
    compute_S12_minus_basis,
    compute_I1_with_shifted_Q,
    DerivedMirrorResult,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)


class TestDerivedMirrorBasic:
    """Basic functionality tests for derived mirror."""

    @pytest.fixture
    def polys_kappa(self):
        """Load κ benchmark polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_I1_mirror_derived_returns_result(self, polys_kappa):
        """compute_I1_mirror_derived should return a DerivedMirrorResult."""
        result = compute_I1_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa
        )
        assert isinstance(result, DerivedMirrorResult)

    def test_I1_mirror_derived_has_T_weight(self, polys_kappa):
        """T_weight should be exp(2R)."""
        R = 1.3036
        result = compute_I1_mirror_derived(
            theta=4/7, R=R, n=30, polynomials=polys_kappa
        )
        expected_T = np.exp(2 * R)
        assert abs(result.T_weight - expected_T) < 1e-10

    def test_I1_mirror_derived_value_is_product(self, polys_kappa):
        """value should equal T_weight × I_shifted_Q_plus_R."""
        result = compute_I1_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa
        )
        expected = result.T_weight * result.I_shifted_Q_plus_R
        assert abs(result.value - expected) < 1e-10

    def test_I2_mirror_derived_returns_result(self, polys_kappa):
        """compute_I2_mirror_derived should return a DerivedMirrorResult."""
        result = compute_I2_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa
        )
        assert isinstance(result, DerivedMirrorResult)


class TestQEqualsOneGivesKernel:
    """
    Gate test: When Q ≡ 1, derived mirror should reduce to pure kernel.

    With Q ≡ 1:
    - Q(1+D) = 1 (identity)
    - I_shifted_Q = I_standard
    - Derived mirror = exp(2R) × I_standard(+R)
    """

    @pytest.fixture
    def polys_with_const_Q(self):
        """Load polynomials with Q = 1 (constant)."""
        P1, P2, P3, _ = load_przz_polynomials()
        # Create Q = 1 (constant polynomial)
        Q_const = Polynomial(np.array([1.0]))
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q_const}

    def test_Q1_I1_shifted_equals_I1_standard(self, polys_with_const_Q):
        """
        With Q=1, I1 with shifted Q should equal I1 with standard Q.

        Both Q(A) and Q(1+A) equal 1 when Q ≡ 1.
        """
        # Compute with shift=1.0 (derived mirror)
        I1_shifted = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=30, polynomials=polys_with_const_Q,
            shift=1.0
        )

        # Compute with shift=0.0 (standard)
        I1_standard = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=30, polynomials=polys_with_const_Q,
            shift=0.0
        )

        # Should be equal when Q=1
        assert abs(I1_shifted - I1_standard) < 1e-10

    def test_Q1_derived_mirror_includes_exp2R(self, polys_with_const_Q):
        """
        With Q=1, derived mirror = exp(2R) × I_standard.
        """
        R = 1.3036
        theta = 4 / 7
        n = 30

        # Derived mirror (with Q=1)
        result = compute_I1_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polys_with_const_Q
        )

        # Standard I1 (with Q=1)
        I1_standard = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polys_with_const_Q,
            shift=0.0
        )

        # Derived should be exp(2R) × standard
        expected = np.exp(2 * R) * I1_standard
        assert abs(result.value - expected) < 1e-8, \
            f"Q=1 derived mismatch: got {result.value}, expected {expected}"


class TestDerivedVsMinusBasis:
    """
    Tests comparing derived mirror to DSL minus basis.

    Key question: What is the relationship between:
    - Derived mirror: exp(2R) × I_shifted_Q(+R)
    - DSL minus basis: I(-R) with Q unchanged

    The implied m₁ from derived is:
        m₁_implied = S12_mirror_derived / S12_minus_basis
    """

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_derived_and_minus_basis_differ(self, polys_kappa):
        """
        Derived mirror should differ from DSL minus basis.

        They use different formulas:
        - Derived: exp(2R) × I_shifted_Q(+R)
        - Minus basis: I(-R) with Q unchanged
        """
        theta = 4 / 7
        R = 1.3036
        n = 30

        derived = compute_S12_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        minus_basis = compute_S12_minus_basis(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        # They should NOT be equal
        ratio = derived / minus_basis if abs(minus_basis) > 1e-15 else float('inf')
        assert abs(ratio - 1.0) > 0.01, \
            f"Derived and minus basis should differ, but ratio={ratio}"

    def test_implied_m1_is_finite(self, polys_kappa):
        """
        Implied m₁ = derived / minus_basis should be finite and positive.
        """
        theta = 4 / 7
        R = 1.3036
        n = 30

        derived = compute_S12_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        minus_basis = compute_S12_minus_basis(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        m1_implied = derived / minus_basis if abs(minus_basis) > 1e-15 else float('inf')

        assert math.isfinite(m1_implied), "implied m1 should be finite"
        # Should be positive (both derived and minus basis should have same sign)
        assert m1_implied > 0, "implied m1 should be positive"


class TestDerivedMirrorAllPairs:
    """Test derived mirror for all pairs."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.mark.parametrize("ell1,ell2", [
        (1, 1), (2, 2), (3, 3),
        (1, 2), (1, 3), (2, 3),
    ])
    def test_I1_mirror_derived_each_pair(self, polys_kappa, ell1, ell2):
        """Each pair should produce finite I1 mirror derived."""
        result = compute_I1_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa,
            ell1=ell1, ell2=ell2
        )
        assert math.isfinite(result.value)

    @pytest.mark.parametrize("ell1,ell2", [
        (1, 1), (2, 2), (3, 3),
        (1, 2), (1, 3), (2, 3),
    ])
    def test_I2_mirror_derived_each_pair(self, polys_kappa, ell1, ell2):
        """Each pair should produce finite I2 mirror derived."""
        result = compute_I2_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa,
            ell1=ell1, ell2=ell2
        )
        assert math.isfinite(result.value)


class TestS12TotalsAreFinite:
    """Test that S12 totals are finite for both methods."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_S12_mirror_derived_kappa(self, polys_kappa):
        """S12_mirror_derived should be finite for κ benchmark."""
        result = compute_S12_mirror_derived(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa
        )
        assert math.isfinite(result)

    def test_S12_mirror_derived_kappa_star(self, polys_kappa_star):
        """S12_mirror_derived should be finite for κ* benchmark."""
        result = compute_S12_mirror_derived(
            theta=4/7, R=1.1167, n=30, polynomials=polys_kappa_star
        )
        assert math.isfinite(result)

    def test_S12_minus_basis_kappa(self, polys_kappa):
        """S12_minus_basis should be finite for κ benchmark."""
        result = compute_S12_minus_basis(
            theta=4/7, R=1.3036, n=30, polynomials=polys_kappa
        )
        assert math.isfinite(result)

    def test_S12_minus_basis_kappa_star(self, polys_kappa_star):
        """S12_minus_basis should be finite for κ* benchmark."""
        result = compute_S12_minus_basis(
            theta=4/7, R=1.1167, n=30, polynomials=polys_kappa_star
        )
        assert math.isfinite(result)


class TestConsistencyWithExistingCode:
    """
    Consistency tests with existing mirror_exact infrastructure.
    """

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_I1_shifted_matches_existing_function(self, polys_kappa):
        """
        The I_shifted_Q_plus_R in derived result should match
        compute_I1_with_shifted_Q directly.
        """
        theta = 4 / 7
        R = 1.3036
        n = 30

        # Via derived mirror
        result = compute_I1_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polys_kappa
        )

        # Directly
        I1_direct = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polys_kappa,
            shift=1.0
        )

        assert abs(result.I_shifted_Q_plus_R - I1_direct) < 1e-10
