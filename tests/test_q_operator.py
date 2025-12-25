"""
tests/test_q_operator.py
Unit tests for Q operator binomial-lift implementation.

These are pure math tests - no dependency on PRZZ integrals.
Tests use seeded randomness for determinism.
"""

import pytest
import numpy as np
from math import comb

from src.q_operator import (
    binomial_lift_coeffs,
    binomial_shift_coeffs,
    verify_binomial_lift,
    przz_basis_to_standard_coeffs,
    standard_to_przz_basis_coeffs,
)


# =============================================================================
# binomial_lift_coeffs tests
# =============================================================================

class TestBinomialLiftCoeffs:
    """Tests for Q(x) -> Q(1+x) coefficient transformation."""

    def test_empty_input(self):
        """Empty coefficient list returns empty."""
        assert binomial_lift_coeffs([]) == []

    def test_constant_polynomial(self):
        """Q(x) = c -> Q(1+x) = c (constant unchanged)."""
        assert binomial_lift_coeffs([5.0]) == [5.0]
        assert binomial_lift_coeffs([0.0]) == [0.0]
        assert binomial_lift_coeffs([-3.14]) == [-3.14]

    def test_linear_polynomial(self):
        """Q(x) = a + bx -> Q(1+x) = (a+b) + bx."""
        # Q(x) = 1 + 2x
        # Q(1+x) = 1 + 2(1+x) = 3 + 2x
        result = binomial_lift_coeffs([1.0, 2.0])
        assert pytest.approx(result[0]) == 3.0
        assert pytest.approx(result[1]) == 2.0

    def test_quadratic_example_from_docstring(self):
        """Q(x) = 1 + 2x + 3x^2 -> Q(1+x) = 6 + 8x + 3x^2."""
        # From docstring example
        result = binomial_lift_coeffs([1.0, 2.0, 3.0])
        assert pytest.approx(result[0]) == 6.0
        assert pytest.approx(result[1]) == 8.0
        assert pytest.approx(result[2]) == 3.0

    def test_pure_power(self):
        """Q(x) = x^n -> Q(1+x) = (1+x)^n, verify binomial expansion."""
        # Q(x) = x^3
        # Q(1+x) = (1+x)^3 = 1 + 3x + 3x^2 + x^3
        q_coeffs = [0.0, 0.0, 0.0, 1.0]  # x^3
        result = binomial_lift_coeffs(q_coeffs)
        expected = [1.0, 3.0, 3.0, 1.0]  # (1+x)^3
        for i in range(4):
            assert pytest.approx(result[i]) == expected[i]

    def test_random_polynomials_verify(self):
        """Verify Q(1+x) == Q_lift(x) for random polynomials at test points."""
        np.random.seed(42)  # Deterministic

        for degree in range(7):
            q_coeffs = list(np.random.randn(degree + 1))
            success, max_error = verify_binomial_lift(q_coeffs)
            assert success, f"Degree {degree}: max_error={max_error}"

    def test_leading_coefficient_preserved(self):
        """Leading coefficient of Q(x) equals leading coefficient of Q(1+x)."""
        np.random.seed(123)
        for _ in range(5):
            n = np.random.randint(2, 8)
            q_coeffs = list(np.random.randn(n))
            q_coeffs[-1] = np.random.randn()  # Ensure non-zero leading

            lifted = binomial_lift_coeffs(q_coeffs)
            assert pytest.approx(lifted[-1]) == q_coeffs[-1]


# =============================================================================
# binomial_shift_coeffs tests
# =============================================================================

class TestBinomialShiftCoeffs:
    """Tests for Q(x) -> Q(shift+x) generalized transformation."""

    def test_shift_one_equals_lift(self):
        """shift=1 should give same result as binomial_lift_coeffs."""
        np.random.seed(456)
        for _ in range(5):
            n = np.random.randint(2, 6)
            q_coeffs = list(np.random.randn(n))

            lifted = binomial_lift_coeffs(q_coeffs)
            shifted = binomial_shift_coeffs(q_coeffs, shift=1.0)

            for i in range(n):
                assert pytest.approx(shifted[i]) == lifted[i]

    def test_shift_zero_is_identity(self):
        """shift=0 should return original coefficients."""
        q_coeffs = [1.0, 2.0, 3.0, 4.0]
        result = binomial_shift_coeffs(q_coeffs, shift=0.0)
        for i in range(len(q_coeffs)):
            assert pytest.approx(result[i]) == q_coeffs[i]

    def test_verify_shift_at_points(self):
        """Verify Q(shift+x) at several test points."""
        np.random.seed(789)

        for shift in [0.5, -0.5, 2.0, -1.5]:
            q_coeffs = list(np.random.randn(5))
            shifted = binomial_shift_coeffs(q_coeffs, shift=shift)

            # Verify at test points
            test_points = [0.0, 0.5, 1.0, -0.5, 0.25]
            for x in test_points:
                # Q(shift + x) evaluated directly
                q_at_shift_plus_x = sum(c * ((shift + x) ** j) for j, c in enumerate(q_coeffs))

                # Q_shifted(x) using transformed coefficients
                q_shifted_at_x = sum(c * (x ** r) for r, c in enumerate(shifted))

                assert pytest.approx(q_at_shift_plus_x, rel=1e-9) == q_shifted_at_x


# =============================================================================
# verify_binomial_lift tests
# =============================================================================

class TestVerifyBinomialLift:
    """Tests for the verification utility."""

    def test_correct_lift_passes(self):
        """Correctly lifted polynomial passes verification."""
        q_coeffs = [1.0, 2.0, 3.0]
        success, max_error = verify_binomial_lift(q_coeffs)
        assert success
        assert max_error < 1e-10

    def test_returns_max_error(self):
        """Returns reasonable max_error value."""
        q_coeffs = [1.0, -1.0, 0.5, -0.25]
        success, max_error = verify_binomial_lift(q_coeffs)
        assert success
        assert 0 <= max_error < 1e-10


# =============================================================================
# PRZZ basis conversion tests
# =============================================================================

class TestPRZZBasisConversion:
    """Tests for PRZZ (1-2x)^k basis <-> standard coefficient conversion."""

    def test_single_power_basis(self):
        """(1-2x)^k expands correctly."""
        # (1-2x)^0 = 1
        result = przz_basis_to_standard_coeffs([(0, 1.0)])
        assert pytest.approx(result[0]) == 1.0

        # (1-2x)^1 = 1 - 2x
        result = przz_basis_to_standard_coeffs([(1, 1.0)])
        assert pytest.approx(result[0]) == 1.0
        assert pytest.approx(result[1]) == -2.0

        # (1-2x)^2 = 1 - 4x + 4x^2
        result = przz_basis_to_standard_coeffs([(2, 1.0)])
        assert pytest.approx(result[0]) == 1.0
        assert pytest.approx(result[1]) == -4.0
        assert pytest.approx(result[2]) == 4.0

    def test_linear_combination(self):
        """Linear combination of basis terms."""
        # 2*(1-2x)^0 + 3*(1-2x)^1 = 2 + 3*(1-2x) = 5 - 6x
        result = przz_basis_to_standard_coeffs([(0, 2.0), (1, 3.0)])
        assert pytest.approx(result[0]) == 5.0
        assert pytest.approx(result[1]) == -6.0

    def test_roundtrip_simple(self):
        """Roundtrip: standard -> PRZZ -> standard recovers original."""
        # Simple polynomial
        original = [1.0, -2.0, 1.0]  # 1 - 2x + x^2

        # Convert to PRZZ basis
        przz_coeffs = standard_to_przz_basis_coeffs(original)

        # Convert back to standard
        recovered = przz_basis_to_standard_coeffs(przz_coeffs)

        # Should recover original (may have trailing zeros)
        for i in range(len(original)):
            assert pytest.approx(recovered[i], abs=1e-10) == original[i]

    def test_verify_at_points(self):
        """Verify standard <-> PRZZ conversions at sample points."""
        original = [0.5, -1.0, 2.0, -0.5]  # 0.5 - x + 2x^2 - 0.5x^3

        przz_coeffs = standard_to_przz_basis_coeffs(original)
        recovered = przz_basis_to_standard_coeffs(przz_coeffs)

        # Evaluate both at test points
        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        for x in test_points:
            val_original = sum(c * (x ** j) for j, c in enumerate(original))
            val_recovered = sum(c * (x ** j) for j, c in enumerate(recovered))
            assert pytest.approx(val_original, rel=1e-9) == val_recovered


# =============================================================================
# Integration test: lift with PRZZ basis
# =============================================================================

class TestLiftWithPRZZBasis:
    """Test binomial lift through PRZZ basis conversion pipeline."""

    def test_lift_przz_polynomial(self):
        """Lift a polynomial defined in PRZZ basis."""
        # Define Q in PRZZ basis: c_1*(1-2x)^1 + c_3*(1-2x)^3
        przz_coeffs = [(1, 0.5), (3, 0.25)]

        # Convert to standard
        standard = przz_basis_to_standard_coeffs(przz_coeffs)

        # Apply binomial lift
        lifted = binomial_lift_coeffs(standard)

        # Verify Q(1+x) = Q_lift(x) at test points
        test_points = [0.0, 0.5, -0.5, 0.25]
        for x in test_points:
            # Q(1+x) using PRZZ basis directly
            q_at_1_plus_x = 0.0
            for k, c_k in przz_coeffs:
                q_at_1_plus_x += c_k * ((1 - 2*(1+x)) ** k)

            # Q_lift(x) using lifted standard coefficients
            q_lift_at_x = sum(c * (x ** r) for r, c in enumerate(lifted))

            assert pytest.approx(q_at_1_plus_x, rel=1e-9) == q_lift_at_x


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_high_degree_stability(self):
        """High-degree polynomials don't cause numerical issues."""
        np.random.seed(999)

        # Degree 10 polynomial
        q_coeffs = list(np.random.randn(11))
        success, max_error = verify_binomial_lift(q_coeffs)
        assert success, f"max_error={max_error}"

    def test_negative_coefficients(self):
        """Negative coefficients handled correctly."""
        q_coeffs = [-1.0, -2.0, -3.0]
        # Q(x) = -1 - 2x - 3x^2
        # Q(1+x) = -1 - 2(1+x) - 3(1+x)^2
        #        = -1 - 2 - 2x - 3(1 + 2x + x^2)
        #        = -1 - 2 - 3 - (2 + 6)x - 3x^2
        #        = -6 - 8x - 3x^2
        result = binomial_lift_coeffs(q_coeffs)
        assert pytest.approx(result[0]) == -6.0
        assert pytest.approx(result[1]) == -8.0
        assert pytest.approx(result[2]) == -3.0

    def test_zero_polynomial(self):
        """Zero polynomial stays zero."""
        result = binomial_lift_coeffs([0.0, 0.0, 0.0])
        for c in result:
            assert c == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
