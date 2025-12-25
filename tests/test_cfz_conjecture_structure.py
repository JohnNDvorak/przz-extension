"""
tests/test_cfz_conjecture_structure.py
Phase 14 Task 1: Tests for CFZ 4-shift object structure.

These tests validate that the CFZ conjecture object has the correct structure
before we implement the full arithmetic factor.
"""

import pytest
import numpy as np
from src.ratios.cfz_conjecture import (
    FourShifts,
    CfzTerms,
    cfz_integrand_terms,
    A_arithmetic_factor,
)


class TestFourShiftsStructure:
    """Test the FourShifts dataclass structure."""

    def test_four_shifts_has_all_parameters(self):
        """FourShifts should have α, β, γ, δ."""
        shifts = FourShifts(alpha=0.1, beta=0.2, gamma=0.3, delta=0.4)
        assert shifts.alpha == 0.1
        assert shifts.beta == 0.2
        assert shifts.gamma == 0.3
        assert shifts.delta == 0.4

    def test_four_shifts_accepts_complex(self):
        """FourShifts should accept complex parameters."""
        shifts = FourShifts(
            alpha=0.1 + 0.2j,
            beta=0.3 + 0.4j,
            gamma=0.5 + 0.6j,
            delta=0.7 + 0.8j
        )
        assert shifts.alpha == 0.1 + 0.2j

    def test_four_shifts_is_immutable(self):
        """FourShifts should be frozen (immutable)."""
        shifts = FourShifts(alpha=0.1, beta=0.2, gamma=0.3, delta=0.4)
        with pytest.raises(AttributeError):
            shifts.alpha = 0.5


class TestCfzTermsStructure:
    """Test the CfzTerms structure."""

    def test_has_two_terms_direct_and_dual(self):
        """CFZ conjecture produces exactly two terms before differentiation."""
        shifts = FourShifts(alpha=0.1, beta=0.1, gamma=0.1, delta=0.1)
        terms = cfz_integrand_terms(shifts, t=1.0)

        # Should have exactly two components
        assert hasattr(terms, 'direct_term')
        assert hasattr(terms, 'dual_term')
        assert len(terms) == 2

    def test_terms_are_numeric(self):
        """Both terms should return numeric values."""
        shifts = FourShifts(alpha=0.1, beta=0.1, gamma=0.1, delta=0.1)
        terms = cfz_integrand_terms(shifts, t=2.0)

        assert isinstance(terms.direct_term, (int, float, complex))
        assert isinstance(terms.dual_term, (int, float, complex))


class TestDualTermStructure:
    """Test that dual term has (t/2π)^{-α-β} structure."""

    def test_dual_term_scales_with_t(self):
        """Dual term should scale as t^{-α-β}."""
        alpha, beta = 0.1, 0.2
        shifts = FourShifts(alpha=alpha, beta=beta, gamma=alpha, delta=beta)

        t1, t2 = 1.0, 2.0
        terms1 = cfz_integrand_terms(shifts, t=t1)
        terms2 = cfz_integrand_terms(shifts, t=t2)

        # Dual term ~ t^{-α-β}, so ratio should be (t2/t1)^{-α-β}
        expected_ratio = (t2 / t1) ** (-(alpha + beta))
        actual_ratio = terms2.dual_term / terms1.dual_term

        # Allow some tolerance for other factors
        assert abs(actual_ratio / expected_ratio - 1.0) < 0.5, (
            f"Dual term doesn't scale as t^{{-α-β}}: "
            f"expected ratio {expected_ratio}, got {actual_ratio}"
        )


class TestArithmeticFactorDiagonal:
    """Test that A(α,β,α,β) = 1 at diagonal."""

    def test_A_diagonal_is_one(self):
        """Paper explicitly states: A(α,β,α,β) = 1."""
        # At diagonal: γ=α, δ=β
        alpha, beta = 0.1, 0.2
        shifts = FourShifts(alpha=alpha, beta=beta, gamma=alpha, delta=beta)

        A_value = A_arithmetic_factor(shifts)

        assert abs(A_value - 1.0) < 1e-10, (
            f"A(α,β,α,β) should be 1, got {A_value}"
        )

    def test_A_diagonal_is_one_at_zero(self):
        """A(0,0,0,0) = 1."""
        shifts = FourShifts(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
        A_value = A_arithmetic_factor(shifts)
        assert abs(A_value - 1.0) < 1e-10

    def test_A_diagonal_is_one_various_values(self):
        """A(α,β,α,β) = 1 for various α,β."""
        for alpha in [0.0, 0.1, 0.5, 1.0]:
            for beta in [0.0, 0.1, 0.5, 1.0]:
                shifts = FourShifts(
                    alpha=alpha, beta=beta,
                    gamma=alpha, delta=beta
                )
                A_value = A_arithmetic_factor(shifts)
                assert abs(A_value - 1.0) < 1e-10, (
                    f"A({alpha},{beta},{alpha},{beta}) should be 1, got {A_value}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
