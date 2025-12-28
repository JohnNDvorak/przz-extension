"""
tests/test_unified_microcase_oracle.py
Phase 26B: Tests for P=Q=1 microcase oracle validation

These tests validate that:
1. The analytic oracle formula is correct
2. The unified P=Q=1 evaluator matches the oracle exactly
3. Both implementations handle all pairs correctly
"""

import pytest
from src.unified_microcase_oracle import (
    oracle_coeff_P1Q1,
    oracle_I1_P1Q1,
    validate_oracle_vs_unified,
)
from src.unified_i1_general import compute_I1_unified_general_P1Q1


class TestOracleCoeffP1Q1:
    """Test the analytic coefficient formula."""

    def test_coeff_at_t_half_is_zero_for_nonzero_ell(self):
        """When t=0.5, a=Rθ(2t-1)=0, so a^{ℓ₁+ℓ₂}=0 for ℓ₁+ℓ₂>0."""
        # At t=0.5, a=0, so coefficient should be 0 for ℓ₁+ℓ₂ > 0
        theta = 4 / 7
        R = 1.3036
        t = 0.5

        for ell1, ell2 in [(1, 1), (2, 2), (1, 2)]:
            coeff = oracle_coeff_P1Q1(theta, R, ell1, ell2, t)
            # a=0 → a^{ℓ₁+ℓ₂}=0 for ℓ₁+ℓ₂>0
            # But the coefficient extraction is more subtle:
            # when a=0, only the constant term in exp survives
            # So coefficient is 0 unless ℓ₁=ℓ₂=0
            assert coeff == 0.0 or abs(coeff) < 1e-15

    def test_coeff_nonzero_at_t_neq_half(self):
        """At t≠0.5, coefficient should be nonzero."""
        theta = 4 / 7
        R = 1.3036
        t = 0.7

        for ell1, ell2 in [(1, 1), (2, 2), (1, 2)]:
            coeff = oracle_coeff_P1Q1(theta, R, ell1, ell2, t)
            assert abs(coeff) > 1e-10, f"Expected nonzero coeff for ({ell1},{ell2})"

    def test_coeff_symmetric_in_ell_for_11(self):
        """For (1,1), swapping doesn't change result."""
        theta = 4 / 7
        R = 1.3036
        t = 0.7

        c_11 = oracle_coeff_P1Q1(theta, R, 1, 1, t)
        # Symmetric case, result should be well-defined
        assert abs(c_11) > 0


class TestOracleI1P1Q1:
    """Test the analytic I₁ computation with P=Q=1."""

    @pytest.mark.parametrize(
        "ell1,ell2",
        [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)],
    )
    def test_oracle_returns_finite(self, ell1, ell2):
        """Oracle returns finite values for all pairs."""
        theta = 4 / 7
        R = 1.3036
        val = oracle_I1_P1Q1(theta, R, ell1, ell2, n_quad_t=40)
        assert not float("inf") == val
        assert not float("-inf") == val
        import math

        assert not math.isnan(val)


class TestOracleVsUnified:
    """Test oracle matches unified P=Q=1 evaluator."""

    @pytest.mark.parametrize(
        "ell1,ell2",
        [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)],
    )
    def test_oracle_matches_unified_per_pair(self, ell1, ell2):
        """Oracle and unified should match exactly for each pair."""
        theta = 4 / 7
        R = 1.3036
        n_quad = 60

        oracle_val = oracle_I1_P1Q1(theta, R, ell1, ell2, n_quad_t=n_quad)
        unified_val = compute_I1_unified_general_P1Q1(
            R, theta, ell1, ell2, n_quad_u=n_quad, n_quad_t=n_quad
        )

        if abs(oracle_val) > 1e-12:
            rel_err = abs(unified_val / oracle_val - 1.0)
        else:
            rel_err = abs(unified_val - oracle_val)

        assert rel_err < 1e-8, f"({ell1},{ell2}): oracle={oracle_val}, unified={unified_val}"

    def test_validate_all_pairs(self):
        """Full validation via validate_oracle_vs_unified."""
        results = validate_oracle_vs_unified(theta=4 / 7, R=1.3036, n_quad=60)

        for (ell1, ell2), data in results.items():
            assert data["match"], f"({ell1},{ell2}): rel_err={data['rel_err']}"


class TestOracleSecondBenchmark:
    """Test oracle at second benchmark R=1.1167."""

    @pytest.mark.parametrize(
        "ell1,ell2",
        [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)],
    )
    def test_oracle_matches_unified_R_1_1167(self, ell1, ell2):
        """Oracle and unified should match at R=1.1167."""
        theta = 4 / 7
        R = 1.1167
        n_quad = 60

        oracle_val = oracle_I1_P1Q1(theta, R, ell1, ell2, n_quad_t=n_quad)
        unified_val = compute_I1_unified_general_P1Q1(
            R, theta, ell1, ell2, n_quad_u=n_quad, n_quad_t=n_quad
        )

        if abs(oracle_val) > 1e-12:
            rel_err = abs(unified_val / oracle_val - 1.0)
        else:
            rel_err = abs(unified_val - oracle_val)

        assert rel_err < 1e-8, f"({ell1},{ell2}): oracle={oracle_val}, unified={unified_val}"
