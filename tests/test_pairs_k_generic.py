"""
tests/test_pairs_k_generic.py
Phase 40: K-Generic Pairs Tests

Tests for the K-generic pairs and normalization module.
These tests verify pair generation and normalization for K=3,4,5.

Created: 2025-12-27 (Phase 40)
"""
import pytest
from math import factorial

from src.evaluator.pairs import (
    get_triangle_pairs,
    pair_count,
    factorial_norm,
    symmetry_factor,
    full_norm,
    pair_key,
    get_all_norms,
    validate_k_pairs,
)


class TestPairCount:
    """Test pair count formula."""

    def test_k3_has_6_pairs(self):
        """K=3 should have 6 pairs."""
        assert pair_count(3) == 6
        assert len(get_triangle_pairs(3)) == 6

    def test_k4_has_10_pairs(self):
        """K=4 should have 10 pairs."""
        assert pair_count(4) == 10
        assert len(get_triangle_pairs(4)) == 10

    def test_k5_has_15_pairs(self):
        """K=5 should have 15 pairs."""
        assert pair_count(5) == 15
        assert len(get_triangle_pairs(5)) == 15

    def test_triangular_formula(self):
        """Pair count should follow K(K+1)/2."""
        for K in range(1, 10):
            expected = K * (K + 1) // 2
            assert pair_count(K) == expected


class TestPairGeneration:
    """Test pair generation."""

    def test_k3_pairs_correct(self):
        """K=3 pairs should be exactly the expected 6 pairs."""
        expected = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        assert get_triangle_pairs(3) == expected

    def test_k4_pairs_correct(self):
        """K=4 pairs should be exactly the expected 10 pairs."""
        expected = [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 2), (2, 3), (2, 4),
            (3, 3), (3, 4),
            (4, 4)
        ]
        assert get_triangle_pairs(4) == expected

    def test_pairs_are_ordered(self):
        """All pairs should satisfy 1 ≤ ℓ₁ ≤ ℓ₂ ≤ K."""
        for K in [3, 4, 5]:
            for l1, l2 in get_triangle_pairs(K):
                assert 1 <= l1 <= l2 <= K


class TestFactorialNorm:
    """Test factorial normalization."""

    def test_norm_11(self):
        """(1,1) should have norm 1/(1!×1!) = 1."""
        assert factorial_norm(1, 1) == 1.0

    def test_norm_12(self):
        """(1,2) should have norm 1/(1!×2!) = 0.5."""
        assert factorial_norm(1, 2) == 0.5

    def test_norm_22(self):
        """(2,2) should have norm 1/(2!×2!) = 0.25."""
        assert factorial_norm(2, 2) == 0.25

    def test_norm_33(self):
        """(3,3) should have norm 1/(3!×3!) = 1/36."""
        assert abs(factorial_norm(3, 3) - 1/36) < 1e-15

    def test_norm_14(self):
        """(1,4) should have norm 1/(1!×4!) = 1/24."""
        assert abs(factorial_norm(1, 4) - 1/24) < 1e-15

    def test_norm_44(self):
        """(4,4) should have norm 1/(4!×4!) = 1/576."""
        assert abs(factorial_norm(4, 4) - 1/576) < 1e-15

    def test_norm_formula(self):
        """All norms should match 1/(ℓ₁!×ℓ₂!)."""
        for K in [3, 4, 5]:
            for l1, l2 in get_triangle_pairs(K):
                expected = 1.0 / (factorial(l1) * factorial(l2))
                assert abs(factorial_norm(l1, l2) - expected) < 1e-15


class TestSymmetryFactor:
    """Test symmetry factor."""

    def test_diagonal_is_1(self):
        """Diagonal pairs should have symmetry factor 1."""
        for ell in range(1, 6):
            assert symmetry_factor(ell, ell) == 1.0

    def test_off_diagonal_is_2(self):
        """Off-diagonal pairs should have symmetry factor 2."""
        for l1 in range(1, 5):
            for l2 in range(l1 + 1, 6):
                assert symmetry_factor(l1, l2) == 2.0


class TestPairKey:
    """Test pair key generation."""

    def test_key_format(self):
        """Keys should be two-digit strings."""
        assert pair_key(1, 1) == "11"
        assert pair_key(1, 2) == "12"
        assert pair_key(2, 3) == "23"
        assert pair_key(4, 4) == "44"

    def test_all_keys_unique(self):
        """All keys for K should be unique."""
        for K in [3, 4, 5]:
            keys = [pair_key(l1, l2) for l1, l2 in get_triangle_pairs(K)]
            assert len(keys) == len(set(keys))


class TestFullNorm:
    """Test combined normalization."""

    def test_full_norm_is_product(self):
        """Full norm should equal factorial_norm × symmetry_factor."""
        for K in [3, 4, 5]:
            for l1, l2 in get_triangle_pairs(K):
                expected = factorial_norm(l1, l2) * symmetry_factor(l1, l2)
                assert abs(full_norm(l1, l2) - expected) < 1e-15


class TestGetAllNorms:
    """Test bulk normalization retrieval."""

    def test_k3_norms(self):
        """K=3 norms should match expected values."""
        norms = get_all_norms(3)
        assert len(norms) == 6

        # Expected: factorial × symmetry
        assert norms["11"] == 1.0 * 1.0  # 1/1 × 1
        assert norms["12"] == 0.5 * 2.0  # 1/2 × 2 = 1.0
        assert norms["13"] == (1/6) * 2.0  # 1/6 × 2
        assert norms["22"] == 0.25 * 1.0  # 1/4 × 1
        assert abs(norms["23"] - (1/12) * 2.0) < 1e-15  # 1/12 × 2
        assert abs(norms["33"] - (1/36) * 1.0) < 1e-15  # 1/36 × 1

    def test_k4_norms_count(self):
        """K=4 should have 10 normalization entries."""
        norms = get_all_norms(4)
        assert len(norms) == 10


class TestValidation:
    """Test validation function."""

    def test_k3_validates(self):
        """K=3 should pass validation."""
        assert validate_k_pairs(3)

    def test_k4_validates(self):
        """K=4 should pass validation."""
        assert validate_k_pairs(4)

    def test_k5_validates(self):
        """K=5 should pass validation."""
        assert validate_k_pairs(5)

    def test_k2_validates(self):
        """K=2 should pass validation."""
        assert validate_k_pairs(2)


class TestK4SpecificNorms:
    """Test K=4 specific normalization values (from K4 implementation plan)."""

    def test_k4_factorial_norms_match_plan(self):
        """Verify K=4 factorial norms match the implementation plan."""
        # From docs/K4_IMPLEMENTATION_PLAN.md
        expected = {
            "11": 1.0,
            "12": 0.5,
            "13": 1/6,
            "14": 1/24,
            "22": 0.25,
            "23": 1/12,
            "24": 1/48,
            "33": 1/36,
            "34": 1/144,
            "44": 1/576,
        }

        for key, expected_val in expected.items():
            l1, l2 = int(key[0]), int(key[1])
            actual = factorial_norm(l1, l2)
            assert abs(actual - expected_val) < 1e-15, (
                f"Mismatch for {key}: expected {expected_val}, got {actual}"
            )
