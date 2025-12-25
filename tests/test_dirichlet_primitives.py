"""
tests/test_dirichlet_primitives.py
Phase 14B Task 1: Tests for Dirichlet arithmetic primitives.

PAPER ANCHORS:
- Λ(n) = log p if n = p^k, else 0 (von Mangoldt)
- Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)  [NOT Λ(n)²!]
- (1⋆Λ₁)(n) = log(n)  [EXACT identity]
- A_{α,β}(0,0;β,α) = 1  [exactly]
"""

import pytest
import math
from src.ratios.dirichlet_primitives import (
    von_mangoldt,
    lambda2,
    lambda_star_lambda,
    one_star_lambda1,
    one_star_lambda2,
    get_divisors,
    A00_at_diagonal,
)


class TestVonMangoldt:
    """Test the von Mangoldt function Λ(n)."""

    def test_von_mangoldt_at_primes(self):
        """Λ(p) = log(p) for primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        for p in primes:
            assert abs(von_mangoldt(p) - math.log(p)) < 1e-10, (
                f"Λ({p}) should be log({p}) = {math.log(p)}"
            )

    def test_von_mangoldt_at_prime_powers(self):
        """Λ(p^k) = log(p) for prime powers."""
        # p^k : expected log(p)
        cases = [
            (4, math.log(2)),    # 2²
            (8, math.log(2)),    # 2³
            (9, math.log(3)),    # 3²
            (27, math.log(3)),   # 3³
            (25, math.log(5)),   # 5²
            (32, math.log(2)),   # 2⁵
        ]
        for n, expected in cases:
            assert abs(von_mangoldt(n) - expected) < 1e-10, (
                f"Λ({n}) should be {expected}, got {von_mangoldt(n)}"
            )

    def test_von_mangoldt_at_composite(self):
        """Λ(n) = 0 for n with multiple distinct prime factors."""
        composites = [6, 10, 12, 14, 15, 18, 20, 21, 22]
        for n in composites:
            assert von_mangoldt(n) == 0.0, (
                f"Λ({n}) should be 0 for composite with distinct primes"
            )

    def test_von_mangoldt_at_1(self):
        """Λ(1) = 0 by convention."""
        assert von_mangoldt(1) == 0.0


class TestLambda2Recurrence:
    """Test Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)."""

    def test_lambda2_at_prime(self):
        """Λ₂(p) = Λ(p)·log(p) + (Λ⋆Λ)(p) = log(p)² + 0."""
        for p in [2, 3, 5, 7]:
            expected = math.log(p) ** 2  # Λ(p)·log(p) = log(p)², (Λ⋆Λ)(p)=0
            actual = lambda2(p)
            assert abs(actual - expected) < 1e-10, (
                f"Λ₂({p}) = {actual}, expected {expected}"
            )

    def test_lambda2_at_prime_square(self):
        """
        Λ₂(p²) = Λ(p²)·log(p²) + (Λ⋆Λ)(p²)
               = log(p)·2log(p) + Λ(1)Λ(p²) + Λ(p)Λ(p)
               = 2log(p)² + 0 + log(p)²
               = 3log(p)²
        """
        for p in [2, 3, 5]:
            expected = 3 * math.log(p) ** 2
            actual = lambda2(p * p)
            assert abs(actual - expected) < 1e-10, (
                f"Λ₂({p}²) = {actual}, expected {expected}"
            )

    def test_lambda2_not_lambda_squared(self):
        """
        Λ₂(4) ≠ Λ(4)².

        Λ(4) = log(2), so Λ(4)² = log(2)² ≈ 0.48
        Λ₂(4) = 3·log(2)² ≈ 1.44
        """
        lambda_4 = von_mangoldt(4)
        lambda2_4 = lambda2(4)
        lambda_4_squared = lambda_4 ** 2

        # They should NOT be equal
        assert abs(lambda2_4 - lambda_4_squared) > 0.1, (
            f"Λ₂(4)={lambda2_4} should differ from Λ(4)²={lambda_4_squared}"
        )

    def test_lambda2_at_composite(self):
        """Λ₂(n) for n with multiple distinct primes."""
        # Λ₂(6) = Λ(6)·log(6) + (Λ⋆Λ)(6)
        #       = 0 + Λ(1)Λ(6) + Λ(2)Λ(3) + Λ(3)Λ(2) + Λ(6)Λ(1)
        #       = 0 + 0 + log(2)log(3) + log(3)log(2) + 0
        #       = 2·log(2)·log(3)
        expected = 2 * math.log(2) * math.log(3)
        actual = lambda2(6)
        assert abs(actual - expected) < 1e-10, (
            f"Λ₂(6) = {actual}, expected {expected}"
        )


class TestLambdaStarLambda:
    """Test the Dirichlet convolution (Λ⋆Λ)(n)."""

    def test_lambda_star_lambda_at_prime(self):
        """(Λ⋆Λ)(p) = 0 for prime p (only divisors 1 and p)."""
        for p in [2, 3, 5, 7]:
            # Divisors: 1, p
            # Λ(1)Λ(p) + Λ(p)Λ(1) = 0
            assert lambda_star_lambda(p) == 0.0

    def test_lambda_star_lambda_at_prime_square(self):
        """
        (Λ⋆Λ)(p²) = Σ_{d|p²} Λ(d)Λ(p²/d)
                  = Λ(1)Λ(p²) + Λ(p)Λ(p) + Λ(p²)Λ(1)
                  = 0 + log(p)² + 0
                  = log(p)²
        """
        for p in [2, 3, 5]:
            expected = math.log(p) ** 2
            actual = lambda_star_lambda(p * p)
            assert abs(actual - expected) < 1e-10

    def test_lambda_star_lambda_at_6(self):
        """
        (Λ⋆Λ)(6) = Σ_{d|6} Λ(d)Λ(6/d)
        Divisors of 6: 1, 2, 3, 6
        = Λ(1)Λ(6) + Λ(2)Λ(3) + Λ(3)Λ(2) + Λ(6)Λ(1)
        = 0 + log(2)log(3) + log(3)log(2) + 0
        = 2·log(2)·log(3)
        """
        expected = 2 * math.log(2) * math.log(3)
        actual = lambda_star_lambda(6)
        assert abs(actual - expected) < 1e-10


class TestOneStarLambda1:
    """Test that (1⋆Λ₁)(n) = log(n) exactly."""

    def test_one_star_lambda1_is_log(self):
        """(1⋆Λ₁)(n) = log(n) for all n ≥ 1."""
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 100]:
            expected = math.log(n) if n > 0 else 0
            actual = one_star_lambda1(n)
            assert abs(actual - expected) < 1e-10, (
                f"(1⋆Λ₁)({n}) = {actual}, expected log({n}) = {expected}"
            )

    def test_one_star_lambda1_at_1(self):
        """(1⋆Λ₁)(1) = log(1) = 0."""
        assert one_star_lambda1(1) == 0.0


class TestOneStarLambda2:
    """Test (1⋆Λ₂)(n) = Σ_{d|n} Λ₂(d)."""

    def test_one_star_lambda2_at_prime(self):
        """
        (1⋆Λ₂)(p) = Λ₂(1) + Λ₂(p) = 0 + log(p)² = log(p)²
        """
        for p in [2, 3, 5, 7]:
            expected = math.log(p) ** 2
            actual = one_star_lambda2(p)
            assert abs(actual - expected) < 1e-10

    def test_one_star_lambda2_at_prime_square(self):
        """
        (1⋆Λ₂)(p²) = Λ₂(1) + Λ₂(p) + Λ₂(p²)
                   = 0 + log(p)² + 3·log(p)²
                   = 4·log(p)²
        """
        for p in [2, 3]:
            expected = 4 * math.log(p) ** 2
            actual = one_star_lambda2(p * p)
            assert abs(actual - expected) < 1e-10, (
                f"(1⋆Λ₂)({p}²) = {actual}, expected {expected}"
            )

    def test_one_star_lambda2_is_positive(self):
        """(1⋆Λ₂)(n) should be non-negative for all n ≥ 1."""
        for n in range(1, 50):
            assert one_star_lambda2(n) >= 0


class TestA00AtDiagonal:
    """Test that A_{α,β}(0,0;β,α) = 1 exactly."""

    def test_A00_is_exactly_one(self):
        """Paper explicitly states A(0,0;β,α) = 1."""
        for alpha in [0.0, 0.1, 0.5, 1.0, -0.5]:
            for beta in [0.0, 0.1, 0.5, 1.0, -0.5]:
                assert A00_at_diagonal(alpha, beta) == 1.0


class TestDivisors:
    """Test the divisor helper function."""

    def test_divisors_of_1(self):
        assert get_divisors(1) == [1]

    def test_divisors_of_prime(self):
        assert set(get_divisors(7)) == {1, 7}

    def test_divisors_of_12(self):
        assert set(get_divisors(12)) == {1, 2, 3, 4, 6, 12}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
