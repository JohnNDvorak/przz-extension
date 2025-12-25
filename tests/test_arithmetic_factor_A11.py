"""
tests/test_arithmetic_factor_A11.py
Phase 14 Task 3: Tests for arithmetic factor A^{(1,1)}.

PAPER ANCHOR:
A^{(1,1)}_{α,β}(0,0;β,α) = Σ_p (log p / (p^{1+α+β} - 1))²

A^{(1,1)}_{0,0}(0,0;0,0) ≈ 1.385603705  (POSITIVE)

SIGN CONVENTION (Phase 14C):
The A11_prime_sum function returns POSITIVE values.
This matches PRZZ TeX Lines 1377-1389: S(0) ≈ 1.385603705.

This gives a NUMERIC ANCHOR independent of the rest of the pipeline.
"""

import pytest
import numpy as np
from src.ratios.arithmetic_factor import (
    primes_up_to,
    A11_prime_sum,
    prime_sum_converges,
)


# Paper's stated value
PAPER_A11_VALUE = 1.385603705


class TestPrimeGeneration:
    """Test the prime number sieve."""

    def test_primes_up_to_10(self):
        """First few primes should be [2, 3, 5, 7]."""
        primes = primes_up_to(10)
        assert primes == [2, 3, 5, 7]

    def test_primes_up_to_30(self):
        """Primes up to 30."""
        primes = primes_up_to(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_primes_count_100(self):
        """There are 25 primes less than 100."""
        primes = primes_up_to(100)
        assert len(primes) == 25


class TestA11PrimeSum:
    """Test the A^{(1,1)} prime sum computation."""

    def test_A11_is_positive(self):
        """
        A^{(1,1)} should be positive.

        SIGN CONVENTION (Phase 14C):
        The function A11_prime_sum returns the POSITIVE value ~1.3856.
        This matches PRZZ TeX Lines 1377-1389.

        Each term (log p / (p^s - 1))² is positive (squared).
        The sum of positive terms is positive.
        """
        result = A11_prime_sum(0.0, prime_cutoff=1000)
        # Paper value is positive - LOCKED CONVENTION
        assert result > 0, f"A11 should be positive, got {result}"

    def test_A11_sign_convention_locked(self):
        """
        GATE TEST (Phase 14C): Sign convention must be positive.

        This test locks the sign convention to prevent accidental
        sign flips that would break the "+5" derivation.

        PRZZ TeX Lines 1377-1389 state S(0) ≈ 1.385603705 (positive).
        """
        result = A11_prime_sum(0.0, prime_cutoff=10000)

        # Must be positive
        assert result > 0, "A11 must be POSITIVE per PRZZ convention"

        # Must match paper value within 1%
        PAPER_VALUE = 1.385603705
        assert abs(result - PAPER_VALUE) / PAPER_VALUE < 0.01, (
            f"A11(0) = {result}, must match paper value {PAPER_VALUE} within 1%"
        )

    def test_A11_matches_paper_at_zero_low_precision(self):
        """
        A^{(1,1)}(0) ≈ 1.385603705 as stated in paper.

        With prime cutoff 1000, we should get close.
        """
        result = A11_prime_sum(0.0, prime_cutoff=1000)

        # Should be within 5% of paper value
        rel_error = abs(result - PAPER_A11_VALUE) / PAPER_A11_VALUE
        assert rel_error < 0.05, (
            f"A11(0) = {result}, expected ~{PAPER_A11_VALUE}, error = {rel_error*100:.1f}%"
        )

    def test_A11_matches_paper_at_zero_high_precision(self):
        """
        With higher prime cutoff, should get closer to paper value.
        """
        result = A11_prime_sum(0.0, prime_cutoff=10000)

        # Should be within 1% of paper value
        rel_error = abs(result - PAPER_A11_VALUE) / PAPER_A11_VALUE
        assert rel_error < 0.01, (
            f"A11(0) = {result}, expected ~{PAPER_A11_VALUE}, error = {rel_error*100:.2f}%"
        )

    def test_A11_convergence_with_cutoff(self):
        """Prime sum should converge as cutoff increases."""
        cutoffs = [100, 500, 1000, 5000, 10000]
        results = [A11_prime_sum(0.0, prime_cutoff=c) for c in cutoffs]

        # Values should be getting closer together (converging)
        for i in range(len(results) - 1):
            diff_current = abs(results[i+1] - results[i])
            if i > 0:
                diff_prev = abs(results[i] - results[i-1])
                # Convergence: later differences should be smaller
                assert diff_current < diff_prev * 1.5, (
                    f"Not converging: diff at {cutoffs[i+1]}={diff_current}, "
                    f"diff at {cutoffs[i]}={diff_prev}"
                )


class TestA11ArgumentDependence:
    """Test how A^{(1,1)} depends on α+β."""

    def test_A11_decreases_with_positive_s(self):
        """
        For s = α+β > 0, each term (log p / (p^{1+s} - 1))² is smaller
        than at s=0, so |A11(s)| < |A11(0)|.
        """
        A11_0 = A11_prime_sum(0.0, prime_cutoff=5000)
        A11_pos = A11_prime_sum(0.5, prime_cutoff=5000)

        # For positive s, the sum should be smaller in magnitude
        assert abs(A11_pos) < abs(A11_0), (
            f"A11(0.5)={A11_pos} should be smaller than A11(0)={A11_0}"
        )

    def test_A11_increases_towards_negative_s(self):
        """
        For s approaching -1, terms blow up (p^{1+s} → p^0 = 1).
        So A11 should increase as s → -1.
        """
        A11_0 = A11_prime_sum(0.0, prime_cutoff=5000)
        A11_neg = A11_prime_sum(-0.5, prime_cutoff=5000)

        # For negative s (but s > -1), the sum should be larger
        assert abs(A11_neg) > abs(A11_0), (
            f"A11(-0.5)={A11_neg} should be larger than A11(0)={A11_0}"
        )


class TestConvergenceHelper:
    """Test the convergence checking helper."""

    def test_prime_sum_converges_returns_true_for_good_convergence(self):
        """Should return True when sum converges."""
        result = prime_sum_converges(
            PAPER_A11_VALUE,
            cutoffs=[1000, 5000, 10000],
            tol=0.02
        )
        # With these cutoffs, should converge to within 2%
        assert result is True

    def test_prime_sum_converges_returns_false_for_bad_tolerance(self):
        """Should return False when tolerance is too tight."""
        result = prime_sum_converges(
            PAPER_A11_VALUE,
            cutoffs=[100, 200],  # Too few primes
            tol=0.001  # Very tight
        )
        # With only 100-200 primes, can't hit 0.1% tolerance
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
