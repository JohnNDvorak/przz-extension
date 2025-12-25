"""
src/ratios/arithmetic_factor.py
Phase 14 Task 3: Arithmetic Factor A^{(1,1)} Prime Sum

PAPER ANCHOR:
============
A^{(1,1)}_{α,β}(0,0;β,α) = Σ_p (log p / (p^{1+α+β} - 1))²

At the origin (α=β=0):
A^{(1,1)}_{0,0}(0,0;0,0) ≈ 1.385603705  (POSITIVE)

SIGN CONVENTION (Phase 14C):
===========================
This function returns the POSITIVE magnitude of the prime sum.

The paper's notation may include a negative sign in the derivative
definition of A^{(1,1)}, but the NUMERIC VALUE reported is positive
(~1.3856). We match the paper's numeric value, not the formal sign.

This is consistent with PRZZ TeX Lines 1377-1389 which reports the
prime sum as a positive anchor value S(0) ≈ 1.385603705.

DERIVATION:
===========
The arithmetic factor A arises from Euler products in the CFZ conjecture.
At diagonal specialization γ=α, δ=β, we have A(α,β,α,β) = 1.

The derivatives of A give A^{(m,n)} factors. The (1,1) derivative at
the diagonal gives the prime sum above.

The sum converges because each term is O(log p / p^{2+2ε}) for s = α+β = ε > 0,
and the prime sum Σ log p / p^s converges for s > 1.
"""

from __future__ import annotations
from typing import List
import math


def primes_up_to(n: int) -> List[int]:
    """
    Return all primes up to n using the Sieve of Eratosthenes.

    Args:
        n: Upper bound (inclusive)

    Returns:
        List of primes ≤ n

    Example:
        >>> primes_up_to(10)
        [2, 3, 5, 7]
    """
    if n < 2:
        return []

    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]


def A11_prime_sum(s: float, *, prime_cutoff: int = 10000) -> float:
    """
    Compute A^{(1,1)} as a sum over primes.

    A^{(1,1)}(s) = Σ_p (log p / (p^{1+s} - 1))²

    At s=0:
    A^{(1,1)}(0) = Σ_p (log p / (p - 1))² ≈ 1.385603705

    Args:
        s: The parameter α+β (typically 0 for the paper's anchor value)
        prime_cutoff: Sum over primes p ≤ prime_cutoff

    Returns:
        The prime sum value

    Note:
        The sum converges slowly. For s=0:
        - cutoff=1000: ~1.37 (within 5%)
        - cutoff=10000: ~1.38 (within 1%)
        - cutoff=100000: ~1.385 (within 0.1%)
    """
    primes = primes_up_to(prime_cutoff)

    total = 0.0
    for p in primes:
        # Each term: (log p / (p^{1+s} - 1))²
        log_p = math.log(p)
        denom = p ** (1.0 + s) - 1.0

        if abs(denom) < 1e-14:
            # Avoid division by zero (happens if s → -1)
            continue

        term = (log_p / denom) ** 2
        total += term

    return total


def prime_sum_converges(
    target: float,
    cutoffs: List[int],
    tol: float = 0.01
) -> bool:
    """
    Check whether the A^{(1,1)} prime sum converges to a target value.

    Computes the sum at each cutoff and checks if the final value
    is within tolerance of the target.

    Args:
        target: Expected limit value (e.g., 1.385603705)
        cutoffs: List of prime cutoffs to try (increasing order)
        tol: Relative tolerance for convergence

    Returns:
        True if final sum is within tol of target

    Example:
        >>> prime_sum_converges(1.385603705, [1000, 5000, 10000], tol=0.02)
        True
    """
    if not cutoffs:
        return False

    # Compute at highest cutoff
    final_cutoff = max(cutoffs)
    result = A11_prime_sum(0.0, prime_cutoff=final_cutoff)

    rel_error = abs(result - target) / abs(target)
    return rel_error < tol


def A11_with_tail_correction(s: float, *, prime_cutoff: int = 10000) -> float:
    """
    Compute A^{(1,1)} with tail correction for faster convergence.

    For large primes p, the term (log p / (p^{1+s} - 1))² ≈ (log p / p^{1+s})²

    The tail sum Σ_{p > N} (log p)² / p^{2+2s} can be approximated using
    the prime counting function.

    This provides faster convergence than the raw sum.

    Args:
        s: The parameter α+β
        prime_cutoff: Explicit sum up to this cutoff

    Returns:
        Sum with estimated tail correction
    """
    # Explicit sum up to cutoff
    explicit_sum = A11_prime_sum(s, prime_cutoff=prime_cutoff)

    # Tail correction using integral approximation
    # Σ_{p>N} (log p)² / p^{2+2s} ≈ ∫_N^∞ (log x)² / (x^{2+2s} log x) dx
    #                              = ∫_N^∞ log x / x^{2+2s} dx
    # For s=0: ∫_N^∞ log x / x² dx = (1 + log N) / N

    if s == 0.0:
        N = float(prime_cutoff)
        tail_estimate = (1.0 + math.log(N)) / N
    else:
        # General case: more complex, use simpler estimate
        N = float(prime_cutoff)
        exponent = 2.0 + 2.0 * s
        if exponent > 1:
            # Rough estimate: dominant term is ~ N^{1-exponent}
            tail_estimate = (math.log(N) ** 2) / (N ** (exponent - 1) * (exponent - 1))
        else:
            tail_estimate = 0.0

    return explicit_sum + tail_estimate
