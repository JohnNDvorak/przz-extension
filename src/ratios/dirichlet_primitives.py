"""
src/ratios/dirichlet_primitives.py
Phase 14B Task 1: Dirichlet Arithmetic Primitives

PAPER ANCHORS:
=============
1. Λ(n) = log p if n = p^k for some prime p and k ≥ 1, else 0
   (von Mangoldt function)

2. Λ_k(n) defined by recurrence:
   Λ₁(n) = Λ(n)
   Λ_{k+1}(n) = Λ_k(n)·log(n) + (Λ ⋆ Λ_k)(n)

   Therefore:
   Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)

   CRITICAL: Λ₂(n) ≠ Λ(n)²

3. (1⋆Λ₁)(n) = Σ_{d|n} Λ(d) = log(n)  [EXACT identity]

4. A_{α,β}(0,0;β,α) = 1 exactly (Euler product cancellation)

The Dirichlet series identity:
   ζ(s) × (-1)^k × (ζ'/ζ)^{(k-1)}(s) = Σ Λ_k(n)/n^s
"""

from __future__ import annotations
from typing import List
from functools import lru_cache
import math


def get_divisors(n: int) -> List[int]:
    """
    Return all positive divisors of n.

    Args:
        n: Positive integer

    Returns:
        List of divisors in ascending order
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]

    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


@lru_cache(maxsize=1024)
def _smallest_prime_factor(n: int) -> int:
    """Return the smallest prime factor of n, or 0 if n <= 1."""
    if n <= 1:
        return 0
    if n == 2:
        return 2
    if n % 2 == 0:
        return 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return i
    return n  # n is prime


def _is_prime_power(n: int) -> tuple:
    """
    Check if n is a prime power p^k.

    Returns:
        (p, k) if n = p^k for some prime p and k ≥ 1
        (0, 0) otherwise
    """
    if n <= 1:
        return (0, 0)

    p = _smallest_prime_factor(n)
    if p == 0:
        return (0, 0)

    k = 0
    m = n
    while m % p == 0:
        m //= p
        k += 1

    if m == 1:
        return (p, k)
    else:
        return (0, 0)  # n has multiple distinct prime factors


def von_mangoldt(n: int) -> float:
    """
    The von Mangoldt function Λ(n).

    Λ(n) = log(p)  if n = p^k for some prime p and k ≥ 1
         = 0       otherwise

    This is the coefficient of n^{-s} in -ζ'(s)/ζ(s).

    Args:
        n: Positive integer

    Returns:
        log(p) if n is a prime power p^k, else 0

    Examples:
        >>> von_mangoldt(2)   # prime
        0.693147...
        >>> von_mangoldt(4)   # 2²
        0.693147...
        >>> von_mangoldt(6)   # 2×3
        0.0
    """
    if n <= 1:
        return 0.0

    p, k = _is_prime_power(n)
    if p > 0:
        return math.log(p)
    else:
        return 0.0


def lambda_star_lambda(n: int) -> float:
    """
    Dirichlet convolution (Λ⋆Λ)(n) = Σ_{d|n} Λ(d)·Λ(n/d).

    Args:
        n: Positive integer

    Returns:
        The convolution value

    Examples:
        >>> lambda_star_lambda(4)  # = Λ(1)Λ(4) + Λ(2)Λ(2) + Λ(4)Λ(1)
        ...                        # = 0 + log(2)² + 0 = log(2)²
    """
    if n <= 1:
        return 0.0

    total = 0.0
    for d in get_divisors(n):
        total += von_mangoldt(d) * von_mangoldt(n // d)
    return total


def lambda2(n: int) -> float:
    """
    Generalized von Mangoldt Λ₂(n) via recurrence.

    Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)

    CRITICAL: This is NOT Λ(n)².

    The recurrence is:
        Λ₁(n) = Λ(n)
        Λ_{k+1}(n) = Λ_k(n)·log(n) + (Λ ⋆ Λ_k)(n)

    For k=1: Λ₂(n) = Λ₁(n)·log(n) + (Λ⋆Λ₁)(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)

    Args:
        n: Positive integer

    Returns:
        Λ₂(n) value

    Examples:
        >>> lambda2(2)  # prime: log(2)² + 0
        0.48045...
        >>> lambda2(4)  # 2²: log(2)·2log(2) + log(2)² = 3·log(2)²
        1.44136...
    """
    if n <= 1:
        return 0.0

    # Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n)
    return von_mangoldt(n) * math.log(n) + lambda_star_lambda(n)


def one_star_lambda1(n: int) -> float:
    """
    Convolution (1⋆Λ₁)(n) = Σ_{d|n} Λ(d) = log(n).

    This is the EXACT identity from the paper. We don't need to
    compute the sum over divisors - it equals log(n) exactly.

    This appears in J₁₃ and J₁₄ bracket terms.

    Args:
        n: Positive integer

    Returns:
        log(n)

    Note:
        The identity (1⋆Λ)(n) = log(n) is classical.
        Proof: Σ_{n≥1} (1⋆Λ)(n)/n^s = ζ(s)·(-ζ'/ζ)(s) = -ζ'(s)
              and -ζ'(s) = Σ_{n≥1} log(n)/n^s.
    """
    if n <= 0:
        return 0.0
    return math.log(n)


def one_star_lambda2(n: int) -> float:
    """
    Convolution (1⋆Λ₂)(n) = Σ_{d|n} Λ₂(d).

    This uses the CORRECT Λ₂ from the recurrence, not Λ².

    Appears in the J₁₁ bracket term.

    Args:
        n: Positive integer

    Returns:
        The convolution sum

    Examples:
        >>> one_star_lambda2(2)  # = Λ₂(1) + Λ₂(2) = 0 + log(2)²
        0.48045...
    """
    if n <= 1:
        return 0.0 if n == 1 else 0.0

    total = 0.0
    for d in get_divisors(n):
        total += lambda2(d)
    return total


def A00_at_diagonal(alpha: complex, beta: complex) -> float:
    """
    Arithmetic factor A_{α,β}(0,0;β,α) = 1 exactly.

    The paper explicitly states this simplification from the Euler
    product cancellation. This is NOT approximate.

    All bracket terms 1-4 (J₁₁ through J₁₄) have this as a prefactor,
    so they effectively have prefactor 1.

    Args:
        alpha: First shift parameter (unused)
        beta: Second shift parameter (unused)

    Returns:
        1.0 (exactly)
    """
    return 1.0


# ============================================================================
# Higher-order Λ_k for completeness
# ============================================================================


def lambda_k(n: int, k: int) -> float:
    """
    Generalized von Mangoldt Λ_k(n) for arbitrary k ≥ 1.

    Defined by recurrence:
        Λ₁(n) = Λ(n)
        Λ_{k+1}(n) = Λ_k(n)·log(n) + (Λ ⋆ Λ_k)(n)

    The Dirichlet series identity:
        ζ(s) × (-1)^k × (d/ds)^k [1/ζ(s)] = Σ Λ_k(n)/n^s

    Args:
        n: Positive integer
        k: Order (k ≥ 1)

    Returns:
        Λ_k(n) value
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    if n <= 1:
        return 0.0

    if k == 1:
        return von_mangoldt(n)

    # Compute iteratively
    lambda_prev = {d: von_mangoldt(d) for d in get_divisors(n)}

    for _ in range(k - 1):
        lambda_curr = {}
        for m in get_divisors(n):
            # Λ_{k+1}(m) = Λ_k(m)·log(m) + (Λ⋆Λ_k)(m)
            term1 = lambda_prev.get(m, 0.0) * math.log(m) if m > 1 else 0.0

            # (Λ⋆Λ_k)(m) = Σ_{d|m} Λ(d)·Λ_k(m/d)
            term2 = 0.0
            for d in get_divisors(m):
                term2 += von_mangoldt(d) * lambda_prev.get(m // d, 0.0)

            lambda_curr[m] = term1 + term2

        lambda_prev = lambda_curr

    return lambda_prev.get(n, 0.0)
