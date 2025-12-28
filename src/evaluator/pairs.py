"""
src/evaluator/pairs.py
Phase 40: K-Generic Triangle Pairs and Normalization

This module provides K-generic utilities for working with (ℓ₁, ℓ₂) pairs,
replacing hardcoded K=3 logic with general formulas.

DESIGN RATIONALE (GPT Guidance):
================================
Creating `terms_k4_d1.py` as a fork of `terms_k3_d1.py` leads to:
- Code path divergence
- Silent semantic drift
- Technical debt (6,700-line evaluate.py style)

Instead, this module provides K-generic functions that drive both K=3 and K=4
with the same machinery.

PAIR STRUCTURE:
===============
For K pieces, we have K(K+1)/2 distinct pairs (ℓ₁, ℓ₂) with 1 ≤ ℓ₁ ≤ ℓ₂ ≤ K:

K=3: 6 pairs  → (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
K=4: 10 pairs → (1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)

Created: 2025-12-27 (Phase 40)
"""
from __future__ import annotations
from math import factorial
from typing import List, Tuple, Dict


def get_triangle_pairs(K: int) -> List[Tuple[int, int]]:
    """
    Return all (ℓ₁, ℓ₂) pairs with 1 ≤ ℓ₁ ≤ ℓ₂ ≤ K.

    Args:
        K: Number of mollifier pieces

    Returns:
        List of (ℓ₁, ℓ₂) tuples in canonical order

    Examples:
        >>> get_triangle_pairs(3)
        [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        >>> len(get_triangle_pairs(4))
        10
    """
    return [(l1, l2) for l1 in range(1, K + 1) for l2 in range(l1, K + 1)]


def pair_count(K: int) -> int:
    """
    Return the number of (ℓ₁, ℓ₂) pairs for given K.

    Formula: K(K+1)/2 (triangular number)

    Args:
        K: Number of mollifier pieces

    Returns:
        Number of pairs

    Examples:
        >>> pair_count(3)
        6
        >>> pair_count(4)
        10
    """
    return K * (K + 1) // 2


def factorial_norm(l1: int, l2: int) -> float:
    """
    Return the factorial normalization 1/(ℓ₁! × ℓ₂!).

    This comes from the Taylor series expansion:
        1/ℓ! from d^ℓ/dx^ℓ at x=0

    Args:
        l1: First piece index (1 ≤ ℓ₁)
        l2: Second piece index (ℓ₁ ≤ ℓ₂)

    Returns:
        1/(ℓ₁! × ℓ₂!)

    Examples:
        >>> factorial_norm(1, 1)
        1.0
        >>> factorial_norm(1, 2)
        0.5
        >>> factorial_norm(2, 2)
        0.25
        >>> factorial_norm(3, 3)
        0.027777777777777776
    """
    return 1.0 / (factorial(l1) * factorial(l2))


def symmetry_factor(l1: int, l2: int) -> float:
    """
    Return the symmetry factor for pair (ℓ₁, ℓ₂).

    For off-diagonal pairs (ℓ₁ < ℓ₂), we count both (ℓ₁, ℓ₂) and (ℓ₂, ℓ₁),
    giving a factor of 2.

    For diagonal pairs (ℓ₁ = ℓ₂), there's no symmetry doubling.

    Args:
        l1: First piece index
        l2: Second piece index

    Returns:
        2.0 for off-diagonal, 1.0 for diagonal

    Examples:
        >>> symmetry_factor(1, 1)
        1.0
        >>> symmetry_factor(1, 2)
        2.0
        >>> symmetry_factor(2, 3)
        2.0
    """
    return 2.0 if l1 < l2 else 1.0


def full_norm(l1: int, l2: int) -> float:
    """
    Return the full normalization: factorial_norm × symmetry_factor.

    Args:
        l1: First piece index
        l2: Second piece index

    Returns:
        Full normalization factor

    Examples:
        >>> full_norm(1, 1)
        1.0
        >>> full_norm(1, 2)
        1.0
        >>> full_norm(2, 3)
        0.16666666666666666
    """
    return factorial_norm(l1, l2) * symmetry_factor(l1, l2)


def pair_key(l1: int, l2: int) -> str:
    """
    Return string key for pair, e.g., "11", "12", "23".

    Args:
        l1: First piece index
        l2: Second piece index

    Returns:
        String key like "11", "12", "33"

    Examples:
        >>> pair_key(1, 1)
        '11'
        >>> pair_key(1, 4)
        '14'
    """
    return f"{l1}{l2}"


def get_all_norms(K: int) -> Dict[str, float]:
    """
    Return dictionary of all normalization factors for K pieces.

    Args:
        K: Number of mollifier pieces

    Returns:
        Dict mapping pair key to full normalization

    Examples:
        >>> norms = get_all_norms(3)
        >>> norms['11']
        1.0
        >>> norms['23']
        0.16666666666666666
    """
    norms = {}
    for l1, l2 in get_triangle_pairs(K):
        norms[pair_key(l1, l2)] = full_norm(l1, l2)
    return norms


def get_all_factorial_norms(K: int) -> Dict[str, float]:
    """
    Return dictionary of factorial normalizations only (no symmetry factor).

    Args:
        K: Number of mollifier pieces

    Returns:
        Dict mapping pair key to 1/(ℓ₁! × ℓ₂!)
    """
    norms = {}
    for l1, l2 in get_triangle_pairs(K):
        norms[pair_key(l1, l2)] = factorial_norm(l1, l2)
    return norms


def get_all_symmetry_factors(K: int) -> Dict[str, float]:
    """
    Return dictionary of symmetry factors.

    Args:
        K: Number of mollifier pieces

    Returns:
        Dict mapping pair key to symmetry factor (1 or 2)
    """
    factors = {}
    for l1, l2 in get_triangle_pairs(K):
        factors[pair_key(l1, l2)] = symmetry_factor(l1, l2)
    return factors


# Validation functions

def validate_k_pairs(K: int) -> bool:
    """
    Validate that pair generation is correct for K.

    Checks:
    1. Pair count matches K(K+1)/2
    2. All pairs satisfy 1 ≤ ℓ₁ ≤ ℓ₂ ≤ K
    3. Normalization factors are correct

    Args:
        K: Number of mollifier pieces

    Returns:
        True if all checks pass
    """
    pairs = get_triangle_pairs(K)

    # Check count
    expected_count = K * (K + 1) // 2
    if len(pairs) != expected_count:
        return False

    # Check ordering
    for l1, l2 in pairs:
        if not (1 <= l1 <= l2 <= K):
            return False

    # Check normalization values are positive
    for l1, l2 in pairs:
        if factorial_norm(l1, l2) <= 0:
            return False
        if symmetry_factor(l1, l2) not in (1.0, 2.0):
            return False

    return True


if __name__ == "__main__":
    # Self-test
    print("K-Generic Pairs Module Self-Test")
    print("=" * 50)

    for K in [3, 4, 5]:
        pairs = get_triangle_pairs(K)
        print(f"\nK={K}: {len(pairs)} pairs")
        print(f"  Pairs: {pairs}")
        print(f"  Validation: {'PASS' if validate_k_pairs(K) else 'FAIL'}")

        norms = get_all_norms(K)
        print(f"  Normalizations:")
        for key, val in norms.items():
            print(f"    {key}: {val:.6f}")
