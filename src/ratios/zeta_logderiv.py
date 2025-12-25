"""
src/ratios/zeta_logderiv.py
Phase 14B Task 2: ζ'/ζ Evaluator

PAPER ANCHORS:
=============
1. ζ'/ζ(s) = -Σ_p log(p)/(p^s - 1) for Re(s) > 1
   (Logarithmic derivative via prime sum)

2. Laurent expansion near s=1:
   ζ'/ζ(1+ε) = -1/ε + γ_E + O(ε)
   where γ_E ≈ 0.5772156649... is Euler-Mascheroni

3. The bracket terms J₁₂, J₁₃, J₁₄ all use ζ'/ζ(1+α+s) or ζ'/ζ(1+β+u)
   evaluated at small arguments, requiring the Laurent expansion.

The logarithmic derivative encodes prime distribution information
and appears throughout the PRZZ ratios framework.
"""

from __future__ import annotations
import math
from typing import List

# Euler-Mascheroni constant γ ≈ 0.5772156649...
# This is the constant term in the Laurent expansion of ζ'/ζ(1+ε)
EULER_MASCHERONI = 0.5772156649015329

# Stieltjes constants γ_1, γ_2 for higher-order expansion
# ζ'/ζ(1+ε) = -1/ε + γ_0 + γ_1·ε + γ_2·ε²/2 + ...
# where γ_0 = γ_E (Euler-Mascheroni)
STIELTJES_GAMMA1 = -0.0728158454836767  # γ_1
STIELTJES_GAMMA2 = -0.0096903631928723  # γ_2


def _sieve_primes(limit: int) -> List[int]:
    """
    Sieve of Eratosthenes for primes up to limit.

    Args:
        limit: Upper bound for primes

    Returns:
        List of primes up to limit
    """
    if limit < 2:
        return []

    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    return [i for i in range(2, limit + 1) if sieve[i]]


def zeta_log_deriv_1_plus_eps(eps: complex, order: int = 2) -> complex:
    """
    Laurent expansion of ζ'/ζ(1+ε) around ε=0.

    ζ'/ζ(1+ε) = -1/ε + γ_E + γ_1·ε + γ_2·ε²/2 + O(ε³)

    where:
    - γ_E = 0.5772... is Euler-Mascheroni (constant term)
    - γ_1, γ_2 are Stieltjes constants

    This expansion is CRITICAL for evaluating the bracket terms
    J₁₂, J₁₃, J₁₄ which involve ζ'/ζ(1+α+s) at small arguments.

    Args:
        eps: The argument, should be small for expansion validity
        order: Expansion order (2 = pole + constant, 3 = adds linear term)

    Returns:
        ζ'/ζ(1+ε) approximated to specified order

    Note:
        For |ε| > 0.5, consider using the prime sum instead.
        The Laurent expansion is most accurate for |ε| < 0.1.
    """
    if abs(eps) < 1e-15:
        # Avoid division by zero - return a large negative number
        # representing the pole
        return complex(-1e15)

    # Pole term: -1/ε
    result = -1.0 / eps

    if order >= 2:
        # Constant term: +γ_E
        result += EULER_MASCHERONI

    if order >= 3:
        # Linear term: γ_1·ε
        result += STIELTJES_GAMMA1 * eps

    if order >= 4:
        # Quadratic term: γ_2·ε²/2
        result += STIELTJES_GAMMA2 * eps * eps / 2.0

    return result


def zeta_log_deriv_prime_sum(
    s: complex,
    prime_cutoff: int = 1000
) -> complex:
    """
    Evaluate ζ'/ζ(s) via prime sum for Re(s) > 1.

    ζ'/ζ(s) = -Σ_p log(p)/(p^s - 1)

    This converges for Re(s) > 1. The convergence is slower
    as Re(s) approaches 1 from above.

    PAPER CONTEXT:
    This formula comes from the Euler product:
        ζ(s) = Π_p (1 - p^{-s})^{-1}

    Taking log: log ζ(s) = -Σ_p log(1 - p^{-s})
    Differentiating: ζ'/ζ(s) = -Σ_p log(p) · p^{-s} / (1 - p^{-s})
                              = -Σ_p log(p) / (p^s - 1)

    Args:
        s: Complex argument with Re(s) > 1
        prime_cutoff: Upper bound for primes in sum

    Returns:
        ζ'/ζ(s) approximated by truncated prime sum

    Note:
        The result is always NEGATIVE for real s > 1.
        As s → ∞, ζ'/ζ(s) → 0.
    """
    primes = _sieve_primes(prime_cutoff)

    total = complex(0.0)
    for p in primes:
        # log(p) / (p^s - 1)
        p_to_s = p ** s
        if abs(p_to_s - 1) > 1e-15:  # Avoid division issues
            total += math.log(p) / (p_to_s - 1)

    return -total  # Note the minus sign in the formula


def zeta_log_deriv(
    s: complex,
    *,
    prime_cutoff: int = 5000,
    use_laurent_threshold: float = 2.0
) -> complex:
    """
    Unified evaluator for ζ'/ζ(s).

    Automatically selects method:
    - For |s - 1| < threshold: use Laurent expansion
    - For Re(s) > 1 and |s-1| >= threshold: use prime sum
    - For other regions: extend Laurent expansion (may be less accurate)

    The Laurent expansion ζ'/ζ(1+ε) = -1/ε + γ + O(ε) is valid for |ε| small,
    but we extend it to larger regions for the PRZZ application where
    α, β can be negative (e.g., α = -R ≈ -1.3).

    Args:
        s: Complex argument
        prime_cutoff: Upper bound for primes in sum method
        use_laurent_threshold: When to switch to Laurent expansion

    Returns:
        ζ'/ζ(s)

    Note:
        For s near 0 or negative, the Laurent expansion is used but may be
        less accurate. The pole at s=1 gives -1/ε behavior.
    """
    eps = s - 1.0

    if s.real > 1 and abs(eps) >= use_laurent_threshold:
        # Well away from pole with Re(s) > 1, use prime sum
        return zeta_log_deriv_prime_sum(s, prime_cutoff=prime_cutoff)
    else:
        # Near s=1 or for s < 1, use Laurent expansion
        # This extends the Laurent expansion to a larger region
        # For the PRZZ bracket terms with α=-R, we need this
        return zeta_log_deriv_1_plus_eps(eps, order=4)


# =============================================================================
# Derived quantities for bracket terms
# =============================================================================


def zeta_log_deriv_product(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    prime_cutoff: int = 1000
) -> complex:
    """
    Product (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u).

    This appears in the J₁₂ bracket term.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        prime_cutoff: For prime sum method

    Returns:
        Product of two ζ'/ζ evaluations
    """
    zeta1 = zeta_log_deriv(1.0 + alpha + s, prime_cutoff=prime_cutoff)
    zeta2 = zeta_log_deriv(1.0 + beta + u, prime_cutoff=prime_cutoff)
    return zeta1 * zeta2


def zeta_log_deriv_at_alpha_s(
    alpha: complex,
    s: complex,
    *,
    prime_cutoff: int = 1000
) -> complex:
    """
    Evaluate (ζ'/ζ)(1+α+s).

    This appears in J₁₄ bracket term (α-side).

    Args:
        alpha: Shift parameter
        s: Contour variable
        prime_cutoff: For prime sum method

    Returns:
        ζ'/ζ(1+α+s)
    """
    return zeta_log_deriv(1.0 + alpha + s, prime_cutoff=prime_cutoff)


def zeta_log_deriv_at_beta_u(
    beta: complex,
    u: complex,
    *,
    prime_cutoff: int = 1000
) -> complex:
    """
    Evaluate (ζ'/ζ)(1+β+u).

    This appears in J₁₃ bracket term (β-side).

    Args:
        beta: Shift parameter
        u: Contour variable
        prime_cutoff: For prime sum method

    Returns:
        ζ'/ζ(1+β+u)
    """
    return zeta_log_deriv(1.0 + beta + u, prime_cutoff=prime_cutoff)
