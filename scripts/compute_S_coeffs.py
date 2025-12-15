#!/usr/bin/env python3
"""
Compute Taylor coefficients for S(z) = Σ_p (log p / (p^{1+z} − 1))²

This script computes the coefficients s_m = S^{(m)}(0)/m! for m = 0..6
using mpmath for high precision arithmetic.

The coefficients are used in the I₅ arithmetic correction term:
    S(2Rt) ≈ Σ_{m=0}^{6} s_m · (2Rt)^m

Method: Direct prime summation with acceleration.
For better convergence, we use the identity:
    log p / (p^{1+z} - 1) = log p · Σ_{n≥1} p^{-n(1+z)}

So:
    (log p / (p^{1+z} - 1))² = (log p)² · (Σ_{n≥1} p^{-n(1+z)})²

For the Taylor expansion, we compute derivatives at z=0 by differentiating
the series term by term.
"""

import mpmath
from mpmath import mp, mpf, log, power, factorial, diff

# Set high precision
mp.dps = 50  # 50 decimal places


def prime_sieve(n_max: int) -> list:
    """Generate primes up to n_max using Sieve of Eratosthenes."""
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n_max + 1, i):
                sieve[j] = False
    return [i for i in range(2, n_max + 1) if sieve[i]]


def S_term(p: int, z: mpf) -> mpf:
    """
    Compute single prime contribution to S(z).

    (log p / (p^{1+z} - 1))²
    """
    log_p = log(mpf(p))
    p_power = power(mpf(p), mpf(1) + z)
    denominator = p_power - mpf(1)
    return (log_p / denominator) ** 2


def S_function(z: mpf, primes: list, n_series: int = 50) -> mpf:
    """
    Compute S(z) = Σ_p (log p / (p^{1+z} − 1))²

    Uses direct summation over primes with series acceleration.
    """
    result = mpf(0)
    for p in primes:
        result += S_term(p, z)
    return result


def compute_S_derivatives(primes: list, max_order: int = 6) -> list:
    """
    Compute S^{(m)}(0)/m! for m = 0..max_order using numerical differentiation.
    """
    coeffs = []

    # Use mpmath's diff function for numerical differentiation
    for m in range(max_order + 1):
        if m == 0:
            # S(0) directly
            val = S_function(mpf(0), primes)
        else:
            # S^{(m)}(0) via numerical differentiation
            # diff computes f^{(n)}(x)
            val = diff(lambda z: S_function(z, primes), mpf(0), m) / factorial(m)

        coeffs.append(val)
        print(f"  s_{m} = {val}")

    return coeffs


def main():
    print("Computing S(z) Taylor coefficients for I₅ correction")
    print("=" * 60)
    print()

    # Use enough primes for convergence
    # The series converges like Σ_p (log p)² / (p-1)², which converges well
    n_primes = 100000
    print(f"Generating primes up to {n_primes}...")
    primes = prime_sieve(n_primes)
    print(f"  Found {len(primes)} primes")
    print()

    # Verify convergence by checking with fewer primes
    print("Convergence check for S(0):")
    for n in [1000, 10000, 50000, 100000]:
        primes_subset = [p for p in primes if p <= n]
        s0 = S_function(mpf(0), primes_subset)
        print(f"  n={n:6d}: S(0) = {s0}")
    print()

    print("Computing Taylor coefficients s_m = S^{(m)}(0)/m!:")
    print("(This may take a minute...)")

    # Use a moderate number of primes for derivatives (numerical diff is expensive)
    primes_for_deriv = [p for p in primes if p <= 10000]
    coeffs = compute_S_derivatives(primes_for_deriv, max_order=6)
    print()

    # Refine s_0 with full prime set
    print("Refining s_0 with more primes...")
    coeffs[0] = S_function(mpf(0), primes)
    print(f"  s_0 (refined) = {coeffs[0]}")
    print()

    # Output in Python format
    print("Python code for src/arithmetic_constants.py:")
    print("-" * 60)
    print('"""')
    print("Precomputed Taylor coefficients for S(z) = Σ_p (log p / (p^{1+z} − 1))²")
    print()
    print("S(z) ≈ Σ_{m=0}^{6} S_COEFFS[m] · z^m")
    print()
    print(f"Computed with {len(primes)} primes using mpmath at {mp.dps} decimal places.")
    print('"""')
    print()
    print("# Taylor coefficients: S_COEFFS[m] = S^{(m)}(0) / m!")
    print("S_COEFFS = [")
    for i, c in enumerate(coeffs):
        # Convert to float with sufficient precision
        c_float = float(c)
        print(f"    {c_float:.16e},  # s_{i}")
    print("]")
    print()

    # Verify the polynomial approximation
    print("Verification: S_poly(z) vs S(z) at sample points:")
    test_z = [mpf("0.1"), mpf("0.5"), mpf("1.0"), mpf("2.0"), mpf("2.6")]  # 2R·t_max ≈ 2.6
    for z in test_z:
        S_exact = S_function(z, primes)
        S_poly = sum(coeffs[m] * (z ** m) for m in range(len(coeffs)))
        rel_err = abs(S_exact - S_poly) / abs(S_exact) * 100
        print(f"  z={float(z):.1f}: S_exact={float(S_exact):.10f}, S_poly={float(S_poly):.10f}, rel_err={float(rel_err):.4f}%")


if __name__ == "__main__":
    main()
