"""
Arithmetic constants for the I₅ correction term.

S(z) = Σ_p (log p / (p^{1+z} − 1))²

For the asymptotic analysis in PRZZ, the relevant value is S(0) since
the shifts α and β are O(1/log T) → 0.

Computed with mpmath using 100,000 primes at 50 decimal places.
"""

# S(0) = Σ_p (log p / (p − 1))²
# This is the prime sum evaluated at z=0
S_AT_ZERO = 1.3854799116100166

# For reference, the first few Taylor coefficients (useful for small z only):
# These diverge rapidly for z > 0.1, so should NOT be used for z = 2Rt
S_TAYLOR_COEFFS = [
    1.3854799116100166,   # s_0 = S(0)
    -5.899717699472018,   # s_1 = S'(0)
    16.70919463616497,    # s_2 = S''(0)/2!
    -40.4850884491245,    # s_3 = S'''(0)/3!
    90.098218493733,      # s_4
    -187.76043698644,     # s_5
    367.44108080488,      # s_6
]


def evaluate_S_at_zero() -> float:
    """
    Return S(0) = Σ_p (log p / (p − 1))².

    This is the constant used in the I₅ arithmetic correction.
    """
    return S_AT_ZERO


def evaluate_S_direct(z: float, n_primes: int = 10000) -> float:
    """
    Evaluate S(z) directly by summing over primes.

    WARNING: This is slow for large n_primes. Use only for verification.

    Args:
        z: Argument (should be >= 0 for convergence)
        n_primes: Number of primes to sum over

    Returns:
        S(z) = Σ_p (log p / (p^{1+z} − 1))²
    """
    import math

    # Simple sieve for primes
    sieve = [True] * (n_primes + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_primes**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n_primes + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, n_primes + 1) if sieve[i]]

    result = 0.0
    for p in primes:
        log_p = math.log(p)
        p_power = p ** (1 + z)
        denominator = p_power - 1
        term = (log_p / denominator) ** 2
        result += term

    return result


# Module-level cache for S(z) at quadrature nodes
# Key: (R, n_quadrature, n_primes)
# Value: np.ndarray of S(2Rt) at each t-node
_S_CACHE: dict = {}


def evaluate_S_at_quadrature_nodes(
    R: float,
    n_quadrature: int,
    n_primes: int = 100000
) -> "np.ndarray":
    """
    Compute S(2Rt) at all Gauss-Legendre quadrature t-nodes.

    This is the principled approach for I₅: compute S(z) exactly at each
    quadrature point where z = 2Rt, NOT via Taylor expansion.

    The result is cached keyed by (R, n_quadrature, n_primes).

    Args:
        R: Shift parameter (typically 1.3036)
        n_quadrature: Number of quadrature points
        n_primes: Number of primes to use in S(z) evaluation

    Returns:
        1D array of shape (n_quadrature,) with S(2Rt) at each t-node
    """
    import numpy as np
    from src.quadrature import gauss_legendre_01

    cache_key = (R, n_quadrature, n_primes)
    if cache_key in _S_CACHE:
        return _S_CACHE[cache_key]

    # Get quadrature nodes (t in [0,1])
    t_nodes, _ = gauss_legendre_01(n_quadrature)

    # Evaluate S at each z = 2Rt
    S_values = np.array([
        evaluate_S_direct(2 * R * t, n_primes) for t in t_nodes
    ])

    _S_CACHE[cache_key] = S_values
    return S_values


def evaluate_S_at_t_grid(
    T: "np.ndarray",
    R: float,
    n_primes: int = 100000
) -> "np.ndarray":
    """
    Compute S(2Rt) for a grid of t values.

    This is for use in full 2D quadrature where T is a grid array.
    Uses vectorized evaluation for efficiency.

    Args:
        T: Grid array of t values (any shape)
        R: Shift parameter
        n_primes: Number of primes to use

    Returns:
        Array with same shape as T, containing S(2Rt) values
    """
    import numpy as np

    # Flatten for evaluation, then reshape
    t_flat = T.ravel()
    z_flat = 2 * R * t_flat

    # Evaluate S at each z value (this could be optimized with caching)
    S_flat = np.array([evaluate_S_direct(z, n_primes) for z in z_flat])

    return S_flat.reshape(T.shape)


def clear_S_cache() -> None:
    """Clear the S(z) quadrature cache."""
    _S_CACHE.clear()
