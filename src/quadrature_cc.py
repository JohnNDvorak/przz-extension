"""
Clenshaw-Curtis quadrature for independent validation.

This module provides an alternative to Gauss-Legendre quadrature
for Gate 2 validation (two independent numerical integrators).

Clenshaw-Curtis uses Chebyshev nodes and provides a different
error distribution than Gauss-Legendre, making it useful for
cross-validation.

Created: 2025-12-28 (GPT Critical Review - Gate 2)
"""

from __future__ import annotations
import functools
import numpy as np
from typing import Tuple


@functools.lru_cache(maxsize=16)
def clenshaw_curtis_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (nodes, weights) for n-point Clenshaw-Curtis quadrature on [0,1].

    Clenshaw-Curtis uses Chebyshev nodes:
        x_k = cos(π k / (n-1))  for k = 0, 1, ..., n-1 on [-1, 1]

    Mapped to [0, 1] via x_01 = 0.5 * (x + 1).

    Args:
        n: Number of quadrature points (must be >= 2)

    Returns:
        Tuple of (nodes, weights) arrays, each of length n.
        Arrays are READ-ONLY to prevent mutation of cached data.

    Raises:
        ValueError: if n < 2
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    # Chebyshev nodes on [-1, 1]
    k = np.arange(n)
    nodes_11 = np.cos(np.pi * k / (n - 1))

    # Compute weights using the standard Clenshaw-Curtis formula
    # See: "Numerical Recipes" or Trefethen's "Spectral Methods in MATLAB"
    weights_11 = _clenshaw_curtis_weights(n)

    # Map to [0, 1]: x_01 = 0.5*(1 - x), w_01 = 0.5*w
    # Note: nodes are in decreasing order on [-1,1], so we flip
    nodes = 0.5 * (1 - nodes_11)  # Maps cos(0)=1 -> 0, cos(pi)=-1 -> 1
    weights = 0.5 * weights_11

    # Sort nodes in increasing order
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]

    # Make arrays read-only
    nodes.flags.writeable = False
    weights.flags.writeable = False

    return nodes, weights


def _clenshaw_curtis_weights(n: int) -> np.ndarray:
    """
    Compute Clenshaw-Curtis weights for n points on [-1, 1].

    Using the algorithm from:
    Trefethen, "Spectral Methods in MATLAB" (2000), Chapter 12.

    The weights integrate polynomials up to degree n-1 exactly.
    Total integral over [-1, 1] is 2, so weights sum to 2.
    """
    if n == 1:
        return np.array([2.0])

    if n == 2:
        return np.array([1.0, 1.0])

    N = n - 1  # degree

    # Clenshaw-Curtis weights via FFT (Waldvogel 2006)
    c = np.zeros(n)
    for k in range(n):
        if k == 0 or k == N:
            c[k] = 1.0 / (N * N - 1) if N > 1 else 0.5
        else:
            # c_k = 2 / (1 - 4k^2) summed with appropriate factor
            pass

    # Alternative: direct formula for weights
    # w_k = c_k * (2/N) for interior, c_0 = c_N = 1/N for endpoints
    # where c_k = sum_{j=0}^{N/2} b_j * cos(2*pi*j*k/N)

    # Simpler explicit formula
    theta = np.pi * np.arange(n) / N

    # Initialize with the direct computation
    # w_k = (2/N) * sum_{j=0}^{N//2} (1/(1-4j^2)) * cos(2jk*pi/N)
    # with endpoint halving for j=0 and j=N/2

    weights = np.zeros(n)

    for k in range(n):
        s = 1.0  # b_0 = 1 (integrates to 1)
        for j in range(1, N // 2 + 1):
            bj = 2.0 / (1.0 - 4.0 * j * j)
            factor = 1.0 if j < N // 2 else 0.5 if N % 2 == 0 else 1.0
            s += factor * bj * np.cos(2 * j * k * np.pi / N)

        weights[k] = (2.0 / N) * s

    # Halve the endpoints
    weights[0] /= 2.0
    weights[-1] /= 2.0

    return weights


def tensor_grid_2d_cc(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (U, T, W) for 2D tensor product Clenshaw-Curtis quadrature on [0,1]^2.

    Args:
        n: Number of quadrature points per dimension (total n^2 points)

    Returns:
        Tuple of (U, T, W) where:
        - U[i,j] = u_i (first coordinate grid)
        - T[i,j] = t_j (second coordinate grid)
        - W[i,j] = w_i * w_j (product weight grid)

        All arrays have shape (n, n).
    """
    nodes, weights = clenshaw_curtis_01(n)

    U, T = np.meshgrid(nodes, nodes, indexing="ij")
    W = np.outer(weights, weights)

    return U, T, W


def tensor_grid_3d_cc(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, Y, Z, W) for 3D tensor product Clenshaw-Curtis quadrature on [0,1]^3.

    Args:
        n: Number of quadrature points per dimension (total n^3 points)

    Returns:
        Tuple of (X, Y, Z, W) where all arrays have shape (n, n, n).
    """
    nodes, weights = clenshaw_curtis_01(n)

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    W = np.einsum("i,j,k->ijk", weights, weights, weights)

    return X, Y, Z, W


# =============================================================================
# VALIDATION
# =============================================================================

def validate_quadrature(nodes: np.ndarray, weights: np.ndarray, max_degree: int = 10) -> dict:
    """
    Validate quadrature by integrating x^k for k = 0, 1, ..., max_degree.

    Exact value: ∫_0^1 x^k dx = 1/(k+1)

    Returns:
        Dictionary with:
        - errors: list of absolute errors for each k
        - max_error: maximum error
        - converged_degree: highest k with error < 1e-10
    """
    errors = []
    for k in range(max_degree + 1):
        exact = 1.0 / (k + 1)
        computed = np.sum(weights * nodes**k)
        errors.append(abs(computed - exact))

    return {
        "errors": errors,
        "max_error": max(errors),
        "converged_degree": max(i for i, e in enumerate(errors) if e < 1e-10) if any(e < 1e-10 for e in errors) else -1,
    }


if __name__ == "__main__":
    # Quick validation
    print("Clenshaw-Curtis Quadrature Validation")
    print("=" * 50)

    for n in [10, 20, 40, 80]:
        nodes, weights = clenshaw_curtis_01(n)
        result = validate_quadrature(nodes, weights, max_degree=2 * n - 2)

        print(f"\nn = {n}:")
        print(f"  Sum of weights: {np.sum(weights):.10f} (should be 1.0)")
        print(f"  Max error (x^k tests): {result['max_error']:.2e}")
        print(f"  Converged up to degree: {result['converged_degree']}")
