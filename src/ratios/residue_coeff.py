"""
src/ratios/residue_coeff.py
Phase 14B Task 3: L_{1,1} and L_{1,2} Coefficient Extraction

PAPER ANCHORS:
=============
1. The bracket terms involve contour integrals:
   ∮ (stuff) × N^{s+u} × ds/s^{i+1} × du/u^{j+1}

2. The N^s factor becomes exp(s·log N) = Σ (s·log N)^k / k!

3. The 1/ζ(1+α+s) factor appears in the residue extraction.

4. PRZZ main-term simplification:
   - For main order: 1/ζ(1+α+s) ≈ (α+s)
   - This is because ζ(1+ε) = 1/ε + O(1), so 1/ζ = ε/(1+O(ε))

TWO MODES:
=========
1. Main-term mode (paper-consistent):
   L_{1,1}^{main} = [s^i] (α+s) · exp(s·log(N/n))

2. Full mode (Laurent expansion):
   L_{1,1}^{full} = [s^i] exp(s·log(N/n)) / ζ(1+α+s)
   using 1/ζ(1+α+s) = (α+s) - γ(α+s)² + ...

The main-term mode is what PRZZ uses. The full mode provides
higher-order refinement.
"""

from __future__ import annotations
import math
from typing import Tuple

from src.ratios.diagonalize import EULER_MASCHERONI


def exp_series_coeff(x: float, i: int) -> float:
    """
    Compute [s^i] exp(s·x) = x^i / i!

    This is the Taylor coefficient of exp(s·x) at s=0.

    Args:
        x: The argument in exp(s·x)
        i: The order of the coefficient

    Returns:
        x^i / i!
    """
    if i < 0:
        return 0.0
    return (x ** i) / math.factorial(i)


def _product_series_coeff(
    coeffs_a: Tuple[float, ...],
    coeffs_b: Tuple[float, ...],
    i: int
) -> float:
    """
    Compute [s^i] of product of two power series.

    If A(s) = Σ a_k s^k and B(s) = Σ b_k s^k, then
    [s^i] A(s)B(s) = Σ_{k=0}^i a_k b_{i-k}

    Args:
        coeffs_a: Tuple of coefficients (a_0, a_1, ..., a_n)
        coeffs_b: Tuple of coefficients (b_0, b_1, ..., b_m)
        i: The order of the coefficient to extract

    Returns:
        The coefficient of s^i in the product
    """
    result = 0.0
    for k in range(i + 1):
        if k < len(coeffs_a) and (i - k) < len(coeffs_b):
            result += coeffs_a[k] * coeffs_b[i - k]
    return result


def L11_main(n: int, N: int, alpha: complex, i: int) -> complex:
    """
    Main-term mode coefficient extraction for L_{1,1}.

    L_{1,1}^{main} = [s^i] (α+s) · exp(s·log(N/n))

    This is the PRZZ paper-consistent simplification where
    1/ζ(1+α+s) is replaced by (α+s).

    The formula expands as:
    (α+s) · exp(s·x) = α·exp(s·x) + s·exp(s·x)
                     = α·Σ(sx)^k/k! + Σ(sx)^k·s/k!

    So [s^i] = α·x^i/i! + x^{i-1}/(i-1)! for i≥1
       [s^0] = α

    Args:
        n: Summation index (from Dirichlet series)
        N: Cutoff parameter
        N: Cutoff parameter
        alpha: Shift parameter
        i: Order of coefficient to extract

    Returns:
        The coefficient [s^i] of (α+s)·exp(s·log(N/n))
    """
    if n <= 0 or N <= 0:
        return complex(0.0)

    x = math.log(N / n)

    # (α+s) · exp(s·x) coefficients:
    # [s^i] α·exp(s·x) = α · x^i/i!
    # [s^i] s·exp(s·x) = [s^{i-1}] exp(s·x) = x^{i-1}/(i-1)! for i≥1, 0 for i=0
    term1 = alpha * exp_series_coeff(x, i)
    term2 = exp_series_coeff(x, i - 1) if i >= 1 else 0.0

    return term1 + term2


def L12_main(n: int, N: int, beta: complex, j: int) -> complex:
    """
    Main-term mode coefficient extraction for L_{1,2}.

    L_{1,2}^{main} = [u^j] (β+u) · exp(u·log(N/n))

    This is the symmetric (β,u,j) version of L11_main.

    Args:
        n: Summation index
        N: Cutoff parameter
        beta: Shift parameter
        j: Order of coefficient to extract

    Returns:
        The coefficient [u^j]
    """
    return L11_main(n, N, beta, j)


def _inv_zeta_coeffs(alpha: complex, order: int) -> Tuple[complex, ...]:
    """
    Compute coefficients of 1/ζ(1+α+s) as power series in s.

    At α=0: ζ(1+s) = 1/s + γ + γ₁s + O(s²)
    So: 1/ζ(1+s) = s / (1 + γs + γ₁s² + ...)
                 = s(1 - γs + (γ²-γ₁)s² + ...)
                 = s - γs² + O(s³)

    For general α:
    ζ(1+α+s) = 1/(α+s) + γ + O(α+s)
    1/ζ(1+α+s) = (α+s) / (1 + γ(α+s) + ...)
               ≈ (α+s)(1 - γ(α+s) + ...)
               = (α+s) - γ(α+s)² + ...

    Expanding in s around s=0:
    (α+s) = α + s
    (α+s)² = α² + 2αs + s²

    So: 1/ζ(1+α+s) ≈ (α - γα²) + (1 - 2γα)s + (-γ)s² + O(s³)

    Args:
        alpha: Shift parameter
        order: Number of terms to compute

    Returns:
        Tuple of coefficients (c_0, c_1, ..., c_{order-1})
    """
    gamma = EULER_MASCHERONI

    # Using the expansion 1/ζ(1+α+s) ≈ (α+s) - γ(α+s)² + ...
    # Let's compute term by term up to specified order

    if order <= 0:
        return ()

    # Coefficients in powers of s
    coeffs = []

    # [s^0] = α - γα² + ...
    c0 = alpha - gamma * alpha * alpha
    coeffs.append(c0)

    if order >= 2:
        # [s^1] = 1 - 2γα + ...
        c1 = 1 - 2 * gamma * alpha
        coeffs.append(c1)

    if order >= 3:
        # [s^2] = -γ + ...
        c2 = -gamma
        coeffs.append(c2)

    # Higher orders need more careful expansion
    # For now, pad with zeros
    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    return tuple(coeffs)


def L11_full(
    n: int,
    N: int,
    alpha: complex,
    i: int,
    *,
    order: int = 3
) -> complex:
    """
    Full mode coefficient extraction using Laurent expansion.

    L_{1,1}^{full} = [s^i] exp(s·log(N/n)) / ζ(1+α+s)

    Uses the expansion:
    1/ζ(1+α+s) = (α+s) - γ(α+s)² + O((α+s)³)

    Expanding in s and multiplying by exp(s·x):
    [s^i] of the product.

    Args:
        n: Summation index
        N: Cutoff parameter
        alpha: Shift parameter
        i: Order of coefficient to extract
        order: Order of Laurent expansion to use

    Returns:
        The coefficient [s^i]
    """
    if n <= 0 or N <= 0:
        return complex(0.0)

    x = math.log(N / n)

    # Get coefficients of 1/ζ(1+α+s) in powers of s
    inv_zeta = _inv_zeta_coeffs(alpha, order)

    # Get coefficients of exp(s·x) in powers of s
    exp_coeffs = tuple(exp_series_coeff(x, k) for k in range(order))

    # Compute [s^i] of the product
    return _product_series_coeff(inv_zeta, exp_coeffs, i)


def L12_full(
    n: int,
    N: int,
    beta: complex,
    j: int,
    *,
    order: int = 3
) -> complex:
    """
    Full mode coefficient extraction for L_{1,2}.

    L_{1,2}^{full} = [u^j] exp(u·log(N/n)) / ζ(1+β+u)

    This is the symmetric (β,u,j) version of L11_full.

    Args:
        n: Summation index
        N: Cutoff parameter
        beta: Shift parameter
        j: Order of coefficient to extract
        order: Order of Laurent expansion to use

    Returns:
        The coefficient [u^j]
    """
    return L11_full(n, N, beta, j, order=order)


# =============================================================================
# Combined coefficient for bracket evaluation
# =============================================================================


def L_product_coefficient(
    n: int,
    N: int,
    alpha: complex,
    beta: complex,
    i: int,
    j: int,
    *,
    mode: str = "main"
) -> complex:
    """
    Combined L_{1,1} × L_{1,2} coefficient extraction.

    This is used in the bracket term evaluations where we have
    double contour integrals in s and u.

    The bracket terms produce sums of form:
    Σ_n [coefficient] × [s^i] × [u^j] × (Dirichlet series term)

    Args:
        n: Summation index
        N: Cutoff parameter
        alpha, beta: Shift parameters
        i, j: Orders of coefficients
        mode: "main" or "full"

    Returns:
        L_{1,1}(i) × L_{1,2}(j)
    """
    if mode == "main":
        L1 = L11_main(n, N, alpha, i)
        L2 = L12_main(n, N, beta, j)
    elif mode == "full":
        L1 = L11_full(n, N, alpha, i)
        L2 = L12_full(n, N, beta, j)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return L1 * L2
