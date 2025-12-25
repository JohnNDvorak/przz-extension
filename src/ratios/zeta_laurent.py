"""
src/ratios/zeta_laurent.py
Phase 14C Task C2: Contour-Lemma Laurent Series

PURPOSE:
========
This module provides proper Laurent/Taylor series expansions for:
- 1/ζ(1+s) around s=0
- ζ'/ζ(1+s) around s=0
- (1/ζ)(ζ'/ζ)(1+s) around s=0

These are the KEY objects for implementing PRZZ's main-term reductions
in J12-J14. The paper doesn't evaluate ζ'/ζ numerically for main terms;
it extracts coefficients from these series expansions.

PAPER ANCHORS:
=============
ζ(1+s) = 1/s + γ + γ₁s + O(s²)

where:
- γ = 0.5772156649... (Euler-Mascheroni)
- γ₁ = -0.0728158454... (Stieltjes constant)

From this:
- 1/ζ(1+s) = s(1 - γs + O(s²)) = s - γs² + O(s³)
- ζ'/ζ(1+s) = -1/s + γ + γ₁s + O(s²)  [pole term + regular part]

USAGE:
======
These series are used for coefficient extraction, NOT for numerical
evaluation at specific s values. The main-term reductions compute
residues by extracting [s^i] coefficients.

TeX REFERENCE:
=============
The Laurent expansion structure is implicit in PRZZ's residue calculus.
See lines 1502-1511 for the contour lemma / integral representation.
"""

from __future__ import annotations
from typing import Tuple, List
import math

# Euler-Mascheroni constant γ
EULER_MASCHERONI = 0.5772156649015329

# Stieltjes constants γ₁, γ₂, γ₃
# ζ(1+s) = 1/s + γ₀ + γ₁s + γ₂s²/2! + γ₃s³/3! + ...
# where γ₀ = γ (Euler-Mascheroni)
STIELTJES_GAMMA1 = -0.0728158454836767
STIELTJES_GAMMA2 = -0.0096903631928723
STIELTJES_GAMMA3 = 0.002053834420303346


class LaurentSeries:
    """
    Represents a Laurent series with a simple pole at s=0.

    f(s) = pole_coeff/s + c₀ + c₁s + c₂s² + ...

    Attributes:
        pole_coeff: Coefficient of 1/s term (0 if no pole)
        coeffs: Tuple of Taylor coefficients (c₀, c₁, c₂, ...)
    """

    def __init__(self, pole_coeff: complex, coeffs: Tuple[complex, ...]):
        self.pole_coeff = pole_coeff
        self.coeffs = coeffs

    def __repr__(self):
        terms = []
        if abs(self.pole_coeff) > 1e-15:
            terms.append(f"{self.pole_coeff:.4f}/s")
        for i, c in enumerate(self.coeffs):
            if abs(c) > 1e-15:
                if i == 0:
                    terms.append(f"{c:.4f}")
                else:
                    terms.append(f"{c:.4f}*s^{i}")
        return " + ".join(terms) if terms else "0"

    def eval_at(self, s: complex) -> complex:
        """Evaluate the series at a specific s value."""
        if abs(s) < 1e-15:
            raise ValueError("Cannot evaluate at s=0 (pole)")
        result = self.pole_coeff / s
        s_power = 1.0
        for c in self.coeffs:
            result += c * s_power
            s_power *= s
        return result

    def taylor_coeff(self, i: int) -> complex:
        """Get the coefficient of s^i in the Taylor part."""
        if i < 0:
            return self.pole_coeff if i == -1 else complex(0.0)
        if i < len(self.coeffs):
            return self.coeffs[i]
        return complex(0.0)


def zeta_series(order: int = 4) -> LaurentSeries:
    """
    Laurent series of ζ(1+s) around s=0.

    ζ(1+s) = 1/s + γ + γ₁s + γ₂s²/2! + ...

    Args:
        order: Number of Taylor coefficients to include

    Returns:
        LaurentSeries with pole_coeff=1, coeffs=(γ, γ₁, γ₂/2!, ...)
    """
    coeffs = []

    # Constant term: γ (Euler-Mascheroni)
    coeffs.append(EULER_MASCHERONI)

    if order >= 2:
        # Linear term: γ₁
        coeffs.append(STIELTJES_GAMMA1)

    if order >= 3:
        # Quadratic term: γ₂/2!
        coeffs.append(STIELTJES_GAMMA2 / 2.0)

    if order >= 4:
        # Cubic term: γ₃/3!
        coeffs.append(STIELTJES_GAMMA3 / 6.0)

    # Pad with zeros if needed
    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    return LaurentSeries(pole_coeff=1.0, coeffs=tuple(coeffs))


def inv_zeta_series(order: int = 4) -> Tuple[complex, ...]:
    """
    Taylor series coefficients of 1/ζ(1+s) in powers of s.

    Since ζ(1+s) = 1/s + γ + O(s), we have:
    ζ(1+s) = (1/s)(1 + γs + O(s²))

    So: 1/ζ(1+s) = s / (1 + γs + O(s²))
                 = s(1 - γs + γ²s² + O(s³))
                 = s - γs² + γ²s³ + O(s⁴)

    Returns:
        Tuple (c₀, c₁, c₂, ...) where 1/ζ(1+s) = Σ cᵢsⁱ

    Note:
        c₀ = 0 (1/ζ(1+s) vanishes at s=0)
        c₁ = 1
        c₂ = -γ
        c₃ = γ² + γ₁
        ...
    """
    gamma = EULER_MASCHERONI
    gamma1 = STIELTJES_GAMMA1

    coeffs: List[complex] = []

    # [s^0] = 0
    coeffs.append(0.0)

    if order >= 2:
        # [s^1] = 1
        coeffs.append(1.0)

    if order >= 3:
        # [s^2] = -γ
        coeffs.append(-gamma)

    if order >= 4:
        # [s^3] = γ² - γ₁ (from expanding 1/(1+γs+γ₁s²+...))
        coeffs.append(gamma * gamma - gamma1)

    # Pad with zeros
    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    return tuple(coeffs)


def zeta_logderiv_series(order: int = 4) -> LaurentSeries:
    """
    Laurent series of ζ'/ζ(1+s) around s=0.

    ζ'/ζ(1+s) = -1/s + γ + γ₁s + O(s²)

    This is the KEY function for J12-J14 reductions.
    The pole at s=0 drives the main-term structure.

    Args:
        order: Number of Taylor coefficients beyond the pole

    Returns:
        LaurentSeries with pole_coeff=-1, coeffs=(γ, γ₁, ...)

    Note:
        The pole coefficient is NEGATIVE: -1/s
        This is crucial for sign conventions in the main-term reductions.
    """
    gamma = EULER_MASCHERONI
    gamma1 = STIELTJES_GAMMA1
    gamma2 = STIELTJES_GAMMA2

    coeffs: List[complex] = []

    # Constant term: γ
    coeffs.append(gamma)

    if order >= 2:
        # Linear term: γ₁
        coeffs.append(gamma1)

    if order >= 3:
        # Quadratic term: γ₂
        coeffs.append(gamma2)

    # Pad with zeros
    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    # Pole coefficient is -1 (NEGATIVE)
    return LaurentSeries(pole_coeff=-1.0, coeffs=tuple(coeffs))


def inv_zeta_times_logderiv_series(order: int = 4) -> Tuple[complex, ...]:
    """
    Taylor series of (1/ζ)(ζ'/ζ)(1+s) = ζ'/ζ²(1+s).

    This is the CRITICAL object for J12-J14 main-term reductions.

    We have:
        1/ζ(1+s) = s - γs² + O(s³)
        ζ'/ζ(1+s) = -1/s + γ + O(s)

    So:
        (1/ζ)(ζ'/ζ)(1+s) = (s - γs² + O(s³))(-1/s + γ + O(s))
                         = -1 + γs + γs - γ²s² + O(s²)
                         = -1 + 2γs + O(s²)

    Returns:
        Tuple (c₀, c₁, c₂, ...) where (1/ζ)(ζ'/ζ)(1+s) = Σ cᵢsⁱ

    Note:
        c₀ = -1 (NEGATIVE constant term!)
        c₁ = 2γ
        This structure is key for understanding how J12-J14 reduce.
    """
    gamma = EULER_MASCHERONI
    gamma1 = STIELTJES_GAMMA1

    # Compute product of series
    # 1/ζ(1+s) = 0 + s - γs² + (γ²-γ₁)s³ + O(s⁴)
    # ζ'/ζ(1+s) = -1/s + γ + γ₁s + O(s²)

    # Product (1/ζ)*(ζ'/ζ):
    # [s^0]: 1*(-1/s) term: s * (-1/s) = -1
    # [s^1]: s*γ + (-γ)*(-1) = γ + γ = 2γ
    # Wait, let me redo this more carefully.

    # Let A(s) = 1/ζ(1+s) = a₁s + a₂s² + a₃s³ + ... (no constant term, no pole)
    # where a₁ = 1, a₂ = -γ, a₃ = γ²-γ₁, ...

    # Let B(s) = ζ'/ζ(1+s) = -1/s + b₀ + b₁s + b₂s² + ...
    # where b₀ = γ, b₁ = γ₁, b₂ = γ₂, ...

    # A(s)*B(s) = (a₁s + a₂s² + ...)(-1/s + b₀ + b₁s + ...)
    #           = a₁*(-1) + a₁*b₀*s + a₁*b₁*s² + ...
    #           + a₂*(-1)*s + a₂*b₀*s² + ...
    #           + a₃*(-1)*s² + ...

    # Collecting by power:
    # [s^0]: a₁*(-1) = 1*(-1) = -1
    # [s^1]: a₁*b₀ + a₂*(-1) = 1*γ + (-γ)*(-1) = γ + γ = 2γ
    # [s^2]: a₁*b₁ + a₂*b₀ + a₃*(-1) = 1*γ₁ + (-γ)*γ + (γ²-γ₁)*(-1)
    #      = γ₁ - γ² - γ² + γ₁ = 2γ₁ - 2γ²

    coeffs: List[complex] = []

    # [s^0] = -1
    coeffs.append(-1.0)

    if order >= 2:
        # [s^1] = 2γ
        coeffs.append(2 * gamma)

    if order >= 3:
        # [s^2] = 2γ₁ - 2γ²
        coeffs.append(2 * gamma1 - 2 * gamma * gamma)

    if order >= 4:
        # Higher order - compute if needed
        # For now, just pad
        coeffs.append(complex(0.0))

    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    return tuple(coeffs)


def logderiv_product_series(
    alpha: complex,
    beta: complex,
    order: int = 4
) -> Tuple[Tuple[complex, ...], Tuple[complex, ...]]:
    """
    Series expansions for (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u).

    This is the product that appears in J12.

    For PRZZ main terms, we need to extract residues at s=u=0.
    This requires understanding the pole structure of the product.

    Returns:
        Tuple of two series: (s-series coeffs, u-series coeffs)
        where the full product is the Cauchy product.

    Note:
        When α, β are small (near 0), the poles at s=-α, u=-β
        are close to s=u=0, making residue extraction complex.

        When α, β = -R (PRZZ point), the poles are at s=R, u=R,
        which are away from s=u=0.
    """
    # For α close to 0: ζ'/ζ(1+α+s) has pole at s = -α
    # Laurent expansion around s=0:
    # ζ'/ζ(1+α+s) = -1/(α+s) + γ + O(α+s)
    #             = -1/α × 1/(1+s/α) + γ + O(α+s)
    #             = -1/α × (1 - s/α + (s/α)² - ...) + γ + ...
    #             = -1/α + s/α² - s²/α³ + ... + γ + ...

    # For α = -R (negative, order 1):
    # -1/(α+s) = -1/(-R+s) = 1/(R-s) = (1/R) × 1/(1-s/R)
    #          = (1/R)(1 + s/R + s²/R² + ...)
    #          = 1/R + s/R² + s²/R³ + ...

    # Series for ζ'/ζ(1+α+s) centered at s=0
    s_coeffs: List[complex] = []

    # If α is close to 0, we have a pole contribution
    # If α is order 1 (like α=-R), we get regular Taylor coefficients

    if abs(alpha) < 1e-10:
        # α ≈ 0: use standard Laurent
        series = zeta_logderiv_series(order)
        s_coeffs = list(series.coeffs)
        # Note: pole at s=0 in this case
    else:
        # α away from 0: ζ'/ζ(1+α+s) regular at s=0
        # Taylor expand around s=0
        for i in range(order):
            # [s^i] of ζ'/ζ(1+α+s) using derivatives
            # This is (1/i!) × (d/ds)^i [ζ'/ζ(1+α+s)]|_{s=0}
            # For now, use simple evaluation approach
            if i == 0:
                # ζ'/ζ(1+α) ≈ -1/α + γ for small α
                s_coeffs.append(-1.0 / alpha + EULER_MASCHERONI)
            elif i == 1:
                # d/ds ζ'/ζ(1+α+s)|_{s=0} = (ζ'/ζ)'(1+α) ≈ 1/α²
                s_coeffs.append(1.0 / (alpha * alpha))
            else:
                # Higher derivatives: (-1)^i × i!/α^{i+1}
                s_coeffs.append((-1) ** (i+1) * math.factorial(i) / (alpha ** (i+1)))

    # Same for u-series with β
    u_coeffs: List[complex] = []
    if abs(beta) < 1e-10:
        series = zeta_logderiv_series(order)
        u_coeffs = list(series.coeffs)
    else:
        for i in range(order):
            if i == 0:
                u_coeffs.append(-1.0 / beta + EULER_MASCHERONI)
            elif i == 1:
                u_coeffs.append(1.0 / (beta * beta))
            else:
                u_coeffs.append((-1) ** (i+1) * math.factorial(i) / (beta ** (i+1)))

    return (tuple(s_coeffs), tuple(u_coeffs))


def extract_coefficient_from_product(
    s_coeffs: Tuple[complex, ...],
    u_coeffs: Tuple[complex, ...],
    i: int,
    j: int
) -> complex:
    """
    Extract [s^i u^j] coefficient from product of series.

    If A(s) = Σ aₖ s^k and B(u) = Σ bₘ u^m, then
    [s^i u^j] A(s)B(u) = aᵢ × bⱼ

    This is for independent variable products (not Cauchy product).
    """
    a_i = s_coeffs[i] if i < len(s_coeffs) else complex(0.0)
    b_j = u_coeffs[j] if j < len(u_coeffs) else complex(0.0)
    return a_i * b_j


# =============================================================================
# Main-term coefficient extraction for J12-J14
# =============================================================================


def j12_main_term_coefficient(
    alpha: complex,
    beta: complex,
    i: int,
    j: int
) -> complex:
    """
    Extract [s^i u^j] coefficient for J12 main-term reduction.

    J12 involves the product (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u).

    At PRZZ point α=β=-R, the main term is dominated by:
    [s^0 u^0]: (-1/α + γ)(-1/β + γ) = (1/R + γ)²

    The paper's key reduction is that this product structure
    collapses in the contour integral to coefficient extraction.

    Args:
        alpha, beta: Shift parameters
        i, j: Orders of s and u coefficients

    Returns:
        The [s^i u^j] coefficient
    """
    s_series, u_series = logderiv_product_series(alpha, beta, order=max(i, j) + 2)
    return extract_coefficient_from_product(s_series, u_series, i, j)


def j13_main_term_sign() -> int:
    """
    Return the sign for J13 main-term reduction.

    The paper's reduction gives a NEGATIVE sign for I13.
    This comes from the residue calculus structure.

    Returns:
        -1 (the J13 main term has a leading minus sign)
    """
    return -1


def j14_main_term_sign() -> int:
    """
    Return the sign for J14 main-term reduction.

    Symmetric with J13: also has a leading minus sign.

    Returns:
        -1 (the J14 main term has a leading minus sign)
    """
    return -1
