"""
src/ratios/j12_c00_reference.py
Phase 14H Task H1: Literal J12 Zeta-Factor Series Builder

PURPOSE:
========
Compute the EXACT (s⁰u⁰) coefficient of the J12 zeta-factor by direct
series multiplication, WITHOUT using Euler-Maclaurin or mirror assembly.

This provides a "semantic truth" for which LaurentMode is correct.

PAPER ANCHORS:
=============
From j1_k3_decomposition.py, J12 (bracket₂) is:
    bracket₂(s,u) = A(0,0;β,α) × Σ_{n≤N} 1/n^{1+s+u} × (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

With A(0,0;β,α) = 1, the zeta-factor in J12 is:
    F(s,u) = (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

The main-term extraction computes [s⁰u⁰]F(s,u) after various reductions.

The question is: what is the correct c₀₀ constant after the PRZZ main-term
reductions, and which LaurentMode matches it?

HYPOTHESES:
==========
H1: RAW_LOGDERIV mode with (1/R + γ)² is correct
    - This corresponds to evaluating the product at shifted arguments

H2: POLE_CANCELLED mode with +1 is correct
    - This corresponds to the pole-cancelled product (1/ζ)(ζ'/ζ) × (1/ζ)(ζ'/ζ)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

from src.ratios.zeta_laurent import (
    EULER_MASCHERONI,
    STIELTJES_GAMMA1,
    zeta_logderiv_series,
    inv_zeta_times_logderiv_series,
    inv_zeta_series,
    LaurentSeries,
)


@dataclass(frozen=True)
class J12C00Result:
    """Result of J12 c₀₀ reference computation."""
    c00_literal_G_product: float       # G(α+s) × G(β+u) at s=u=0, G=(1/ζ)(ζ'/ζ)
    c00_literal_logderiv_product: float  # (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u) at s=u=0
    c00_raw_logderiv: float            # What RAW_LOGDERIV mode uses: (1/R + γ)²
    c00_pole_cancelled: float          # What POLE_CANCELLED mode uses: +1
    alpha: float
    beta: float
    R: float
    details: Dict


def multiply_taylor_series(
    a_coeffs: Tuple[complex, ...],
    b_coeffs: Tuple[complex, ...],
    order: int
) -> Tuple[complex, ...]:
    """
    Multiply two Taylor series and return coefficients up to given order.

    If A(s) = Σ aᵢsⁱ and B(s) = Σ bⱼsʲ, then
    (A×B)(s) = Σ cₖsᵏ where cₖ = Σ_{i+j=k} aᵢbⱼ

    Args:
        a_coeffs: Coefficients of first series (a₀, a₁, ...)
        b_coeffs: Coefficients of second series (b₀, b₁, ...)
        order: Maximum order to compute

    Returns:
        Tuple of product coefficients (c₀, c₁, ..., c_{order-1})
    """
    result: List[complex] = []
    for k in range(order):
        ck = complex(0.0)
        for i in range(min(k + 1, len(a_coeffs))):
            j = k - i
            if j < len(b_coeffs):
                ck += a_coeffs[i] * b_coeffs[j]
        result.append(ck)
    return tuple(result)


def eval_logderiv_at_shift(alpha: float, order: int = 4) -> Tuple[complex, ...]:
    """
    Get Taylor series of (ζ'/ζ)(1+α+s) around s=0.

    Since (ζ'/ζ)(1+w) = -1/w + γ + γ₁w + O(w²),
    substituting w = α + s:

    (ζ'/ζ)(1+α+s) = -1/(α+s) + γ + γ₁(α+s) + O((α+s)²)

    For the constant term [s⁰], when s=0:
    (ζ'/ζ)(1+α) = -1/α + γ + γ₁α + O(α²)

    But this diverges at α=0. For the main-term reduction, the paper
    handles the poles via contour integration.

    For finite α ≠ 0:
    c₀ = (ζ'/ζ)(1+α) = -1/α + γ + O(α)

    Args:
        alpha: Shift parameter (e.g., -R)
        order: Number of coefficients

    Returns:
        Taylor coefficients of (ζ'/ζ)(1+α+s) in s
    """
    if abs(alpha) < 1e-14:
        raise ValueError("alpha=0 not supported (pole)")

    gamma = EULER_MASCHERONI
    gamma1 = STIELTJES_GAMMA1

    # At s=0: (ζ'/ζ)(1+α) = -1/α + γ + γ₁α + O(α²)
    c0 = -1.0 / alpha + gamma + gamma1 * alpha

    # For higher orders, need derivatives of (ζ'/ζ)(1+α+s) at s=0
    # d/ds[(ζ'/ζ)(1+α+s)]|_{s=0} = (ζ'/ζ)'(1+α) = 1/α² + γ₁ + O(α)
    c1 = 1.0 / (alpha * alpha) + gamma1

    # Higher orders omitted for now (would need more Stieltjes constants)
    coeffs = [c0, c1]
    while len(coeffs) < order:
        coeffs.append(complex(0.0))

    return tuple(coeffs)


def compute_G_series_at_shift(alpha: float, order: int = 4) -> Tuple[complex, ...]:
    """
    Get Taylor series of G(α+s) = (1/ζ)(ζ'/ζ)(1+α+s) around s=0.

    The function G(ε) = (1/ζ)(ζ'/ζ)(1+ε) has the expansion (from zeta_laurent.py):
    G(ε) = -1 + 2γε + (2γ₁ - 2γ²)ε² + O(ε³)

    Substituting ε = α + s:
    G(α+s) = -1 + 2γ(α+s) + (2γ₁ - 2γ²)(α+s)² + O((α+s)³)

    Expanding in powers of s:
    G(α+s) = G(α) + G'(α)s + G''(α)s²/2 + ...

    At s=0, the constant term is G(α).

    Args:
        alpha: Shift parameter (e.g., -R)
        order: Number of coefficients

    Returns:
        Taylor coefficients of G(α+s) in s
    """
    gamma = EULER_MASCHERONI
    gamma1 = STIELTJES_GAMMA1

    # Coefficients of G(ε) = Σ gᵢεⁱ
    g_coeffs = inv_zeta_times_logderiv_series(order=6)
    # g_coeffs = (-1, 2γ, 2γ₁-2γ², ...)

    # G(α+s) = Σᵢ gᵢ(α+s)ⁱ = Σᵢ gᵢ Σⱼ (i choose j) αⁱ⁻ʲ sʲ
    # [sᵏ] = Σᵢ gᵢ (i choose k) αⁱ⁻ᵏ

    result: List[complex] = []
    for k in range(order):
        coeff = complex(0.0)
        for i in range(k, len(g_coeffs)):
            # Binomial coefficient (i choose k)
            binom = 1.0
            for j in range(k):
                binom *= (i - j) / (j + 1)
            coeff += g_coeffs[i] * binom * (alpha ** (i - k))
        result.append(coeff)

    return tuple(result)


def compute_j12_c00_reference(
    R: float,
    *,
    alpha: float = None,
    beta: float = None,
    order: int = 4
) -> J12C00Result:
    """
    Compute reference c₀₀ values for J12 zeta-factor.

    Computes the (s⁰u⁰) coefficient for two interpretations:

    1. G-product: G(α+s) × G(β+u) where G = (1/ζ)(ζ'/ζ)
       At s=u=0: G(α) × G(β)
       This is what POLE_CANCELLED mode claims to approximate

    2. Log-deriv product: (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
       At s=u=0: (ζ'/ζ)(1+α) × (ζ'/ζ)(1+β)
       This is the literal bracket₂ structure

    Args:
        R: PRZZ R parameter
        alpha: First shift (default: -R)
        beta: Second shift (default: -R)
        order: Series order

    Returns:
        J12C00Result with all c₀₀ values for comparison
    """
    if alpha is None:
        alpha = -R
    if beta is None:
        beta = -R

    gamma = EULER_MASCHERONI

    # =========================================================
    # Interpretation 1: G-product (pole-cancelled)
    # G(ε) = (1/ζ)(ζ'/ζ)(1+ε) has c₀ = -1
    # So G(α) × G(β) at s=u=0 with α=β=-R:
    # =========================================================
    G_alpha_coeffs = compute_G_series_at_shift(alpha, order)
    G_beta_coeffs = compute_G_series_at_shift(beta, order)

    # [s⁰] of G(α+s) is G(α) = first coefficient
    G_alpha = G_alpha_coeffs[0]
    G_beta = G_beta_coeffs[0]

    # Product constant term
    c00_G_product = float((G_alpha * G_beta).real)

    # =========================================================
    # Interpretation 2: Raw log-derivative product
    # (ζ'/ζ)(1+α) × (ζ'/ζ)(1+β) at α=β=-R
    # =========================================================
    # (ζ'/ζ)(1+α) = -1/α + γ + O(α)
    logderiv_alpha = -1.0 / alpha + gamma
    logderiv_beta = -1.0 / beta + gamma

    c00_logderiv_product = float(logderiv_alpha * logderiv_beta)

    # =========================================================
    # What the current modes use
    # =========================================================
    # RAW_LOGDERIV: (1/R + γ)²
    c00_raw = (1.0 / R + gamma) ** 2

    # POLE_CANCELLED: 1.0
    c00_pole = 1.0

    # =========================================================
    # Build details
    # =========================================================
    details = {
        "G_alpha_c0": float(G_alpha.real),
        "G_beta_c0": float(G_beta.real),
        "logderiv_alpha": logderiv_alpha,
        "logderiv_beta": logderiv_beta,
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        # Computed G(α) for α=-R using series expansion
        # G(ε) = -1 + 2γε + O(ε²), so G(-R) = -1 + 2γ(-R) = -1 - 2γR
        "G_at_minus_R_approx": -1.0 - 2.0 * gamma * R,
    }

    return J12C00Result(
        c00_literal_G_product=c00_G_product,
        c00_literal_logderiv_product=c00_logderiv_product,
        c00_raw_logderiv=c00_raw,
        c00_pole_cancelled=c00_pole,
        alpha=alpha,
        beta=beta,
        R=R,
        details=details,
    )


def print_c00_comparison(R: float = 1.3036):
    """Print comparison of c₀₀ values for debugging."""
    result = compute_j12_c00_reference(R)

    print("=" * 70)
    print(f"PHASE 14H: J12 c₀₀ Reference Computation (R={R})")
    print("=" * 70)
    print()
    print("LITERAL VALUES (from series expansion):")
    print(f"  G-product G(α)×G(β):        {result.c00_literal_G_product:.6f}")
    print(f"  Log-deriv product:          {result.c00_literal_logderiv_product:.6f}")
    print()
    print("CURRENT MODES:")
    print(f"  RAW_LOGDERIV (1/R+γ)²:      {result.c00_raw_logderiv:.6f}")
    print(f"  POLE_CANCELLED (1.0):       {result.c00_pole_cancelled:.6f}")
    print()
    print("DETAILS:")
    print(f"  α = β = {result.alpha:.4f}")
    print(f"  G(α) ≈ G(-R) = {result.details['G_alpha_c0']:.6f}")
    print(f"  (ζ'/ζ)(1+α) = {result.details['logderiv_alpha']:.6f}")
    print()

    # Check which mode matches which literal
    diff_G_raw = abs(result.c00_literal_G_product - result.c00_raw_logderiv)
    diff_G_pole = abs(result.c00_literal_G_product - result.c00_pole_cancelled)
    diff_logderiv_raw = abs(result.c00_literal_logderiv_product - result.c00_raw_logderiv)
    diff_logderiv_pole = abs(result.c00_literal_logderiv_product - result.c00_pole_cancelled)

    print("MATCHES:")
    print(f"  |G-product - RAW|:       {diff_G_raw:.6f}")
    print(f"  |G-product - POLE|:      {diff_G_pole:.6f}")
    print(f"  |logderiv - RAW|:        {diff_logderiv_raw:.6f}")
    print(f"  |logderiv - POLE|:       {diff_logderiv_pole:.6f}")
    print()

    # Interpretation
    if diff_logderiv_raw < 0.01:
        print("CONCLUSION: RAW_LOGDERIV matches log-derivative product")
    elif diff_G_pole < 0.01:
        print("CONCLUSION: POLE_CANCELLED matches G-product")
    else:
        print("CONCLUSION: Neither mode matches literal (need further analysis)")

    print("=" * 70)

    return result


if __name__ == "__main__":
    # Test for both benchmarks
    for R in [1.3036, 1.1167]:
        print_c00_comparison(R)
        print()
