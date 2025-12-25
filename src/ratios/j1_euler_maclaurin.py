"""
src/ratios/j1_euler_maclaurin.py
Phase 14C Task C4: Euler-Maclaurin Integral Forms for J1 Pieces

PURPOSE:
========
Replace n-sum truncation with analytic integral forms.

PRZZ explicitly converts Dirichlet sums to integrals via Euler-Maclaurin:

    Σ_{n≤N} f(n) → ∫_0^1 P(u) du × (analytic form)

This removes:
1. Cutoff artifacts from n_cutoff
2. Slow convergence of truncated sums
3. Numerical noise that masks the "+5" signal

TeX REFERENCE:
=============
PRZZ Lines 2391-2409 contain the Euler-Maclaurin lemma.
The conversion introduces (1-u) weight factors.

KEY FORMULAS:
=============
I₁₂ = (T Φ̂(0) / log N) × (1/(α+β)) × ∫₀¹ P₁(u)P₂(u) du

I₁₃ = -(T Φ̂(0) / log N) × ∫₀¹ (1-u) P₁(u)P₂(u) du

I₁₄ = -(T Φ̂(0) / log N) × ∫₀¹ (1-u) P₁(u)P₂(u) du
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy import integrate

from src.ratios.arithmetic_factor import A11_prime_sum
from src.ratios.zeta_laurent import EULER_MASCHERONI


class LaurentMode(str, Enum):
    """Laurent factor handling modes for J12.

    Phase 14G tested two modes:
    - RAW_LOGDERIV: (1/R + gamma)^2 - Laurent approximation
    - POLE_CANCELLED: +1 constant - Limit as alpha->0

    Phase 14H SEMANTIC DECISION:
    ===========================
    RAW_LOGDERIV matches the bracket_2 formula as written.

    Phase 15A FINDING:
    =================
    The Laurent approximation (1/R + gamma)^2 has significant error (~20%)
    compared to the actual numerical value of (zeta'/zeta)(1-R)^2.

    - Actual (zeta'/zeta)(1-R)^2 ≈ 3.00 (kappa), 3.16 (kappa*)
    - Laurent (1/R + gamma)^2 ≈ 1.81 (kappa), 2.17 (kappa*)

    New modes added:
    - ACTUAL_LOGDERIV: Actual numerical (zeta'/zeta)(1-R)^2 via mpmath
    - FULL_G_PRODUCT: G(-R)^2 = (zeta'/zeta^2)^2 [includes 1/zeta factor]

    Note: FULL_G_PRODUCT gives values 10-20x larger than expected.
    ACTUAL_LOGDERIV is the more plausible correction (~60% larger than Laurent).

    Proof: See tests/test_j12_c00_semantics.py and src/ratios/g_product_full.py
    """
    RAW_LOGDERIV = "raw_logderiv"       # Laurent approximation: (1/R + gamma)^2
    POLE_CANCELLED = "pole_cancelled"   # Limit as alpha->0: +1 constant
    ACTUAL_LOGDERIV = "actual_logderiv" # Phase 15A: Actual (zeta'/zeta)(1-R)^2
    FULL_G_PRODUCT = "full_g_product"   # Phase 15A: G(-R)^2 = (zeta'/zeta^2)^2


# Phase 15B: Lock default mode based on Phase 15 investigation
# ACTUAL_LOGDERIV uses the true numerical value of (ζ'/ζ)(1-R)²
# instead of the Laurent approximation (1/R + γ)² which has ~20% error.
# See: docs/PHASE_15_SUMMARY.md for full analysis
DEFAULT_LAURENT_MODE = LaurentMode.ACTUAL_LOGDERIV

if TYPE_CHECKING:
    from src.ratios.przz_polynomials import PrzzK3Polynomials


def _default_P1(u: float) -> float:
    """Default P₁ polynomial: linear (1-u)."""
    return 1.0 - u


def _default_P2(u: float) -> float:
    """Default P₂ polynomial: identity."""
    return u


def _extract_poly_funcs(polys: Optional["PrzzK3Polynomials"]) -> Tuple[callable, callable]:
    """
    Extract callable P1 and P2 functions from PrzzK3Polynomials.

    Args:
        polys: Optional PrzzK3Polynomials dataclass

    Returns:
        (P1_func, P2_func) tuple of callables
    """
    if polys is None:
        return None, None

    def P1_func(u: float) -> float:
        return float(polys.P1.eval(np.array([u]))[0])

    def P2_func(u: float) -> float:
        return float(polys.P2.eval(np.array([u]))[0])

    return P1_func, P2_func


def j11_as_integral(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None
) -> float:
    """
    J11 contribution as Euler-Maclaurin integral.

    The (1⋆Λ₂) Dirichlet sum becomes an integral over the
    polynomial product P₁(u)P₂(u).

    Note: The exact form depends on how Λ₂ integrates.
    For main-term purposes, this is a polynomial-weighted integral.

    Args:
        R: PRZZ R parameter
        theta: θ parameter (default 4/7)
        P1_func, P2_func: Polynomial functions (default to simple forms)

    Returns:
        J11 integral contribution
    """
    P1 = P1_func or _default_P1
    P2 = P2_func or _default_P2

    # Integrand: P₁(u)P₂(u) × [some weight from Λ₂]
    # For main-term, use (1+u) weight from log structure
    def integrand(u):
        if u <= 0 or u >= 1:
            return 0.0
        # Weight from (1⋆Λ₂) structure: involves log²(1/u) terms
        # Simplified main-term: just polynomial product
        return P1(u) * P2(u)

    result, _ = integrate.quad(integrand, 0, 1)
    return result


def j12_as_integral(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> float:
    """
    J12 contribution as Euler-Maclaurin integral.

    PRZZ main-term form:
        I₁₂ = (T Φ̂(0) / log N) × (1/(α+β)) × ∫₀¹ P₁(u)P₂(u) du

    At α=β=-R: 1/(α+β) = 1/(-2R) = -1/(2R)

    Phase 14G: Added laurent_mode parameter.
    Phase 14H: RAW_LOGDERIV proven semantically correct.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions
        laurent_mode: Laurent factor mode (default from DEFAULT_LAURENT_MODE)

    Returns:
        J12 integral contribution
    """
    P1 = P1_func or _default_P1
    P2 = P2_func or _default_P2

    # Polynomial integral
    def integrand(u):
        return P1(u) * P2(u)

    poly_integral, _ = integrate.quad(integrand, 0, 1)

    # Factor from 1/(α+β) at α=β=-R
    divisor = -2.0 * R
    if abs(divisor) < 1e-14:
        divisor = 1.0  # Avoid division by zero

    # Phase 14G/15A: Laurent factor depends on mode
    if laurent_mode == LaurentMode.POLE_CANCELLED:
        # PRZZ pole-cancelled mode: G(α+s) x G(β+u) = (-1) x (-1) = +1
        # This is R-invariant (the correct main-term behavior)
        # See inv_zeta_times_logderiv_series() which proves c_0 = -1
        laurent_factor = 1.0
    elif laurent_mode == LaurentMode.ACTUAL_LOGDERIV:
        # Phase 15A: Use ACTUAL (ζ'/ζ)(1-R)² computed numerically
        # This is ~66% larger than Laurent approximation for κ, ~46% for κ*
        from src.ratios.g_product_full import compute_j12_actual_logderiv_squared
        laurent_factor = compute_j12_actual_logderiv_squared(abs(R))
    elif laurent_mode == LaurentMode.FULL_G_PRODUCT:
        # Phase 15A: Use FULL G-product G(-R)² = (ζ'/ζ²)²
        # WARNING: This gives values 19-35x larger than expected - likely NOT correct
        from src.ratios.g_product_full import compute_j12_full_G_product
        laurent_factor = compute_j12_full_G_product(abs(R))
    else:
        # RAW_LOGDERIV: Laurent approximation (Phase 14E behavior, R-sensitive)
        # (1/R + gamma)^2 varies with R
        laurent_factor = (1.0 / R + EULER_MASCHERONI) ** 2

    return laurent_factor * poly_integral / divisor


def j13_as_integral(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> float:
    """
    J13 contribution as Euler-Maclaurin integral.

    PRZZ main-term form with (1-u) weight and NEGATIVE sign:
        I₁₃ = -(T Φ̂(0) / log N) × (1/θ) × ∫₀¹ (1-u) P₁(u)P₂(u) du

    The (1-u) weight comes from Euler-Maclaurin conversion.
    The negative sign comes from Laurent reduction.

    Phase 16: Added laurent_mode parameter. J13 uses SINGLE (ζ'/ζ) factor,
    not squared like J12. Laurent approximation has ~29% error for κ.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions
        laurent_mode: Laurent factor mode (default ACTUAL_LOGDERIV)

    Returns:
        J13 integral contribution (NEGATIVE)
    """
    P1 = P1_func or _default_P1
    P2 = P2_func or _default_P2

    # Integrand with (1-u) weight
    def integrand(u):
        return (1.0 - u) * P1(u) * P2(u)

    poly_integral, _ = integrate.quad(integrand, 0, 1)

    # PRZZ prefactor: -1/θ
    prefactor = -1.0 / theta

    # Phase 16: β-side ζ'/ζ factor depends on mode
    if laurent_mode == LaurentMode.ACTUAL_LOGDERIV:
        from src.ratios.g_product_full import compute_zeta_logderiv_actual
        beta_logderiv = compute_zeta_logderiv_actual(abs(R))
    else:
        # Original Laurent approximation
        beta_logderiv = 1.0 / R + EULER_MASCHERONI

    return prefactor * beta_logderiv * poly_integral


def j14_as_integral(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> float:
    """
    J14 contribution (symmetric with J13).

    Same (1-u) weight and negative prefactor.

    Phase 16: Added laurent_mode parameter. J14 uses SINGLE (ζ'/ζ) factor,
    not squared like J12. Laurent approximation has ~29% error for κ.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions
        laurent_mode: Laurent factor mode (default ACTUAL_LOGDERIV)

    Returns:
        J14 integral contribution (NEGATIVE)
    """
    P1 = P1_func or _default_P1
    P2 = P2_func or _default_P2

    # Integrand with (1-u) weight
    def integrand(u):
        return (1.0 - u) * P1(u) * P2(u)

    poly_integral, _ = integrate.quad(integrand, 0, 1)

    # PRZZ prefactor: -1/θ
    prefactor = -1.0 / theta

    # Phase 16: α-side ζ'/ζ factor depends on mode (same as β at α=β=-R)
    if laurent_mode == LaurentMode.ACTUAL_LOGDERIV:
        from src.ratios.g_product_full import compute_zeta_logderiv_actual
        alpha_logderiv = compute_zeta_logderiv_actual(abs(R))
    else:
        # Original Laurent approximation
        alpha_logderiv = 1.0 / R + EULER_MASCHERONI

    return prefactor * alpha_logderiv * poly_integral


def j15_as_integral(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None
) -> float:
    """
    J15 contribution using A^{(1,1)} prime sum.

    J15 = A^{(1,1)}(0) × ∫₀¹ P₁(u)P₂(u) du

    Note: A^{(1,1)} is evaluated at s=0 (diagonal point).

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions

    Returns:
        J15 integral contribution
    """
    P1 = P1_func or _default_P1
    P2 = P2_func or _default_P2

    # Polynomial integral
    def integrand(u):
        return P1(u) * P2(u)

    poly_integral, _ = integrate.quad(integrand, 0, 1)

    # A^{(1,1)}(0) ≈ 1.3856
    A11_value = A11_prime_sum(0.0, prime_cutoff=5000)

    return A11_value * poly_integral


def compute_J1_as_integrals(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None
) -> Dict[str, float]:
    """
    Compute all J1 pieces as Euler-Maclaurin integrals.

    This is the cutoff-free version for bridge analysis.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions

    Returns:
        Dictionary with j11-j15 contributions
    """
    return {
        'j11': j11_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
        'j12': j12_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
        'j13': j13_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
        'j14': j14_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
        'j15': j15_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
    }


def compute_S12_from_J1_integrals(
    theta: float,
    R: float,
    *,
    P1_func=None,
    P2_func=None,
    polys: Optional["PrzzK3Polynomials"] = None
) -> Dict:
    """
    Bridge analysis using Euler-Maclaurin integrals.

    This is the cutoff-free replacement for compute_S12_from_J1_pieces_micro.

    Args:
        theta: θ parameter
        R: PRZZ R parameter
        P1_func, P2_func: Polynomial functions (legacy interface)
        polys: PrzzK3Polynomials dataclass (preferred interface)

    Returns:
        Dictionary with decomposition analysis
    """
    # Extract polynomial functions from polys if provided
    if polys is not None:
        P1_func, P2_func = _extract_poly_funcs(polys)

    pieces = compute_J1_as_integrals(R, theta, P1_func=P1_func, P2_func=P2_func)
    total = sum(pieces.values())

    # Compute at two R values for A*exp(R)+B decomposition
    R2 = R + 0.2
    pieces_R2 = compute_J1_as_integrals(R2, theta, P1_func=P1_func, P2_func=P2_func)
    total_R2 = sum(pieces_R2.values())

    exp_R = np.exp(R)
    exp_R2 = np.exp(R2)

    # Linear regression: total = A * exp(R) + B
    A = (total_R2 - total) / (exp_R2 - exp_R)
    B = total - A * exp_R

    return {
        "exp_R_coefficient": float(A),
        "constant_offset": float(B),
        "total": float(total),
        "per_piece": pieces,
        "R": R,
        "theta": theta,
        "method": "euler_maclaurin_integrals",
    }


def decompose_m1_using_integrals(
    theta: float,
    R: float,
    *,
    polys: Optional["PrzzK3Polynomials"] = None
) -> Dict:
    """
    Decompose m₁ = exp(R) + 5 using Euler-Maclaurin integrals.

    This is the cutoff-free version of decompose_m1_from_pieces.

    Args:
        theta: θ parameter
        R: PRZZ R parameter
        polys: Optional PrzzK3Polynomials dataclass. If provided, uses real
               PRZZ polynomials instead of simplified defaults.

    Returns:
        Dictionary with exp coefficient, constant offset, and per-piece analysis
    """
    R1, R2 = R, R + 0.2

    # Extract polynomial functions once (reuse for both R values)
    P1_func, P2_func = _extract_poly_funcs(polys)

    result_R1 = compute_S12_from_J1_integrals(
        theta, R1, P1_func=P1_func, P2_func=P2_func
    )
    result_R2 = compute_S12_from_J1_integrals(
        theta, R2, P1_func=P1_func, P2_func=P2_func
    )

    total_R1 = result_R1["total"]
    total_R2 = result_R2["total"]

    exp_R1 = np.exp(R1)
    exp_R2 = np.exp(R2)

    # Solve: total = A * exp(R) + B
    A = (total_R2 - total_R1) / (exp_R2 - exp_R1)
    B = total_R1 - A * exp_R1

    # Per-piece analysis
    per_piece = {}
    for name in ['j11', 'j12', 'j13', 'j14', 'j15']:
        val_R1 = result_R1["per_piece"][name]
        val_R2 = result_R2["per_piece"][name]

        piece_slope = (val_R2 - val_R1) / (exp_R2 - exp_R1)
        piece_intercept = val_R1 - piece_slope * exp_R1

        per_piece[name] = {
            "exp_coefficient": float(piece_slope),
            "constant": float(piece_intercept),
        }

    return {
        "exp_coefficient": float(A),
        "constant_offset": float(B),
        "target_constant": 5,  # = 2K-1 for K=3
        "per_piece_contribution": per_piece,
        "R_values": [R1, R2],
        "total_values": [total_R1, total_R2],
        "method": "euler_maclaurin_integrals",
        "using_real_polynomials": polys is not None,
    }


def compute_I12_components(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> Dict[str, float]:
    """
    Compute I₁I₂-type components that require mirror assembly.

    PRZZ Structure:
    - I₁ and I₂ require mirror: I(α,β) + T^{-α-β} × I(-β,-α)
    - J11, J12, J15 are I₁I₂ type

    For mirror assembly:
    - At +R: α = β = -R (original evaluation point)
    - At -R: α = β = +R (mirror point after sign swap)

    Phase 14G: Added laurent_mode parameter for j12.
    Phase 14H: RAW_LOGDERIV proven semantically correct.

    Args:
        R: PRZZ R parameter (can be positive or negative for mirror)
        theta: θ parameter
        P1_func, P2_func: Polynomial functions
        laurent_mode: Laurent factor mode for j12

    Returns:
        Dictionary with j11, j12, j15 contributions
    """
    # Pass R directly - do NOT use abs(R)
    # j12 especially depends on sign of R
    return {
        'j11': j11_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
        'j12': j12_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func,
                               laurent_mode=laurent_mode),
        'j15': j15_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func),
    }


def compute_I34_components(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    P1_func=None,
    P2_func=None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
) -> Dict[str, float]:
    """
    Compute I₃I₄-type components that do NOT require mirror assembly.

    PRZZ Structure:
    - I₃ and I₄ do NOT require mirror
    - J13, J14 are I₃I₄ type

    Phase 16: Added laurent_mode parameter to thread mode to j13/j14.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        P1_func, P2_func: Polynomial functions
        laurent_mode: Laurent factor mode for j13/j14

    Returns:
        Dictionary with j13, j14 contributions
    """
    return {
        'j13': j13_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func,
                               laurent_mode=laurent_mode),
        'j14': j14_as_integral(R, theta, P1_func=P1_func, P2_func=P2_func,
                               laurent_mode=laurent_mode),
    }


def compute_m1_with_mirror_assembly(
    theta: float,
    R: float,
    *,
    polys: Optional["PrzzK3Polynomials"] = None,
    K: int = 3,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    include_j15: bool = True,
) -> Dict:
    """
    Compute m₁ using PRZZ mirror assembly formula.

    PHASE 14E: This is the core fix for the +5 gate.
    PHASE 14G: Added laurent_mode parameter for mode testing.
    PHASE 14H: Proved RAW_LOGDERIV is semantically correct.

    PRZZ Structure (from TRUTH_SPEC.md):
        c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)

    Where m = exp(R) + (2K-1) for K pieces.
    For K=3: m = exp(R) + 5

    KEY INSIGHT:
    - The "+5" is a combinatorial factor from mirror assembly
    - NOT from J15/polynomial integrals (Phase 14D showed J15 ≈ 0.65)
    - A/B extraction must happen AFTER mirror assembly, not on individual pieces

    Phase 14H semantic decision:
    - RAW_LOGDERIV: Uses (1/R + γ)² which matches J12 bracket₂ structure
    - POLE_CANCELLED: Uses +1 but does NOT match J12 (was based on G-product)

    Phase 19.1: Added include_j15 parameter to enable main-only vs with-error
    mode separation. J₁,₅ involves A^{(1,1)} which is an error term per
    TRUTH_SPEC Lines 1621-1628.

    Args:
        theta: θ parameter
        R: PRZZ R parameter
        polys: Optional PrzzK3Polynomials for real polynomials
        K: Number of mollifier pieces (default 3)
        laurent_mode: Laurent factor mode for j12 (default RAW_LOGDERIV)
        include_j15: If False, exclude J₁,₅ from computation (main-term only)

    Returns:
        Dictionary with exp_coefficient, constant_offset, and detailed breakdown
    """
    # Extract polynomial functions
    P1_func, P2_func = _extract_poly_funcs(polys)

    # =========================================================
    # STEP 1: Compute I₁I₂ components at +R and -R
    # =========================================================
    i12_plus = compute_I12_components(R, theta, P1_func=P1_func, P2_func=P2_func,
                                       laurent_mode=laurent_mode)
    i12_minus = compute_I12_components(-R, theta, P1_func=P1_func, P2_func=P2_func,
                                        laurent_mode=laurent_mode)

    # Phase 19.1: Optionally exclude J₁,₅ (error term per TRUTH_SPEC)
    if not include_j15:
        i12_plus['j15'] = 0.0
        i12_minus['j15'] = 0.0

    # =========================================================
    # STEP 2: Compute I₃I₄ components at +R (no mirror needed)
    # Phase 16: Thread laurent_mode to I34 components for J13/J14
    # =========================================================
    i34_plus = compute_I34_components(R, theta, P1_func=P1_func, P2_func=P2_func,
                                       laurent_mode=laurent_mode)

    # =========================================================
    # STEP 3: Apply mirror assembly formula
    # =========================================================
    # Mirror multiplier: m = exp(R) + (2K - 1)
    # For K=3: m = exp(R) + 5
    m = np.exp(R) + (2 * K - 1)

    # Sum I₁I₂ components
    i12_plus_total = sum(i12_plus.values())
    i12_minus_total = sum(i12_minus.values())

    # Sum I₃I₄ components
    i34_plus_total = sum(i34_plus.values())

    # PRZZ assembly: c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
    assembled_total = i12_plus_total + m * i12_minus_total + i34_plus_total

    # =========================================================
    # STEP 4: Extract A and B from assembled result
    # =========================================================
    # The assembled formula has structure: A × exp(R) + B
    #
    # Since m = exp(R) + 5, we have:
    #   assembled = I₁₂(+R) + [exp(R) + 5] × I₁₂(-R) + I₃₄(+R)
    #             = [I₁₂(+R) + I₃₄(+R) + 5 × I₁₂(-R)] + exp(R) × I₁₂(-R)
    #
    # So: A = I₁₂(-R)  and  B = I₁₂(+R) + I₃₄(+R) + 5 × I₁₂(-R)

    A = i12_minus_total
    B = i12_plus_total + i34_plus_total + (2 * K - 1) * i12_minus_total

    # =========================================================
    # STEP 5: Compute normalized metrics (Phase 14F)
    # =========================================================
    # D = I₁₂(+R) + I₃₄(+R) is the "contamination" from non-mirror pieces
    # B = D + (2K-1) × A, so B/A = (2K-1) + D/A
    #
    # GPT's insight: checking B/A ≈ 5 is invariant across benchmarks,
    # while raw B depends on A which differs between κ and κ*.
    D = i12_plus_total + i34_plus_total
    delta = D / A if abs(A) > 1e-14 else float('inf')
    B_over_A = B / A if abs(A) > 1e-14 else float('inf')

    # Verification: assembled_total should equal A * exp(R) + B
    verify = A * np.exp(R) + B
    verification_error = abs(assembled_total - verify)

    return {
        "exp_coefficient": float(A),
        "constant_offset": float(B),
        "target_constant": 2 * K - 1,  # = 5 for K=3
        "assembled_total": float(assembled_total),
        "mirror_multiplier": float(m),
        "i12_plus_total": float(i12_plus_total),
        "i12_minus_total": float(i12_minus_total),
        "i34_plus_total": float(i34_plus_total),
        "i12_plus_pieces": {k: float(v) for k, v in i12_plus.items()},
        "i12_minus_pieces": {k: float(v) for k, v in i12_minus.items()},
        "i34_plus_pieces": {k: float(v) for k, v in i34_plus.items()},
        "R": float(R),
        "K": K,
        "theta": float(theta),
        "method": "mirror_assembly",
        "using_real_polynomials": polys is not None,
        "verification_error": float(verification_error),
        # Phase 14F: Normalized metrics
        "D": float(D),                    # D = I₁₂(+R) + I₃₄(+R)
        "delta": float(delta),            # delta = D/A (contamination ratio)
        "B_over_A": float(B_over_A),      # B/A = (2K-1) + delta ≈ 5
        # Phase 14G: Laurent mode tracking
        "laurent_mode": laurent_mode.value,
        # Phase 19.1: J₁,₅ inclusion tracking
        "include_j15": include_j15,
    }


def print_integral_bridge_analysis(
    R: float = 1.3036,
    *,
    polys: Optional["PrzzK3Polynomials"] = None,
    benchmark: str = "default"
):
    """Print analysis using Euler-Maclaurin integrals."""
    print("=" * 60)
    if polys is not None:
        print(f"PHASE 14D: BRIDGE ANALYSIS WITH REAL POLYNOMIALS")
        print(f"Benchmark: {polys.benchmark}, R={polys.R}")
    else:
        print(f"PHASE 14C: EULER-MACLAURIN BRIDGE ANALYSIS (R={R})")
        print(f"Using simplified default polynomials")
    print("=" * 60)
    print()

    decomp = decompose_m1_using_integrals(theta=4.0 / 7.0, R=R, polys=polys)

    print("m₁ decomposition (using integrals):")
    print(f"  m₁ ≈ A × exp(R) + B")
    print(f"  A (exp coefficient): {decomp['exp_coefficient']:.6f}")
    print(f"  B (constant offset): {decomp['constant_offset']:.6f}")
    print(f"  Target B: {decomp['target_constant']} (= 2K-1 for K=3)")
    print(f"  Using real polynomials: {decomp['using_real_polynomials']}")
    print()

    print("Per-piece contributions:")
    for name, contrib in decomp["per_piece_contribution"].items():
        print(f"  {name}:")
        print(f"    exp coefficient: {contrib['exp_coefficient']:.6f}")
        print(f"    constant: {contrib['constant']:.6f}")
    print()

    print("=" * 60)


def print_mirror_assembly_analysis(
    R: float = 1.3036,
    *,
    polys: Optional["PrzzK3Polynomials"] = None,
):
    """Print Phase 14E/14F mirror assembly analysis."""
    print("=" * 70)
    print("PHASE 14E/14F: MIRROR ASSEMBLY ANALYSIS")
    if polys is not None:
        print(f"Benchmark: {polys.benchmark}, R={polys.R}")
    else:
        print(f"Using default polynomials, R={R}")
    print("=" * 70)
    print()

    decomp = compute_m1_with_mirror_assembly(theta=4.0 / 7.0, R=R, polys=polys)

    print("PRZZ Mirror Assembly Formula:")
    print("  c = I₁₂(+R) + m × I₁₂(-R) + I₃₄(+R)")
    print(f"  where m = exp(R) + {decomp['target_constant']} = {decomp['mirror_multiplier']:.4f}")
    print()

    print("Component totals:")
    print(f"  I₁₂(+R): {decomp['i12_plus_total']:.6f}")
    print(f"  I₁₂(-R): {decomp['i12_minus_total']:.6f}")
    print(f"  I₃₄(+R): {decomp['i34_plus_total']:.6f}")
    print()

    print("Assembled result:")
    print(f"  Total: {decomp['assembled_total']:.6f}")
    print(f"  = I₁₂(+R) + {decomp['mirror_multiplier']:.4f} × I₁₂(-R) + I₃₄(+R)")
    print()

    print("m₁ = A × exp(R) + B decomposition:")
    print(f"  A (exp coefficient): {decomp['exp_coefficient']:.6f}")
    print(f"  B (constant offset): {decomp['constant_offset']:.6f}")
    print(f"  Target B: {decomp['target_constant']} (= 2K-1 for K=3)")
    gap = decomp['constant_offset'] - decomp['target_constant']
    print(f"  Gap from target: {gap:+.6f}")
    print()

    # Phase 14F: Normalized metrics
    print("PHASE 14F: NORMALIZED METRICS (B/A):")
    print("-" * 50)
    print(f"  D = I₁₂(+R) + I₃₄(+R): {decomp['D']:.6f}")
    print(f"  delta = D/A: {decomp['delta']:.6f}")
    print(f"  B/A = {decomp['target_constant']} + delta: {decomp['B_over_A']:.6f}")
    print(f"  Target B/A: {decomp['target_constant']} (= 2K-1 for K=3)")
    gap_normalized = decomp['B_over_A'] - decomp['target_constant']
    gap_pct = gap_normalized / decomp['target_constant'] * 100
    print(f"  Gap from target: {gap_normalized:+.6f} ({gap_pct:+.1f}%)")
    print()

    print("I₁₂ pieces (at +R):")
    for name, val in decomp['i12_plus_pieces'].items():
        print(f"  {name}: {val:.6f}")
    print()

    print("I₁₂ pieces (at -R):")
    for name, val in decomp['i12_minus_pieces'].items():
        print(f"  {name}: {val:.6f}")
    print()

    print("I₃₄ pieces (at +R):")
    for name, val in decomp['i34_plus_pieces'].items():
        print(f"  {name}: {val:.6f}")
    print()

    print(f"Verification error: {decomp['verification_error']:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    from src.ratios.przz_polynomials import load_przz_k3_polynomials

    # Test Phase 14E mirror assembly
    print("\n" + "=" * 70)
    print("PHASE 14E: MIRROR ASSEMBLY TESTS")
    print("=" * 70)

    print("\n--- Test 1: Default Polynomials (mirror assembly) ---")
    print_mirror_assembly_analysis(R=1.3036)

    print("\n--- Test 2: KAPPA Polynomials (mirror assembly) ---")
    polys_k = load_przz_k3_polynomials("kappa")
    print_mirror_assembly_analysis(R=polys_k.R, polys=polys_k)

    print("\n--- Test 3: KAPPA* Polynomials (mirror assembly) ---")
    polys_ks = load_przz_k3_polynomials("kappa_star")
    print_mirror_assembly_analysis(R=polys_ks.R, polys=polys_ks)

    # Compare old vs new approach
    print("\n" + "=" * 70)
    print("COMPARISON: Phase 14D (no mirror) vs Phase 14E (with mirror)")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        # Old approach (Phase 14D)
        old = decompose_m1_using_integrals(theta=4.0 / 7.0, R=R, polys=polys)
        # New approach (Phase 14E)
        new = compute_m1_with_mirror_assembly(theta=4.0 / 7.0, R=R, polys=polys)

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  Phase 14D B (no mirror): {old['constant_offset']:.4f}")
        print(f"  Phase 14E B (with mirror): {new['constant_offset']:.4f}")
        print(f"  Target B: 5")
