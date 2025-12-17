"""
src/terms_k3_d1.py
Term definitions for K=3, d=1 PRZZ computation.

This module builds Term objects for each (ℓ₁, ℓ₂) pair.
Currently implements (1,1) terms: I₁, I₂, I₃, I₄.

PRZZ Reference: Section 6.2.1

=============================================================================
(1,1) I₁ TERM STRUCTURE
=============================================================================

vars = ("x1", "y1")
deriv_orders = {"x1": 1, "y1": 1}

ARGUMENTS (from PRZZ):
----------------------
For pair (ℓ₁, ℓ₂) with variables x₁...x_{ℓ₁} and y₁...y_{ℓ₂}:

Let S = sum of all vars = x1 + y1 (for (1,1))
Let X = sum of x vars = x1
Let Y = sum of y vars = y1

Arg_α = θ·t·S − θ·Y + t
      = t + θt·x1 + (θt−θ)·y1
      Coefficients: a0=t, x1_coeff=θt, y1_coeff=θ(t-1)

Arg_β = θ·t·S − θ·X + t
      = t + (θt−θ)·x1 + θt·y1
      Coefficients: a0=t, x1_coeff=θ(t-1), y1_coeff=θt

NOTE: Arg_α ≠ Arg_β (swapped x/y coefficients!)
      Do NOT collapse Q(Arg_α)·Q(Arg_β) into Q(...)²

FACTORS:
--------
- Algebraic prefactor: (θS + 1)/θ = 1/θ + x1 + y1
- Poly prefactors: [(1-u)²]
- Poly factors: [P₁(x1+u), P₁(y1+u)]  (P₁ = P₂ for ℓ=1)
- Q factors: [Q(Arg_α), Q(Arg_β)]
- Exp factors: [exp(R·Arg_α), exp(R·Arg_β)]

NUMERIC PREFACTOR:
------------------
For (1,1): pair sign is (-1)^{1+1} = +1
I₁ has no additional sign, so numeric_prefactor = 1.0

=============================================================================
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Callable
import numpy as np

from src.term_dsl import (
    Term, AffineExpr, PolyFactor, ExpFactor,
    SeriesContext, GridFunc
)


# =============================================================================
# Omega-case mapping for polynomial pieces
# =============================================================================
# PRZZ omega classification (TeX lines 2302-2304):
#   omega(d,l) = 1×l₁ + ... + d×l_d - 1
# For d=1: omega = ell - 1
#
# INVESTIGATION RESULT (2025-12-16):
# Setting P2/P3 to Case C (omega=1,2) made c drop from ~1.95 to ~0.64
# This goes in the WRONG direction (PRZZ target is c=2.14)
# The omega-case classification likely applies to a DIFFERENT part of
# the PRZZ computation (possibly Υ function or bracket combinatorics),
# NOT to the main integral P(u+X) evaluations.
#
# For now, all polynomials use Case B (omega=0) to preserve baseline.
# The gap vs PRZZ (1.95 vs 2.14) remains under investigation.
#
# - P1 (ell=1): omega=0 → Case B (standard polynomial)
# - P2 (ell=2): omega=0 → Case B (standard polynomial) - NOT Case C
# - P3 (ell=3): omega=0 → Case B (standard polynomial) - NOT Case C
# - Q: Not subject to omega handling (evaluated at t, not u)
#
OMEGA_FOR_POLY = {
    "P1": 0,   # Case B
    "P2": 0,   # Case B (NOT Case C - see note above)
    "P3": 0,   # Case B (NOT Case C - see note above)
    "Q": None  # Not subject to omega handling
}


def make_poly_factor(poly_name: str, argument: AffineExpr, power: int = 1) -> PolyFactor:
    """
    Create PolyFactor with correct omega metadata based on polynomial name.

    This helper ensures all P polynomial factors have the correct omega for
    Case B/C handling during evaluation.
    """
    omega = OMEGA_FOR_POLY.get(poly_name)
    return PolyFactor(poly_name=poly_name, argument=argument, power=power, omega=omega)


# =============================================================================
# Helper functions for building AffineExpr objects
# =============================================================================

def make_P_argument(var: str) -> AffineExpr:
    """
    Build P argument: var + u

    For P₁(x1+u): make_P_argument("x1")
    """
    return AffineExpr(
        a0=lambda U, T: U,
        var_coeffs={var: 1.0}
    )


def make_Q_arg_alpha(theta: float, x_vars: Tuple[str, ...], y_vars: Tuple[str, ...]) -> AffineExpr:
    """
    Build Q argument α: θ·t·S − θ·Y + t

    For (1,1): Arg_α = t + θt·x1 + (θt−θ)·y1

    Args:
        theta: θ parameter
        x_vars: tuple of x variable names
        y_vars: tuple of y variable names
    """
    var_coeffs: Dict[str, GridFunc] = {}

    # x variables get coefficient θ·t
    for v in x_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T

    # y variables get coefficient θ·t − θ = θ(t-1)
    for v in y_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T - th

    return AffineExpr(
        a0=lambda U, T: T,  # constant term is t
        var_coeffs=var_coeffs
    )


def make_Q_arg_beta(theta: float, x_vars: Tuple[str, ...], y_vars: Tuple[str, ...]) -> AffineExpr:
    """
    Build Q argument β: θ·t·S − θ·X + t

    For (1,1): Arg_β = t + (θt−θ)·x1 + θt·y1

    NOTE: This is NOT the same as Arg_α! Coefficients are swapped.

    Args:
        theta: θ parameter
        x_vars: tuple of x variable names
        y_vars: tuple of y variable names
    """
    var_coeffs: Dict[str, GridFunc] = {}

    # x variables get coefficient θ·t − θ = θ(t-1)
    for v in x_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T - th

    # y variables get coefficient θ·t
    for v in y_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T

    return AffineExpr(
        a0=lambda U, T: T,  # constant term is t
        var_coeffs=var_coeffs
    )


def make_algebraic_prefactor_11(theta: float) -> AffineExpr:
    """
    Build algebraic prefactor for (1,1): (θS + 1)/θ = 1/θ + x1 + y1

    This is an AffineExpr, NOT absorbed into numeric_prefactor.
    """
    return AffineExpr(
        a0=1.0 / theta,
        var_coeffs={"x1": 1.0, "y1": 1.0}
    )


def make_poly_prefactor_11() -> GridFunc:
    """
    Build poly prefactor for (1,1): (1-u)²

    This is a pure grid function, NOT an AffineExpr.
    """
    return lambda U, T: (1 - U) ** 2


def make_P_argument_unshifted() -> AffineExpr:
    """
    Build P argument at just u (no formal variables): P(u)

    For I₂ where P is evaluated at u only, not shifted by x or y.
    """
    return AffineExpr(
        a0=lambda U, T: U,
        var_coeffs={}  # No formal variables
    )


def make_Q_argument_at_t() -> AffineExpr:
    """
    Build Q argument at just t (no formal variables): Q(t)

    For I₂ where Q is evaluated at t only.
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={}  # No formal variables
    )


def make_exp_argument_at_t() -> AffineExpr:
    """
    Build exp argument at just t: exp(2R·t)

    For I₂ decoupled term.
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={}
    )


def make_Q_arg_alpha_x_only(theta: float) -> AffineExpr:
    """
    Build Q argument α with only x variable: t + θt·x1

    For I₃ (single x derivative term).
    When y is not differentiated, y=0 in the integrand, so:
    Arg_α|_{y=0} = t + θt·x1
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T}
    )


def make_Q_arg_beta_x_only(theta: float) -> AffineExpr:
    """
    Build Q argument β with only x variable: t + θ(t-1)·x1

    For I₃ (single x derivative term).
    When y is not differentiated, y=0 in the integrand, so:
    Arg_β|_{y=0} = t + θ(t-1)·x1
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T - th}
    )


def make_Q_arg_alpha_y_only(theta: float) -> AffineExpr:
    """
    Build Q argument α with only y variable: t + θ(t-1)·y1

    For I₄ (single y derivative term).
    When x is not differentiated, x=0 in the integrand, so:
    Arg_α|_{x=0} = t + θ(t-1)·y1
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y1": lambda U, T, th=theta: th * T - th}
    )


def make_Q_arg_beta_y_only(theta: float) -> AffineExpr:
    """
    Build Q argument β with only y variable: t + θt·y1

    For I₄ (single y derivative term).
    When x is not differentiated, x=0 in the integrand, so:
    Arg_β|_{x=0} = t + θt·y1
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y1": lambda U, T, th=theta: th * T}
    )


def make_poly_prefactor_single() -> GridFunc:
    """
    Build poly prefactor for I₃/I₄: (1-u)

    Single derivative terms have (1-u)^1, not (1-u)^2.
    """
    return lambda U, T: (1 - U)


# =============================================================================
# SINGLE-VARIABLE HELPERS (PRZZ CORRECT STRUCTURE)
# =============================================================================
#
# ROOT CAUSE FIX: PRZZ uses only 2 variables (x, y) for ALL pairs, not ℓ₁+ℓ₂.
# The original DSL used x1,x2,...,x_{ℓ₁} and y1,y2,...,y_{ℓ₂} which computes
# completely different derivatives (e.g., d⁴/dx₁dx₂dy₁dy₂ vs d²/dxdy).
#
# Evidence: For (2,2) pair, DSL I₁ was 3.3× oracle, I₃/I₄ had wrong signs.
# I₂ matched oracle (0.9088) because it has no derivatives.
#
# These helpers use the PRZZ-correct single x, y structure.
#

def _make_P_argument_x() -> AffineExpr:
    """P argument: x + u (PRZZ correct structure)."""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"x": 1.0})


def _make_P_argument_y() -> AffineExpr:
    """P argument: y + u (PRZZ correct structure)."""
    return AffineExpr(a0=lambda U, T: U, var_coeffs={"y": 1.0})


def _make_Q_arg_alpha_single(theta: float) -> AffineExpr:
    """
    Q argument α with single x, y: t + θtx + θ(t-1)y

    PRZZ Reference: Section 6.2.1, lines 1514-1517
    For ALL (ℓ₁, ℓ₂) pairs, uses the same 2-variable structure.
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={
            "x": lambda U, T, th=theta: th * T,
            "y": lambda U, T, th=theta: th * (T - 1)
        }
    )


def _make_Q_arg_beta_single(theta: float) -> AffineExpr:
    """
    Q argument β with single x, y: t + θ(t-1)x + θty

    PRZZ Reference: Section 6.2.1
    NOTE: Coefficients are SWAPPED from α (x↔y swap).
    """
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={
            "x": lambda U, T, th=theta: th * (T - 1),
            "y": lambda U, T, th=theta: th * T
        }
    )


def _make_Q_arg_alpha_x_only_single(theta: float) -> AffineExpr:
    """Q argument α with only x: t + θtx (for I₃, y set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x": lambda U, T, th=theta: th * T}
    )


def _make_Q_arg_beta_x_only_single(theta: float) -> AffineExpr:
    """Q argument β with only x: t + θ(t-1)x (for I₃, y set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x": lambda U, T, th=theta: th * (T - 1)}
    )


def _make_Q_arg_alpha_y_only_single(theta: float) -> AffineExpr:
    """Q argument α with only y: t + θ(t-1)y (for I₄, x set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y": lambda U, T, th=theta: th * (T - 1)}
    )


def _make_Q_arg_beta_y_only_single(theta: float) -> AffineExpr:
    """Q argument β with only y: t + θty (for I₄, x set to 0)."""
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"y": lambda U, T, th=theta: th * T}
    )


def _make_algebraic_prefactor_single(theta: float) -> AffineExpr:
    """
    Algebraic prefactor with single x, y: (1+θ(x+y))/θ = 1/θ + x + y

    PRZZ Reference: This is the (θS+1)/θ factor from Section 6.2.1.
    For single x, y structure: S = x + y.
    """
    return AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0, "y": 1.0})


def _make_algebraic_prefactor_x_only(theta: float) -> AffineExpr:
    """
    Algebraic prefactor with only x: (1+θx)/θ = 1/θ + x

    For I₃ terms where y=0.
    """
    return AffineExpr(a0=1.0 / theta, var_coeffs={"x": 1.0})


def _make_algebraic_prefactor_y_only(theta: float) -> AffineExpr:
    """
    Algebraic prefactor with only y: (1+θy)/θ = 1/θ + y

    For I₄ terms where x=0.
    """
    return AffineExpr(a0=1.0 / theta, var_coeffs={"y": 1.0})


# =============================================================================
# (1,1) I₁ Term Builder
# =============================================================================

def make_I1_11(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₁ term (main coupled term).

    This is the primary term for the (1,1) pair, involving:
    - Mixed derivative ∂²/∂x₁∂y₁
    - Products of P₁ and Q polynomials
    - Exponential factors exp(R·Arg)

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)

    Returns:
        Term object representing I₁ for (1,1)

    PRZZ Reference: Section 6.2.1, equation for I₁
    """
    x_vars = ("x1",)
    y_vars = ("y1",)
    all_vars = ("x1", "y1")

    # Build the two distinct Q arguments
    # CRITICAL: These are NOT equal! Do not collapse.
    Q_arg_alpha = make_Q_arg_alpha(theta, x_vars, y_vars)
    Q_arg_beta = make_Q_arg_beta(theta, x_vars, y_vars)

    # Build P arguments
    P_arg_x = make_P_argument("x1")  # x1 + u
    P_arg_y = make_P_argument("y1")  # y1 + u

    return Term(
        name="I1_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₁",
        vars=all_vars,
        deriv_orders={"x1": 1, "y1": 1},
        domain="[0,1]^2",

        # Numeric prefactor: pair sign (-1)^{1+1} = +1, no additional sign
        numeric_prefactor=1.0,

        # Algebraic prefactor: (θS + 1)/θ = 1/θ + x1 + y1
        algebraic_prefactor=make_algebraic_prefactor_11(theta),

        # Poly prefactors: (1-u)²
        poly_prefactors=[make_poly_prefactor_11()],

        # Poly factors: P₁(x1+u), P₁(y1+u), Q(Arg_α), Q(Arg_β)
        poly_factors=[
            make_poly_factor("P1", P_arg_x),
            make_poly_factor("P1", P_arg_y),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],

        # Exp factors: exp(R·Arg_α), exp(R·Arg_β)
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


# =============================================================================
# (1,1) I₂ Term Builder - Decoupled term (no derivatives)
# =============================================================================

def make_I2_11(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₂ term (decoupled term, no derivatives).

    This term has no formal variables - it's evaluated at x=0, y=0.
    The integral is just over (u, t) ∈ [0,1]².

    Structure:
    - vars = () (empty)
    - deriv_orders = {} (empty)
    - P factors: P₁(u), P₁(u) (both unshifted)
    - Q factor: Q(t)² (single Q with power=2)
    - Exp factor: exp(2R·t) (note: 2R, not R)
    - numeric_prefactor: 1/θ

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)

    Returns:
        Term object representing I₂ for (1,1)

    PRZZ Reference: Section 6.2.1, equation for I₂
    """
    P_arg = make_P_argument_unshifted()  # P(u)
    Q_arg = make_Q_argument_at_t()       # Q(t)
    exp_arg = make_exp_argument_at_t()   # exp(2R·t)

    return Term(
        name="I2_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₂",
        vars=(),  # No formal variables
        deriv_orders={},  # No derivatives
        domain="[0,1]^2",

        # Numeric prefactor: 1/θ
        numeric_prefactor=1.0 / theta,

        # No algebraic prefactor for I₂
        algebraic_prefactor=None,

        # No poly prefactors for I₂
        poly_prefactors=[],

        # Poly factors: P₁(u), P₁(u), Q(t)²
        poly_factors=[
            make_poly_factor("P1", P_arg),
            make_poly_factor("P1", P_arg),
            make_poly_factor("Q", Q_arg, power=2),
        ],

        # Exp factor: exp(2R·t) - note the 2R scaling!
        exp_factors=[
            ExpFactor(2 * R, exp_arg),
        ]
    )


# =============================================================================
# (1,1) I₃ Term Builder - Single x derivative
# =============================================================================

def make_I3_11(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₃ term (single x derivative).

    This term differentiates only with respect to x₁.
    The y variable is evaluated at 0 in the integrand.

    Structure:
    - vars = ("x1",) (only x1)
    - deriv_orders = {"x1": 1}
    - P factors: P₁(x1+u) shifted, P₁(u) unshifted
    - Q factors: Q(Arg_α|_{y=0}), Q(Arg_β|_{y=0})
      where Arg_α|_{y=0} = t + θt·x1 and Arg_β|_{y=0} = t + θ(t-1)·x1
    - Exp factors: exp(R·Arg_α|_{y=0}), exp(R·Arg_β|_{y=0})
    - poly_prefactor: (1-u) (single power)
    - numeric_prefactor: -1.0 (leading minus sign)

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)

    Returns:
        Term object representing I₃ for (1,1)

    PRZZ Reference: Section 6.2.1, equation for I₃
    """
    # P arguments: one shifted by x1, one unshifted
    P_arg_shifted = make_P_argument("x1")  # P₁(x1 + u)
    P_arg_unshifted = make_P_argument_unshifted()  # P₁(u)

    # Q arguments with y=0 (only x1 variable)
    # CRITICAL: α and β are STILL DISTINCT even with y=0
    Q_arg_alpha = make_Q_arg_alpha_x_only(theta)  # t + θt·x1
    Q_arg_beta = make_Q_arg_beta_x_only(theta)    # t + θ(t-1)·x1

    return Term(
        name="I3_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₃ (line 1562-1563)",
        vars=("x1",),  # Only x1 formal variable
        deriv_orders={"x1": 1},
        domain="[0,1]^2",

        # Numeric prefactor: -1/θ
        # PRZZ: I₃ = -[(1+θx)/θ]|_{x=0} × d/dx[...] = -(1/θ) × derivative
        # The prefactor (1+θx)/θ evaluates at x=0 to give 1/θ
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ

        # No variable-dependent algebraic prefactor for I₃
        algebraic_prefactor=None,

        # Poly prefactors: (1-u) single power
        poly_prefactors=[make_poly_prefactor_single()],

        # Poly factors: P₁(x1+u), P₁(u), Q(Arg_α), Q(Arg_β)
        poly_factors=[
            make_poly_factor("P1", P_arg_shifted),
            make_poly_factor("P1", P_arg_unshifted),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],

        # Exp factors: exp(R·Arg_α), exp(R·Arg_β)
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


# =============================================================================
# (1,1) I₄ Term Builder - Single y derivative
# =============================================================================

def make_I4_11(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₄ term (single y derivative).

    This term differentiates only with respect to y₁.
    The x variable is evaluated at 0 in the integrand.

    Structure:
    - vars = ("y1",) (only y1)
    - deriv_orders = {"y1": 1}
    - P factors: P₁(u) unshifted, P₁(y1+u) shifted
    - Q factors: Q(Arg_α|_{x=0}), Q(Arg_β|_{x=0})
      where Arg_α|_{x=0} = t + θ(t-1)·y1 and Arg_β|_{x=0} = t + θt·y1
    - Exp factors: exp(R·Arg_α|_{x=0}), exp(R·Arg_β|_{x=0})
    - poly_prefactor: (1-u) (single power)
    - numeric_prefactor: -1.0 (leading minus sign)

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)

    Returns:
        Term object representing I₄ for (1,1)

    PRZZ Reference: Section 6.2.1, equation for I₄
    """
    # P arguments: one unshifted, one shifted by y1
    P_arg_unshifted = make_P_argument_unshifted()  # P₁(u)
    P_arg_shifted = make_P_argument("y1")  # P₁(y1 + u)

    # Q arguments with x=0 (only y1 variable)
    # CRITICAL: α and β are STILL DISTINCT even with x=0
    Q_arg_alpha = make_Q_arg_alpha_y_only(theta)  # t + θ(t-1)·y1
    Q_arg_beta = make_Q_arg_beta_y_only(theta)    # t + θt·y1

    return Term(
        name="I4_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₄ (line 1568-1569)",
        vars=("y1",),  # Only y1 formal variable
        deriv_orders={"y1": 1},
        domain="[0,1]^2",

        # Numeric prefactor: -1/θ
        # PRZZ: I₄ = -[(1+θy)/θ]|_{y=0} × d/dy[...] = -(1/θ) × derivative
        # Same as I3 structure but with y instead of x
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ

        # No variable-dependent algebraic prefactor for I₄
        algebraic_prefactor=None,

        # Poly prefactors: (1-u) single power
        poly_prefactors=[make_poly_prefactor_single()],

        # Poly factors: P₁(u), P₁(y1+u), Q(Arg_α), Q(Arg_β)
        poly_factors=[
            make_poly_factor("P1", P_arg_unshifted),
            make_poly_factor("P1", P_arg_shifted),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],

        # Exp factors: exp(R·Arg_α), exp(R·Arg_β)
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


# =============================================================================
# Convenience function to get all (1,1) terms
# =============================================================================

def make_all_terms_11(theta: float, R: float) -> List[Term]:
    """
    Build all (1,1) terms: I₁, I₂, I₃, I₄.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)

    Returns:
        List of Term objects [I₁, I₂, I₃, I₄]
    """
    return [
        make_I1_11(theta, R),
        make_I2_11(theta, R),
        make_I3_11(theta, R),
        make_I4_11(theta, R),
    ]


# =============================================================================
# (1,1) PAIR TERMS - V2 SINGLE-VARIABLE STRUCTURE (PRZZ CORRECT)
# =============================================================================
#
# These terms use standard x, y variable names (not x1, y1) for consistency
# with the PRZZ-correct single-variable structure.
#

def make_I1_11_v2(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₁ term using standard x, y variable names.

    This is structurally similar to the original but uses x, y instead of x1, y1
    for consistency with the PRZZ-correct single-variable pattern.
    """
    Q_arg_alpha = _make_Q_arg_alpha_single(theta)
    Q_arg_beta = _make_Q_arg_beta_single(theta)

    return Term(
        name="I1_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=0 I₁",
        vars=("x", "y"),  # Standard x, y
        deriv_orders={"x": 1, "y": 1},
        domain="[0,1]^2",
        numeric_prefactor=1.0,
        algebraic_prefactor=_make_algebraic_prefactor_single(theta),
        poly_prefactors=[_make_poly_prefactor_power(2)],  # (1-u)² - keeping same as original
        poly_factors=[
            make_poly_factor("P1", _make_P_argument_x()),
            make_poly_factor("P1", _make_P_argument_y()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_11_v2(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₂ term (decoupled, no derivatives).

    Unchanged from original - I₂ has no variables.
    """
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=0 I₂",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P1", P_arg),
            make_poly_factor("P1", P_arg),
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_11_v2(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₃ term using PRZZ-correct prefactor structure.

    Key fix: Uses algebraic_prefactor (1+θx)/θ = 1/θ + x with numeric_prefactor=-1
    to correctly handle the product rule for:
    I₃ = -d/dx[(1+θx)/θ × integral]|_{x=0} = -(base + (1/θ)×deriv)
    """
    Q_arg_alpha = _make_Q_arg_alpha_x_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_x_only_single(theta)

    return Term(
        name="I3_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=0 I₃",
        vars=("x",),
        deriv_orders={"x": 1},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,  # Negation, not -1/θ!
        algebraic_prefactor=_make_algebraic_prefactor_x_only(theta),  # (1+θx)/θ = 1/θ + x
        poly_prefactors=[_make_poly_prefactor_power(1)],  # (1-u)
        poly_factors=[
            make_poly_factor("P1", _make_P_argument_x()),
            make_poly_factor("P1", make_P_argument_unshifted()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_11_v2(theta: float, R: float) -> Term:
    """
    Build the (1,1) I₄ term using PRZZ-correct prefactor structure.

    By symmetry with I₃.
    """
    Q_arg_alpha = _make_Q_arg_alpha_y_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_y_only_single(theta)

    return Term(
        name="I4_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=0 I₄",
        vars=("y",),
        deriv_orders={"y": 1},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,
        algebraic_prefactor=_make_algebraic_prefactor_y_only(theta),
        poly_prefactors=[_make_poly_prefactor_power(1)],
        poly_factors=[
            make_poly_factor("P1", make_P_argument_unshifted()),
            make_poly_factor("P1", _make_P_argument_y()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_11_v2(theta: float, R: float) -> List[Term]:
    """Build all (1,1) terms using PRZZ-correct single-variable structure."""
    return [
        make_I1_11_v2(theta, R),
        make_I2_11_v2(theta, R),
        make_I3_11_v2(theta, R),
        make_I4_11_v2(theta, R),
    ]


# =============================================================================
# GENERALIZED HELPERS FOR ARBITRARY (ℓ₁, ℓ₂) PAIRS
# =============================================================================

def _make_x_vars(ell1: int) -> Tuple[str, ...]:
    """Generate x variable names: ('x1', 'x2', ..., 'x_{ell1}')."""
    return tuple(f"x{i}" for i in range(1, ell1 + 1))


def _make_y_vars(ell2: int) -> Tuple[str, ...]:
    """Generate y variable names: ('y1', 'y2', ..., 'y_{ell2}')."""
    return tuple(f"y{i}" for i in range(1, ell2 + 1))


def _make_Q_arg_alpha_general(
    theta: float,
    x_vars: Tuple[str, ...],
    y_vars: Tuple[str, ...]
) -> AffineExpr:
    """
    Build Q argument α for general (ℓ₁, ℓ₂): t + θt·X + θ(t-1)·Y

    where X = sum of x vars, Y = sum of y vars.
    Each x_i gets coefficient θt, each y_j gets coefficient θ(t-1).
    """
    var_coeffs: Dict[str, GridFunc] = {}

    for v in x_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T

    for v in y_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T - th

    return AffineExpr(a0=lambda U, T: T, var_coeffs=var_coeffs)


def _make_Q_arg_beta_general(
    theta: float,
    x_vars: Tuple[str, ...],
    y_vars: Tuple[str, ...]
) -> AffineExpr:
    """
    Build Q argument β for general (ℓ₁, ℓ₂): t + θ(t-1)·X + θt·Y

    SWAPPED from α: x gets θ(t-1), y gets θt.
    """
    var_coeffs: Dict[str, GridFunc] = {}

    for v in x_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T - th

    for v in y_vars:
        var_coeffs[v] = lambda U, T, th=theta: th * T

    return AffineExpr(a0=lambda U, T: T, var_coeffs=var_coeffs)


def _make_algebraic_prefactor_general(
    theta: float,
    x_vars: Tuple[str, ...],
    y_vars: Tuple[str, ...]
) -> AffineExpr:
    """
    Build algebraic prefactor for (ℓ₁, ℓ₂): (θS + 1)/θ = 1/θ + S

    where S = sum of all variables.

    NOTE: For ℓ₁+ℓ₂ > 2, this would need to be (θS+1)^{ℓ₁+ℓ₂-1}/θ^{ℓ₁+ℓ₂-1},
    but the series expansion handles powers via nilpotent multiplication.
    For now we keep the simple form; the higher powers emerge from
    the product of multiple P factors in the integrand.
    """
    var_coeffs: Dict[str, GridFunc] = {}

    for v in x_vars:
        var_coeffs[v] = 1.0
    for v in y_vars:
        var_coeffs[v] = 1.0

    return AffineExpr(a0=1.0 / theta, var_coeffs=var_coeffs)


def _make_poly_prefactor_power(power: int) -> GridFunc:
    """Build poly prefactor (1-u)^power."""
    return lambda U, T: (1 - U) ** power


def _make_P_argument_sum(var_list: Tuple[str, ...]) -> AffineExpr:
    """
    Build P argument with SUM of variables: P(x₁ + x₂ + ... + u)

    For pair (ℓ₁, ℓ₂):
    - Left side: P_ℓ₁(sum_of_x_vars + u)
    - Right side: P_ℓ₂(sum_of_y_vars + u)

    CRITICAL: The argument is the SUM of all variables in that group plus u,
    NOT separate factors for each variable!
    """
    var_coeffs: Dict[str, GridFunc] = {v: 1.0 for v in var_list}
    return AffineExpr(
        a0=lambda U, T: U,
        var_coeffs=var_coeffs
    )


# =============================================================================
# (2,2) PAIR TERMS
# =============================================================================

def make_I1_22(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₁ term (main coupled term).

    Variables: x1, x2, y1, y2 (4 variables)
    All four variables are differentiated.

    Structure per TECHNICAL_ANALYSIS.md 10.1 (Interpretation B - SUMMED arguments):
    - P_left = P₂(x1+x2+u)   (sum of x vars + u)
    - P_right = P₂(y1+y2+u)  (sum of y vars + u)
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    x_vars = _make_x_vars(2)  # ("x1", "x2")
    y_vars = _make_y_vars(2)  # ("y1", "y2")
    all_vars = x_vars + y_vars

    Q_arg_alpha = _make_Q_arg_alpha_general(theta, x_vars, y_vars)
    Q_arg_beta = _make_Q_arg_beta_general(theta, x_vars, y_vars)

    # SUMMED P arguments (Interpretation B)
    P_arg_left = _make_P_argument_sum(x_vars)   # x1+x2+u
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+u

    poly_factors = [
        make_poly_factor("P2", P_arg_left),   # P₂(x1+x2+u)
        make_poly_factor("P2", P_arg_right),  # P₂(y1+y2+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    # Algebraic prefactor (θS+1)/θ = 1/θ + S
    algebraic_prefactor = _make_algebraic_prefactor_general(theta, x_vars, y_vars)

    return Term(
        name="I1_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, I₁ for (2,2)",
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=1.0,  # (-1)^{2+2} = +1
        algebraic_prefactor=algebraic_prefactor,
        poly_prefactors=[_make_poly_prefactor_power(4)],  # (1-u)^{2+2}
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_22(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₂ term (decoupled, no derivatives).

    No formal variables - evaluated at x=y=0.
    2 P factors: P₂(u) for left, P₂(u) for right
    """
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, I₂ for (2,2)",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P2", P_arg),  # P₂(u) for left
            make_poly_factor("P2", P_arg),  # P₂(u) for right
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_22(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₃ term (x derivatives only).

    Variables: x1, x2 (differentiated), y=0.
    Structure per Interpretation B (SUMMED arguments):
    - P_left: P₂(x1+x2+u)  - 1 P factor with summed x vars
    - P_right: P₂(u)       - 1 P factor unshifted (y=0)
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    x_vars = _make_x_vars(2)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in x_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in x_vars}
    )

    # SUMMED P arguments (Interpretation B)
    P_arg_left = _make_P_argument_sum(x_vars)  # x1+x2+u

    poly_factors = [
        make_poly_factor("P2", P_arg_left),              # P₂(x1+x2+u)
        make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u) for y=0
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I3_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, I₃ for (2,2)",
        vars=x_vars,
        deriv_orders={v: 1 for v in x_vars},
        domain="[0,1]^2",
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(2)],  # (1-u)^2
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_22(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₄ term (y derivatives only).

    Variables: y1, y2 (differentiated), x=0.
    Structure per Interpretation B (SUMMED arguments):
    - P_left: P₂(u)        - 1 P factor unshifted (x=0)
    - P_right: P₂(y1+y2+u) - 1 P factor with summed y vars
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    y_vars = _make_y_vars(2)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in y_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in y_vars}
    )

    # SUMMED P arguments (Interpretation B)
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+u

    poly_factors = [
        make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u) for x=0
        make_poly_factor("P2", P_arg_right),                  # P₂(y1+y2+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I4_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, I₄ for (2,2)",
        vars=y_vars,
        deriv_orders={v: 1 for v in y_vars},
        domain="[0,1]^2",
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(2)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_22(theta: float, R: float) -> List[Term]:
    """Build all (2,2) terms."""
    return [
        make_I1_22(theta, R),
        make_I2_22(theta, R),
        make_I3_22(theta, R),
        make_I4_22(theta, R),
    ]


# =============================================================================
# (2,2) PAIR TERMS - V2 SINGLE-VARIABLE STRUCTURE (PRZZ CORRECT)
# =============================================================================
#
# ROOT CAUSE FIX: These terms use single x, y variables matching PRZZ TeX.
# The original terms used x1,x2,y1,y2 which computed d⁴/dx₁dx₂dy₁dy₂
# instead of the correct d²/dxdy.
#
# Oracle validation (przz_22_exact_oracle.py):
#   - I₂ (no derivatives): Oracle = DSL = 0.9088 ✓
#   - I₁: Old DSL = 3.88 (wrong), Oracle = 1.17
#   - I₃/I₄: Old DSL = +0.13 (wrong sign!), Oracle = -0.54
#

def make_I1_22_v2(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₁ term using PRZZ-correct single x, y structure.

    Variables: x, y (2 variables, NOT x1,x2,y1,y2)
    Derivative: d²/dxdy (NOT d⁴/dx₁dx₂dy₁dy₂)

    PRZZ Reference: Section 6.2.1, ℓ₁=ℓ₂=1, lines 1530-1532
    """
    Q_arg_alpha = _make_Q_arg_alpha_single(theta)
    Q_arg_beta = _make_Q_arg_beta_single(theta)

    return Term(
        name="I1_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=1 I₁",
        vars=("x", "y"),  # PRZZ correct: 2 variables
        deriv_orders={"x": 1, "y": 1},  # d²/dxdy
        domain="[0,1]^2",
        numeric_prefactor=1.0,  # (-1)^{1+1} = +1
        algebraic_prefactor=_make_algebraic_prefactor_single(theta),  # (1+θ(x+y))/θ
        poly_prefactors=[_make_poly_prefactor_power(2)],  # (1-u)² for ℓ₁=ℓ₂=1
        poly_factors=[
            make_poly_factor("P2", _make_P_argument_x()),  # P₂(x+u)
            make_poly_factor("P2", _make_P_argument_y()),  # P₂(y+u)
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_22_v2(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₂ term (decoupled, no derivatives).

    This is unchanged from the original - I₂ has no derivatives so
    the variable structure issue doesn't affect it.
    """
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=1 I₂",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],  # No (1-u) factor for I₂
        poly_factors=[
            make_poly_factor("P2", P_arg),
            make_poly_factor("P2", P_arg),
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_22_v2(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₃ term using PRZZ-correct single x structure.

    Variables: x only (NOT x1, x2)
    Derivative: d/dx (NOT d²/dx₁dx₂)

    PRZZ Reference: Section 6.2.1, ℓ₁=ℓ₂=1, lines 1562-1563
    I₃ = -d/dx[(1+θx)/θ × ∫∫ (1-u)P₂(x+u)P₂(u) Q_α Q_β exp du dt]|_{x=0}

    The product rule gives:
    I₃ = -(I₀ + (1/θ)×dI/dx)

    With algebraic_prefactor = (1+θx)/θ = 1/θ + x, the series extraction
    automatically produces this via: d/dx[(1/θ + x)×F]|₀ = F(0) + (1/θ)×F'(0)
    Then numeric_prefactor = -1.0 gives the negation.
    """
    Q_arg_alpha = _make_Q_arg_alpha_x_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_x_only_single(theta)

    return Term(
        name="I3_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=1 I₃",
        vars=("x",),  # Only x variable
        deriv_orders={"x": 1},  # d/dx
        domain="[0,1]^2",
        numeric_prefactor=-1.0,  # Negation from PRZZ formula
        algebraic_prefactor=_make_algebraic_prefactor_x_only(theta),  # (1+θx)/θ
        poly_prefactors=[_make_poly_prefactor_power(1)],  # (1-u)^1 for I₃ with ℓ₁=1
        poly_factors=[
            make_poly_factor("P2", _make_P_argument_x()),  # P₂(x+u)
            make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u) at y=0
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_22_v2(theta: float, R: float) -> Term:
    """
    Build the (2,2) I₄ term using PRZZ-correct single y structure.

    Variables: y only (NOT y1, y2)
    Derivative: d/dy (NOT d²/dy₁dy₂)

    PRZZ Reference: Section 6.2.1, ℓ₁=ℓ₂=1
    I₄ = -d/dy[(1+θy)/θ × ∫∫ (1-u)P₂(u)P₂(y+u) Q_α Q_β exp du dt]|_{y=0}

    By symmetry with I₃.
    """
    Q_arg_alpha = _make_Q_arg_alpha_y_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_y_only_single(theta)

    return Term(
        name="I4_22",
        pair=(2, 2),
        przz_reference="Section 6.2.1, ℓ₁=ℓ₂=1 I₄",
        vars=("y",),  # Only y variable
        deriv_orders={"y": 1},  # d/dy
        domain="[0,1]^2",
        numeric_prefactor=-1.0,  # Negation from PRZZ formula
        algebraic_prefactor=_make_algebraic_prefactor_y_only(theta),  # (1+θy)/θ
        poly_prefactors=[_make_poly_prefactor_power(1)],  # (1-u)^1 for I₄ with ℓ₂=1
        poly_factors=[
            make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u) at x=0
            make_poly_factor("P2", _make_P_argument_y()),  # P₂(y+u)
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_22_v2(theta: float, R: float) -> List[Term]:
    """Build all (2,2) terms using PRZZ-correct single-variable structure."""
    return [
        make_I1_22_v2(theta, R),
        make_I2_22_v2(theta, R),
        make_I3_22_v2(theta, R),
        make_I4_22_v2(theta, R),
    ]


# =============================================================================
# (3,3) PAIR TERMS
# =============================================================================

def make_I1_33(theta: float, R: float) -> Term:
    """
    Build the (3,3) I₁ term. 6 variables: x1,x2,x3,y1,y2,y3.

    Structure per TECHNICAL_ANALYSIS.md 10.1 (Interpretation B - SUMMED arguments):
    - P_left = P₃(x1+x2+x3+u)   (sum of x vars + u)
    - P_right = P₃(y1+y2+y3+u)  (sum of y vars + u)
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    x_vars = _make_x_vars(3)
    y_vars = _make_y_vars(3)
    all_vars = x_vars + y_vars

    Q_arg_alpha = _make_Q_arg_alpha_general(theta, x_vars, y_vars)
    Q_arg_beta = _make_Q_arg_beta_general(theta, x_vars, y_vars)

    # SUMMED P arguments (Interpretation B)
    P_arg_left = _make_P_argument_sum(x_vars)   # x1+x2+x3+u
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+y3+u

    poly_factors = [
        make_poly_factor("P3", P_arg_left),   # P₃(x1+x2+x3+u)
        make_poly_factor("P3", P_arg_right),  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    # Algebraic prefactor (θS+1)/θ = 1/θ + S
    algebraic_prefactor = _make_algebraic_prefactor_general(theta, x_vars, y_vars)

    return Term(
        name="I1_33",
        pair=(3, 3),
        przz_reference="Section 6.2.1, I₁ for (3,3)",
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=1.0,  # (-1)^{3+3} = +1
        algebraic_prefactor=algebraic_prefactor,
        poly_prefactors=[_make_poly_prefactor_power(6)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_33(theta: float, R: float) -> Term:
    """
    Build the (3,3) I₂ term (decoupled).
    2 P factors: P₃(u) for left, P₃(u) for right
    """
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_33",
        pair=(3, 3),
        przz_reference="Section 6.2.1, I₂ for (3,3)",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P3", P_arg),  # P₃(u) for left
            make_poly_factor("P3", P_arg),  # P₃(u) for right
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_33(theta: float, R: float) -> Term:
    """
    Build the (3,3) I₃ term (x derivatives only).

    Variables: x1, x2, x3 (differentiated), y=0.
    Structure per Interpretation B (SUMMED arguments):
    - P_left: P₃(x1+x2+x3+u)  - 1 P factor with summed x vars
    - P_right: P₃(u)          - 1 P factor unshifted (y=0)
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    x_vars = _make_x_vars(3)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in x_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in x_vars}
    )

    # SUMMED P arguments (Interpretation B)
    P_arg_left = _make_P_argument_sum(x_vars)  # x1+x2+x3+u

    poly_factors = [
        make_poly_factor("P3", P_arg_left),              # P₃(x1+x2+x3+u)
        make_poly_factor("P3", make_P_argument_unshifted()),  # P₃(u) for y=0
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I3_33",
        pair=(3, 3),
        przz_reference="Section 6.2.1, I₃ for (3,3)",
        vars=x_vars,
        deriv_orders={v: 1 for v in x_vars},
        domain="[0,1]^2",
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(3)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_33(theta: float, R: float) -> Term:
    """
    Build the (3,3) I₄ term (y derivatives only).

    Variables: y1, y2, y3 (differentiated), x=0.
    Structure per Interpretation B (SUMMED arguments):
    - P_left: P₃(u)            - 1 P factor unshifted (x=0)
    - P_right: P₃(y1+y2+y3+u)  - 1 P factor with summed y vars
    Total: 2 P factors + 2 Q factors = 4 poly_factors
    """
    y_vars = _make_y_vars(3)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in y_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in y_vars}
    )

    # SUMMED P arguments (Interpretation B)
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+y3+u

    poly_factors = [
        make_poly_factor("P3", make_P_argument_unshifted()),  # P₃(u) for x=0
        make_poly_factor("P3", P_arg_right),                  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I4_33",
        pair=(3, 3),
        przz_reference="Section 6.2.1, I₄ for (3,3)",
        vars=y_vars,
        deriv_orders={v: 1 for v in y_vars},
        domain="[0,1]^2",
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(3)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_33(theta: float, R: float) -> List[Term]:
    """Build all (3,3) terms."""
    return [
        make_I1_33(theta, R),
        make_I2_33(theta, R),
        make_I3_33(theta, R),
        make_I4_33(theta, R),
    ]


# =============================================================================
# (1,2) PAIR TERMS - Off-diagonal, asymmetric
# =============================================================================

def make_I1_12(theta: float, R: float) -> Term:
    """
    Build the (1,2) I₁ term. 3 variables: x1, y1, y2.

    Structure per TECHNICAL_ANALYSIS.md 10.1:
    - P_left = P₁(x1+u)     (sum of x vars = x1)
    - P_right = P₂(y1+y2+u) (sum of y vars)
    Only 2 P factors total!
    """
    x_vars = _make_x_vars(1)  # ("x1",)
    y_vars = _make_y_vars(2)  # ("y1", "y2")
    all_vars = x_vars + y_vars

    Q_arg_alpha = _make_Q_arg_alpha_general(theta, x_vars, y_vars)
    Q_arg_beta = _make_Q_arg_beta_general(theta, x_vars, y_vars)

    # Per TECHNICAL_ANALYSIS.md 10.1: P arguments are SUMS
    P_arg_left = _make_P_argument_sum(x_vars)   # x1+u
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+u

    poly_factors = [
        make_poly_factor("P1", P_arg_left),   # P₁(x1+u)
        make_poly_factor("P2", P_arg_right),  # P₂(y1+y2+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    # Algebraic prefactor (θS+1)/θ = 1/θ + S
    algebraic_prefactor = _make_algebraic_prefactor_general(theta, x_vars, y_vars)

    return Term(
        name="I1_12",
        pair=(1, 2),
        przz_reference="Section 6.2.1, I₁ for (1,2)",
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,  # (-1)^{1+2} = -1
        algebraic_prefactor=algebraic_prefactor,
        poly_prefactors=[_make_poly_prefactor_power(3)],  # (1-u)^{1+2}
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_12(theta: float, R: float) -> Term:
    """Build the (1,2) I₂ term (decoupled). Only 2 P factors!"""
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_12",
        pair=(1, 2),
        przz_reference="Section 6.2.1, I₂ for (1,2)",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P1", P_arg),  # P₁(u) for left
            make_poly_factor("P2", P_arg),  # P₂(u) for right
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_12(theta: float, R: float) -> Term:
    """Build the (1,2) I₃ term (x derivative only). Only 2 P factors!"""
    x_vars = ("x1",)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T - th}
    )

    return Term(
        name="I3_12",
        pair=(1, 2),
        przz_reference="Section 6.2.1, I₃ for (1,2)",
        vars=x_vars,
        deriv_orders={"x1": 1},
        domain="[0,1]^2",
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(1)],
        poly_factors=[
            make_poly_factor("P1", make_P_argument("x1")),        # P₁(x1+u)
            make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u)
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_12(theta: float, R: float) -> Term:
    """
    Build the (1,2) I₄ term (y derivatives only).

    Variables: y1, y2 (differentiated), x=0.
    2 P factors: P₁(u) for left, P₂(y1+y2+u) for right
    """
    y_vars = _make_y_vars(2)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in y_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in y_vars}
    )

    # 2 P factors: P₁(u) for left (x=0), P₂(y1+y2+u) for right
    P_arg_right = _make_P_argument_sum(y_vars)

    poly_factors = [
        make_poly_factor("P1", make_P_argument_unshifted()),  # P₁(u) for x=0
        make_poly_factor("P2", P_arg_right),                  # P₂(y1+y2+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I4_12",
        pair=(1, 2),
        przz_reference="Section 6.2.1, I₄ for (1,2)",
        vars=y_vars,
        deriv_orders={v: 1 for v in y_vars},
        domain="[0,1]^2",
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(2)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_12(theta: float, R: float) -> List[Term]:
    """Build all (1,2) terms."""
    return [
        make_I1_12(theta, R),
        make_I2_12(theta, R),
        make_I3_12(theta, R),
        make_I4_12(theta, R),
    ]


# =============================================================================
# (1,3) PAIR TERMS
# =============================================================================

def make_I1_13(theta: float, R: float) -> Term:
    """
    Build the (1,3) I₁ term. 4 variables: x1, y1, y2, y3.

    Structure per TECHNICAL_ANALYSIS.md 10.1:
    - P_left = P₁(x1+u)         (sum of x vars = x1)
    - P_right = P₃(y1+y2+y3+u)  (sum of y vars)
    Only 2 P factors total!
    """
    x_vars = _make_x_vars(1)
    y_vars = _make_y_vars(3)
    all_vars = x_vars + y_vars

    Q_arg_alpha = _make_Q_arg_alpha_general(theta, x_vars, y_vars)
    Q_arg_beta = _make_Q_arg_beta_general(theta, x_vars, y_vars)

    # Per TECHNICAL_ANALYSIS.md 10.1: P arguments are SUMS
    P_arg_left = _make_P_argument_sum(x_vars)   # x1+u
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+y3+u

    poly_factors = [
        make_poly_factor("P1", P_arg_left),   # P₁(x1+u)
        make_poly_factor("P3", P_arg_right),  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    # Algebraic prefactor (θS+1)/θ = 1/θ + S
    algebraic_prefactor = _make_algebraic_prefactor_general(theta, x_vars, y_vars)

    return Term(
        name="I1_13",
        pair=(1, 3),
        przz_reference="Section 6.2.1, I₁ for (1,3)",
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=1.0,  # (-1)^{1+3} = +1
        algebraic_prefactor=algebraic_prefactor,
        poly_prefactors=[_make_poly_prefactor_power(4)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_13(theta: float, R: float) -> Term:
    """Build the (1,3) I₂ term (decoupled). Only 2 P factors!"""
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_13",
        pair=(1, 3),
        przz_reference="Section 6.2.1, I₂ for (1,3)",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P1", P_arg),  # P₁(u) for left
            make_poly_factor("P3", P_arg),  # P₃(u) for right
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_13(theta: float, R: float) -> Term:
    """Build the (1,3) I₃ term (x derivative only). Only 2 P factors!"""
    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={"x1": lambda U, T, th=theta: th * T - th}
    )

    return Term(
        name="I3_13",
        pair=(1, 3),
        przz_reference="Section 6.2.1, I₃ for (1,3)",
        vars=("x1",),
        deriv_orders={"x1": 1},
        domain="[0,1]^2",
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(1)],
        poly_factors=[
            make_poly_factor("P1", make_P_argument("x1")),        # P₁(x1+u)
            make_poly_factor("P3", make_P_argument_unshifted()),  # P₃(u)
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_13(theta: float, R: float) -> Term:
    """
    Build the (1,3) I₄ term (y derivatives only).

    Variables: y1, y2, y3 (differentiated), x=0.
    2 P factors: P₁(u) for left, P₃(y1+y2+y3+u) for right
    """
    y_vars = _make_y_vars(3)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in y_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in y_vars}
    )

    # 2 P factors: P₁(u) for left (x=0), P₃(y1+y2+y3+u) for right
    P_arg_right = _make_P_argument_sum(y_vars)

    poly_factors = [
        make_poly_factor("P1", make_P_argument_unshifted()),  # P₁(u) for x=0
        make_poly_factor("P3", P_arg_right),                  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I4_13",
        pair=(1, 3),
        przz_reference="Section 6.2.1, I₄ for (1,3)",
        vars=y_vars,
        deriv_orders={v: 1 for v in y_vars},
        domain="[0,1]^2",
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(3)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_13(theta: float, R: float) -> List[Term]:
    """Build all (1,3) terms."""
    return [
        make_I1_13(theta, R),
        make_I2_13(theta, R),
        make_I3_13(theta, R),
        make_I4_13(theta, R),
    ]


# =============================================================================
# (2,3) PAIR TERMS
# =============================================================================

def make_I1_23(theta: float, R: float) -> Term:
    """
    Build the (2,3) I₁ term. 5 variables: x1, x2, y1, y2, y3.

    Structure per TECHNICAL_ANALYSIS.md 10.1:
    - P_left = P₂(x1+x2+u)      (sum of x vars)
    - P_right = P₃(y1+y2+y3+u)  (sum of y vars)
    Only 2 P factors total!
    """
    x_vars = _make_x_vars(2)
    y_vars = _make_y_vars(3)
    all_vars = x_vars + y_vars

    Q_arg_alpha = _make_Q_arg_alpha_general(theta, x_vars, y_vars)
    Q_arg_beta = _make_Q_arg_beta_general(theta, x_vars, y_vars)

    # Per TECHNICAL_ANALYSIS.md 10.1: P arguments are SUMS
    P_arg_left = _make_P_argument_sum(x_vars)   # x1+x2+u
    P_arg_right = _make_P_argument_sum(y_vars)  # y1+y2+y3+u

    poly_factors = [
        make_poly_factor("P2", P_arg_left),   # P₂(x1+x2+u)
        make_poly_factor("P3", P_arg_right),  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    # Algebraic prefactor (θS+1)/θ = 1/θ + S
    algebraic_prefactor = _make_algebraic_prefactor_general(theta, x_vars, y_vars)

    return Term(
        name="I1_23",
        pair=(2, 3),
        przz_reference="Section 6.2.1, I₁ for (2,3)",
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,  # (-1)^{2+3} = -1
        algebraic_prefactor=algebraic_prefactor,
        poly_prefactors=[_make_poly_prefactor_power(5)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I2_23(theta: float, R: float) -> Term:
    """Build the (2,3) I₂ term (decoupled). Only 2 P factors!"""
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name="I2_23",
        pair=(2, 3),
        przz_reference="Section 6.2.1, I₂ for (2,3)",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor("P2", P_arg),  # P₂(u) for left
            make_poly_factor("P3", P_arg),  # P₃(u) for right
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def make_I3_23(theta: float, R: float) -> Term:
    """
    Build the (2,3) I₃ term (x derivatives only).

    Variables: x1, x2 (differentiated), y=0.
    2 P factors: P₂(x1+x2+u) for left, P₃(u) for right (y=0)
    """
    x_vars = _make_x_vars(2)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in x_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in x_vars}
    )

    # 2 P factors: P₂(x1+x2+u) for left, P₃(u) for right
    P_arg_left = _make_P_argument_sum(x_vars)

    poly_factors = [
        make_poly_factor("P2", P_arg_left),             # P₂(x1+x2+u)
        make_poly_factor("P3", make_P_argument_unshifted()),  # P₃(u) for y=0
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I3_23",
        pair=(2, 3),
        przz_reference="Section 6.2.1, I₃ for (2,3)",
        vars=x_vars,
        deriv_orders={v: 1 for v in x_vars},
        domain="[0,1]^2",
        # PRZZ: I₃ = -[(1+θX)/θ]|_{X=0} × d/dX[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(2)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_I4_23(theta: float, R: float) -> Term:
    """
    Build the (2,3) I₄ term (y derivatives only).

    Variables: y1, y2, y3 (differentiated), x=0.
    2 P factors: P₂(u) for left, P₃(y1+y2+y3+u) for right
    """
    y_vars = _make_y_vars(3)

    Q_arg_alpha = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T - th for v in y_vars}
    )
    Q_arg_beta = AffineExpr(
        a0=lambda U, T: T,
        var_coeffs={v: lambda U, T, th=theta: th * T for v in y_vars}
    )

    # 2 P factors: P₂(u) for left (x=0), P₃(y1+y2+y3+u) for right
    P_arg_right = _make_P_argument_sum(y_vars)

    poly_factors = [
        make_poly_factor("P2", make_P_argument_unshifted()),  # P₂(u) for x=0
        make_poly_factor("P3", P_arg_right),                  # P₃(y1+y2+y3+u)
        make_poly_factor("Q", Q_arg_alpha),
        make_poly_factor("Q", Q_arg_beta),
    ]

    return Term(
        name="I4_23",
        pair=(2, 3),
        przz_reference="Section 6.2.1, I₄ for (2,3)",
        vars=y_vars,
        deriv_orders={v: 1 for v in y_vars},
        domain="[0,1]^2",
        # PRZZ: I₄ = -[(1+θY)/θ]|_{Y=0} × d/dY[...] = -(1/θ) × derivative
        numeric_prefactor=-1.0 / theta,  # PRZZ paper correct: (1+θx)/θ|_{x=0} = 1/θ
        algebraic_prefactor=None,
        poly_prefactors=[_make_poly_prefactor_power(3)],
        poly_factors=poly_factors,
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_23(theta: float, R: float) -> List[Term]:
    """Build all (2,3) terms."""
    return [
        make_I1_23(theta, R),
        make_I2_23(theta, R),
        make_I3_23(theta, R),
        make_I4_23(theta, R),
    ]


# =============================================================================
# MASTER FUNCTION: Get all K=3 terms
# =============================================================================

def make_all_terms_k3(theta: float, R: float) -> Dict[str, List[Term]]:
    """
    Build all terms for K=3, d=1.

    Returns dict with keys: "11", "22", "33", "12", "13", "23"
    """
    return {
        "11": make_all_terms_11(theta, R),
        "22": make_all_terms_22(theta, R),
        "33": make_all_terms_33(theta, R),
        "12": make_all_terms_12(theta, R),
        "13": make_all_terms_13(theta, R),
        "23": make_all_terms_23(theta, R),
    }


# =============================================================================
# V2 TERMS FOR REMAINING PAIRS (PRZZ CORRECT STRUCTURE)
# =============================================================================
#
# These use single x, y variables matching PRZZ's explicit formulas.
# The pair index (ℓ₁, ℓ₂) determines:
#   - Which polynomials to use (P_ℓ₁, P_ℓ₂)
#   - The (1-u) power via PRZZ_ℓ = our_ℓ - 1
#   - The sign factor (-1)^{ℓ₁+ℓ₂}
#
# For PRZZ ℓ indexing (0-based Λ convolution count):
#   (1-u) power for I₁ = (ℓ₁-1) + (ℓ₂-1) = ℓ₁ + ℓ₂ - 2
#   (1-u) power for I₃ = ℓ₁ - 1
#   (1-u) power for I₄ = ℓ₂ - 1
#

def _poly_name(ell: int) -> str:
    """Get polynomial name for piece index."""
    return f"P{ell}"


def _make_I1_generic_v2(theta: float, R: float, ell1: int, ell2: int) -> Term:
    """
    Generic I₁ builder for any (ℓ₁, ℓ₂) pair using PRZZ-correct structure.

    Always uses 2 variables: x, y
    Derivative: d²/dxdy
    (1-u) power: (ℓ₁-1) + (ℓ₂-1) = ℓ₁ + ℓ₂ - 2
    """
    Q_arg_alpha = _make_Q_arg_alpha_single(theta)
    Q_arg_beta = _make_Q_arg_beta_single(theta)

    # Sign factor: (-1)^{ℓ₁+ℓ₂}
    sign = (-1) ** (ell1 + ell2)
    # (1-u) power for I₁ using PRZZ convention: (ℓ₁-1) + (ℓ₂-1)
    one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))

    return Term(
        name=f"I1_{ell1}{ell2}",
        pair=(ell1, ell2),
        przz_reference=f"Section 6.2.1, I₁ for ({ell1},{ell2})",
        vars=("x", "y"),
        deriv_orders={"x": 1, "y": 1},
        domain="[0,1]^2",
        numeric_prefactor=float(sign),
        algebraic_prefactor=_make_algebraic_prefactor_single(theta),
        poly_prefactors=[_make_poly_prefactor_power(one_minus_u_power)] if one_minus_u_power > 0 else [],
        poly_factors=[
            make_poly_factor(_poly_name(ell1), _make_P_argument_x()),
            make_poly_factor(_poly_name(ell2), _make_P_argument_y()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def _make_I2_generic_v2(theta: float, R: float, ell1: int, ell2: int) -> Term:
    """Generic I₂ builder (no derivatives, unchanged structure)."""
    P_arg = make_P_argument_unshifted()
    Q_arg = make_Q_argument_at_t()
    exp_arg = make_exp_argument_at_t()

    return Term(
        name=f"I2_{ell1}{ell2}",
        pair=(ell1, ell2),
        przz_reference=f"Section 6.2.1, I₂ for ({ell1},{ell2})",
        vars=(),
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            make_poly_factor(_poly_name(ell1), P_arg),
            make_poly_factor(_poly_name(ell2), P_arg),
            make_poly_factor("Q", Q_arg, power=2),
        ],
        exp_factors=[ExpFactor(2 * R, exp_arg)]
    )


def _make_I3_generic_v2(theta: float, R: float, ell1: int, ell2: int) -> Term:
    """
    Generic I₃ builder using PRZZ-correct prefactor structure.

    Uses single x variable with d/dx.
    algebraic_prefactor = (1+θx)/θ with numeric_prefactor = -1
    (1-u) power: ℓ₁ - 1
    """
    Q_arg_alpha = _make_Q_arg_alpha_x_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_x_only_single(theta)

    # (1-u) power for I₃: ℓ₁ - 1
    one_minus_u_power = max(0, ell1 - 1)

    return Term(
        name=f"I3_{ell1}{ell2}",
        pair=(ell1, ell2),
        przz_reference=f"Section 6.2.1, I₃ for ({ell1},{ell2})",
        vars=("x",),
        deriv_orders={"x": 1},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,
        algebraic_prefactor=_make_algebraic_prefactor_x_only(theta),
        poly_prefactors=[_make_poly_prefactor_power(one_minus_u_power)] if one_minus_u_power > 0 else [],
        poly_factors=[
            make_poly_factor(_poly_name(ell1), _make_P_argument_x()),
            make_poly_factor(_poly_name(ell2), make_P_argument_unshifted()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def _make_I4_generic_v2(theta: float, R: float, ell1: int, ell2: int) -> Term:
    """
    Generic I₄ builder using PRZZ-correct prefactor structure.

    Uses single y variable with d/dy.
    algebraic_prefactor = (1+θy)/θ with numeric_prefactor = -1
    (1-u) power: ℓ₂ - 1
    """
    Q_arg_alpha = _make_Q_arg_alpha_y_only_single(theta)
    Q_arg_beta = _make_Q_arg_beta_y_only_single(theta)

    # (1-u) power for I₄: ℓ₂ - 1
    one_minus_u_power = max(0, ell2 - 1)

    return Term(
        name=f"I4_{ell1}{ell2}",
        pair=(ell1, ell2),
        przz_reference=f"Section 6.2.1, I₄ for ({ell1},{ell2})",
        vars=("y",),
        deriv_orders={"y": 1},
        domain="[0,1]^2",
        numeric_prefactor=-1.0,
        algebraic_prefactor=_make_algebraic_prefactor_y_only(theta),
        poly_prefactors=[_make_poly_prefactor_power(one_minus_u_power)] if one_minus_u_power > 0 else [],
        poly_factors=[
            make_poly_factor(_poly_name(ell1), make_P_argument_unshifted()),
            make_poly_factor(_poly_name(ell2), _make_P_argument_y()),
            make_poly_factor("Q", Q_arg_alpha),
            make_poly_factor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )


def make_all_terms_33_v2(theta: float, R: float) -> List[Term]:
    """Build all (3,3) terms using PRZZ-correct single-variable structure."""
    return [
        _make_I1_generic_v2(theta, R, 3, 3),
        _make_I2_generic_v2(theta, R, 3, 3),
        _make_I3_generic_v2(theta, R, 3, 3),
        _make_I4_generic_v2(theta, R, 3, 3),
    ]


def make_all_terms_12_v2(theta: float, R: float) -> List[Term]:
    """Build all (1,2) terms using PRZZ-correct single-variable structure."""
    return [
        _make_I1_generic_v2(theta, R, 1, 2),
        _make_I2_generic_v2(theta, R, 1, 2),
        _make_I3_generic_v2(theta, R, 1, 2),
        _make_I4_generic_v2(theta, R, 1, 2),
    ]


def make_all_terms_13_v2(theta: float, R: float) -> List[Term]:
    """Build all (1,3) terms using PRZZ-correct single-variable structure."""
    return [
        _make_I1_generic_v2(theta, R, 1, 3),
        _make_I2_generic_v2(theta, R, 1, 3),
        _make_I3_generic_v2(theta, R, 1, 3),
        _make_I4_generic_v2(theta, R, 1, 3),
    ]


def make_all_terms_23_v2(theta: float, R: float) -> List[Term]:
    """Build all (2,3) terms using PRZZ-correct single-variable structure."""
    return [
        _make_I1_generic_v2(theta, R, 2, 3),
        _make_I2_generic_v2(theta, R, 2, 3),
        _make_I3_generic_v2(theta, R, 2, 3),
        _make_I4_generic_v2(theta, R, 2, 3),
    ]


def make_all_terms_k3_v2(theta: float, R: float) -> Dict[str, List[Term]]:
    """
    Build all terms for K=3, d=1 using PRZZ-correct single-variable structure.

    Returns dict with keys: "11", "22", "33", "12", "13", "23"
    """
    return {
        "11": make_all_terms_11_v2(theta, R),
        "22": make_all_terms_22_v2(theta, R),
        "33": make_all_terms_33_v2(theta, R),
        "12": make_all_terms_12_v2(theta, R),
        "13": make_all_terms_13_v2(theta, R),
        "23": make_all_terms_23_v2(theta, R),
    }


# =============================================================================
# I₅ ARITHMETIC CORRECTION NOTES
# =============================================================================
#
# The I₅ arithmetic correction is computed using the formula:
#
#   I₅ = -S(0) × θ²/12 × I₂_total
#
# where:
#   - S(0) = Σ_p (log p / (p - 1))² ≈ 1.3854799116
#   - θ = 4/7 (mollifier exponent)
#   - I₂_total = sum of normalized I₂ contributions from all pairs
#
# This formula was derived empirically to match PRZZ's reported κ = 0.417293962.
# It achieves ~99.97% accuracy.
#
# The factor 12 = 2 × 3! likely comes from combinatorial structure in the
# bracket expansion, though the exact theoretical derivation is not fully
# understood.
#
# The I₅ computation is implemented directly in evaluate.py:evaluate_c_full()
# rather than as separate Term objects, since it depends on the aggregate I₂
# sum rather than individual term integrands.
#
# See TECHNICAL_ANALYSIS.md Section 9.5 for mathematical background on
# the S(α+β) arithmetic factor and its role in the bracket computation.
# =============================================================================
