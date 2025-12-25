"""
src/ratios/j1_k3_decomposition.py
Phase 14B Task 4: Five-Piece J₁ Decomposition with REAL Bracket Terms

PAPER ANCHORS:
=============
J₁ = J_{1,1} + J_{1,2} + J_{1,3} + J_{1,4} + J_{1,5}

The five bracket terms are explicitly defined in PRZZ:

1. J₁₁ (bracket₁): The (1⋆Λ₂)(n) Dirichlet series term
   bracket₁(s,u) = A(0,0;β,α) × Σ_{n≤N} (1⋆Λ₂)(n)/n^{1+s+u}
   where (1⋆Λ₂)(n) = Σ_{d|n} Λ₂(d) with Λ₂ from recurrence

2. J₁₂ (bracket₂): The double log-derivative term
   bracket₂(s,u) = A(0,0;β,α) × Σ_{n≤N} 1/n^{1+s+u} × (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

3. J₁₃ (bracket₃): The (1⋆Λ)(n) with β-side log-derivative
   bracket₃(s,u) = A(0,0;β,α) × Σ_{n≤N} log(n)/n^{1+s+u} × (ζ'/ζ)(1+β+u)
   [using the identity (1⋆Λ₁)(n) = log(n)]

4. J₁₄ (bracket₄): Symmetric α-side version
   bracket₄(s,u) = A(0,0;β,α) × Σ_{n≤N} log(n)/n^{1+s+u} × (ζ'/ζ)(1+α+s)

5. J₁₅ (bracket₅): The A^{(1,1)} prime sum term
   bracket₅(s,u) = A^{(1,1)}(0,0;β,α) × Σ_{n≤N} 1/n^{1+s+u}
   where A^{(1,1)}(0) ≈ 1.3856 (verified prime sum)

KEY SIMPLIFICATIONS:
==================
- A_{α,β}(0,0;β,α) = 1 EXACTLY (Euler product cancellation)
- (1⋆Λ₁)(n) = log(n) EXACTLY (use directly, don't compute sum)
- Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n), NOT Λ(n)²
"""

from __future__ import annotations
from typing import NamedTuple
import math

from src.ratios.arithmetic_factor import A11_prime_sum
from src.ratios.dirichlet_primitives import (
    one_star_lambda1,
    one_star_lambda2,
    A00_at_diagonal,
)
from src.ratios.zeta_logderiv import (
    zeta_log_deriv,
    EULER_MASCHERONI,
)


class J1Pieces(NamedTuple):
    """
    The five pieces of J₁ decomposition for K=3.

    Each piece represents a distinct bracket term from the paper.

    Attributes:
        j11: bracket₁ - uses (1⋆Λ₂)(n) Dirichlet series
        j12: bracket₂ - double (ζ'/ζ) product
        j13: bracket₃ - log(n) with β-side (ζ'/ζ)
        j14: bracket₄ - log(n) with α-side (ζ'/ζ)
        j15: bracket₅ - A^{(1,1)} prime sum term
    """
    j11: complex
    j12: complex
    j13: complex
    j14: complex
    j15: complex


def _dirichlet_sum_1_over_n(
    s: complex,
    u: complex,
    n_cutoff: int = 100
) -> complex:
    """
    Compute Σ_{n=1}^{n_cutoff} 1/n^{1+s+u}.

    This is the basic Dirichlet series Σ 1/n^w with w = 1+s+u.

    Args:
        s, u: Contour variables
        n_cutoff: Truncation for the sum

    Returns:
        The truncated Dirichlet series
    """
    w = 1.0 + s + u
    total = complex(0.0)
    for n in range(1, n_cutoff + 1):
        total += 1.0 / (n ** w)
    return total


def _dirichlet_sum_one_star_lambda2(
    s: complex,
    u: complex,
    n_cutoff: int = 100
) -> complex:
    """
    Compute Σ_{n=1}^{n_cutoff} (1⋆Λ₂)(n) / n^{1+s+u}.

    Uses the CORRECT Λ₂ from recurrence: Λ₂(n) = Λ(n)·log(n) + (Λ⋆Λ)(n).

    Args:
        s, u: Contour variables
        n_cutoff: Truncation for the sum

    Returns:
        The Dirichlet series with (1⋆Λ₂) coefficients
    """
    w = 1.0 + s + u
    total = complex(0.0)
    for n in range(1, n_cutoff + 1):
        coeff = one_star_lambda2(n)
        total += coeff / (n ** w)
    return total


def _dirichlet_sum_log_n(
    s: complex,
    u: complex,
    n_cutoff: int = 100
) -> complex:
    """
    Compute Σ_{n=1}^{n_cutoff} log(n) / n^{1+s+u}.

    Uses the identity (1⋆Λ₁)(n) = log(n).

    Args:
        s, u: Contour variables
        n_cutoff: Truncation for the sum

    Returns:
        The Dirichlet series with log(n) coefficients
    """
    w = 1.0 + s + u
    total = complex(0.0)
    for n in range(1, n_cutoff + 1):
        # Using (1⋆Λ₁)(n) = log(n) directly
        coeff = one_star_lambda1(n)  # = log(n)
        total += coeff / (n ** w)
    return total


def bracket_j11(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    Compute bracket₁(s,u) = A(0,0;β,α) × Σ_{n≤N} (1⋆Λ₂)(n)/n^{1+s+u}.

    Since A(0,0;β,α) = 1 exactly, this is just the (1⋆Λ₂) Dirichlet series.

    This bracket term uses the CORRECT Λ₂ from the paper's recurrence.

    Args:
        alpha, beta: Shift parameters (used for A₀₀, which is 1)
        s, u: Contour variables
        n_cutoff: Truncation for Dirichlet sum

    Returns:
        The bracket₁ contribution
    """
    # A_{α,β}(0,0;β,α) = 1 exactly
    A00 = A00_at_diagonal(alpha, beta)

    # Σ (1⋆Λ₂)(n) / n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_one_star_lambda2(s, u, n_cutoff)

    return A00 * dirichlet_sum


def bracket_j12(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    Compute bracket₂(s,u) = A(0,0;β,α) × Σ 1/n^{1+s+u} × (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u).

    This is the double log-derivative term: product of two ζ'/ζ evaluations.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation for Dirichlet sum

    Returns:
        The bracket₂ contribution
    """
    # A_{α,β}(0,0;β,α) = 1 exactly
    A00 = A00_at_diagonal(alpha, beta)

    # Σ 1/n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_1_over_n(s, u, n_cutoff)

    # (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
    zeta1 = zeta_log_deriv(1.0 + alpha + s)
    zeta2 = zeta_log_deriv(1.0 + beta + u)

    return A00 * dirichlet_sum * zeta1 * zeta2


def bracket_j13(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    Compute bracket₃(s,u) = A(0,0;β,α) × Σ log(n)/n^{1+s+u} × (ζ'/ζ)(1+β+u).

    This uses (1⋆Λ₁)(n) = log(n) with the β-side log-derivative.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation for Dirichlet sum

    Returns:
        The bracket₃ contribution
    """
    # A_{α,β}(0,0;β,α) = 1 exactly
    A00 = A00_at_diagonal(alpha, beta)

    # Σ log(n)/n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_log_n(s, u, n_cutoff)

    # (ζ'/ζ)(1+β+u)
    zeta_beta = zeta_log_deriv(1.0 + beta + u)

    return A00 * dirichlet_sum * zeta_beta


def bracket_j14(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    Compute bracket₄(s,u) = A(0,0;β,α) × Σ log(n)/n^{1+s+u} × (ζ'/ζ)(1+α+s).

    This is the symmetric α-side version of bracket₃.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation for Dirichlet sum

    Returns:
        The bracket₄ contribution
    """
    # A_{α,β}(0,0;β,α) = 1 exactly
    A00 = A00_at_diagonal(alpha, beta)

    # Σ log(n)/n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_log_n(s, u, n_cutoff)

    # (ζ'/ζ)(1+α+s)
    zeta_alpha = zeta_log_deriv(1.0 + alpha + s)

    return A00 * dirichlet_sum * zeta_alpha


def bracket_j15(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100,
    prime_cutoff: int = 5000
) -> complex:
    """
    Compute bracket₅(s,u) = A^{(1,1)}(0,0;β,α) × Σ 1/n^{1+s+u}.

    This uses the verified A^{(1,1)} prime sum (≈ 1.3856 at the diagonal).

    IMPORTANT: A^{(1,1)} is evaluated at the DIAGONAL point (s=0), not at α+β.
    The paper's anchor value 1.3856 comes from A^{(1,1)}(0), which is what we use.

    For the general formula with shifts, the A^{(1,1)} factor depends on
    the evaluation scheme, but at main-term level, we use the diagonal value.

    Args:
        alpha, beta: Shift parameters (for reference, not used in A^{(1,1)} eval)
        s, u: Contour variables
        n_cutoff: Truncation for Dirichlet sum
        prime_cutoff: Cutoff for A^{(1,1)} prime sum

    Returns:
        The bracket₅ contribution
    """
    # A^{(1,1)} evaluated at the diagonal point s=0
    # This gives the paper's anchor value ≈ 1.3856
    A11_value = A11_prime_sum(0.0, prime_cutoff=prime_cutoff)

    # Σ 1/n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_1_over_n(s, u, n_cutoff)

    return A11_value * dirichlet_sum


def build_J1_pieces_K3(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100,
    include_A11: bool = True
) -> J1Pieces:
    """
    Build the five pieces of J₁ for K=3 regime using REAL bracket formulas.

    This replaces the placeholder formulas with the actual paper expressions.

    Args:
        alpha: First shift parameter
        beta: Second shift parameter
        s: First contour variable
        u: Second contour variable (can be integration variable)
        n_cutoff: Truncation for Dirichlet sums
        include_A11: If True, include the A^{(1,1)} contribution in j15

    Returns:
        J1Pieces with five contributions from real bracket terms
    """
    # Bracket 1: (1⋆Λ₂)(n) Dirichlet series
    j11 = bracket_j11(alpha, beta, s, u, n_cutoff=n_cutoff)

    # Bracket 2: double (ζ'/ζ) product
    j12 = bracket_j12(alpha, beta, s, u, n_cutoff=n_cutoff)

    # Bracket 3: log(n) with β-side (ζ'/ζ)
    j13 = bracket_j13(alpha, beta, s, u, n_cutoff=n_cutoff)

    # Bracket 4: log(n) with α-side (ζ'/ζ) (symmetric)
    j14 = bracket_j14(alpha, beta, s, u, n_cutoff=n_cutoff)

    # Bracket 5: A^{(1,1)} prime sum term
    if include_A11:
        j15 = bracket_j15(alpha, beta, s, u, n_cutoff=n_cutoff)
    else:
        j15 = complex(0.0)

    return J1Pieces(j11=j11, j12=j12, j13=j13, j14=j14, j15=j15)


def sum_J1(pieces: J1Pieces) -> complex:
    """
    Sum all five pieces of J₁.

    Args:
        pieces: J1Pieces tuple

    Returns:
        Total J₁ value
    """
    return sum(pieces)


def count_active_pieces(pieces: J1Pieces, threshold: float = 1e-14) -> int:
    """
    Count how many pieces have magnitude above threshold.

    This verifies that all five pieces contribute at generic parameter values.

    Args:
        pieces: J1Pieces tuple
        threshold: Minimum magnitude to count as active

    Returns:
        Number of active pieces (0-5)
    """
    return sum(1 for piece in pieces if abs(piece) > threshold)


# ============================================================================
# Phase 14C: Main-Term Variants using Laurent Reductions
# ============================================================================

# Import Laurent series machinery
from src.ratios.zeta_laurent import (
    logderiv_product_series,
    j13_main_term_sign,
    j14_main_term_sign,
    EULER_MASCHERONI,
)


def bracket_j12_main(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    J12 MAIN TERM using paper's contour-lemma reduction.

    Paper derivation (Phase 14C):
    ============================
    The literal formula:
        J12 = A00 × Σ(1/n^{1+s+u}) × (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

    gets reduced via Laurent expansion. When extracting residues
    at s=u=0, the ζ'/ζ poles combine with the coefficient extraction
    to collapse the main term.

    At PRZZ point α=β=-R, the (ζ'/ζ)² product evaluates to:
        (1/R + γ)² ≈ constant

    The paper's main-term reduction shows this collapses to:
        J12_main = (1/(α+β)) × Σ(1/n) × [coeff extraction]

    NOT the literal product of (ζ'/ζ) values.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation

    Returns:
        J12 main-term contribution
    """
    # Get Laurent series coefficients
    s_coeffs, u_coeffs = logderiv_product_series(alpha, beta, order=3)

    # The [s^0 u^0] coefficient captures the main-term constant
    c00 = s_coeffs[0] * u_coeffs[0]

    # The main-term reduction: factor of 1/(α+β) appears
    # from the difference quotient in the contour lemma
    if abs(alpha + beta) < 1e-14:
        # At diagonal α+β=0, this needs limiting behavior
        # Use the integral representation from PRZZ lines 1502-1511
        divisor = 1.0  # Placeholder for limit
    else:
        divisor = alpha + beta

    # Dirichlet sum Σ 1/n^{1+s+u}
    dirichlet_sum = _dirichlet_sum_1_over_n(s, u, n_cutoff)

    # Main-term: c00 × (1/divisor) × sum
    # Note: The literal formula multiplies by ζ'/ζ values directly.
    # The main-term formula uses the Laurent coefficient c00 instead.
    return c00 * dirichlet_sum / divisor


def bracket_j13_main(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    J13 MAIN TERM with negative sign from Laurent reduction.

    Paper derivation (Phase 14C):
    ============================
    The literal formula:
        J13 = A00 × Σ log(n)/n^{1+s+u} × (ζ'/ζ)(1+β+u)

    The main-term reduction gives a LEADING MINUS SIGN.
    This comes from:
    1. The (ζ'/ζ) pole at u=0 has coefficient -1
    2. The residue extraction flips the sign
    3. PRZZ I₃ prefactor is -1/θ (lines 1551-1564)

    The main term becomes:
        J13_main = -1 × [integration weight (1-u)] × [log(n) sum]

    NOT: positive times (ζ'/ζ) product.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation

    Returns:
        J13 main-term contribution (NEGATIVE)
    """
    # Get the sign from Laurent reduction
    sign = j13_main_term_sign()  # Returns -1

    # The β-side (ζ'/ζ) evaluated at s=0 for main term
    # At α+β=-2R, the value is (-1/β + γ) = (1/R + γ)
    beta_logderiv = -1.0 / beta + EULER_MASCHERONI if abs(beta) > 1e-10 else 0.0

    # Dirichlet sum with log(n) weights
    dirichlet_sum = _dirichlet_sum_log_n(s, u, n_cutoff)

    # Main-term: sign × (ζ'/ζ)(1+β) × sum
    # The key difference from literal formula is the SIGN.
    return sign * beta_logderiv * dirichlet_sum


def bracket_j14_main(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100
) -> complex:
    """
    J14 MAIN TERM (symmetric α-side of J13).

    Same negative sign convention as J13.

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation

    Returns:
        J14 main-term contribution (NEGATIVE)
    """
    # Get the sign from Laurent reduction
    sign = j14_main_term_sign()  # Returns -1

    # The α-side (ζ'/ζ) evaluated at u=0 for main term
    alpha_logderiv = -1.0 / alpha + EULER_MASCHERONI if abs(alpha) > 1e-10 else 0.0

    # Dirichlet sum with log(n) weights
    dirichlet_sum = _dirichlet_sum_log_n(s, u, n_cutoff)

    # Main-term: sign × (ζ'/ζ)(1+α) × sum
    return sign * alpha_logderiv * dirichlet_sum


def build_J1_pieces_K3_main_terms(
    alpha: complex,
    beta: complex,
    s: complex,
    u: complex,
    *,
    n_cutoff: int = 100,
    include_A11: bool = True
) -> J1Pieces:
    """
    Build J1 pieces using MAIN-TERM reductions (Phase 14C).

    This is what PRZZ actually uses for computing κ.
    The key differences from literal evaluation:

    1. J12: Uses Laurent coefficient instead of raw (ζ'/ζ)² product
    2. J13, J14: Have leading MINUS signs from residue calculus
    3. J11: Same as literal (no ζ'/ζ factors)
    4. J15: Same as literal (A^{(1,1)} prime sum)

    Args:
        alpha, beta: Shift parameters
        s, u: Contour variables
        n_cutoff: Truncation
        include_A11: Include J15 term

    Returns:
        J1Pieces with main-term reductions applied
    """
    # J11: No ζ'/ζ factors, same as literal
    j11 = bracket_j11(alpha, beta, s, u, n_cutoff=n_cutoff)

    # J12: Main-term reduction (Laurent coefficient, not raw product)
    j12 = bracket_j12_main(alpha, beta, s, u, n_cutoff=n_cutoff)

    # J13: Main-term with NEGATIVE sign
    j13 = bracket_j13_main(alpha, beta, s, u, n_cutoff=n_cutoff)

    # J14: Main-term with NEGATIVE sign
    j14 = bracket_j14_main(alpha, beta, s, u, n_cutoff=n_cutoff)

    # J15: Same as literal (A^{(1,1)} is independent of ζ'/ζ structure)
    if include_A11:
        j15 = bracket_j15(alpha, beta, s, u, n_cutoff=n_cutoff)
    else:
        j15 = complex(0.0)

    return J1Pieces(j11=j11, j12=j12, j13=j13, j14=j14, j15=j15)


# ============================================================================
# Analysis helpers
# ============================================================================


def analyze_piece_contributions(
    alpha: complex = 0.0,
    beta: complex = 0.0,
    s: complex = 0.0,
    u_samples: int = 5,
    *,
    n_cutoff: int = 50
) -> dict:
    """
    Analyze how each bracket term contributes across u values.

    Args:
        alpha, beta, s: Parameter values
        u_samples: Number of u points to sample
        n_cutoff: Truncation for Dirichlet sums

    Returns:
        Dictionary with piece-by-piece analysis
    """
    import numpy as np

    u_values = np.linspace(0.1, 0.9, u_samples)

    results = {
        'u_values': u_values.tolist(),
        'j11': [], 'j12': [], 'j13': [], 'j14': [], 'j15': [],
        'totals': []
    }

    for u_val in u_values:
        pieces = build_J1_pieces_K3(
            alpha, beta, s, complex(u_val),
            n_cutoff=n_cutoff
        )
        results['j11'].append(float(pieces.j11.real))
        results['j12'].append(float(pieces.j12.real))
        results['j13'].append(float(pieces.j13.real))
        results['j14'].append(float(pieces.j14.real))
        results['j15'].append(float(pieces.j15.real))
        results['totals'].append(float(sum_J1(pieces).real))

    return results


def get_piece_contributions_at_R(
    R: float,
    theta: float = 4.0 / 7.0,
    *,
    n_cutoff: int = 50
) -> dict:
    """
    Get piece contributions at PRZZ R value.

    Args:
        R: The PRZZ R parameter (e.g., 1.3036)
        theta: The θ parameter (default 4/7)
        n_cutoff: Truncation for Dirichlet sums

    Returns:
        Dictionary with contributions and their relationship to exp(R)+5
    """
    import numpy as np

    # At PRZZ point: α = β = -R/L
    alpha = -R
    beta = -R
    s_val = 0.1  # Small positive s for evaluation
    u_val = 0.1  # Small positive u for evaluation

    pieces = build_J1_pieces_K3(
        alpha, beta, complex(s_val), complex(u_val),
        n_cutoff=n_cutoff
    )

    return {
        'R': R,
        'alpha': float(alpha),
        'beta': float(beta),
        's': s_val,
        'u': u_val,
        'exp_R': float(np.exp(R)),
        'exp_R_plus_5': float(np.exp(R) + 5),
        'pieces': {
            'j11': float(pieces.j11.real),
            'j12': float(pieces.j12.real),
            'j13': float(pieces.j13.real),
            'j14': float(pieces.j14.real),
            'j15': float(pieces.j15.real),
        },
        'total': float(sum_J1(pieces).real),
        'num_pieces': 5,
        'note': 'Using real bracket formulas from paper'
    }
