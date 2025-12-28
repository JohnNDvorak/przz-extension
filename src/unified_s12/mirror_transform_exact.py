"""
src/unified_s12/mirror_transform_exact.py
Phase 45: Exact Mirror Operator Implementation

This module computes S12_mirror_exact using the TRUE TeX operator semantics,
not the scalar m approximation.

TEX MIRROR SEMANTICS (from PRZZ lines 1502-1511):
================================================

The identity has form:
    I(alpha, beta) + T^{-(alpha+beta)} * I(-beta, -alpha)

THREE COMPONENTS REQUIRED:
=========================

1. T^{-(alpha+beta)} FACTOR
   At evaluation point alpha = beta = -R/L, this gives exp(2R).
   This is a PREFACTOR multiplying the entire mirror integrand.

2. (-beta, -alpha) SUBSTITUTION
   This is NOT just "evaluate at -R". It changes how derivatives act:
   - D_alpha acting on F(-beta,-alpha) -> -D_{second arg} of F
   - D_beta acting on F(-beta,-alpha) -> -D_{first arg} of F
   
   In eigenvalue terms, this means:
   - The eigenvalue for Q(D_alpha) in mirror = -(eigenvalue for Q(D_beta) in direct)
   - The eigenvalue for Q(D_beta) in mirror = -(eigenvalue for Q(D_alpha) in direct)
   
   With sign flips from the chain rule!

3. EIGENVALUE SUBSTITUTION
   After mapping the operators, apply eigenvalue substitution.
   
   DIRECT:     A_alpha(t) = t + theta*(t-1)*x + theta*t*y
               A_beta(t)  = t + theta*t*x + theta*(t-1)*y
   
   MIRROR:     A'_alpha = -A_beta  (negated!)
               A'_beta  = -A_alpha (negated!)

KEY INSIGHT FROM PHASE 41-43:
============================
The scalar m approximation fails because kappa and kappa* need OPPOSITE corrections.
The exact mirror gives polynomial-dependent m_eff values, resolving this.

DIAGNOSTIC ONLY:
================
m_eff := S12_mirror_exact / S12(-R)

This is computed AFTER the fact, not used as input.

Created: 2025-12-27 (Phase 45)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.series_bivariate import BivariateSeries, build_exp_bracket, build_log_factor
from src.unified_i1_paper import build_P_factor_paper, omega_for_ell, _extract_poly_coeffs
from src.unified_i2_paper import compute_I2_unified_paper


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class S12ExactResult:
    """Result of exact S12 computation."""
    
    S12_direct: float       # I1 + I2 at +R (direct term)
    S12_mirror_exact: float # Mirror term with exact operator semantics
    S12_total_exact: float  # S12_direct + S12_mirror_exact
    
    # Diagnostic (NOT an input)
    S12_minus_R: float      # S12 at -R (for m_eff computation)
    m_eff: float            # S12_mirror_exact / S12_minus_R
    
    # For comparison
    m_scalar: float         # exp(R) + (2K-1) from baseline formula
    m_corrected: float      # From I1-fraction correction (Phase 44)
    
    # Parameters
    R: float
    theta: float
    K: int


@dataclass
class CExactResult:
    """Result of exact c computation."""
    
    c_exact: float          # S12_direct + S12_mirror_exact + S34
    c_scalar: float         # Using scalar m (for comparison)
    
    kappa_exact: float      # 1 - log(c_exact)/R
    kappa_scalar: float     # 1 - log(c_scalar)/R
    
    # Components
    S12_direct: float
    S12_mirror_exact: float
    S34: float
    
    # Diagnostic
    m_eff: float
    m_scalar: float
    
    # Parameters
    R: float
    theta: float


# =============================================================================
# Q FACTOR WITH MIRROR EIGENVALUES
# =============================================================================

def build_Q_factor_direct(
    Q_coeffs: List[float],
    t: float,
    theta: float,
    max_dx: int,
    max_dy: int,
    which: str = "alpha",
) -> BivariateSeries:
    """
    Build Q factor for DIRECT term with standard eigenvalues.
    
    DIRECT EIGENVALUES:
    - Q_alpha: A_alpha(t) = t + theta*(t-1)*x + theta*t*y
    - Q_beta:  A_beta(t)  = t + theta*t*x + theta*(t-1)*y
    """
    from src.series_bivariate import build_Q_factor
    
    if which == "alpha":
        return build_Q_factor(
            Q_coeffs,
            a0=t,
            ax=theta * (t - 1),
            ay=theta * t,
            max_dx=max_dx,
            max_dy=max_dy,
        )
    else:  # beta
        return build_Q_factor(
            Q_coeffs,
            a0=t,
            ax=theta * t,
            ay=theta * (t - 1),
            max_dx=max_dx,
            max_dy=max_dy,
        )


def build_Q_factor_mirror(
    Q_coeffs: List[float],
    t: float,
    theta: float,
    max_dx: int,
    max_dy: int,
    which: str = "alpha",
) -> BivariateSeries:
    """
    Build Q factor for MIRROR term with transformed eigenvalues.
    
    MIRROR TRANSFORMATION:
    =====================
    The substitution (-beta, -alpha) means:
    - D_alpha in mirror acts like -D_beta in direct
    - D_beta in mirror acts like -D_alpha in direct
    
    So the eigenvalue for Q(D_alpha) in mirror is:
        -A_beta(t) = -(t + theta*t*x + theta*(t-1)*y)
                   = -t - theta*t*x - theta*(t-1)*y
    
    And for Q(D_beta) in mirror:
        -A_alpha(t) = -(t + theta*(t-1)*x + theta*t*y)
                    = -t - theta*(t-1)*x - theta*t*y
    
    The SIGN FLIP is crucial - this is what the scalar m misses!
    """
    from src.series_bivariate import build_Q_factor
    
    if which == "alpha":
        # Mirror: A'_alpha = -A_beta(direct)
        return build_Q_factor(
            Q_coeffs,
            a0=-t,                    # Sign flip!
            ax=-theta * t,            # Sign flip + swap from beta
            ay=-theta * (t - 1),      # Sign flip + swap from beta
            max_dx=max_dx,
            max_dy=max_dy,
        )
    else:  # beta
        # Mirror: A'_beta = -A_alpha(direct)
        return build_Q_factor(
            Q_coeffs,
            a0=-t,                    # Sign flip!
            ax=-theta * (t - 1),      # Sign flip + swap from alpha
            ay=-theta * t,            # Sign flip + swap from alpha
            max_dx=max_dx,
            max_dy=max_dy,
        )


# =============================================================================
# I1 COMPUTATION (MIRROR VERSION)
# =============================================================================

def compute_I1_mirror(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 40,
    include_Q: bool = True,
    apply_factorial_norm: bool = True,
) -> float:
    """
    Compute I1 for the MIRROR term with exact operator semantics.
    
    This implements:
    1. T^{-(alpha+beta)} = exp(2R) prefactor
    2. (-beta, -alpha) substitution -> eigenvalue sign flip and swap
    3. Same integrand structure but with mirror eigenvalues
    """
    max_dx = ell1
    max_dy = ell2
    
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)
    
    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)
    
    # Get polynomial objects
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")
    
    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")
    
    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None and include_Q else None
    
    # PRZZ (1-u) power
    one_minus_u_power = ell1 + ell2
    
    # T^{-(alpha+beta)} = exp(2R) prefactor for mirror
    T_prefactor = math.exp(2 * R)
    
    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power
        
        for t, t_w in zip(t_nodes, t_weights):
            # 1. Exp factor (same structure, the eigenvalue change is in Q)
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)
            
            # 2. Log factor: 1/theta + x + y (same)
            log_factor = build_log_factor(theta, max_dx, max_dy)
            
            # 3. P factors (same as direct)
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a
            )
            
            # Build bracket
            bracket = exp_factor * log_factor * P_x * P_y
            
            # 4. Q factors with MIRROR EIGENVALUES
            if include_Q and Q_coeffs is not None:
                Q_alpha_mirror = build_Q_factor_mirror(
                    Q_coeffs, t, theta, max_dx, max_dy, which="alpha"
                )
                Q_beta_mirror = build_Q_factor_mirror(
                    Q_coeffs, t, theta, max_dx, max_dy, which="beta"
                )
                bracket = bracket * Q_alpha_mirror * Q_beta_mirror
            
            # 5. Extract x^ell1 y^ell2 coefficient
            coeff = bracket.extract(ell1, ell2)
            
            # 6. Add to integral
            total += coeff * one_minus_u_factor * u_w * t_w
    
    # 7. Apply factorial normalization
    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)
    
    # 8. Sign convention for off-diagonal
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign
    
    # 9. Apply T^{-(alpha+beta)} prefactor
    total *= T_prefactor
    
    return total


# =============================================================================
# I2 MIRROR (simpler - no derivatives)
# =============================================================================

def compute_I2_mirror(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 40,
    include_Q: bool = True,
) -> float:
    """
    Compute I2 for the MIRROR term.
    
    I2 has no log factor derivatives, so it's simpler.
    But it still needs the T^{-(alpha+beta)} prefactor and mirror eigenvalues.
    """
    max_dx = ell1
    max_dy = ell2
    
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)
    
    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)
    
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")
    
    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None and include_Q else None
    
    one_minus_u_power = ell1 + ell2
    T_prefactor = math.exp(2 * R)
    
    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power
        
        for t, t_w in zip(t_nodes, t_weights):
            # Exp factor
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)
            
            # P factors
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a
            )
            
            # For I2, there's no log factor, so we need the 1/theta term only
            # (the "constant" part of L = 1/theta + x + y)
            inv_theta = 1.0 / theta
            
            bracket = exp_factor * P_x * P_y
            # Scale by 1/theta (constant part of L)
            bracket = inv_theta * bracket
            
            # Q factors with MIRROR eigenvalues
            if include_Q and Q_coeffs is not None:
                Q_alpha_mirror = build_Q_factor_mirror(
                    Q_coeffs, t, theta, max_dx, max_dy, which="alpha"
                )
                Q_beta_mirror = build_Q_factor_mirror(
                    Q_coeffs, t, theta, max_dx, max_dy, which="beta"
                )
                bracket = bracket * Q_alpha_mirror * Q_beta_mirror
            
            # Extract x^ell1 y^ell2
            coeff = bracket.extract(ell1, ell2)
            
            total += coeff * one_minus_u_factor * u_w * t_w
    
    # Factorial normalization (I2 doesn't have derivative ell1!ell2!)
    # Apply T prefactor
    total *= T_prefactor
    
    return total


# =============================================================================
# S12 COMPUTATION
# =============================================================================

def compute_s12_direct(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """
    Compute S12_direct = sum over all pairs of (I1 + I2) at +R.
    
    This is the standard direct term.
    """
    from src.unified_i1_paper import compute_I1_unified_paper
    from src.unified_i2_paper import compute_I2_unified_paper
    
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }
    
    total = 0.0
    pairs = ["11", "22", "33", "12", "13", "23"]
    
    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        
        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm
        
        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True, apply_factorial_norm=True,
        )
        
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )
        
        pair_contrib = (I1_result.I1_value + I2_result.I2_value) * norm * sym
        total += pair_contrib
    
    return total


def compute_s12_mirror_exact(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """
    Compute S12_mirror with EXACT operator semantics.
    
    This uses mirror eigenvalues and includes T^{-(alpha+beta)} = exp(2R) prefactor.
    
    This is the key Phase 45 contribution: polynomial-dependent mirror!
    """
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }
    
    total = 0.0
    pairs = ["11", "22", "33", "12", "13", "23"]
    
    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        
        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm
        
        # Use mirror versions with exact eigenvalues
        I1_mirror = compute_I1_mirror(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True, apply_factorial_norm=True,
        )
        
        I2_mirror = compute_I2_mirror(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )
        
        pair_contrib = (I1_mirror + I2_mirror) * norm * sym
        total += pair_contrib
    
    return total


def compute_s12_total_exact(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """
    Compute S12_total = S12_direct + S12_mirror_exact.
    
    This is the full S12 contribution without scalar m approximation.
    """
    S12_direct = compute_s12_direct(theta, R, K, polynomials, n_quad)
    S12_mirror = compute_s12_mirror_exact(theta, R, K, polynomials, n_quad)
    return S12_direct + S12_mirror


def compute_m_eff(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """
    Compute m_eff = S12_mirror_exact / S12(-R).
    
    This is a DIAGNOSTIC, not an input.
    It shows what scalar m would have been needed to match the exact result.
    """
    S12_mirror = compute_s12_mirror_exact(theta, R, K, polynomials, n_quad)
    S12_minus_R = compute_s12_direct(theta, -R, K, polynomials, n_quad)
    
    if abs(S12_minus_R) < 1e-15:
        return float('inf')
    
    return S12_mirror / S12_minus_R


# =============================================================================
# FULL C COMPUTATION
# =============================================================================

def compute_S34(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute S34 = I3 + I4 (no mirror for these)."""
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term
    
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    
    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }
    
    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm
        
        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += full_norm * result.value
    
    return S34


def compute_c_exact(
    polynomials: Dict,
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> CExactResult:
    """
    Compute c with EXACT mirror operator semantics.
    
    c_exact = S12_direct + S12_mirror_exact + S34
    
    This is the Phase 45 production evaluator.
    No empirical parameters, everything derived from TeX semantics.
    
    Args:
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        R: The R parameter
        theta: θ parameter (default: 4/7)
        K: Number of mollifier pieces (default: 3)
        n_quad: Quadrature points (default: 60)
    
    Returns:
        CExactResult with c_exact and diagnostics
    """
    # Compute S12 components
    S12_direct = compute_s12_direct(theta, R, K, polynomials, n_quad)
    S12_mirror_exact = compute_s12_mirror_exact(theta, R, K, polynomials, n_quad)
    S34 = compute_S34(theta, R, polynomials, n_quad)
    
    # Exact c
    c_exact = S12_direct + S12_mirror_exact + S34
    kappa_exact = 1 - math.log(c_exact) / R if c_exact > 0 else float('-inf')
    
    # Scalar m for comparison
    S12_minus_R = compute_s12_direct(theta, -R, K, polynomials, n_quad)
    m_scalar = math.exp(R) + (2 * K - 1)
    m_eff = S12_mirror_exact / S12_minus_R if abs(S12_minus_R) > 1e-15 else float('inf')
    
    c_scalar = S12_direct + m_scalar * S12_minus_R + S34
    kappa_scalar = 1 - math.log(c_scalar) / R if c_scalar > 0 else float('-inf')
    
    return CExactResult(
        c_exact=c_exact,
        c_scalar=c_scalar,
        kappa_exact=kappa_exact,
        kappa_scalar=kappa_scalar,
        S12_direct=S12_direct,
        S12_mirror_exact=S12_mirror_exact,
        S34=S34,
        m_eff=m_eff,
        m_scalar=m_scalar,
        R=R,
        theta=theta,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_exact_mirror(verbose: bool = True) -> Tuple[bool, str]:
    """
    Validate the exact mirror evaluator against κ and κ* benchmarks.
    
    Returns:
        (passed, message)
    """
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
    
    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112
    
    theta = 4 / 7
    K = 3
    n_quad = 60
    
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    
    # Evaluate
    result_kappa = compute_c_exact(polys_kappa, 1.3036, theta=theta, K=K, n_quad=n_quad)
    result_kappa_star = compute_c_exact(polys_kappa_star, 1.1167, theta=theta, K=K, n_quad=n_quad)
    
    gap_kappa = (result_kappa.c_exact / c_target_kappa - 1) * 100
    gap_kappa_star = (result_kappa_star.c_exact / c_target_kappa_star - 1) * 100
    
    gap_scalar_kappa = (result_kappa.c_scalar / c_target_kappa - 1) * 100
    gap_scalar_kappa_star = (result_kappa_star.c_scalar / c_target_kappa_star - 1) * 100
    
    if verbose:
        print("Phase 45: Exact Mirror Operator Validation")
        print("=" * 70)
        print(f"κ (R=1.3036):")
        print(f"  c_target = {c_target_kappa:.6f}")
        print(f"  c_exact  = {result_kappa.c_exact:.6f} (gap = {gap_kappa:+.4f}%)")
        print(f"  c_scalar = {result_kappa.c_scalar:.6f} (gap = {gap_scalar_kappa:+.4f}%)")
        print(f"  m_eff    = {result_kappa.m_eff:.6f}")
        print(f"  m_scalar = {result_kappa.m_scalar:.6f}")
        print()
        print(f"κ* (R=1.1167):")
        print(f"  c_target = {c_target_kappa_star:.6f}")
        print(f"  c_exact  = {result_kappa_star.c_exact:.6f} (gap = {gap_kappa_star:+.4f}%)")
        print(f"  c_scalar = {result_kappa_star.c_scalar:.6f} (gap = {gap_scalar_kappa_star:+.4f}%)")
        print(f"  m_eff    = {result_kappa_star.m_eff:.6f}")
        print(f"  m_scalar = {result_kappa_star.m_scalar:.6f}")
    
    # Phase 45 target: exact should be better than scalar
    improved_kappa = abs(gap_kappa) < abs(gap_scalar_kappa)
    improved_kappa_star = abs(gap_kappa_star) < abs(gap_scalar_kappa_star)
    
    # Ultimate target: <0.01%
    passed = abs(gap_kappa) < 0.01 and abs(gap_kappa_star) < 0.01
    
    if passed:
        msg = f"PASS: Both benchmarks within 0.01% (κ: {gap_kappa:+.4f}%, κ*: {gap_kappa_star:+.4f}%)"
    else:
        # Check improvement
        if improved_kappa and improved_kappa_star:
            msg = f"IMPROVED: Exact better than scalar (κ: {gap_kappa:+.4f}% vs {gap_scalar_kappa:+.4f}%, κ*: {gap_kappa_star:+.4f}% vs {gap_scalar_kappa_star:+.4f}%)"
        else:
            msg = f"NEEDS WORK: κ gap={gap_kappa:+.4f}%, κ* gap={gap_kappa_star:+.4f}%"
    
    return passed, msg
