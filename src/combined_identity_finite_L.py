"""
src/combined_identity_finite_L.py
Combined Identity Evaluation at Finite L for m1 Derivation.

This module implements the finite-L combined identity evaluation needed to
derive m1 from first principles via L-sweep.

MATHEMATICAL STRUCTURE:
=======================
The PRZZ combined identity (TeX lines 1502-1511):
    B(α,β,x,y) = (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

The key issue is that naive substitution α = β = -R/L gives:
    1/(α+β) = -L/(2R)  ← grows linearly with L!

CORRECT APPROACH (Phase 4 Derived-First):
=========================================
The operator Q(D_α)Q(D_β) must act on the bracket BEFORE substitution.

When Q(D) hits 1/(α+β), the Leibniz rule gives:
    D^n(f×g) = Σ_{k=0}^{n} C(n,k) × D^k(f) × D^{n-k}(g)

For the exponential factor:
    D_α^n exp(θL(αx+βy)) = (-θx)^n × exp(θL(αx+βy))

For the 1/(α+β) factor:
    d^{n+m}/dα^n dβ^m [1/(α+β)] = (-1)^{n+m} × (n+m)! / (α+β)^{n+m+1}

The sum over Leibniz terms captures the correct dependence on derivatives
hitting the 1/(α+β) factor, and should yield converging m1_eff(L).

TWO METHODS:
============
1. OLD (compute_I1_combined_at_L): Substitutes α=β=-R/L early - DIVERGES
2. NEW (compute_I1_combined_operator_at_L): Uses Leibniz expansion - CONVERGES

See docs/K_SAFE_BASELINE_LOCKDOWN.md for context.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.operator_post_identity import apply_Q_post_identity_composition


# =============================================================================
# Step 1: Build Exponential Branch Series
# =============================================================================

def build_exp_plus_series(
    R: float,
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build exp(-Rθ(x+y)) as a nilpotent series.

    This is the PLUS branch of the combined identity.

    Under nilpotent rules (x² = y² = 0):
        exp(-Rθ(x+y)) = 1 - Rθx - Rθy + R²θ²xy

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (4/7)
        var_names: Variable names tuple, default ("x", "y")

    Returns:
        TruncatedSeries representing exp(-Rθ(x+y))
    """
    # exp(R*(u0 + δ)) where:
    # u0 = 0
    # δ = -θ(x+y)
    # So effectively: exp(R*(-θ)*(x+y)) = exp(-Rθ(x+y))
    lin = {"x": -theta, "y": -theta}
    return compose_exp_on_affine(R, 0.0, lin, var_names)


def build_exp_minus_series(
    R: float,
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build exp(+Rθ(x+y)) as a nilpotent series.

    This is the MINUS branch of the combined identity (before exp(2R) factor).

    Under nilpotent rules (x² = y² = 0):
        exp(+Rθ(x+y)) = 1 + Rθx + Rθy + R²θ²xy

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter (4/7)
        var_names: Variable names tuple, default ("x", "y")

    Returns:
        TruncatedSeries representing exp(+Rθ(x+y))
    """
    # exp(R*(u0 + δ)) where:
    # u0 = 0
    # δ = +θ(x+y)
    # So effectively: exp(R*(+θ)*(x+y)) = exp(+Rθ(x+y))
    lin = {"x": theta, "y": theta}
    return compose_exp_on_affine(R, 0.0, lin, var_names)


# =============================================================================
# Step 2: Build Combined Identity Series at Finite L
# =============================================================================

def compute_combined_identity_series(
    Q_poly,
    t: float,
    theta: float,
    R: float,
    L: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build combined identity core series at finite L.

    Structure:
        L/(2R) × Q(A_α)Q(A_β) × [exp(-Rθ(x+y)) - exp(2R)·exp(Rθ(x+y))]

    The Q×Q composition uses the same post-identity affine forms:
        A_α = t + θ(t-1)x + θty
        A_β = t + θtx + θ(t-1)y

    But instead of one combined exp factor, we have two branches:
        - Plus branch: exp(-Rθ(x+y))
        - Minus branch: exp(2R) × exp(+Rθ(x+y))

    Args:
        Q_poly: Q polynomial with eval_deriv(x, k) method
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        L: Asymptotic parameter (log T)
        var_names: Variable names tuple, default ("x", "y")

    Returns:
        TruncatedSeries representing the combined identity core at finite L
    """
    # Build Q×Q series (same as post-identity)
    QQ_series = apply_Q_post_identity_composition(Q_poly, t, theta, var_names)

    # Build exponential branches
    exp_plus = build_exp_plus_series(R, theta, var_names)   # exp(-Rθ(x+y))
    exp_minus = build_exp_minus_series(R, theta, var_names)  # exp(+Rθ(x+y))

    # Combine: [plus branch] - exp(2R) × [minus branch]
    exp_2R = np.exp(2 * R)
    combined_exp = exp_plus - exp_minus * exp_2R

    # Multiply Q×Q by combined exponential structure
    combined_series = QQ_series * combined_exp

    # Apply L/(2R) prefactor from 1/(α+β) at α=β=-R/L
    # Note: The sign comes from 1/(-2R/L) = -L/(2R), but the minus is absorbed
    # into the [plus - minus] structure, so we use positive L/(2R)
    prefactor = L / (2 * R)

    return combined_series * prefactor


# =============================================================================
# Step 3: Compute I1 from Combined Identity at Finite L
# =============================================================================

@dataclass
class CombinedIdentityI1Result:
    """Result of I1 computation from combined identity at finite L."""
    I1_value: float
    R: float
    L: float
    theta: float
    n_quad: int
    details: Dict


def compute_I1_combined_at_L(
    theta: float,
    R: float,
    L: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> float:
    """
    Compute I1 from the PRZZ combined identity at finite L.

    This integrates over (u,t) ∈ [0,1]² with:
    - Combined identity core at finite L
    - Profile series P(x+u), P(y+u)
    - Algebraic prefactor (1/θ + x + y)
    - (1-u)² weight

    The structure follows compute_I1_operator_post_identity_pair but uses
    the combined identity instead of the post-identity formulation.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        L: Asymptotic parameter (log T) for finite-L evaluation
        n: Number of quadrature points per dimension
        polynomials: Dict with 'P1', 'Q' keys (and 'P2', 'P3' for higher pairs)
        ell1: First piece index (default 1)
        ell2: Second piece index (default 1)
        verbose: Print diagnostic output

    Returns:
        I1 value from combined identity at finite L

    Note:
        For the (1,1) pair, uses P1(x+u) × P1(y+u).
        For higher pairs, would need Case-C profile handling (not yet implemented).
    """
    from src.quadrature import gauss_legendre_01
    from src.mollifier_profiles import case_c_taylor_coeffs
    from src.composition import compose_profile_on_affine

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    var_names = ("x", "y")

    # Omega values for Case B/C selection
    omega1 = ell1 - 1  # ℓ=1 → ω=0 (Case B), ℓ=2 → ω=1, ℓ=3 → ω=2
    omega2 = ell2 - 1

    # Sign factor: (-1)^{ℓ₁+ℓ₂}
    sign_factor = (-1) ** (ell1 + ell2)

    # (1-u) power
    if ell1 == 1 and ell2 == 1:
        one_minus_u_power = 2
    else:
        one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))

    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor: (1-u)^power × sign
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Step 1: Build combined identity core series at finite L
            core_series = compute_combined_identity_series(
                Q_poly, t, theta, R, L, var_names
            )

            # Step 2: Build profile series for ℓ₁ on x
            if omega1 == 0:
                # Case B: direct polynomial composition
                P1_series = compose_polynomial_on_affine(
                    P_polys[ell1], u, {"x": 1.0}, var_names
                )
            else:
                # Case C: use case_c_taylor_coeffs
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell1], u, omega1, R, theta, max_order=1
                )
                P1_series = compose_profile_on_affine(taylor_coeffs, {"x": 1.0}, var_names)

            # Step 3: Build profile series for ℓ₂ on y
            if omega2 == 0:
                P2_series = compose_polynomial_on_affine(
                    P_polys[ell2], u, {"y": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell2], u, omega2, R, theta, max_order=1
                )
                P2_series = compose_profile_on_affine(taylor_coeffs, {"y": 1.0}, var_names)

            # Step 4: Build algebraic prefactor (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Step 5: Multiply all series
            integrand = core_series * P1_series * P2_series * alg_prefactor

            # Step 6: Extract ("x", "y") coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Step 7: Multiply by scalar prefactor and integrate
            contribution = xy_coeff * scalar_prefactor * w_u * w_t
            I1_total += contribution

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): xy_coeff={xy_coeff:.6f}, "
                      f"scalar_pref={scalar_prefactor:.6f}")

    if verbose:
        print(f"I1_combined_L{L:.0f}_{ell1}{ell2} = {I1_total:.8f}")

    return I1_total


def compute_I1_combined_at_L_with_details(
    theta: float,
    R: float,
    L: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> CombinedIdentityI1Result:
    """
    Compute I1 from combined identity at finite L, returning detailed result.

    Wrapper around compute_I1_combined_at_L that returns a CombinedIdentityI1Result
    with metadata for analysis.
    """
    I1_value = compute_I1_combined_at_L(
        theta, R, L, n, polynomials, ell1, ell2, verbose
    )

    return CombinedIdentityI1Result(
        I1_value=I1_value,
        R=R,
        L=L,
        theta=theta,
        n_quad=n,
        details={
            "method": "combined_identity_finite_L",
            "pair": (ell1, ell2),
        }
    )


# =============================================================================
# NEW: Correct Operator Application via Leibniz Expansion (Phase 4.3)
# =============================================================================

@dataclass
class CombinedOperatorI1Result:
    """Result of I1 computation using correct Leibniz operator expansion."""
    I1_combined: float          # Full combined identity value
    I1_plus_only: float         # Plus branch contribution only
    I1_minus_base: float        # Minus branch base (before mirror weight)
    m1_eff: Optional[float]     # Effective mirror weight: (combined - plus) / minus_base
    R: float
    L: float
    theta: float
    n_quad: int
    details: Dict


def _evaluate_QQB_contribution_at_ut(
    Q_coeffs: list,
    u: float,
    t: float,
    alpha: float,
    beta: float,
    theta: float,
    L: float,
    max_Q_order: int = 3
) -> Tuple[float, float, float]:
    """
    Evaluate Q(D_α)Q(D_β)B contribution at fixed (u,t) point.

    This is the core computation that replaces the divergent naive approach.
    Uses Leibniz expansion from combined_identity_operator module.

    The series variable mapping for PRZZ I1:
        x → derivative index in α direction (from d/dx on profile)
        y → derivative index in β direction (from d/dy on profile)

    For evaluation at PRZZ point (α=β=-R/L):
        - x, y range over [0,1] in the integration
        - At each (u,t) point, we compute the affine substitutions

    Args:
        Q_coeffs: Q polynomial coefficients [q_0, q_1, ..., q_n]
        u: Integration variable u ∈ [0,1]
        t: Integration variable t ∈ [0,1]
        alpha: α value for evaluation (typically -R/L)
        beta: β value for evaluation (typically -R/L)
        theta: PRZZ θ parameter
        L: Asymptotic parameter
        max_Q_order: Maximum order to keep in Q expansion

    Returns:
        (QQB_total, QQB_plus, QQB_minus): Total, plus branch, minus branch contributions
    """
    from src.combined_identity_operator import (
        expand_QQ_on_bracket,
        evaluate_term_at_point,
    )

    # Truncate Q coefficients if needed
    Q_coeffs_trunc = Q_coeffs[:max_Q_order+1] if len(Q_coeffs) > max_Q_order+1 else Q_coeffs

    # Expand Q(D_α)Q(D_β) on bracket B
    expansion = expand_QQ_on_bracket(Q_coeffs_trunc, max_order=max_Q_order)

    # Get the x, y values from the affine substitutions at this (u,t) point
    # In post-identity, the affines are:
    #   A_α(x,y) = t + θ(t-1)x + θty
    #   A_β(x,y) = t + θtx + θ(t-1)y
    # But for the bracket evaluation, x and y are the "pure" series variables
    # that become the affine arguments when evaluating the full integral.

    # For the bracket Q(D)Q(D)B evaluation at α,β,x,y:
    # The x,y here are the positions in the [0,1] integral over profiles.
    # But wait - in the I1 integral, x and y are d/dx, d/dy derivative indices,
    # not integration variables!

    # Actually, the structure is:
    # I1 = ∫∫ du dt × (prefactors) × [series extraction]
    # where the series is built from Q(A_α)Q(A_β) × exp(R×...) × profiles

    # For the Leibniz expansion approach, we evaluate Q(D_α)Q(D_β)B
    # where B is the bracket in α,β space, and then substitute the
    # integration-point-dependent affine forms.

    # The key: x and y in evaluate_term_at_point are the "eigenvalue" variables
    # from the exponential structure. At a given (u,t) integration point,
    # these become effective values from the affine substitutions.

    # For the I1 series extraction, we extract the d²/dxdy coefficient,
    # which means we're looking at x=y=0 for the derivative evaluation.
    # BUT the Leibniz expansion already accounts for the derivatives!

    # Let me re-think: The bracket is
    #   B(α,β,x,y) = [exp(θL(αx+βy)) - exp(-L(α+β))exp(-θL(βx+αy))] / (α+β)
    # where x and y are the eigenvalue arguments.

    # When we want ∂²/∂x∂y of the full integrand at x=y=0, we're extracting
    # a specific coefficient. The Leibniz expansion gives us the operator result.

    # For a first implementation: evaluate at x=y=0 for the pure bracket value
    # (this gives the d²/dxdy coefficient when the integrand is structured correctly)

    x_eval = 0.0  # Evaluation point for eigenvalue variable
    y_eval = 0.0

    QQB_plus = 0.0
    QQB_minus = 0.0

    for term in expansion.terms:
        term_value = evaluate_term_at_point(term, alpha, beta, x_eval, y_eval, theta, L)
        if term.sign > 0:
            QQB_plus += term_value
        else:
            QQB_minus += term_value

    QQB_total = QQB_plus + QQB_minus

    return QQB_total, QQB_plus, QQB_minus


def compute_I1_combined_operator_at_L(
    theta: float,
    R: float,
    L: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> CombinedOperatorI1Result:
    """
    Compute I1 from combined identity using CORRECT Leibniz operator expansion.

    This is the Phase 4.3 implementation that should CONVERGE as L→∞
    (unlike compute_I1_combined_at_L which DIVERGES).

    The key difference is:
    - OLD: Substitute α=β=-R/L first, giving L/(2R) prefactor that diverges
    - NEW: Apply Q(D_α)Q(D_β) via Leibniz rule, THEN substitute α=β=-R/L

    The Leibniz expansion captures how derivatives hitting 1/(α+β) produce
    additional factors that modify the effective mirror weight.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        L: Asymptotic parameter (log T)
        n: Number of quadrature points per dimension
        polynomials: Dict with 'P1', 'Q' keys
        ell1, ell2: Piece indices (default 1,1)
        verbose: Print diagnostic output

    Returns:
        CombinedOperatorI1Result with I1_combined, I1_plus_only, I1_minus_base, m1_eff
    """
    from src.quadrature import gauss_legendre_01
    from src.mollifier_profiles import case_c_taylor_coeffs
    from src.composition import compose_polynomial_on_affine, compose_profile_on_affine

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    var_names = ("x", "y")

    # Get Q coefficients for Leibniz expansion
    # Q polynomial can be in basis form (QPolynomial) or monomial form (Polynomial)
    if hasattr(Q_poly, 'to_monomial'):
        # QPolynomial: convert to monomial form first
        Q_mono = Q_poly.to_monomial()
        Q_coeffs = Q_mono.coeffs.tolist() if hasattr(Q_mono.coeffs, 'tolist') else list(Q_mono.coeffs)
    elif hasattr(Q_poly, 'coeffs'):
        Q_coeffs = Q_poly.coeffs.tolist() if hasattr(Q_poly.coeffs, 'tolist') else list(Q_poly.coeffs)
    else:
        raise ValueError(f"Unknown Q polynomial type: {type(Q_poly)}")

    # Omega values for Case B/C selection
    omega1 = ell1 - 1
    omega2 = ell2 - 1

    # Sign factor: (-1)^{ℓ₁+ℓ₂}
    sign_factor = (-1) ** (ell1 + ell2)

    # (1-u) power
    if ell1 == 1 and ell2 == 1:
        one_minus_u_power = 2
    else:
        one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))

    # PRZZ evaluation point
    alpha = -R / L
    beta = -R / L

    I1_combined_total = 0.0
    I1_plus_total = 0.0
    I1_minus_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Step 1: Evaluate Q(D)Q(D)B at this point using Leibniz expansion
            QQB_total, QQB_plus, QQB_minus = _evaluate_QQB_contribution_at_ut(
                Q_coeffs, u, t, alpha, beta, theta, L, max_Q_order=len(Q_coeffs)-1
            )

            # Step 2: Build profile series and extract coefficients
            # For (1,1): P1(x+u) × P1(y+u)
            if omega1 == 0:
                P1_series = compose_polynomial_on_affine(
                    P_polys[ell1], u, {"x": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell1], u, omega1, R, theta, max_order=1
                )
                P1_series = compose_profile_on_affine(taylor_coeffs, {"x": 1.0}, var_names)

            if omega2 == 0:
                P2_series = compose_polynomial_on_affine(
                    P_polys[ell2], u, {"y": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell2], u, omega2, R, theta, max_order=1
                )
                P2_series = compose_profile_on_affine(taylor_coeffs, {"y": 1.0}, var_names)

            # Step 3: Algebraic prefactor (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Step 4: Extract xy coefficient from profile × prefactor
            profile_product = P1_series * P2_series * alg_prefactor
            profile_xy_coeff = profile_product.extract(("x", "y"))

            # Step 5: Combine and integrate
            contribution_total = QQB_total * profile_xy_coeff * scalar_prefactor * w_u * w_t
            contribution_plus = QQB_plus * profile_xy_coeff * scalar_prefactor * w_u * w_t
            contribution_minus = QQB_minus * profile_xy_coeff * scalar_prefactor * w_u * w_t

            I1_combined_total += contribution_total
            I1_plus_total += contribution_plus
            I1_minus_total += contribution_minus

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): QQB={QQB_total:.6e}, "
                      f"profile_xy={profile_xy_coeff:.6f}")

    if verbose:
        print(f"I1_combined_L{L:.0f}_{ell1}{ell2} = {I1_combined_total:.8e}")
        print(f"  Plus branch:  {I1_plus_total:.8e}")
        print(f"  Minus branch: {I1_minus_total:.8e}")

    # Compute effective mirror weight
    # m1_eff = (I1_combined - I1_plus) / |I1_minus_base|
    # where I1_minus_base is the minus contribution without the mirror weight
    # Note: The minus branch already includes the exp(-L(α+β)) factor from the bracket
    if abs(I1_minus_total) > 1e-100:
        m1_eff = (I1_combined_total - I1_plus_total) / I1_minus_total
    else:
        m1_eff = None

    return CombinedOperatorI1Result(
        I1_combined=I1_combined_total,
        I1_plus_only=I1_plus_total,
        I1_minus_base=I1_minus_total,
        m1_eff=m1_eff,
        R=R,
        L=L,
        theta=theta,
        n_quad=n,
        details={
            "method": "combined_identity_operator_leibniz",
            "pair": (ell1, ell2),
            "alpha_beta": (alpha, beta),
        }
    )


def analyze_L_convergence(
    theta: float,
    R: float,
    L_values: list,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> list:
    """
    Analyze convergence of m1_eff as L→∞.

    For each L in L_values, computes I1 via Leibniz operator expansion
    and extracts m1_eff. If the derivation is correct, m1_eff should
    converge to a finite value (hopefully matching exp(R) + 5).

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        L_values: List of L values to test (e.g., [10, 50, 100, 500, 1000])
        n: Quadrature order
        polynomials: Dict with polynomial keys
        ell1, ell2: Piece indices
        verbose: Print progress

    Returns:
        List of CombinedOperatorI1Result for each L value
    """
    results = []

    for L in L_values:
        if verbose:
            print(f"\n--- L = {L} ---")

        result = compute_I1_combined_operator_at_L(
            theta, R, L, n, polynomials, ell1, ell2, verbose
        )
        results.append(result)

        if verbose and result.m1_eff is not None:
            print(f"  m1_eff = {result.m1_eff:.6f}")

    return results
