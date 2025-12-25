"""
src/combined_identity_regularized.py
U-Regularized Combined Identity for Pole-Free Operator Application.

MATHEMATICAL INSIGHT (GPT 2025-12-22):
======================================
The Leibniz expansion approach fails because differentiating 1/(α+β) produces
factorial amplification: d^n[1/(α+β)] = (-1)^n × n! / (α+β)^{n+1}

The solution is to ELIMINATE the pole BEFORE applying Q(D_α)Q(D_β), by rewriting
the bracket as an auxiliary integral.

TRANSFORMATION:
===============
Original bracket:
    B(α,β;x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)

where N = T^θ = exp(θL), T = exp(L), s = α+β.

Define interpolation H(u) = T^{-s(1-u)} N^{-βx-αy+us(x+y)}, u ∈ [0,1]

Then H(1) = N^{αx+βy} and H(0) = T^{-s}N^{-βx-αy}

So: B = (H(1) - H(0)) / s = ∫₀¹ (dH/du) du / s

Computing dH/du:
    d/du log H(u) = s log T + s(x+y) log N = s L (1 + θ(x+y))

Therefore:
    B(α,β;x,y) = L(1+θ(x+y)) ∫₀¹ E(α,β;x,y,u) du

where E(α,β;x,y,u) = T^{-s(1-u)} N^{-βx-αy+us(x+y)}
                   = exp(-L s(1-u) - θL(βx+αy-us(x+y)))

KEY INSIGHT: E is a pure exponential kernel with NO POLE!

EIGENVALUE SUBSTITUTION:
========================
For D_α = -1/L × ∂/∂α and D_β = -1/L × ∂/∂β:

D_α E = A_α(u,x,y) × E
D_β E = A_β(u,x,y) × E

where:
    A_α(u,x,y) = (1-u) + θ((1-u)y - ux)
    A_β(u,x,y) = (1-u) + θ((1-u)x - uy)

So for any polynomial Q:
    Q(D_α)Q(D_β) E = Q(A_α)Q(A_β) E

And therefore:
    Q(D_α)Q(D_β) B(α,β;x,y) = L(1+θ(x+y)) ∫₀¹ Q(A_α)Q(A_β) E(α,β;x,y,u) du

This is:
- POLE-FREE (no factorial amplification)
- OPERATOR-STABLE (pure eigenvalue substitution)
- Compatible with existing quadrature infrastructure
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Eigenvalue Functions for the Regularized Kernel
# =============================================================================

def compute_A_alpha(u: float, x: float, y: float, theta: float) -> float:
    """
    Compute eigenvalue A_α(u,x,y) for D_α on the regularized kernel E.

    A_α(u,x,y) = (1-u) + θ((1-u)y - ux)

    Args:
        u: Regularization parameter ∈ [0, 1]
        x, y: Series variables
        theta: PRZZ θ parameter (4/7)

    Returns:
        Eigenvalue for D_α
    """
    return (1 - u) + theta * ((1 - u) * y - u * x)


def compute_A_beta(u: float, x: float, y: float, theta: float) -> float:
    """
    Compute eigenvalue A_β(u,x,y) for D_β on the regularized kernel E.

    A_β(u,x,y) = (1-u) + θ((1-u)x - uy)

    Args:
        u: Regularization parameter ∈ [0, 1]
        x, y: Series variables
        theta: PRZZ θ parameter (4/7)

    Returns:
        Eigenvalue for D_β
    """
    return (1 - u) + theta * ((1 - u) * x - u * y)


def compute_kernel_E(
    alpha: float,
    beta: float,
    x: float,
    y: float,
    u: float,
    theta: float,
    L: float
) -> float:
    """
    Compute the regularized kernel E(α,β;x,y,u).

    E(α,β;x,y,u) = T^{-s(1-u)} × N^{-βx-αy+us(x+y)}
                 = exp(-L s(1-u)) × exp(-θL(βx+αy-us(x+y)))
                 = exp(-L s(1-u) - θL(βx+αy-us(x+y)))

    where s = α + β.

    Args:
        alpha, beta: Evaluation point
        x, y: Series variables
        u: Regularization parameter ∈ [0, 1]
        theta: PRZZ θ parameter
        L: Asymptotic parameter

    Returns:
        Value of kernel E at the given point
    """
    s = alpha + beta
    exponent = -L * s * (1 - u) - theta * L * (beta * x + alpha * y - u * s * (x + y))
    return np.exp(exponent)


# =============================================================================
# Affine Coefficient Extraction for TruncatedSeries Approach
# =============================================================================

def get_A_alpha_affine_coeffs_regularized(u: float, theta: float) -> Tuple[float, float, float]:
    """
    Extract affine coefficients for A_α in the u-regularized form.

    A_α(u,x,y) = (1-u) + θ((1-u)y - ux)
               = (1-u) + θ(1-u)·y + (-θu)·x

    Returns:
        (u0, x_coeff, y_coeff) where A_α = u0 + x_coeff·x + y_coeff·y
    """
    u0 = 1 - u
    x_coeff = -theta * u
    y_coeff = theta * (1 - u)
    return u0, x_coeff, y_coeff


def get_A_beta_affine_coeffs_regularized(u: float, theta: float) -> Tuple[float, float, float]:
    """
    Extract affine coefficients for A_β in the u-regularized form.

    A_β(u,x,y) = (1-u) + θ((1-u)x - uy)
               = (1-u) + θ(1-u)·x + (-θu)·y

    Returns:
        (u0, x_coeff, y_coeff) where A_β = u0 + x_coeff·x + y_coeff·y
    """
    u0 = 1 - u
    x_coeff = theta * (1 - u)
    y_coeff = -theta * u
    return u0, x_coeff, y_coeff


def get_exp_affine_coeffs_regularized(u: float, theta: float, R: float) -> Tuple[float, float, float]:
    """
    Extract affine coefficients for exp factor at α=β=-R/L.

    At α=β=-R/L, s = -2R/L:
        E = exp(2R(1-u) + θR(1-2u)(x+y))

    Returns:
        (exp_u0, x_coeff, y_coeff) for the exp exponent (not multiplied by R)
    """
    exp_u0 = 2 * R * (1 - u)
    # Coefficient for x and y in the exponent: θR(1-2u)
    lin_coeff = theta * R * (1 - 2 * u)
    return exp_u0, lin_coeff, lin_coeff


def compute_QQexp_series_regularized_at_u(
    Q_poly,
    u: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y")
):
    """
    Build Q(A_α)Q(A_β)×exp as a TruncatedSeries at fixed u.

    This is the CORRECT approach using nilpotent series algebra
    instead of finite differences.

    Mathematical basis:
    - A_α and A_β are affine in (x, y)
    - Q(affine) expands via Taylor on nilpotent variables
    - exp(affine) expands similarly
    - Multiply series, extract xy coefficient directly

    Args:
        Q_poly: Q polynomial with eval_deriv method
        u: Regularization parameter ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        var_names: Variable names tuple

    Returns:
        TruncatedSeries representing Q(A_α)Q(A_β)×exp
    """
    from src.composition import compose_polynomial_on_affine, compose_exp_on_affine

    # Build Q(A_α) series
    u0_alpha, x_alpha, y_alpha = get_A_alpha_affine_coeffs_regularized(u, theta)
    Q_alpha = compose_polynomial_on_affine(
        Q_poly, u0_alpha, {"x": x_alpha, "y": y_alpha}, var_names
    )

    # Build Q(A_β) series
    u0_beta, x_beta, y_beta = get_A_beta_affine_coeffs_regularized(u, theta)
    Q_beta = compose_polynomial_on_affine(
        Q_poly, u0_beta, {"x": x_beta, "y": y_beta}, var_names
    )

    # Build exp series
    exp_u0, exp_x, exp_y = get_exp_affine_coeffs_regularized(u, theta, R)
    # compose_exp_on_affine expects R × (u0 + lin), but we already have the full exponent
    # So use R=1.0 and pass pre-scaled values
    exp_series = compose_exp_on_affine(1.0, exp_u0, {"x": exp_x, "y": exp_y}, var_names)

    return Q_alpha * Q_beta * exp_series


# =============================================================================
# Q(D)Q(D) via Eigenvalue Substitution on Regularized Kernel (LEGACY - numeric only)
# =============================================================================

def evaluate_QQ_on_kernel_E(
    Q_poly,
    alpha: float,
    beta: float,
    x: float,
    y: float,
    u: float,
    theta: float,
    L: float
) -> float:
    """
    Compute Q(D_α)Q(D_β) E using eigenvalue substitution.

    Q(D_α)Q(D_β) E = Q(A_α) Q(A_β) E

    where A_α, A_β are the eigenvalues at (u,x,y).

    NOTE: This is the LEGACY numeric version for verification.
    For derivative extraction, use compute_QQexp_series_regularized_at_u().

    Args:
        Q_poly: Q polynomial with eval(x) method
        alpha, beta: Evaluation point for kernel
        x, y: Series variables
        u: Regularization parameter
        theta: PRZZ θ parameter
        L: Asymptotic parameter

    Returns:
        Value of Q(D_α)Q(D_β) E at the given point
    """
    # Compute eigenvalues
    A_alpha = compute_A_alpha(u, x, y, theta)
    A_beta = compute_A_beta(u, x, y, theta)

    # Evaluate Q at eigenvalues
    Q_at_A_alpha = Q_poly.eval(A_alpha)
    Q_at_A_beta = Q_poly.eval(A_beta)

    # Compute kernel
    E = compute_kernel_E(alpha, beta, x, y, u, theta, L)

    return Q_at_A_alpha * Q_at_A_beta * E


def compute_QQB_regularized(
    Q_poly,
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float,
    n_quad: int = 50
) -> float:
    """
    Compute Q(D_α)Q(D_β) B using the u-regularized form.

    Q(D_α)Q(D_β) B(α,β;x,y) = L(1+θ(x+y)) ∫₀¹ Q(A_α)Q(A_β) E(α,β;x,y,u) du

    Args:
        Q_poly: Q polynomial with eval(x) method
        alpha, beta: Evaluation point
        x, y: Series variables
        theta: PRZZ θ parameter
        L: Asymptotic parameter
        n_quad: Number of quadrature points for u-integration

    Returns:
        Value of Q(D_α)Q(D_β) B at the given point
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n_quad)

    # Prefactor: L(1 + θ(x+y))
    prefactor = L * (1 + theta * (x + y))

    # Integrate over u
    integral = 0.0
    for u, w in zip(u_nodes, u_weights):
        QQE = evaluate_QQ_on_kernel_E(Q_poly, alpha, beta, x, y, u, theta, L)
        integral += QQE * w

    return prefactor * integral


# =============================================================================
# Verification: Original Bracket for Q=1
# =============================================================================

def compute_bracket_original(
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float
) -> float:
    """
    Compute the original bracket B(α,β;x,y) directly.

    B = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)

    This is used for verification against the regularized form.
    """
    s = alpha + beta
    if abs(s) < 1e-100:
        raise ValueError(f"α+β too small: {s}")

    # Plus term: N^{αx+βy} = exp(θL(αx+βy))
    exp_plus = np.exp(theta * L * (alpha * x + beta * y))

    # Minus term: T^{-(α+β)} N^{-βx-αy} = exp(-L(α+β)) exp(-θL(βx+αy))
    exp_minus = np.exp(-L * s - theta * L * (beta * x + alpha * y))

    return (exp_plus - exp_minus) / s


def compute_bracket_regularized(
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float,
    n_quad: int = 50
) -> float:
    """
    Compute the bracket using the regularized u-integral form.

    B(α,β;x,y) = L(1+θ(x+y)) ∫₀¹ E(α,β;x,y,u) du

    For Q=1, Q(D_α)Q(D_β)B = B, so this should match the original bracket.
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n_quad)

    prefactor = L * (1 + theta * (x + y))

    integral = 0.0
    for u, w in zip(u_nodes, u_weights):
        E = compute_kernel_E(alpha, beta, x, y, u, theta, L)
        integral += E * w

    return prefactor * integral


# =============================================================================
# Dataclass for Results
# =============================================================================

@dataclass
class RegularizedI1Result:
    """Result of I1 computation using u-regularized kernel."""
    I1_combined: float          # Full combined identity value
    R: float
    L: float
    theta: float
    n_quad_ut: int              # Quadrature for (u,t) integration
    n_quad_reg: int             # Quadrature for regularization u-integral
    m1_eff: Optional[float]     # Effective mirror weight if computable
    details: Dict


# =============================================================================
# Main I1 Computation with Regularized Kernel (REFACTORED - Phase 5)
# =============================================================================

def compute_I1_combined_regularized_at_L(
    theta: float,
    R: float,
    L: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    n_quad_reg: int = 30,
    verbose: bool = False
) -> RegularizedI1Result:
    """
    Compute I1 from combined identity using u-regularized (pole-free) form.

    PHASE 5 REFACTORED VERSION:
    - Uses TruncatedSeries for derivative extraction (no finite differences)
    - Uses TeX-normalized prefactor (1/θ + x + y), NOT L(1+θ(x+y))
    - Single integration variable for combined-identity (u_reg = TeX t)
    - Should match operator_post_identity.py results exactly

    Mathematical structure (TeX 1529-1533):
        I₁ = d²/dxdy [(1/θ + x + y) × ∫∫ (1-u)² P(x+u)P(y+u) × Q(A_α)Q(A_β)×exp × du dt]

    The regularization parameter u corresponds to TeX t via t = 1-u.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        L: Asymptotic parameter (log T) - NOTE: result should be L-invariant!
        n: Number of quadrature points for profile integration
        polynomials: Dict with 'P1', 'Q' keys
        ell1, ell2: Piece indices (default 1,1)
        n_quad_reg: Number of quadrature points for combined-identity integration
        verbose: Print diagnostic output

    Returns:
        RegularizedI1Result with I1_combined and metadata
    """
    from src.quadrature import gauss_legendre_01
    from src.mollifier_profiles import case_c_taylor_coeffs
    from src.composition import compose_polynomial_on_affine, compose_profile_on_affine
    from src.series import TruncatedSeries

    # Profile integration (mollifier variable)
    u_profile_nodes, u_profile_weights = gauss_legendre_01(n)

    # Combined-identity integration (TeX t, represented as u_reg where t ≈ 1-u)
    # NOTE: We use u_reg directly as the integration variable, matching TeX structure
    u_reg_nodes, u_reg_weights = gauss_legendre_01(n_quad_reg)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    var_names = ("x", "y")

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

    I1_total = 0.0

    for i_up, (u_profile, w_up) in enumerate(zip(u_profile_nodes, u_profile_weights)):
        # Scalar prefactor for this profile point: (1-u)^power × sign
        scalar_prefactor = (1 - u_profile) ** one_minus_u_power * sign_factor

        # Build profile series for ℓ₁ on x
        if omega1 == 0:
            P1_series = compose_polynomial_on_affine(
                P_polys[ell1], u_profile, {"x": 1.0}, var_names
            )
        else:
            taylor_coeffs = case_c_taylor_coeffs(
                P_polys[ell1], u_profile, omega1, R, theta, max_order=1
            )
            P1_series = compose_profile_on_affine(taylor_coeffs, {"x": 1.0}, var_names)

        # Build profile series for ℓ₂ on y
        if omega2 == 0:
            P2_series = compose_polynomial_on_affine(
                P_polys[ell2], u_profile, {"y": 1.0}, var_names
            )
        else:
            taylor_coeffs = case_c_taylor_coeffs(
                P_polys[ell2], u_profile, omega2, R, theta, max_order=1
            )
            P2_series = compose_profile_on_affine(taylor_coeffs, {"y": 1.0}, var_names)

        # Algebraic prefactor (1/θ + x + y) - TeX-normalized, NO L factor!
        alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
        alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
        alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

        # Product of profiles and algebraic prefactor
        profile_product = P1_series * P2_series * alg_prefactor

        # Integrate over combined-identity parameter u_reg
        # Using TruncatedSeries for EXACT derivative extraction (no FD!)
        for i_t, (u_reg, w_t) in enumerate(zip(u_reg_nodes, u_reg_weights)):
            # Build Q(A_α)Q(A_β)×exp series at this u_reg value
            # This uses nilpotent series algebra - no finite differences!
            QQexp_series = compute_QQexp_series_regularized_at_u(
                Q_poly, u_reg, theta, R, var_names
            )

            # Full integrand series = QQexp × profiles × algebraic_prefactor
            full_series = QQexp_series * profile_product

            # Extract xy coefficient directly from series - EXACT!
            xy_coeff = full_series.extract(("x", "y"))

            # Accumulate contribution
            contribution = xy_coeff * scalar_prefactor * w_up * w_t
            I1_total += contribution

            if verbose and i_up == 0 and i_t < 3:
                # Also show individual series coefficients for debugging
                QQ_xy = QQexp_series.extract(("x", "y"))
                print(f"  (u_p={u_profile:.3f}, u_reg={u_reg:.3f}): "
                      f"QQexp_xy={QQ_xy:.6e}, full_xy={xy_coeff:.6e}")

    if verbose:
        print(f"I1_regularized_L{L:.0f}_{ell1}{ell2} = {I1_total:.8e}")

    return RegularizedI1Result(
        I1_combined=I1_total,
        R=R,
        L=L,
        theta=theta,
        n_quad_ut=n,
        n_quad_reg=n_quad_reg,
        m1_eff=None,
        details={
            "method": "u_regularized_series_extraction",
            "pair": (ell1, ell2),
            "fixes_applied": ["no_FD", "tex_normalized_prefactor", "single_t_integral"],
        }
    )


def analyze_L_convergence_regularized(
    theta: float,
    R: float,
    L_values: list,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    n_quad_reg: int = 30,
    verbose: bool = False
) -> list:
    """
    Analyze convergence of I1 as L→∞ using the regularized method.

    This should NOT diverge like the Leibniz method because the pole
    has been eliminated via u-regularization.
    """
    results = []

    for L in L_values:
        if verbose:
            print(f"\n--- L = {L} (regularized) ---")

        result = compute_I1_combined_regularized_at_L(
            theta, R, L, n, polynomials, ell1, ell2, n_quad_reg, verbose
        )
        results.append(result)

    return results
