"""
src/mirror_exact.py
Exact Mirror Computation via Operator Shift Q→Q(1+·)

Phase 6 Implementation: Replace empirical m₁ with derived mirror-operator transformation.

MATHEMATICAL BASIS (from docs/TEX_MIRROR_OPERATOR_SHIFT.md):
=============================================================
The operator shift identity:
    Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F

When operators act on the mirror term (which has T^{-s} factor), the polynomial Q
must be shifted: Q → Q(1+·).

This module provides:
1. compute_I1_with_shifted_Q() - I₁ computed using Q_shifted = Q(1+z)
2. compute_decomposition_components() - Separate direct and mirror contributions
3. validate_decomposition() - Gate test that combined = direct + mirror
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.q_operator import lift_poly_by_shift
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.series import TruncatedSeries
from src.quadrature import gauss_legendre_01


@dataclass
class MirrorDecompositionResult:
    """Result from mirror decomposition computation."""
    I1_combined: float          # Standard I₁ (via post-identity)
    I1_with_shifted_Q: float    # I₁ computed with Q(1+·) operators
    R: float
    theta: float
    ell1: int
    ell2: int
    details: Dict


def get_mirror_eigenvalue_coeffs(t: float, theta: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get eigenvalue affine coefficients for the MIRROR term's N^{-βx-αy}.

    For the mirror exponential N^{-βx-αy} = exp(-θL(βx+αy)):
        D_α[N^{-βx-αy}] = -1/L × ∂/∂α[exp(-θL(βx+αy))]
                        = -1/L × (-θL×y) × exp(...)
                        = θy × exp(...)

    So the mirror eigenvalues are SWAPPED compared to direct:
        A_α^{mirror} = θy  (affine: u0=0, x_coeff=0, y_coeff=θ)
        A_β^{mirror} = θx  (affine: u0=0, x_coeff=θ, y_coeff=0)

    Note: The t-parameter doesn't directly enter the mirror eigenvalues
    because the N^{-βx-αy} factor doesn't depend on t. However, the
    combined identity integral does have t-dependence in the overall
    structure.

    For the combined identity at parameter t:
        A_α = t + θ(t-1)x + θt·y  (includes both direct and mirror)
        A_β = t + θt·x + θ(t-1)·y

    These are the POST-combined-identity eigenvalues from operator_post_identity.py.

    Returns:
        ((u0_alpha, x_alpha, y_alpha), (u0_beta, x_beta, y_beta))
    """
    # For pure mirror (N^{-βx-αy} only), no t-dependence:
    # A_α^{mirror} = θy
    # A_β^{mirror} = θx
    u0_alpha = 0.0
    x_alpha = 0.0
    y_alpha = theta

    u0_beta = 0.0
    x_beta = theta
    y_beta = 0.0

    return ((u0_alpha, x_alpha, y_alpha), (u0_beta, x_beta, y_beta))


def apply_QQexp_shifted_composition(
    Q_poly,
    t: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y"),
    shift: float = 1.0
) -> TruncatedSeries:
    """
    Compute Q_shifted(A_α) × Q_shifted(A_β) × exp(...) as a TruncatedSeries.

    This uses Q_shifted(z) = Q(shift + z) instead of Q(z).

    The shifted polynomial accounts for the operator shift identity:
        Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F

    So when operators act on a term with T^{-s} factor, we use Q(1+·).

    Args:
        Q_poly: Q polynomial object
        t: Integration variable t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        var_names: Variable names tuple
        shift: Shift amount (default 1.0 for Q(1+·))

    Returns:
        TruncatedSeries representing Q_shifted(A_α) × Q_shifted(A_β) × exp
    """
    from src.operator_post_identity import get_A_alpha_affine_coeffs, get_A_beta_affine_coeffs, get_exp_affine_coeffs

    # Create shifted polynomial Q_shifted(z) = Q(shift + z)
    Q_shifted = lift_poly_by_shift(Q_poly, shift=shift)

    # Get affine coefficients (same as standard post-identity)
    u0_alpha, x_alpha, y_alpha = get_A_alpha_affine_coeffs(t, theta)
    u0_beta, x_beta, y_beta = get_A_beta_affine_coeffs(t, theta)

    # Build Q_shifted(A_α) series
    lin_alpha = {"x": x_alpha, "y": y_alpha}
    Q_alpha_series = compose_polynomial_on_affine(Q_shifted, u0_alpha, lin_alpha, var_names)

    # Build Q_shifted(A_β) series
    lin_beta = {"x": x_beta, "y": y_beta}
    Q_beta_series = compose_polynomial_on_affine(Q_shifted, u0_beta, lin_beta, var_names)

    # Build exp series (same as standard)
    exp_u0, exp_x, exp_y = get_exp_affine_coeffs(t, theta, R)
    exp_lin = {"x": exp_x, "y": exp_y}
    exp_series = compose_exp_on_affine(1.0, exp_u0, exp_lin, var_names)

    return Q_alpha_series * Q_beta_series * exp_series


def compute_I1_with_shifted_Q(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    shift: float = 1.0,
    verbose: bool = False
) -> float:
    """
    Compute I₁ using SHIFTED polynomial Q(shift + ·).

    This is the same structure as compute_I1_operator_post_identity_pair()
    but uses Q_shifted = Q(shift + z) instead of Q(z).

    When shift=1.0, this computes what the mirror contribution would be
    if we properly accounted for the operator shift identity.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with 'P1', 'P2', 'P3', 'Q' keys
        ell1, ell2: Piece indices
        shift: Polynomial shift amount (default 1.0)
        verbose: Print diagnostic output

    Returns:
        I₁ value computed with shifted Q
    """
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

    # Omega values for Case B/C selection
    omega1 = ell1 - 1
    omega2 = ell2 - 1

    var_names = ("x", "y")

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
            # Scalar prefactor
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Build Q_shifted(A_α)Q_shifted(A_β) × exp series
            core_series = apply_QQexp_shifted_composition(
                Q_poly, t, theta, R, var_names, shift=shift
            )

            # Build profile series for ℓ₁ on x
            if omega1 == 0:
                P1_series = compose_polynomial_on_affine(
                    P_polys[ell1], u, {"x": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell1], u, omega1, R, theta, max_order=1
                )
                P1_series = compose_profile_on_affine(taylor_coeffs, {"x": 1.0}, var_names)

            # Build profile series for ℓ₂ on y
            if omega2 == 0:
                P2_series = compose_polynomial_on_affine(
                    P_polys[ell2], u, {"y": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell2], u, omega2, R, theta, max_order=1
                )
                P2_series = compose_profile_on_affine(taylor_coeffs, {"y": 1.0}, var_names)

            # Algebraic prefactor (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Full integrand
            integrand = core_series * P1_series * P2_series * alg_prefactor

            # Extract xy coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Accumulate
            I1_total += xy_coeff * scalar_prefactor * w_u * w_t

    if verbose:
        print(f"I1_shifted_Q_{ell1}{ell2} (shift={shift}) = {I1_total:.8f}")

    return I1_total


def compute_I1_standard(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
) -> float:
    """
    Compute standard I₁ using unshifted Q (shift=0).

    This is equivalent to compute_I1_operator_post_identity_pair() but
    uses the unified shifted infrastructure with shift=0.

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with polynomial objects
        ell1, ell2: Piece indices

    Returns:
        Standard I₁ value
    """
    return compute_I1_with_shifted_Q(
        theta=theta, R=R, n=n, polynomials=polynomials,
        ell1=ell1, ell2=ell2, shift=0.0
    )


def compute_mirror_decomposition(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> MirrorDecompositionResult:
    """
    Compute both standard I₁ and I₁ with shifted Q for comparison.

    This allows us to study how the Q shift affects I₁ values.

    The theory predicts:
    - I₁_standard uses Q(A_α)Q(A_β)
    - I₁_shifted uses Q(1+A_α)Q(1+A_β)
    - The ratio should relate to the empirical m₁

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with polynomial objects
        ell1, ell2: Piece indices
        verbose: Print diagnostics

    Returns:
        MirrorDecompositionResult with both I₁ values
    """
    from src.operator_post_identity import compute_I1_operator_post_identity_pair

    # Compute standard I₁ via validated post-identity path
    result_standard = compute_I1_operator_post_identity_pair(
        theta=theta, R=R, ell1=ell1, ell2=ell2, n=n, polynomials=polynomials
    )
    I1_combined = result_standard.I1_value

    # Compute I₁ with shifted Q
    I1_shifted = compute_I1_with_shifted_Q(
        theta=theta, R=R, n=n, polynomials=polynomials,
        ell1=ell1, ell2=ell2, shift=1.0, verbose=verbose
    )

    if verbose:
        ratio = I1_shifted / I1_combined if abs(I1_combined) > 1e-15 else float('inf')
        print(f"I₁ comparison ({ell1},{ell2}):")
        print(f"  Standard (Q):     {I1_combined:.8f}")
        print(f"  Shifted (Q(1+·)): {I1_shifted:.8f}")
        print(f"  Ratio:            {ratio:.4f}")

    return MirrorDecompositionResult(
        I1_combined=I1_combined,
        I1_with_shifted_Q=I1_shifted,
        R=R,
        theta=theta,
        ell1=ell1,
        ell2=ell2,
        details={
            "method": "mirror_decomposition",
            "shift": 1.0,
        }
    )


def analyze_shift_effect_on_m1(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    verbose: bool = True
) -> Dict:
    """
    Analyze how the Q shift affects the implied mirror weight.

    Theory: The empirical m₁ = exp(R) + 5 should be explained by:
    1. The T^{-s} weight: exp(2R) at α=β=-R/L
    2. The Q shift effect: Q(1+·) vs Q(·)

    This function computes the ratio of I₁ values to understand
    what the Q shift contributes.

    Returns:
        Dict with analysis results
    """
    from src.m1_policy import M1_EMPIRICAL_KAPPA, M1_EMPIRICAL_KAPPA_STAR

    # Compute for (1,1) pair
    result = compute_mirror_decomposition(
        theta=theta, R=R, n=n, polynomials=polynomials,
        ell1=1, ell2=1, verbose=False
    )

    I1_standard = result.I1_combined
    I1_shifted = result.I1_with_shifted_Q

    # Compute theoretical weights
    exp_2R = np.exp(2 * R)
    exp_R = np.exp(R)
    m1_empirical = exp_R + 5  # K=3 formula

    # What is the Q-shift effect?
    if abs(I1_standard) > 1e-15:
        q_shift_ratio = I1_shifted / I1_standard
    else:
        q_shift_ratio = float('inf')

    # If the assembly formula is:
    #   c = I₁(+R) + m₁ × I₁(-R)
    # And the theory says:
    #   mirror_contribution = exp(2R) × I₁_shifted
    # Then:
    #   m₁ × I₁(-R) ≈ exp(2R) × I₁_shifted
    # So:
    #   m₁ ≈ exp(2R) × (I₁_shifted / I₁(-R))

    results = {
        "R": R,
        "I1_standard": I1_standard,
        "I1_shifted": I1_shifted,
        "q_shift_ratio": q_shift_ratio,
        "exp_2R": exp_2R,
        "exp_R": exp_R,
        "m1_empirical": m1_empirical,
        "exp_2R_over_m1": exp_2R / m1_empirical,
    }

    if verbose:
        print("\n=== Q-Shift Effect Analysis ===")
        print(f"R = {R}")
        print(f"I₁ (standard Q):     {I1_standard:.6f}")
        print(f"I₁ (shifted Q(1+·)): {I1_shifted:.6f}")
        print(f"Q-shift ratio:       {q_shift_ratio:.4f}")
        print(f"exp(2R) = {exp_2R:.4f}")
        print(f"exp(R) = {exp_R:.4f}")
        print(f"m₁_empirical = {m1_empirical:.4f}")
        print(f"exp(2R) / m₁ = {exp_2R / m1_empirical:.4f}")

    return results


def compute_I1_at_minus_R(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    use_shifted_Q: bool = False
) -> float:
    """
    Compute I₁ at the -R evaluation point.

    This evaluates the integral with α = β = +R/L (instead of -R/L),
    which is relevant for understanding the mirror assembly formula.

    Note: The exp factor in the core integrand changes sign with R:
    - At α=β=-R/L: exp(+2Rt + θR(2t-1)(x+y))
    - At α=β=+R/L: exp(-2Rt - θR(2t-1)(x+y))

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter (will use -R internally)
        n: Quadrature points
        polynomials: Dict with polynomial objects
        ell1, ell2: Piece indices
        use_shifted_Q: If True, use Q(1+·) instead of Q

    Returns:
        I₁ value at -R evaluation point
    """
    # Simply call with -R
    shift = 1.0 if use_shifted_Q else 0.0

    # The exp_affine_coeffs function uses R directly in:
    #   u0 = 2*R*t, lin = R*(2θt - θ)
    # So passing -R gives the "negative R" contribution
    return compute_I1_with_shifted_Q(
        theta=theta, R=-R, n=n, polynomials=polynomials,
        ell1=ell1, ell2=ell2, shift=shift
    )


# =============================================================================
# Phase 9.2: Derived Mirror Term Implementation
# =============================================================================

@dataclass
class DerivedMirrorResult:
    """
    Result from derived mirror term computation.

    The derived mirror is the TRUE TeX mirror term:
        T^{-(α+β)} × I(-β,-α) with Q(1+D) operators

    At the PRZZ evaluation point α=β=-R/L:
        T^{-(α+β)} = exp(2R)

    And the I(-β,-α) with Q(1+D) is computed using the shifted Q polynomial
    evaluated at the STANDARD +R parameters.
    """
    value: float
    """Full derived mirror: exp(2R) × I_shifted_Q(+R)."""

    I_shifted_Q_plus_R: float
    """I value with Q(1+·) at +R (before T weight)."""

    T_weight: float
    """T^{-(α+β)} = exp(2R)."""

    R: float
    """R parameter used."""

    ell1: int
    """First piece index."""

    ell2: int
    """Second piece index."""


def compute_I1_mirror_derived(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> DerivedMirrorResult:
    """
    Compute the TeX-derived mirror term for I₁.

    This computes the TRUE mirror term from PRZZ:
        T^{-(α+β)} × I₁(-β,-α) with Q(1+D) operators

    At α = β = -R/L:
        T^{-(α+β)} = exp(2R)
        The I(-β,-α) part evaluates at swapped/negated arguments

    Implementation:
    1. Compute I₁ with Q(1+·) at +R parameters
    2. Multiply by exp(2R)

    This differs from the DSL "-R branch" which:
    - Flips R → -R in exp factors
    - Uses Q (not Q(1+·))

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with polynomial objects
        ell1, ell2: Piece indices
        verbose: Print diagnostics

    Returns:
        DerivedMirrorResult with full mirror value and breakdown
    """
    # Compute I₁ with shifted Q at +R
    I_shifted = compute_I1_with_shifted_Q(
        theta=theta,
        R=R,  # Use +R, not -R
        n=n,
        polynomials=polynomials,
        ell1=ell1,
        ell2=ell2,
        shift=1.0,  # Q → Q(1+·)
        verbose=verbose
    )

    # T^{-(α+β)} = exp(2R) at PRZZ point
    T_weight = np.exp(2 * R)

    # Full derived mirror
    value = T_weight * I_shifted

    if verbose:
        print(f"I1_mirror_derived ({ell1},{ell2}):")
        print(f"  I_shifted_Q(+R) = {I_shifted:.8f}")
        print(f"  T_weight = exp(2R) = {T_weight:.4f}")
        print(f"  Full derived = {value:.8f}")

    return DerivedMirrorResult(
        value=value,
        I_shifted_Q_plus_R=I_shifted,
        T_weight=T_weight,
        R=R,
        ell1=ell1,
        ell2=ell2
    )


def compute_I2_mirror_derived(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False
) -> DerivedMirrorResult:
    """
    Compute the TeX-derived mirror term for I₂.

    I₂ has no formal variables (no derivatives), so it's simpler than I₁.

    The I₂ term structure:
        ∫ P₁(u) P₂(u) Q²(t) exp(2Rt) du dt

    For the mirror term with Q(1+D):
        Uses Q(1+t)² instead of Q(t)²

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with polynomial objects
        ell1, ell2: Piece indices
        verbose: Print diagnostics

    Returns:
        DerivedMirrorResult with full mirror value
    """
    # I₂ is computed using compute_I2_standard with shifted Q
    # Since I₂ has no x,y variables, the "shift" affects Q(t) → Q(1+t)
    I_shifted = _compute_I2_with_shifted_Q(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        ell1=ell1,
        ell2=ell2,
        shift=1.0
    )

    T_weight = np.exp(2 * R)
    value = T_weight * I_shifted

    if verbose:
        print(f"I2_mirror_derived ({ell1},{ell2}):")
        print(f"  I2_shifted_Q(+R) = {I_shifted:.8f}")
        print(f"  T_weight = exp(2R) = {T_weight:.4f}")
        print(f"  Full derived = {value:.8f}")

    return DerivedMirrorResult(
        value=value,
        I_shifted_Q_plus_R=I_shifted,
        T_weight=T_weight,
        R=R,
        ell1=ell1,
        ell2=ell2
    )


def _compute_I2_with_shifted_Q(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int,
    ell2: int,
    shift: float = 1.0
) -> float:
    """
    Compute I₂ with shifted Q polynomial.

    I₂ = (1/θ) ∫∫ P_{ℓ₁}(u) P_{ℓ₂}(u) Q²(t) exp(2Rt) du dt

    With shift: Q(t) → Q(shift + t)

    For shift=1.0: Q²(1+t) instead of Q²(t)
    """
    from src.q_operator import lift_poly_by_shift

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    # Shift Q: Q(t) → Q(shift + t)
    Q_shifted = lift_poly_by_shift(Q_poly, shift=shift)

    I2_total = 0.0

    for u, w_u in zip(u_nodes, u_weights):
        # P factors at u
        P_ell1_u = P_polys[ell1].eval(np.array([u]))[0]
        P_ell2_u = P_polys[ell2].eval(np.array([u]))[0]
        PP_u = P_ell1_u * P_ell2_u

        for t, w_t in zip(t_nodes, t_weights):
            # Q²(shift + t)
            Q_val = Q_shifted.eval(np.array([t]))[0]
            QQ_t = Q_val ** 2

            # exp(2Rt)
            exp_factor = np.exp(2 * R * t)

            # Integrate
            integrand = (1.0 / theta) * PP_u * QQ_t * exp_factor
            I2_total += integrand * w_u * w_t

    return I2_total


def compute_S12_mirror_derived(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    K: int = 3,
    verbose: bool = False
) -> float:
    """
    Compute the full S12 derived mirror contribution.

    This computes exp(2R) × [I₁ + I₂]_shifted_Q for all pairs,
    using triangle×2 convention.

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with polynomial objects
        K: Number of mollifier pieces (default 3)
        verbose: Print diagnostics

    Returns:
        Total S12 derived mirror value
    """
    # Factorial normalization (same as production evaluator)
    factorial_norm = {
        "11": 1.0,
        "22": 0.25,
        "33": 1.0 / 36.0,
        "12": 0.5,
        "13": 1.0 / 6.0,
        "23": 1.0 / 12.0,
    }

    # Symmetry factors (triangle×2)
    symmetry = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    pairs = ["11", "22", "33", "12", "13", "23"]
    total = 0.0

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        norm = factorial_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm

        # I₁ derived mirror
        I1_result = compute_I1_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )

        # I₂ derived mirror
        I2_result = compute_I2_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )

        pair_total = full_norm * (I1_result.value + I2_result.value)
        total += pair_total

        if verbose:
            print(f"Pair {pair_key}: I1_der={I1_result.value:.6f}, "
                  f"I2_der={I2_result.value:.6f}, "
                  f"norm={full_norm:.4f}, contrib={pair_total:.6f}")

    if verbose:
        print(f"S12_mirror_derived total: {total:.8f}")

    return total


def compute_S12_minus_basis(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    K: int = 3,
    verbose: bool = False
) -> float:
    """
    Compute the DSL minus basis (S12 at -R with Q unchanged).

    This is the EMPIRICAL basis used in the production evaluator.
    It differs from the derived mirror.

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter (will use -R internally)
        n: Quadrature points
        polynomials: Dict with polynomial objects
        K: Number of mollifier pieces
        verbose: Print diagnostics

    Returns:
        Total S12 at -R (DSL minus basis)
    """
    # Use same normalization as derived
    factorial_norm = {
        "11": 1.0,
        "22": 0.25,
        "33": 1.0 / 36.0,
        "12": 0.5,
        "13": 1.0 / 6.0,
        "23": 1.0 / 12.0,
    }

    symmetry = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    pairs = ["11", "22", "33", "12", "13", "23"]
    total = 0.0

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        norm = factorial_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm

        # I₁ at -R (no Q shift)
        I1_minus = compute_I1_at_minus_R(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, use_shifted_Q=False
        )

        # I₂ at -R (no Q shift)
        I2_minus = _compute_I2_with_shifted_Q(
            theta=theta, R=-R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )

        pair_total = full_norm * (I1_minus + I2_minus)
        total += pair_total

        if verbose:
            print(f"Pair {pair_key}: I1(-R)={I1_minus:.6f}, "
                  f"I2(-R)={I2_minus:.6f}, contrib={pair_total:.6f}")

    if verbose:
        print(f"S12_minus_basis total: {total:.8f}")

    return total
