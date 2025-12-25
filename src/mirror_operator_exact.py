"""
src/mirror_operator_exact.py
Phase 10.2: Derived Mirror Operator with Swap/Sign Conjugation

This module implements the CORRECT derived mirror operator that was missing in Phase 9.

KEY INSIGHT (from GPT analysis, 2025-12-23):
============================================
Phase 9 applied Q(1+D) shift but used the DIRECT eigenvalues:
    A_α = t + θ(t-1)x + θty
    A_β = t + θtx + θ(t-1)y

This caused Q(1+A_α) to evaluate in the range [1+t+..., 2+...], where Q polynomials explode
(the 112× blowup observed in Phase 9).

THE MISSING PIECE:
When mirroring via (α, β) → (-β, -α), the differential operators undergo conjugation:
    D_α → -D_β
    D_β → -D_α

For the mirror exponential N^{-βx-αy} = exp(-θL(βx+αy)):
    D_α[N^{-βx-αy}] = θy × N^{-βx-αy}  (SWAPPED!)
    D_β[N^{-βx-αy}] = θx × N^{-βx-αy}  (SWAPPED!)

So the MIRROR eigenvalues are:
    A_α^mirror = θy  (pure y-linear, no x, no t-dependence)
    A_β^mirror = θx  (pure x-linear, no y, no t-dependence)

These keep arguments in [0, θ] ≈ [0, 0.57] instead of [1+t, 2+...].

MATHEMATICAL STRUCTURE:
=======================
The TeX mirror term is:
    T^{-(α+β)} × I(-β,-α)

At the PRZZ evaluation point α = β = -R/L:
    T^{-(α+β)} = exp(2R)

The I(-β,-α) part involves:
1. Swapped eigenvalues for the Q operators
2. Swapped/negated arguments in the exponential factor

COMPARISON WITH PHASE 9:
========================
Phase 9 approach (WRONG):
    - Used Q(1+A_α), Q(1+A_β) with DIRECT eigenvalues
    - Result: 112× blowup

Phase 10 approach (CORRECT):
    - Use Q(A_α^mirror), Q(A_β^mirror) with SWAPPED eigenvalues
    - Expected: well-behaved values
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.series import TruncatedSeries
from src.quadrature import gauss_legendre_01


@dataclass(frozen=True)
class MirrorEigenvalues:
    """
    Mirror eigenvalue coefficients for A_α^mirror and A_β^mirror.

    For the mirror term with N^{-βx-αy}:
        A_α^mirror = θy  (pure y-linear)
        A_β^mirror = θx  (pure x-linear)

    These are fundamentally different from the direct eigenvalues:
        A_α = t + θ(t-1)x + θty
        A_β = t + θtx + θ(t-1)y

    The swap x↔y is the key conjugation induced by (-β,-α) substitution.
    """
    # A_α^mirror = u0_alpha + x_alpha*x + y_alpha*y
    y_alpha: float  # = θ (coefficient of y for A_α^mirror)
    x_beta: float   # = θ (coefficient of x for A_β^mirror)

    u0_alpha: float = 0.0
    x_alpha: float = 0.0

    # A_β^mirror = u0_beta + x_beta*x + y_beta*y
    u0_beta: float = 0.0
    y_beta: float = 0.0


def get_mirror_eigenvalues_with_swap(theta: float) -> MirrorEigenvalues:
    """
    Compute the SWAPPED eigenvalues for the mirror term (Phase 10, STATIC version).

    DEPRECATED: Use get_mirror_eigenvalues_complement_t() for the correct
    t-dependent transform (Phase 12).

    For the mirror exponential N^{-βx-αy} = exp(-θL(βx+αy)):
        D_α[N^{-βx-αy}] = -1/L × ∂/∂α[exp(-θL(βx+αy))]
                        = -1/L × (-θL×y) × exp(...)
                        = θy × exp(...)

        D_β[N^{-βx-αy}] = -1/L × ∂/∂β[exp(-θL(βx+αy))]
                        = -1/L × (-θL×x) × exp(...)
                        = θx × exp(...)

    So the eigenvalues are:
        A_α^mirror = θy  (pure y-linear, NO x, NO constant term)
        A_β^mirror = θx  (pure x-linear, NO y, NO constant term)

    These are SWAPPED compared to direct: α acts on y, β acts on x.

    Args:
        theta: PRZZ θ parameter (typically 4/7)

    Returns:
        MirrorEigenvalues with the swap structure
    """
    return MirrorEigenvalues(
        y_alpha=theta,    # A_α^mirror = θy
        x_beta=theta,     # A_β^mirror = θx
        u0_alpha=0.0,
        x_alpha=0.0,
        u0_beta=0.0,
        y_beta=0.0
    )


def get_mirror_eigenvalues_complement_t(t: float, theta: float) -> MirrorEigenvalues:
    """
    Compute t-DEPENDENT mirror eigenvalues using complement structure (Phase 12).

    The complement structure derives from the combined identity, which shows
    that after regularization, the mirror term should preserve t-dependence.

    Direct eigenvalues (from operator_post_identity.py):
        A_α(t) = t + θ(t-1)x + θt·y
        A_β(t) = t + θt·x + θ(t-1)·y

    Complement transform (Phase 12 hypothesis):
        A_α^mirror(t) = 1 - A_β(t) = (1-t) - θt·x - θ(t-1)·y
                                   = (1-t) - θt·x + θ(1-t)·y

        A_β^mirror(t) = 1 - A_α(t) = (1-t) - θ(t-1)·x - θt·y
                                   = (1-t) + θ(1-t)·x - θt·y

    Why complement structure:
    - Q(A_α) + Q(1-A_α) may have nice cancellation properties
    - Preserves the Q argument range in [0,1] approximately
    - Maintains t-dependence which Phase 10 dropped

    Args:
        t: Integration variable t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)

    Returns:
        MirrorEigenvalues with t-dependent complement structure
    """
    # A_α^mirror = (1-t) - θt·x + θ(1-t)·y
    u0_alpha = 1 - t
    x_alpha = -theta * t
    y_alpha = theta * (1 - t)

    # A_β^mirror = (1-t) + θ(1-t)·x - θt·y
    u0_beta = 1 - t
    x_beta = theta * (1 - t)
    y_beta = -theta * t

    return MirrorEigenvalues(
        y_alpha=y_alpha,
        x_beta=x_beta,
        u0_alpha=u0_alpha,
        x_alpha=x_alpha,
        u0_beta=u0_beta,
        y_beta=y_beta
    )


def get_mirror_exp_affine_coeffs(
    t: float,
    theta: float,
    R: float
) -> Tuple[float, float, float]:
    """
    Compute exp factor affine coefficients for the MIRROR term.

    For the mirror term, the exponential factor is different because:
    1. The N factor has swapped/negated arguments: N^{βx+αy} instead of N^{-αx-βy}
    2. The T factor: T^{-t(α+β)} evaluated at α=β=-R/L gives exp(2Rt)

    The MIRROR exponential contribution (before the T^{-(α+β)}=exp(2R) weight):
        exp(-t(α+β)L(1+θ(x+y))) × exp(θL(βx+αy))

    At α = β = -R/L:
        = exp(2Rt) × exp(-θR(x+y))

    The sign of the N^{...} term is POSITIVE (βx+αy at α=β=-R/L gives -R(x+y)/L,
    and with θL factor: -θR(x+y)).

    Wait, let me be more careful:
        At α = β = -R/L:
        βx + αy = (-R/L)x + (-R/L)y = -R(x+y)/L
        θL(βx+αy) = θL × (-R(x+y)/L) = -θR(x+y)

    So the mirror exp factor is:
        exp(2Rt - θR(x+y))
        = exp(2Rt) × exp(-θR(x+y))
        = exp(2Rt) × exp(-θRx) × exp(-θRy)

    This is DIFFERENT from direct which has:
        exp(2Rt + θR(2t-1)(x+y))

    The mirror has a SIMPLER structure - no (2t-1) factor!

    Returns:
        (u0, lin_x, lin_y) for exp(u0 + lin_x*x + lin_y*y)

    NOTE: This is the PHASE 10 (static) version with -θR(x+y).
    For Phase 13+ (t-flip consistent), use get_mirror_exp_affine_coeffs_t_flip().
    """
    # exp factor: exp(2Rt - θR(x+y))
    u0 = 2 * R * t      # constant term
    lin_x = -theta * R  # coefficient of x (NEGATIVE!)
    lin_y = -theta * R  # coefficient of y (NEGATIVE!)

    return (u0, lin_x, lin_y)


def get_mirror_exp_affine_coeffs_t_flip(
    t: float,
    theta: float,
    R: float,
    include_T_weight: bool = False
) -> Tuple[float, float, float]:
    """
    Compute CORRECTED exp factor affine coefficients for mirror term (Phase 13).

    Phase 13 insight: For consistency with the complement eigenvalues,
    the mirror exp must match the direct exp evaluated at t' = 1-t.

    Direct exp at 1-t:  exp(2R(1-t) + θR(1-2t)(x+y))

    For mirror to match direct(1-t), we have two options:

    Option A (include_T_weight=False): Mirror integrand × T_weight = Direct(1-t)
        Mirror integrand has: u0 = 2R(1-t) - 2R = -2Rt
                             lin = θR(1-2t)
        Then T_weight × exp(-2Rt) = exp(2R) × exp(-2Rt) = exp(2R(1-t)) ✓

    Option B (include_T_weight=True): Mirror integrand already includes T_weight
        u0 = 2R(1-t) [directly from direct at 1-t]
        lin = θR(1-2t)

    Args:
        t: Integration variable t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        include_T_weight: If True, u0 already includes T-weight (no separate exp(2R))
                         If False, u0 does not include T-weight (apply exp(2R) separately)

    Returns:
        (u0, lin_x, lin_y) for exp(u0 + lin_x*x + lin_y*y)
    """
    if include_T_weight:
        # Option B: u0 directly matches direct at 1-t (T-weight included)
        u0 = 2 * R * (1 - t)
    else:
        # Option A: u0 requires separate T-weight factor (DEFAULT for compatibility)
        # After T_weight × exp(u0) = exp(2R) × exp(-2Rt) = exp(2R(1-t))
        u0 = -2 * R * t

    # Lin coefficients match direct at 1-t
    lin_coeff = theta * R * (1 - 2 * t)
    lin_x = lin_coeff
    lin_y = lin_coeff

    return (u0, lin_x, lin_y)


def apply_QQexp_mirror_composition(
    Q_poly,
    t: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y"),
    use_t_dependent: bool = False,
    use_t_flip_exp: bool = False
) -> TruncatedSeries:
    """
    Compute Q(A_α^mirror) × Q(A_β^mirror) × exp_mirror as a TruncatedSeries.

    Args:
        Q_poly: Q polynomial object
        t: Integration variable t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        var_names: Variable names tuple
        use_t_dependent: If True, use Phase 12 complement eigenvalues (t-dependent).
                        If False (default), use Phase 10 static eigenvalues.
        use_t_flip_exp: If True, use Phase 13 t-flip consistent exp coefficients.
                       If False (default), use Phase 10/12 static exp coefficients.

    Phase 10 (static eigenvalues, static exp):
        A_α^mirror = θy, A_β^mirror = θx (no t-dependence)
        Exp: exp(2Rt - θR(x+y))

    Phase 12 (t-dependent eigenvalues, static exp):
        A_α^mirror(t) = 1 - A_β(t), A_β^mirror(t) = 1 - A_α(t)
        Exp: exp(2Rt - θR(x+y)) ← BUGGY! Static exp doesn't match eigenvalues.

    Phase 13 (t-dependent eigenvalues, t-flip exp):
        A_α^mirror(t) = 1 - A_β(t), A_β^mirror(t) = 1 - A_α(t)
        Exp: exp(2Rt + θR(1-2t)(x+y)) ← CORRECT! Matches direct at t' = 1-t.

    Returns:
        TruncatedSeries representing Q(A_α^mirror) × Q(A_β^mirror) × exp_mirror
    """
    if use_t_dependent:
        # Phase 12/13: t-dependent complement eigenvalues
        eig = get_mirror_eigenvalues_complement_t(t, theta)
    else:
        # Phase 10: static swapped eigenvalues
        eig = get_mirror_eigenvalues_with_swap(theta)

    # Build Q(A_α^mirror) series
    lin_alpha = {"x": eig.x_alpha, "y": eig.y_alpha}
    Q_alpha_series = compose_polynomial_on_affine(
        Q_poly, eig.u0_alpha, lin_alpha, var_names
    )

    # Build Q(A_β^mirror) series
    lin_beta = {"x": eig.x_beta, "y": eig.y_beta}
    Q_beta_series = compose_polynomial_on_affine(
        Q_poly, eig.u0_beta, lin_beta, var_names
    )

    # Build exp series for mirror
    if use_t_flip_exp:
        # Phase 13: t-flip consistent exp (matches direct at t' = 1-t)
        exp_u0, exp_x, exp_y = get_mirror_exp_affine_coeffs_t_flip(t, theta, R)
    else:
        # Phase 10/12: static exp (different structure than direct)
        exp_u0, exp_x, exp_y = get_mirror_exp_affine_coeffs(t, theta, R)

    exp_lin = {"x": exp_x, "y": exp_y}
    exp_series = compose_exp_on_affine(1.0, exp_u0, exp_lin, var_names)

    return Q_alpha_series * Q_beta_series * exp_series


@dataclass
class MirrorOperatorResult:
    """
    Result from mirror operator computation.

    The derived mirror operator correctly implements:
        T^{-(α+β)} × Q(D_α)Q(D_β)[F(-β,-α)]

    where F(-β,-α) uses the SWAPPED eigenvalues.
    """
    value: float
    """Full derived mirror: exp(2R) × I_swapped(+R)."""

    I_swapped: float
    """I value with swapped eigenvalues (before T weight)."""

    T_weight: float
    """T^{-(α+β)} = exp(2R) at PRZZ point."""

    R: float
    ell1: int
    ell2: int

    # Diagnostic info
    Q_alpha_range: Tuple[float, float]
    """Range of Q(A_α^mirror) = Q(θy) for y ∈ [0, 1]."""

    Q_beta_range: Tuple[float, float]
    """Range of Q(A_β^mirror) = Q(θx) for x ∈ [0, 1]."""


def compute_I1_mirror_operator_exact(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False,
    use_t_dependent: bool = False,
    use_t_flip_exp: bool = False
) -> MirrorOperatorResult:
    """
    Compute I₁ mirror contribution using the CORRECT operator transformation.

    This is the Phase 10.2 implementation that fixes Phase 9's blowup.
    Phase 12 adds t-dependent eigenvalues via use_t_dependent=True.
    Phase 13 adds t-flip consistent exp via use_t_flip_exp=True.

    Steps:
    1. Use swapped eigenvalues (Phase 10) or complement eigenvalues (Phase 12/13)
    2. Compose Q on mirror eigenvalues (NOT Q(1+·) on direct eigenvalues)
    3. Include proper mirror exponential factor (static or t-flip)
    4. Extract xy coefficient and integrate
    5. Multiply by T^{-(α+β)} = exp(2R)

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with 'P1', 'P2', 'P3', 'Q' keys
        ell1, ell2: Piece indices
        verbose: Print diagnostic output
        use_t_dependent: If True, use Phase 12/13 complement eigenvalues.
                        If False (default), use Phase 10 static eigenvalues.
        use_t_flip_exp: If True, use Phase 13 t-flip consistent exp coefficients.
                       If False (default), use Phase 10/12 static exp.

    Returns:
        MirrorOperatorResult with full mirror value and diagnostics
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

    I1_swapped = 0.0

    # Track Q value ranges for diagnostics
    Q_alpha_vals = []
    Q_beta_vals = []

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Build Q(A_α^mirror)Q(A_β^mirror) × exp_mirror series
            core_series = apply_QQexp_mirror_composition(
                Q_poly, t, theta, R, var_names,
                use_t_dependent=use_t_dependent,
                use_t_flip_exp=use_t_flip_exp
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
            I1_swapped += xy_coeff * scalar_prefactor * w_u * w_t

            # Track Q ranges for diagnostics
            if i_u == 0:
                # Q(A_α^mirror) = Q(θy) evaluated at y in [0,1] via t-grid point
                Q_alpha_vals.append(Q_poly.eval(np.array([theta * t]))[0])
                # Q(A_β^mirror) = Q(θx) - use u as proxy for x
                Q_beta_vals.append(Q_poly.eval(np.array([theta * u]))[0])

    # T^{-(α+β)} = exp(2R) at PRZZ point
    T_weight = np.exp(2 * R)

    # Full derived mirror
    value = T_weight * I1_swapped

    # Compute Q value ranges
    Q_alpha_range = (min(Q_alpha_vals), max(Q_alpha_vals)) if Q_alpha_vals else (0, 0)
    Q_beta_range = (min(Q_beta_vals), max(Q_beta_vals)) if Q_beta_vals else (0, 0)

    if verbose:
        print(f"I1_mirror_operator_exact ({ell1},{ell2}):")
        print(f"  I_swapped = {I1_swapped:.8f}")
        print(f"  T_weight = exp(2R) = {T_weight:.4f}")
        print(f"  Full derived = {value:.8f}")
        print(f"  Q(A_α^mirror) range: [{Q_alpha_range[0]:.4f}, {Q_alpha_range[1]:.4f}]")
        print(f"  Q(A_β^mirror) range: [{Q_beta_range[0]:.4f}, {Q_beta_range[1]:.4f}]")

    return MirrorOperatorResult(
        value=value,
        I_swapped=I1_swapped,
        T_weight=T_weight,
        R=R,
        ell1=ell1,
        ell2=ell2,
        Q_alpha_range=Q_alpha_range,
        Q_beta_range=Q_beta_range
    )


def compute_I2_mirror_operator_exact(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = False,
    use_t_dependent: bool = False,
    use_t_flip_exp: bool = False
) -> MirrorOperatorResult:
    """
    Compute I₂ mirror contribution using the CORRECT operator transformation.

    I₂ has no formal variables (no derivatives), so it's simpler.
    The mirror term for I₂ uses:
    - Q² evaluated at the mirror eigenvalues (at x=y=0)
    - The mirror exponential factor

    Phase 10 (static, use_t_dependent=False):
        At x=y=0: A_α^mirror = θ×0 = 0, A_β^mirror = θ×0 = 0
        Q factor: Q(0)²

    Phase 12 (t-dependent, use_t_dependent=True):
        At x=y=0: A_α^mirror = (1-t), A_β^mirror = (1-t)
        Q factor: Q(1-t)²
        This t-dependence is KEY for closing the 36x gap!

    Phase 13 (t-flip exp, use_t_flip_exp=True):
        Exp factor at x=y=0: exp(-2Rt) (matches direct at 1-t after T-weight)

    Args:
        theta, R, n, polynomials, ell1, ell2, verbose: Same as I₁
        use_t_dependent: If True, use Phase 12/13 complement eigenvalues.
        use_t_flip_exp: If True, use Phase 13 t-flip consistent exp.

    Returns:
        MirrorOperatorResult
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    I2_swapped = 0.0

    for u, w_u in zip(u_nodes, u_weights):
        # P factors at u
        P_ell1_u = P_polys[ell1].eval(np.array([u]))[0]
        P_ell2_u = P_polys[ell2].eval(np.array([u]))[0]
        PP_u = P_ell1_u * P_ell2_u

        for t, w_t in zip(t_nodes, t_weights):
            # For I₂ mirror, the Q factor evaluates at the mirror eigenvalues at x=y=0
            if use_t_dependent:
                # Phase 12: A_α^mirror(t)|_{x=y=0} = (1-t), same for A_β^mirror
                eig = get_mirror_eigenvalues_complement_t(t, theta)
                Q_alpha_at_0 = Q_poly.eval(np.array([eig.u0_alpha]))[0]
                Q_beta_at_0 = Q_poly.eval(np.array([eig.u0_beta]))[0]
                QQ_t = Q_alpha_at_0 * Q_beta_at_0
            else:
                # Phase 10: A_α^mirror = θy, so at y=0 it's 0
                Q0 = Q_poly.eval(np.array([0.0]))[0]
                QQ_t = Q0 ** 2

            # Mirror exp factor at x=y=0
            if use_t_flip_exp:
                # Phase 13: exp(-2Rt) - T-weight applied separately
                exp_factor = np.exp(-2 * R * t)
            else:
                # Phase 10/12: exp(2Rt)
                exp_factor = np.exp(2 * R * t)

            # Integrate
            integrand = (1.0 / theta) * PP_u * QQ_t * exp_factor
            I2_swapped += integrand * w_u * w_t

    T_weight = np.exp(2 * R)
    value = T_weight * I2_swapped

    if verbose:
        print(f"I2_mirror_operator_exact ({ell1},{ell2}):")
        print(f"  I2_swapped = {I2_swapped:.8f}")
        print(f"  T_weight = {T_weight:.4f}")
        print(f"  Full derived = {value:.8f}")

    return MirrorOperatorResult(
        value=value,
        I_swapped=I2_swapped,
        T_weight=T_weight,
        R=R,
        ell1=ell1,
        ell2=ell2,
        Q_alpha_range=(Q_poly.eval(np.array([0.0]))[0], Q_poly.eval(np.array([0.0]))[0]),
        Q_beta_range=(Q_poly.eval(np.array([0.0]))[0], Q_poly.eval(np.array([0.0]))[0])
    )


def compute_S12_mirror_operator_exact(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    K: int = 3,
    verbose: bool = False
) -> float:
    """
    Compute the full S12 mirror contribution using the CORRECT derived operator.

    This computes exp(2R) × [I₁ + I₂]_swapped for all pairs,
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

        # I₁ derived mirror with swap
        I1_result = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )

        # I₂ derived mirror with swap
        I2_result = compute_I2_mirror_operator_exact(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )

        pair_total = full_norm * (I1_result.value + I2_result.value)
        total += pair_total

        if verbose:
            print(f"Pair {pair_key}: I1_swap={I1_result.value:.6f}, "
                  f"I2_swap={I2_result.value:.6f}, "
                  f"norm={full_norm:.4f}, contrib={pair_total:.6f}")

    if verbose:
        print(f"S12_mirror_operator_exact total: {total:.8f}")

    return total


# =============================================================================
# Comparison with Phase 9 approach
# =============================================================================

def compare_with_phase9_approach(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Compare the new swap-based approach with Phase 9's Q-shift approach.

    This diagnostic function shows why Phase 9 blew up and why swap works.

    Returns:
        Dict with comparison values
    """
    from src.mirror_exact import compute_I1_mirror_derived

    # Phase 9 approach: Q(1+D) on DIRECT eigenvalues
    phase9_result = compute_I1_mirror_derived(
        theta=theta, R=R, n=n, polynomials=polynomials,
        ell1=ell1, ell2=ell2
    )

    # Phase 10 approach: Q(D) on SWAPPED eigenvalues
    phase10_result = compute_I1_mirror_operator_exact(
        theta=theta, R=R, n=n, polynomials=polynomials,
        ell1=ell1, ell2=ell2
    )

    # Compute ratio
    ratio = phase9_result.value / phase10_result.value if abs(phase10_result.value) > 1e-15 else float('inf')

    results = {
        "phase9_value": phase9_result.value,
        "phase9_I_shifted": phase9_result.I_shifted_Q_plus_R,
        "phase10_value": phase10_result.value,
        "phase10_I_swapped": phase10_result.I_swapped,
        "ratio_phase9_over_phase10": ratio,
        "T_weight": phase10_result.T_weight,
        "Q_alpha_range": phase10_result.Q_alpha_range,
        "Q_beta_range": phase10_result.Q_beta_range,
    }

    if verbose:
        print("\n=== Phase 9 vs Phase 10 Comparison ===")
        print(f"Pair: ({ell1},{ell2}), R = {R}")
        print(f"")
        print(f"Phase 9 (Q-shift on direct eigenvalues):")
        print(f"  I_shifted = {phase9_result.I_shifted_Q_plus_R:.8f}")
        print(f"  Full value = {phase9_result.value:.8f}")
        print(f"")
        print(f"Phase 10 (standard Q on swapped eigenvalues):")
        print(f"  I_swapped = {phase10_result.I_swapped:.8f}")
        print(f"  Full value = {phase10_result.value:.8f}")
        print(f"  Q(A_α^mirror) range: {phase10_result.Q_alpha_range}")
        print(f"  Q(A_β^mirror) range: {phase10_result.Q_beta_range}")
        print(f"")
        print(f"Ratio Phase9/Phase10: {ratio:.4f}")
        if ratio > 10:
            print("  ⚠️ Phase 9 is significantly larger - confirms the blowup!")
        elif 0.5 < ratio < 2:
            print("  ✓ Values are comparable")

    return results
