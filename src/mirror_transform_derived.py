"""
src/mirror_transform_derived.py
Phase 27: Derived Mirror Transform Using Bivariate Engine

Implements the PRZZ mirror transform I(-β,-α) directly using the bivariate series
engine, WITHOUT using R → -R as a proxy.

PRZZ TeX lines 1502-1511 define the mirror structure:
    I(α,β) + T^{-(α+β)} × I(-β,-α)

KEY DIFFERENCES between direct I(α,β) and mirror I(-β,-α):

1. EXPONENTIAL STRUCTURE:
   - Direct: N^{αx+βy} × T^{-t(α+β)}
   - Mirror: N^{-βx-αy} × T^{-t(-β-α)} × T^{-(α+β)}
           = N^{-βx-αy} × T^{t(α+β)} × T^{-(α+β)}
           = N^{-βx-αy} × T^{(t-1)(α+β)}

   At α=β=-R/L:
   - Direct N factor: exp(-θR(x+y))  [inside the (2t-1) structure]
   - Mirror N factor: exp(+θR(x+y))  ← sign flip!

2. T PREFACTOR:
   - Direct: integrates T^{-t(α+β)} over t ∈ [0,1]
   - Mirror: T^{-(α+β)} × T^{t(α+β)} = T^{(t-1)(α+β)}

   At α=β=-R/L:
   - Direct T factor: T^{2tR/L} → exp(2Rt) in the exponent
   - Mirror T factor: T^{2(t-1)R/L} → exp(2R(t-1)) = exp(2Rt - 2R)

3. Q EIGENVALUE STRUCTURE:
   - Direct: A_α(t) = t + θ(t-1)x + θty   [couples x to (t-1), y to t]
            A_β(t) = t + θtx + θ(t-1)y   [couples x to t, y to (t-1)]

   - Mirror with (-β,-α) substitution swaps the operator coupling:
            A_α^M(t) = t + θty + θ(t-1)x  [swapped from direct A_β]
            A_β^M(t) = t + θ(t-1)y + θtx  [swapped from direct A_α]

   In code: swap ax↔ay for Q_alpha, Q_beta

Created: 2025-12-26 (Phase 27)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from src.quadrature import gauss_legendre_01
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_log_factor,
    build_P_factor,
    build_Q_factor,
)
from src.unified_i1_general import get_polynomial_coeffs


@dataclass
class MirrorI1Result:
    """Result of mirror I₁ evaluation."""

    ell1: int
    ell2: int
    I1_mirror_value: float

    # T prefactor applied
    T_prefactor: float
    I1_mirror_raw: float  # Before T prefactor

    # Diagnostics
    n_quad_u: int
    n_quad_t: int


@dataclass
class MirrorI2Result:
    """Result of mirror I₂ evaluation."""

    ell1: int
    ell2: int
    I2_mirror_value: float

    T_prefactor: float
    I2_mirror_raw: float

    n_quad_u: int
    n_quad_t: int


def compute_I1_mirror_derived(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    include_Q: bool = True,
    apply_factorial_norm: bool = True,
    T_prefactor_mode: str = "exp_2R",  # "exp_2R", "exp_2R_theta", "none", "absorbed"
) -> MirrorI1Result:
    """
    Compute mirror I₁ for pair (ℓ₁, ℓ₂) using derived PRZZ transform.

    This implements I(-β,-α) directly, not via R → -R substitution.

    REVISED UNDERSTANDING (after diagnostic):
    The PRZZ difference quotient structure means both direct and mirror
    share the same t-integration structure. The key differences are:
    1. Sign flip on the (x+y) coefficient in the N^{...} exponential
    2. Q eigenvalue swap (x↔y coupling)
    3. T^{-(α+β)} prefactor applied after integration

    T_prefactor_mode options:
    - "exp_2R": Apply T^{-(α+β)} = exp(2R) after integration
    - "exp_2R_theta": Apply T^{-(α+β)} = exp(2R/θ) after integration
    - "none": No prefactor (T_prefactor = 1.0)
    - "absorbed": Old mode with t-structure change (wrong)

    Args:
        R, theta, ell1, ell2, polynomials: PRZZ parameters
        n_quad_u, n_quad_t: Quadrature points
        include_Q: Whether to include Q factors
        apply_factorial_norm: Whether to multiply by ℓ₁!ℓ₂!
        T_prefactor_mode: How to apply the T^{-(α+β)} prefactor

    Returns:
        MirrorI1Result with mirror I₁ value
    """
    max_dx = ell1
    max_dy = ell2

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # Get polynomial coefficient lists
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    P_ell1_coeffs = get_polynomial_coeffs(P_ell1)
    P_ell2_coeffs = get_polynomial_coeffs(P_ell2)
    Q_coeffs = get_polynomial_coeffs(Q) if Q is not None and include_Q else None

    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # MIRROR EXP FACTOR:
            # Two interpretations being tested:
            #
            # Mode "absorbed" (original): Change t-structure
            #   exp(2R(t-1) - Rθ(2t-1)(x+y))
            #
            # Mode "exp_2R" (revised): Same t-structure, flip xy sign, multiply by exp(2R)
            #   exp(2Rt - Rθ(2t-1)(x+y)) × exp(2R)
            #
            # The key insight is that both direct and mirror terms share
            # the same ∫ T^{-t(α+β)} dt integration, so the t-structure
            # should be the same. The difference is only in the N^{...} sign.
            if T_prefactor_mode == "absorbed":
                # Old mode: absorb T prefactor into t-structure
                a0_mirror = 2 * R * (t - 1)  # = 2Rt - 2R
            else:
                # New mode: same t-structure as direct
                a0_mirror = 2 * R * t

            # Sign flip on (x+y) coefficient
            a_xy_mirror = -R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0_mirror, a_xy_mirror, a_xy_mirror, max_dx, max_dy)

            # Log factor: same as direct (1/θ + x + y)
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # P factors: same structure (depends on u, not on R sign)
            P_x = build_P_factor(P_ell1_coeffs, u, "x", max_dx, max_dy)
            P_y = build_P_factor(P_ell2_coeffs, u, "y", max_dx, max_dy)

            bracket = exp_factor * log_factor * P_x * P_y

            # MIRROR Q FACTORS:
            # Direct eigenvalues:
            #   A_α = t + θ(t-1)x + θty
            #   A_β = t + θtx + θ(t-1)y
            #
            # Mirror eigenvalues (x↔y swap in coupling):
            #   A_α^M = t + θty + θ(t-1)x  [swap ax↔ay from A_α]
            #   A_β^M = t + θ(t-1)y + θtx  [swap ax↔ay from A_β]
            #
            # Actually, the (-β,-α) substitution means:
            # - What was the α-slot now gets -β
            # - What was the β-slot now gets -α
            # Since α=β=-R/L (symmetric), this mainly affects the operator coupling
            if include_Q and Q_coeffs is not None:
                # Mirror Q_alpha: swap ax, ay from direct Q_alpha
                # Direct Q_alpha: ax=θ(t-1), ay=θt
                # Mirror Q_alpha: ax=θt, ay=θ(t-1) ← swapped
                Q_alpha_mirror = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * t,       # was θ(t-1)
                    ay=theta * (t - 1), # was θt
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                # Mirror Q_beta: swap ax, ay from direct Q_beta
                # Direct Q_beta: ax=θt, ay=θ(t-1)
                # Mirror Q_beta: ax=θ(t-1), ay=θt ← swapped
                Q_beta_mirror = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * (t - 1), # was θt
                    ay=theta * t,       # was θ(t-1)
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                bracket = bracket * Q_alpha_mirror * Q_beta_mirror

            # Extract x^ℓ₁ y^ℓ₂ coefficient
            coeff = bracket.extract(ell1, ell2)

            total += coeff * one_minus_u_factor * u_w * t_w

    # Apply factorial normalization
    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    # Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    # Store raw value before T prefactor
    raw_value = total

    # T^{-(α+β)} prefactor at α=β=-R/L
    # Different interpretations based on mode:
    if T_prefactor_mode == "exp_2R":
        # T^{-(α+β)} = T^{2R/L}, assuming L=1 gives T^{2R} = e^{2R}
        T_prefactor = math.exp(2 * R)
    elif T_prefactor_mode == "exp_2R_theta":
        # T^{2R/L} with L=θ gives T^{2R/θ} = e^{2R/θ}
        T_prefactor = math.exp(2 * R / theta)
    elif T_prefactor_mode == "absorbed":
        # Old mode: T prefactor already absorbed into t-structure
        T_prefactor = 1.0
    else:  # "none"
        T_prefactor = 1.0

    total *= T_prefactor

    return MirrorI1Result(
        ell1=ell1,
        ell2=ell2,
        I1_mirror_value=total,
        T_prefactor=T_prefactor,
        I1_mirror_raw=raw_value,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
    )


def compute_I2_mirror_derived(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    include_Q: bool = True,
    apply_factorial_norm: bool = True,
) -> MirrorI2Result:
    """
    Compute mirror I₂ for pair (ℓ₁, ℓ₂) using derived PRZZ transform.

    I₂ has similar mirror structure to I₁, with the main difference being
    the (1-u) power in the integrand (coming from Euler-Maclaurin).

    For I₂: uses ∫ u^{ℓ₁+ℓ₂} du instead of ∫ (1-u)^{ℓ₁+ℓ₂} du
    """
    max_dx = ell1
    max_dy = ell2

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    P_ell1_coeffs = get_polynomial_coeffs(P_ell1)
    P_ell2_coeffs = get_polynomial_coeffs(P_ell2)
    Q_coeffs = get_polynomial_coeffs(Q) if Q is not None and include_Q else None

    u_power = ell1 + ell2  # I₂ uses u^{ℓ₁+ℓ₂} not (1-u)^{ℓ₁+ℓ₂}

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        u_factor = u ** u_power  # Different from I₁

        for t, t_w in zip(t_nodes, t_weights):
            # Mirror exp factor (same as I₁ mirror)
            a0_mirror = 2 * R * (t - 1)
            a_xy_mirror = -R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0_mirror, a_xy_mirror, a_xy_mirror, max_dx, max_dy)

            # Log factor
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # P factors
            P_x = build_P_factor(P_ell1_coeffs, u, "x", max_dx, max_dy)
            P_y = build_P_factor(P_ell2_coeffs, u, "y", max_dx, max_dy)

            bracket = exp_factor * log_factor * P_x * P_y

            # Mirror Q factors
            if include_Q and Q_coeffs is not None:
                Q_alpha_mirror = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta * t, ay=theta * (t - 1),
                    max_dx=max_dx, max_dy=max_dy,
                )
                Q_beta_mirror = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta * (t - 1), ay=theta * t,
                    max_dx=max_dx, max_dy=max_dy,
                )
                bracket = bracket * Q_alpha_mirror * Q_beta_mirror

            coeff = bracket.extract(ell1, ell2)
            total += coeff * u_factor * u_w * t_w

    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    raw_value = total
    T_prefactor = 1.0
    total *= T_prefactor

    return MirrorI2Result(
        ell1=ell1,
        ell2=ell2,
        I2_mirror_value=total,
        T_prefactor=T_prefactor,
        I2_mirror_raw=raw_value,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
    )


# =============================================================================
# P=Q=1 MICROCASE ORACLES
# =============================================================================

def compute_I1_mirror_P1Q1(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    apply_factorial_norm: bool = True,
) -> float:
    """
    Compute mirror I₁ with P=Q=1 for oracle validation.

    With P=Q=1, the bracket simplifies to:
        exp(2R(t-1) - Rθ(2t-1)(x+y)) × (1/θ + x + y)

    This can be compared against analytical results.
    """
    max_dx = ell1
    max_dy = ell2

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # Mirror exp factor
            a0_mirror = 2 * R * (t - 1)
            a_xy_mirror = -R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0_mirror, a_xy_mirror, a_xy_mirror, max_dx, max_dy)

            # Log factor
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # P=Q=1: just exp × log
            bracket = exp_factor * log_factor

            coeff = bracket.extract(ell1, ell2)
            total += coeff * one_minus_u_factor * u_w * t_w

    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return total


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

@dataclass
class MirrorComparisonResult:
    """Result of comparing derived mirror to -R proxy."""

    ell1: int
    ell2: int

    # Direct term at +R
    I1_direct_plusR: float

    # Mirror approaches
    I1_mirror_derived: float   # Using this module
    I1_proxy_minusR: float     # Using compute_I1_unified_general with -R

    # Effective multiplier (if proxy != 0)
    m_eff: Optional[float]

    # Empirical reference
    m_empirical: float


def compare_mirror_to_proxy(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> MirrorComparisonResult:
    """
    Compare derived mirror transform to the -R proxy approach.

    This diagnostic helps understand the relationship between:
    1. The exact mirror I(-β,-α)
    2. The proxy I(+R, but with -R in formulas)

    Args:
        R, theta, ell1, ell2, polynomials: PRZZ parameters
        n_quad: Quadrature points

    Returns:
        MirrorComparisonResult with comparison metrics
    """
    from src.unified_i1_general import compute_I1_unified_general

    # Direct at +R
    direct_result = compute_I1_unified_general(
        R=R, theta=theta, ell1=ell1, ell2=ell2,
        polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
    )

    # Derived mirror
    mirror_result = compute_I1_mirror_derived(
        R=R, theta=theta, ell1=ell1, ell2=ell2,
        polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
    )

    # Proxy using -R (existing approach)
    proxy_result = compute_I1_unified_general(
        R=-R, theta=theta, ell1=ell1, ell2=ell2,
        polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
    )

    # Effective multiplier
    m_eff = None
    if abs(proxy_result.I1_value) > 1e-15:
        m_eff = mirror_result.I1_mirror_value / proxy_result.I1_value

    # Empirical reference (K=3)
    m_empirical = math.exp(R) + 5

    return MirrorComparisonResult(
        ell1=ell1,
        ell2=ell2,
        I1_direct_plusR=direct_result.I1_value,
        I1_mirror_derived=mirror_result.I1_mirror_value,
        I1_proxy_minusR=proxy_result.I1_value,
        m_eff=m_eff,
        m_empirical=m_empirical,
    )
