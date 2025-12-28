"""
src/mirror_transform/microcase_11_pq1.py
Phase 31C.2: Microcase Mirror Transform for (1,1) with P=Q=1

This is the minimal test case for operator-level mirror transforms.
By setting P=Q=1, we isolate:
1. The bracket identity structure
2. The operator swap/sign behavior (chain rule)
3. Any missing factors (2's, θ's, normalizations)

If this microcase can't reproduce the scalar identity, it never will
at full K=3 with actual polynomials.

MICROCASE SETUP:
===============
- Pair: (ℓ₁,ℓ₂) = (1,1)
- P₁ = 1 (constant polynomial)
- Q = 1 (constant polynomial)
- Only I₁ term (d²/dαdβ structure)

THE KEY IDENTITY:
================
For the microcase, the integrand should satisfy:

    I_total = I_direct(α,β) + T^{-(α+β)} · I_mirror(-β,-α)

At α=β=-R:
    I_direct = I(-R,-R)
    I_mirror = exp(2R) · I(R,R) with swap/chain rule

The ratio I_mirror/I_direct should give insight into m.

Created: 2025-12-26 (Phase 31C)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.mirror_transform.spec import (
    MirrorTransformPieces,
    SwapTransformSpec,
    ChainRuleResult,
)


@dataclass
class MicrocaseResult:
    """Result of microcase evaluation."""

    # Direct term at α=β=-R
    direct_value: float

    # Mirror term (with swap and prefactor)
    mirror_value: float
    swap_prefactor: float  # T^{-(α+β)} = exp(2R)

    # Total
    total_value: float

    # Ratio (diagnostic)
    mirror_to_direct_ratio: float

    # What m would be if mirror = m × direct
    effective_m: float

    # Metadata
    R: float
    theta: float


def _compute_bracket_integrand_pq1(
    u: float,
    t: float,
    R: float,
    theta: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Compute the bracket integrand for P=Q=1 microcase.

    For P=Q=1, the integrand simplifies to:
        exp(αu + βu) × log_factor × algebraic_prefactor

    where log_factor comes from ζ'/ζ expansion.

    For the (1,1) term with d²/dαdβ, we extract the xy coefficient
    from the Taylor series.

    Args:
        u: Integration variable in [0,1]
        t: Integration variable in [0,1] (from log factor)
        R: The R parameter
        theta: The θ parameter
        alpha, beta: Evaluation point

    Returns:
        The integrand value
    """
    # For P=Q=1 microcase, the log-free part is just:
    # ∫∫ exp(αu + βu) du dt

    # But we need the xy coefficient from the Taylor expansion.
    # The exponential exp(αx + βy) evaluated at x=y=u gives:
    # exp(αu + βu) = exp(u(α+β))

    # The xy coefficient in exp(αx + βy) is:
    # ∂²/∂x∂y[exp(αx + βy)] = αβ exp(αx + βy)

    # At x=y=0 (for Taylor coefficient): αβ
    # But integrated over u with the exponential factor...

    # For simplicity, use the unified bracket approach:
    # The difference quotient identity already combines direct+mirror

    exp_factor = math.exp((alpha + beta) * u)

    # Log factor from ζ'/ζ structure
    # At leading order: log(N^{x+y}T) → (1/θ + x + y) log T → ~(1/θ)
    # For microcase, this is just (1/θ)
    log_factor = 1 / theta

    # Algebraic prefactor for (1,1): just 1
    alg_prefactor = 1.0

    # The xy coefficient from the series is αβ times the integral
    # (This is a simplification - full version needs series expansion)

    return exp_factor * log_factor * alg_prefactor


def compute_microcase_direct(
    R: float,
    theta: float = 4 / 7,
    n_quad: int = 40,
) -> float:
    """
    Compute the direct term I(-R,-R) for microcase.

    This is the simpler of the two terms - no swap needed.
    """
    alpha = -R
    beta = -R

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    total = 0.0

    for i, u in enumerate(u_nodes):
        for j, t in enumerate(t_nodes):
            integrand = _compute_bracket_integrand_pq1(
                u, t, R, theta, alpha, beta
            )

            # For (1,1) d²/dαdβ structure, multiply by αβ
            # (This extracts the xy coefficient)
            xy_coeff = alpha * beta

            total += xy_coeff * integrand * u_weights[i] * t_weights[j]

    return total


def compute_microcase_mirror(
    R: float,
    theta: float = 4 / 7,
    n_quad: int = 40,
) -> Tuple[float, float]:
    """
    Compute the mirror term T^{-(α+β)} · I(-β,-α) for microcase.

    This involves:
    1. Swap: (α,β) = (-R,-R) → (-β,-α) = (R,R)
    2. Prefactor: T^{-(α+β)} = T^{2R} = exp(2R)
    3. Chain rule: d²/dαdβ → (+1)·d²/dβdα = d²/dαdβ (signs cancel)

    Returns:
        (mirror_value, prefactor)
    """
    spec = SwapTransformSpec.at_przz_point(R, theta)
    chain = ChainRuleResult.for_I1()

    # Swapped evaluation point
    alpha_swapped = spec.alpha_swapped  # = R
    beta_swapped = spec.beta_swapped    # = R

    # Prefactor from T^{-(α+β)}
    prefactor = spec.prefactor  # = exp(2R)

    # Chain rule sign for d²/dαdβ
    chain_sign = chain.sign  # = +1 (signs cancel)

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    integral = 0.0

    for i, u in enumerate(u_nodes):
        for j, t in enumerate(t_nodes):
            # Evaluate at SWAPPED point
            integrand = _compute_bracket_integrand_pq1(
                u, t, R, theta, alpha_swapped, beta_swapped
            )

            # xy coefficient at swapped point
            # After swap, ∂²/∂α∂β becomes ∂²/∂β∂α with sign from chain rule
            # For symmetric case α_swapped = β_swapped = R:
            xy_coeff = alpha_swapped * beta_swapped  # = R²

            integral += chain_sign * xy_coeff * integrand * u_weights[i] * t_weights[j]

    # Apply prefactor
    mirror_value = prefactor * integral

    return mirror_value, prefactor


def compute_microcase_total(
    R: float,
    theta: float = 4 / 7,
    n_quad: int = 40,
) -> MicrocaseResult:
    """
    Compute total mirror transform for (1,1) microcase with P=Q=1.

    This is the key function: if it can reproduce the scalar identity,
    we understand the mirror structure.
    """
    # Direct term
    direct = compute_microcase_direct(R, theta, n_quad)

    # Mirror term with prefactor
    mirror, prefactor = compute_microcase_mirror(R, theta, n_quad)

    # Total
    total = direct + mirror

    # Ratio
    ratio = mirror / direct if abs(direct) > 1e-15 else float('inf')

    # Effective m (what scalar would make mirror = m × |direct|)
    # Note: mirror and direct may have same or opposite signs
    m_eff = abs(mirror / direct) if abs(direct) > 1e-15 else float('inf')

    return MicrocaseResult(
        direct_value=direct,
        mirror_value=mirror,
        swap_prefactor=prefactor,
        total_value=total,
        mirror_to_direct_ratio=ratio,
        effective_m=m_eff,
        R=R,
        theta=theta,
    )


def analyze_microcase_at_przz_points():
    """Analyze microcase at both PRZZ benchmark points."""
    print("=" * 60)
    print("MICROCASE MIRROR ANALYSIS: (1,1) with P=Q=1")
    print("=" * 60)

    for name, R in [("kappa", 1.3036), ("kappa_star", 1.1167)]:
        result = compute_microcase_total(R)

        print(f"\n--- {name.upper()} (R={R}) ---")
        print(f"Direct I(-R,-R):       {result.direct_value:.6f}")
        print(f"Prefactor exp(2R):     {result.swap_prefactor:.6f}")
        print(f"Mirror (with swap):    {result.mirror_value:.6f}")
        print(f"Total:                 {result.total_value:.6f}")
        print(f"Mirror/Direct ratio:   {result.mirror_to_direct_ratio:.6f}")
        print(f"Effective m:           {result.effective_m:.6f}")
        print(f"Target m (exp(R)+5):   {math.exp(R) + 5:.6f}")

        m_gap_pct = 100 * (result.effective_m - (math.exp(R) + 5)) / (math.exp(R) + 5)
        print(f"m gap:                 {m_gap_pct:+.2f}%")


if __name__ == "__main__":
    analyze_microcase_at_przz_points()
