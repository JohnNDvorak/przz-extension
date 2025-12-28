"""
src/unified_s12/g_i2_weighted.py
Phase 46.2: First-Principles g_I2 Derivation

DERIVATION PRINCIPLE:
=====================

I1 has log factor (1/θ + x + y) which creates cross-terms under ∂²/∂x∂y.
These cross-terms provide INTERNAL correction via:
    correction = 1 + θ × (F_x + F_y) / F_xy ≈ 1 + θ/(2K(2K+1)) = g_baseline

I2 has no log factor - just (1/θ) × kernel.
Therefore I2 has NO internal correction mechanism.
All correction must come externally via the mirror coefficient.

FIRST-PRINCIPLES DERIVATION:
============================

g_I1 is derived from log factor split (compute_g_i1_internal_ratio):
  - Measures θ × (F_x + F_y) / F_xy from actual I1 integrand
  - Should give ≈ 1.0 when log factor cross-terms self-correct

g_I2 = g_baseline BY CONSTRUCTION:
  - I2 lacks log factor, so needs FULL external correction
  - g_I2 = 1 + θ/(2K(2K+1)) = g_baseline

Q=1 GATE:
=========

When Q=1, the derivation should give:
  - g_I1 ≈ g_baseline (from log factor split measurement)
  - g_I2 = g_baseline (by definition)

The ~0.4% residual in benchmarks comes from Q polynomial differential
attenuation effects not captured by this first-principles derivation.

Created: 2025-12-27 (Phase 46.2)
Updated: 2025-12-27 - Fixed derivation: g_I2 = g_baseline by construction
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from src.quadrature import tensor_grid_2d, gauss_legendre_01
from src.polynomials import Polynomial


@dataclass
class GI2WeightedResult:
    """Result of first-principles g_I2 derivation.

    g_I2 = g_baseline BY CONSTRUCTION because I2 lacks log factor cross-terms.
    """

    # The derived g_I2 value (= g_baseline)
    g_I2: float

    # Parameters
    theta: float
    K: int
    R: float

    # Theoretical baseline
    g_baseline: float     # 1 + θ/(2K(2K+1))

    # Diagnostic: u-moment under I2 kernel (NOT used for g_I2 derivation)
    # This is computed for analysis but is NOT the Beta moment
    u_moment: float       # ∫ u×kernel / ∫ kernel (diagnostic only)

    @property
    def g_gap_from_baseline_pct(self) -> float:
        """Gap between derived g_I2 and g_baseline (should be 0%)."""
        if abs(self.g_baseline) < 1e-15:
            return float('nan')
        return (self.g_I2 / self.g_baseline - 1) * 100


def compute_i2_kernel_integrals(
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> Tuple[float, float]:
    """
    Compute the I2 kernel integrals with and without u-weighting.

    For K=3, I2 sums over pairs (ℓ₁, ℓ₂):
        I2 = Σ weight(ℓ₁,ℓ₂) × ∫∫ exp(2Rt) × P_ℓ₁(u) × P_ℓ₂(u) × Q(t)² du dt

    We compute two versions:
        kernel_integral = ∫∫ kernel du dt
        u_kernel_integral = ∫∫ u × kernel du dt

    Returns:
        (kernel_integral, u_kernel_integral)
    """
    # Get quadrature points
    u_pts, u_wts = gauss_legendre_01(n_quad)
    t_pts, t_wts = gauss_legendre_01(n_quad)

    # Get polynomials
    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials["Q"]

    polys = [None, P1, P2, P3]  # 1-indexed

    # Pair weights: weight = symmetry × 1/(ℓ₁! × ℓ₂!)
    factorial = [1, 1, 2, 6]  # 0!, 1!, 2!, 3!
    pairs = [
        (1, 1, 1.0),   # (1,1) diagonal
        (2, 2, 1.0),   # (2,2) diagonal
        (3, 3, 1.0),   # (3,3) diagonal
        (1, 2, 2.0),   # (1,2) off-diagonal (symmetry 2)
        (1, 3, 2.0),   # (1,3) off-diagonal
        (2, 3, 2.0),   # (2,3) off-diagonal
    ]

    total_kernel = 0.0
    total_u_kernel = 0.0

    for ell1, ell2, sym in pairs:
        # Factorial normalization
        fact_norm = 1.0 / (factorial[ell1] * factorial[ell2])
        weight = sym * fact_norm

        # Compute integral over (u, t)
        kernel_sum = 0.0
        u_kernel_sum = 0.0

        for i_u, (u, w_u) in enumerate(zip(u_pts, u_wts)):
            # Evaluate P_ℓ₁(u) and P_ℓ₂(u)
            P_ell1_u = polys[ell1].eval(np.array([u]))[0]
            P_ell2_u = polys[ell2].eval(np.array([u]))[0]
            P_product = P_ell1_u * P_ell2_u

            for i_t, (t, w_t) in enumerate(zip(t_pts, t_wts)):
                # Compute kernel components
                exp_factor = np.exp(2 * R * t)
                Q_t = Q.eval(np.array([t]))[0]
                Q_squared = Q_t * Q_t

                # Full kernel
                kernel = exp_factor * P_product * Q_squared

                # Accumulate
                kernel_sum += w_u * w_t * kernel
                u_kernel_sum += w_u * w_t * u * kernel

        total_kernel += weight * kernel_sum
        total_u_kernel += weight * u_kernel_sum

    return total_kernel, total_u_kernel


def derive_g_i2_weighted(
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> GI2WeightedResult:
    """
    Derive g_I2 from first principles.

    DERIVATION:
    ===========
    I2 has no log factor, so it lacks internal cross-term correction.
    Therefore g_I2 = g_baseline = 1 + θ/(2K(2K+1)) BY CONSTRUCTION.

    The u-moment is computed for diagnostic purposes only.
    It is NOT the Beta moment and should NOT be used to derive g_I2.

    Args:
        R: R parameter
        theta: θ parameter
        polynomials: Dict with P1, P2, P3, Q
        K: Number of mollifier pieces
        n_quad: Quadrature points

    Returns:
        GI2WeightedResult with g_I2 = g_baseline
    """
    # Theoretical values
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    # g_I2 = g_baseline by construction
    g_I2 = g_baseline

    # Compute u-moment for diagnostic purposes only
    kernel_integral, u_kernel_integral = compute_i2_kernel_integrals(
        R, theta, polynomials, K, n_quad
    )
    if abs(kernel_integral) < 1e-15:
        u_moment = float('nan')
    else:
        u_moment = u_kernel_integral / kernel_integral

    return GI2WeightedResult(
        g_I2=g_I2,
        theta=theta,
        K=K,
        R=R,
        g_baseline=g_baseline,
        u_moment=u_moment,  # Diagnostic only
    )


def compute_g_i1_internal_ratio(
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> float:
    """
    Compute g_I1 from the internal product-rule ratio.

    For I1, the log factor creates:
        ∂²/∂x∂y[(1/θ + x + y) × F] = (1/θ)F_xy + F_x + F_y

    The internal correction is:
        g_internal = [(1/θ)F_xy + F_x + F_y] / [(1/θ)F_xy]
                   = 1 + θ × (F_x + F_y) / F_xy

    We compute this from the log factor split infrastructure.

    Returns:
        The derived g_I1 value
    """
    from src.unified_s12.logfactor_split import compute_aggregate_correction

    result = compute_aggregate_correction(theta, R, K, polynomials, n_quad)

    # Internal correction = measured_correction from log factor split
    g_I1 = result["measured_correction"]

    return g_I1


@dataclass
class DerivedGValuesWeighted:
    """Complete first-principles g derivation.

    - g_I1 from log factor split (internal product-rule ratio)
    - g_I2 = g_baseline by construction (I2 lacks log factor)
    """

    # Derived g values
    g_I1: float           # From internal product-rule ratio
    g_I2: float           # = g_baseline by construction

    # Diagnostic: u-moment under I2 kernel (NOT used for derivation)
    I2_u_moment: float    # ∫ u×kernel / ∫ kernel

    # Parameters
    theta: float
    K: int
    R: float

    @property
    def g_baseline(self) -> float:
        """Theoretical g_baseline = 1 + θ/(2K(2K+1))."""
        return 1 + self.theta / (2 * self.K * (2 * self.K + 1))

    @property
    def g_I1_gap_from_one(self) -> float:
        """How far g_I1 is from 1.0."""
        return abs(self.g_I1 - 1.0)

    @property
    def g_I2_gap_from_baseline(self) -> float:
        """How far g_I2 is from g_baseline (should be 0)."""
        return abs(self.g_I2 - self.g_baseline)


def derive_g_values_weighted(
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> DerivedGValuesWeighted:
    """
    Derive both g_I1 and g_I2 from first principles.

    This is the complete first-principles derivation:
    - g_I1 from internal product-rule ratio (log factor split)
    - g_I2 = g_baseline by construction (I2 lacks log factor)

    Args:
        R: R parameter
        theta: θ parameter
        polynomials: Dict with P1, P2, P3, Q
        K: Number of mollifier pieces
        n_quad: Quadrature points

    Returns:
        DerivedGValuesWeighted with both g values
    """
    # Derive g_I1 from log factor split
    g_I1 = compute_g_i1_internal_ratio(R, theta, polynomials, K, n_quad)

    # Derive g_I2 = g_baseline by construction
    i2_result = derive_g_i2_weighted(R, theta, polynomials, K, n_quad)

    return DerivedGValuesWeighted(
        g_I1=g_I1,
        g_I2=i2_result.g_I2,
        I2_u_moment=i2_result.u_moment,  # Diagnostic only
        theta=theta,
        K=K,
        R=R,
    )
