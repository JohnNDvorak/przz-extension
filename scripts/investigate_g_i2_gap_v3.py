"""
scripts/investigate_g_i2_gap_v3.py
Phase 46.3: Final Investigation - Where is Q Attenuation Absorbed?

FINDINGS SO FAR:
================
1. Q has MASSIVE asymmetric attenuation on I2:
   - I2(+R) drops by ~85%
   - I2(-R) drops by ~43%
   - Ratio changes by ~74%

2. But g_I2 calibration gap is only 0.58%

3. This means Q attenuation is already mostly absorbed somewhere

HYPOTHESIS:
===========
The current formula ALREADY includes Q in the I2 computation!
So comparing "real Q" to "Q=1" is not the right comparison.

The RIGHT comparison is:
- What g_baseline predicts based on Beta moment (which assumes Q=1)
- What g_I2 actually needs when Q is included

The Beta moment formula assumes:
    I2 = (1/θ) × ∫∫ exp(2Rt) × P_ℓ₁(u) × P_ℓ₂(u) du dt

But the actual formula includes Q:
    I2 = (1/θ) × ∫∫ exp(2Rt) × P_ℓ₁(u) × P_ℓ₂(u) × Q(t)² du dt

The u-moment under the Q-attenuated kernel is DIFFERENT from the Beta moment.

Let's measure:
1. The u-moment with Q=1: ∫ u×kernel / ∫ kernel
2. The u-moment with real Q: ∫ u×kernel×Q² / ∫ kernel×Q²
3. See if the ratio explains the 0.58% gap

Created: 2025-12-27 (Phase 46.3)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict
import math

from src.polynomials import load_przz_polynomials, Polynomial
from src.quadrature import gauss_legendre_01


@dataclass
class UMomentResult:
    """Result of u-moment measurement under I2 kernel."""

    # U-moments (average u-value under the kernel)
    u_moment_Q1: float        # With Q=1
    u_moment_real: float      # With real Q

    # Kernel integrals (normalization)
    kernel_integral_Q1: float
    kernel_integral_real: float

    # Weighted u integrals
    u_kernel_integral_Q1: float
    u_kernel_integral_real: float

    # Derived
    @property
    def u_moment_ratio(self) -> float:
        """Ratio of u-moments: real/Q1."""
        if abs(self.u_moment_Q1) < 1e-15:
            return float('nan')
        return self.u_moment_real / self.u_moment_Q1

    @property
    def u_moment_shift_pct(self) -> float:
        """How much Q shifts the u-moment (percentage)."""
        return (self.u_moment_ratio - 1.0) * 100


def measure_u_moment_for_I2_kernel(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    include_Q: bool = True,
) -> UMomentResult:
    """
    Measure the u-moment under the I2 kernel.

    I2 kernel = exp(2Rt) × Σ w(ℓ₁,ℓ₂) × P_ℓ₁(u) × P_ℓ₂(u) × Q(t)²

    We compute:
        u_moment = ∫∫ u × kernel du dt / ∫∫ kernel du dt

    Args:
        R: R parameter
        theta: theta parameter
        polynomials: Dict with P1, P2, P3, Q
        n_quad: Quadrature points
        include_Q: Whether to include Q factor

    Returns:
        UMomentResult with u-moments and integrals
    """
    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials.get("Q")

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Pair weights for I2
    pairs = [
        (1, 1, 1.0),  # (ell1, ell2, symmetry)
        (2, 2, 1.0),
        (3, 3, 1.0),
        (1, 2, 2.0),
        (1, 3, 2.0),
        (2, 3, 2.0),
    ]

    factorial = [1, 1, 2, 6]
    polys = [None, P1, P2, P3]

    kernel_integral = 0.0
    u_kernel_integral = 0.0

    for ell1, ell2, symmetry in pairs:
        fact_norm = 1.0 / (factorial[ell1] * factorial[ell2])
        weight = symmetry * fact_norm

        for u, u_w in zip(u_nodes, u_weights):
            # Evaluate P_ℓ₁(u) and P_ℓ₂(u)
            P1_u = polys[ell1].eval(np.array([u]))[0]
            P2_u = polys[ell2].eval(np.array([u]))[0]
            P_product = P1_u * P2_u

            for t, t_w in zip(t_nodes, t_weights):
                # exp(2Rt)
                exp_val = math.exp(2 * R * t)

                # Q(t)²
                if include_Q and Q is not None:
                    Q_val = Q.eval(np.array([t]))[0]
                    Q_sq = Q_val ** 2
                else:
                    Q_sq = 1.0

                # Full kernel
                kernel = exp_val * P_product * Q_sq

                # Accumulate
                kernel_integral += weight * kernel * u_w * t_w
                u_kernel_integral += weight * u * kernel * u_w * t_w

    # Compute u-moment
    if abs(kernel_integral) > 1e-15:
        u_moment = u_kernel_integral / kernel_integral
    else:
        u_moment = float('nan')

    return UMomentResult(
        u_moment_Q1=u_moment if not include_Q else float('nan'),
        u_moment_real=u_moment if include_Q else float('nan'),
        kernel_integral_Q1=kernel_integral if not include_Q else float('nan'),
        kernel_integral_real=kernel_integral if include_Q else float('nan'),
        u_kernel_integral_Q1=u_kernel_integral if not include_Q else float('nan'),
        u_kernel_integral_real=u_kernel_integral if include_Q else float('nan'),
    )


def run_u_moment_investigation(
    R: float = 1.3036,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> None:
    """
    Run u-moment investigation to understand g_I2 gap.

    Args:
        R: R parameter
        theta: theta parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points
    """
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_real = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    Q1 = Polynomial(coeffs=[1.0])
    polynomials_Q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q1}

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    g_I2_calibrated = 1.01945154
    gap_pct = (g_I2_calibrated / g_baseline - 1) * 100

    # Expected Beta moment
    # For I2, the Beta moment should be: B(ℓ₁+ℓ₂+1, 1) = 1/(ℓ₁+ℓ₂+2)
    # But with weighted sum over pairs, the effective moment is different
    # The u-moment formula is: u_avg = 1/(ℓ+2) where ℓ = ℓ₁+ℓ₂
    # For the weighted sum, we need to compute the effective average

    print("=" * 80)
    print("PHASE 46.3: U-MOMENT INVESTIGATION FOR g_I2 GAP")
    print("=" * 80)
    print()
    print(f"Parameters: R={R}, θ={theta:.10f}, K={K}")
    print()
    print(f"g_baseline:      {g_baseline:.8f}")
    print(f"g_I2_calibrated: {g_I2_calibrated:.8f}")
    print(f"Gap:             {gap_pct:+.4f}%")
    print()

    print("CONTEXT:")
    print("--------")
    print("The g_baseline formula assumes Beta moment: B(ℓ₁+ℓ₂+1, 1) = 1/(ℓ₁+ℓ₂+2)")
    print("This comes from ∫₀¹ u × u^(ℓ₁+ℓ₂) du = 1/(ℓ₁+ℓ₂+2)")
    print()
    print("But when Q(t)² is included, the u-distribution changes:")
    print("  ∫∫ u × exp(2Rt) × P_ℓ₁(u) × P_ℓ₂(u) × Q(t)² du dt")
    print()
    print("The Q factor reweights the t-integration, which may shift the")
    print("effective u-moment and thus require a different g correction.")
    print()

    # Measure u-moments
    result_Q1 = measure_u_moment_for_I2_kernel(R, theta, polynomials_Q1, n_quad, include_Q=False)
    result_real = measure_u_moment_for_I2_kernel(R, theta, polynomials_real, n_quad, include_Q=True)

    print("=" * 80)
    print("U-MOMENT MEASUREMENTS")
    print("=" * 80)
    print()
    print(f"With Q=1:")
    print(f"  Kernel integral:     {result_Q1.kernel_integral_Q1:.8f}")
    print(f"  u×Kernel integral:   {result_Q1.u_kernel_integral_Q1:.8f}")
    print(f"  u-moment:            {result_Q1.u_moment_Q1:.8f}")
    print()
    print(f"With real Q:")
    print(f"  Kernel integral:     {result_real.kernel_integral_real:.8f}")
    print(f"  u×Kernel integral:   {result_real.u_kernel_integral_real:.8f}")
    print(f"  u-moment:            {result_real.u_moment_real:.8f}")
    print()

    # Compute the shift
    u_moment_shift = result_real.u_moment_real - result_Q1.u_moment_Q1
    u_moment_shift_pct = (result_real.u_moment_real / result_Q1.u_moment_Q1 - 1) * 100

    print(f"U-moment shift: {u_moment_shift:+.8f} ({u_moment_shift_pct:+.4f}%)")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # The g_baseline formula is: g = 1 + θ/(2K(2K+1))
    # This comes from the Beta moment correction

    # If Q shifts the u-moment, it might shift the required g

    # The Beta moment for pair (ℓ₁, ℓ₂) is: B(ℓ₁+ℓ₂+1, 1) = 1/(ℓ₁+ℓ₂+2)
    # The g formula accounts for this via the 1/(2K(2K+1)) term

    # If the u-moment shifts from u_Q1 to u_real, the effective correction shifts

    print(f"1. g_baseline assumes u-moment = {result_Q1.u_moment_Q1:.8f} (with Q=1)")
    print(f"2. Actual u-moment with Q = {result_real.u_moment_real:.8f}")
    print(f"3. Shift: {u_moment_shift_pct:+.4f}%")
    print()
    print(f"4. g_I2 calibration gap: {gap_pct:+.4f}%")
    print()

    # Check correlation
    if abs(u_moment_shift_pct) > 0.01:
        correlation = u_moment_shift_pct / gap_pct if abs(gap_pct) > 1e-6 else float('nan')
        print(f"Correlation: {correlation:.2f}x")
        print()

        if abs(correlation - 1.0) < 0.2:
            print("STRONG CORRELATION!")
            print("The u-moment shift explains the g_I2 calibration gap!")
            print()
            print("INTERPRETATION:")
            print("---------------")
            print("The Q polynomial reweights the t-integration in I2, which shifts")
            print("the effective u-moment. This shifts the required g correction.")
            print()
            print("Formula:")
            print(f"  g_I2 = g_baseline × (u_real / u_Q1)")
            print(f"       = {g_baseline:.8f} × {result_real.u_moment_real / result_Q1.u_moment_Q1:.8f}")
            print(f"       = {g_baseline * result_real.u_moment_real / result_Q1.u_moment_Q1:.8f}")
            print()
            print(f"  vs calibrated g_I2 = {g_I2_calibrated:.8f}")
        else:
            print("The u-moment shift doesn't fully explain the calibration gap.")
            print()
            if abs(correlation) < 0.1:
                print("Very low correlation - different mechanism at play.")
            elif correlation < 0:
                print("NEGATIVE correlation - u-moment shift is in wrong direction!")
            else:
                print("Partial correlation - u-moment is one factor among several.")
    else:
        print("U-moment shift is negligible (<0.01%).")
        print("This is NOT the mechanism for the g_I2 gap.")

    print()

    # Additional diagnostic: look at per-R behavior
    print("=" * 80)
    print("MECHANISM IDENTIFICATION")
    print("=" * 80)
    print()
    print("To identify the mechanism, we need to understand:")
    print()
    print("A. Does Q shift the u-moment? (measured above)")
    print("B. Does Q change HOW I2(-R) relates to I2(+R)?")
    print("C. Is there a Q×mirror interaction in the formula assembly?")
    print()

    # From v2, we know Q attenuates I2(+R) and I2(-R) differently
    # This changes the ratio by ~74%

    # But the mirror formula is: c = I12(+R) + m × I12(-R)
    # where m = g × base

    # If Q attenuates both sides, it might cancel out in the ratio

    # Let's think about it differently:
    # The g correction is supposed to account for the Beta moment
    # But with Q, the effective Beta moment changes

    # The Beta moment is: ∫ u × u^ℓ du = 1/(ℓ+2)
    # With Q: ∫∫ u × u^ℓ × Q(t)² × exp(2Rt) dt du / ∫∫ u^ℓ × Q(t)² × exp(2Rt) dt du

    # This is NOT just 1/(ℓ+2) anymore!

    print("The g_baseline formula assumes:")
    print(f"  ∫ u × P_ℓ₁(u) × P_ℓ₂(u) du / ∫ P_ℓ₁(u) × P_ℓ₂(u) du")
    print()
    print("But I2 integrand has Q(t)² factor:")
    print(f"  ∫∫ u × P_ℓ₁(u) × P_ℓ₂(u) × Q(t)² × exp(2Rt) du dt")
    print(f"  / ∫∫ P_ℓ₁(u) × P_ℓ₂(u) × Q(t)² × exp(2Rt) du dt")
    print()
    print("If Q(t)² is not constant, it changes the weighting of different u regions.")
    print()

    # Since exp(2Rt) varies with t, and Q(t)² varies with t,
    # different t-values contribute differently to the u-moment

    # At t near 0: exp(2Rt) ≈ 1, Q(t) ≈ 1
    # At t near 1: exp(2Rt) = exp(2R), Q(t) = Q(1)

    # If Q(1) < 1, then high-t contributions are suppressed
    # This shifts the effective t-distribution toward t=0

    # But how does this affect the u-moment?

    # The polynomials P_ℓ(u) have u-dependence
    # If certain t-regions are weighted differently, and those regions
    # couple differently to u via the integrand structure, then the
    # effective u-moment changes

    print("HYPOTHESIS:")
    print("-----------")
    print("Q(t)² reweights the t-integration, suppressing high-t contributions.")
    print("This changes the effective u-distribution under the I2 kernel.")
    print()
    print(f"Measured effect: u-moment shifts by {u_moment_shift_pct:+.4f}%")
    print(f"Required effect: g must shift by {gap_pct:+.4f}%")
    print()

    if abs(u_moment_shift_pct - gap_pct) < 0.2:
        print("MATCH! The u-moment shift explains the g_I2 gap.")
    else:
        print("NO MATCH. The mechanism is more subtle.")


if __name__ == "__main__":
    print()
    print("BENCHMARK 1: κ case (R=1.3036)")
    print()
    run_u_moment_investigation(R=1.3036, theta=4/7, K=3, n_quad=60)
    print()
    print()
    print("=" * 80)
    print("=" * 80)
    print()
    print("BENCHMARK 2: κ* case (R=1.1167)")
    print()
    run_u_moment_investigation(R=1.1167, theta=4/7, K=3, n_quad=60)
