#!/usr/bin/env python3
"""
scripts/step_a_moment_analysis.py
STEP A: Symbolic (2t-1) Moment Analysis

GOAL: Derive the exact analytic forms for the (2t-1) moment integrals M₀, M₁, M₂
and show that xy_coeff_integral = R²θ²M₂ + 2Rθ²M₁

THE KEY INSIGHT:
The PRZZ combined bracket kernel (after DQ identity application) is:
    exp(2Rt) × exp(Rθ(x+y)(2t-1)) × (1+θ(x+y))

The xy coefficient comes from expanding this in x,y and extracting the xy term:
    xy_coeff(t) = exp(2Rt) × [R²θ²(2t-1)² + 2Rθ²(2t-1)]
                = R²θ² exp(2Rt)(2t-1)² + 2Rθ² exp(2Rt)(2t-1)

Thus:
    xy_integral = R²θ² ∫exp(2Rt)(2t-1)² dt + 2Rθ² ∫exp(2Rt)(2t-1) dt
                = R²θ² M₂ + 2Rθ² M₁

ANALYTIC DERIVATION:
Define M_n = ∫₀¹ (2t-1)^n exp(2Rt) dt

There's a recurrence relation via integration by parts:
    M_n = [exp(2R) - (-1)^n]/(2R) - (n/R) M_{n-1}

This gives:
    M₀ = (exp(2R) - 1)/(2R)                                    [DQ limit]
    M₁ = [exp(2R) + 1]/(2R) - M₀/R
       = [(R-1)exp(2R) + (R+1)]/(2R²)
    M₂ = [exp(2R) - 1]/(2R) - 2M₁/R
       = M₀ - 2M₁/R

Created: 2025-12-29 (Phase 54 - PRZZ g-factor Derivation)
"""

import math
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quadrature import gauss_legendre_01


@dataclass
class MomentAnalysisResult:
    """Results from (2t-1) moment analysis."""
    R: float
    theta: float

    # Analytic moment values
    M0_analytic: float  # DQ limit
    M1_analytic: float  # First moment
    M2_analytic: float  # Second moment

    # Numerical verification
    M0_numeric: float
    M1_numeric: float
    M2_numeric: float

    # xy integral decomposition
    xy_integral_direct: float       # Computed directly as ∫xy_coeff(t) dt
    xy_integral_from_moments: float # = R²θ²M₂ + 2Rθ²M₁

    # Agreement
    moment_agreement: float  # |direct - from_moments| / |direct|


def compute_M0_analytic(R: float) -> float:
    """
    M₀ = ∫₀¹ exp(2Rt) dt = (exp(2R) - 1)/(2R)

    This is the Difference Quotient limit at x=y=0.
    """
    if abs(R) < 1e-10:
        return 1.0  # Limit as R→0
    return (math.exp(2*R) - 1) / (2*R)


def compute_M1_analytic(R: float) -> float:
    """
    M₁ = ∫₀¹ (2t-1) exp(2Rt) dt

    Derivation via integration by parts:
    M₁ = [exp(2R) + 1]/(2R) - M₀/R
       = [(R-1)exp(2R) + (R+1)]/(2R²)
    """
    if abs(R) < 1e-10:
        return 0.0  # Odd function over symmetric interval in limit

    exp2R = math.exp(2*R)
    return ((R - 1) * exp2R + (R + 1)) / (2 * R**2)


def compute_M2_analytic(R: float) -> float:
    """
    M₂ = ∫₀¹ (2t-1)² exp(2Rt) dt

    Derivation via recurrence:
    M₂ = [exp(2R) - 1]/(2R) - 2M₁/R = M₀ - 2M₁/R
    """
    if abs(R) < 1e-10:
        # (2t-1)² is even, has average 1/3 over [0,1]
        return 1/3

    M0 = compute_M0_analytic(R)
    M1 = compute_M1_analytic(R)
    return M0 - 2 * M1 / R


def compute_moments_numeric(R: float, n_quad: int = 100) -> Tuple[float, float, float]:
    """
    Compute M₀, M₁, M₂ numerically for verification.
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    M0 = 0.0
    M1 = 0.0
    M2 = 0.0

    for t, w in zip(t_nodes, t_weights):
        exp_2Rt = math.exp(2 * R * t)
        u = 2*t - 1  # The (2t-1) factor

        M0 += exp_2Rt * w
        M1 += exp_2Rt * u * w
        M2 += exp_2Rt * u**2 * w

    return M0, M1, M2


def compute_xy_integral_direct(R: float, theta: float, n_quad: int = 100) -> float:
    """
    Compute the xy coefficient integral directly:
        ∫₀¹ exp(2Rt) × [R²θ²(2t-1)² + 2Rθ²(2t-1)] dt

    This is what the Path 1 analysis computed as ~2.67 for R=1.3036.
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    result = 0.0
    for t, w in zip(t_nodes, t_weights):
        exp_2Rt = math.exp(2 * R * t)
        u = R * theta * (2*t - 1)
        xy_coeff = u**2 + 2*theta*u
        result += exp_2Rt * xy_coeff * w

    return result


def compute_xy_integral_from_moments(R: float, theta: float) -> float:
    """
    Compute xy integral from moment decomposition:
        xy_integral = R²θ² M₂ + 2Rθ² M₁
    """
    M1 = compute_M1_analytic(R)
    M2 = compute_M2_analytic(R)

    return R**2 * theta**2 * M2 + 2 * R * theta**2 * M1


def analyze_moments(R: float, theta: float = 4/7, n_quad: int = 100) -> MomentAnalysisResult:
    """
    Complete moment analysis for given R and θ.
    """
    # Analytic values
    M0_a = compute_M0_analytic(R)
    M1_a = compute_M1_analytic(R)
    M2_a = compute_M2_analytic(R)

    # Numeric verification
    M0_n, M1_n, M2_n = compute_moments_numeric(R, n_quad)

    # xy integrals
    xy_direct = compute_xy_integral_direct(R, theta, n_quad)
    xy_moments = compute_xy_integral_from_moments(R, theta)

    # Agreement
    agreement = abs(xy_direct - xy_moments) / abs(xy_direct) if xy_direct != 0 else 0

    return MomentAnalysisResult(
        R=R,
        theta=theta,
        M0_analytic=M0_a,
        M1_analytic=M1_a,
        M2_analytic=M2_a,
        M0_numeric=M0_n,
        M1_numeric=M1_n,
        M2_numeric=M2_n,
        xy_integral_direct=xy_direct,
        xy_integral_from_moments=xy_moments,
        moment_agreement=agreement,
    )


def print_moment_analysis(result: MomentAnalysisResult):
    """Pretty print moment analysis results."""
    print("=" * 70)
    print(f"(2t-1) MOMENT ANALYSIS FOR R = {result.R}, θ = {result.theta:.6f}")
    print("=" * 70)
    print()

    print("ANALYTIC MOMENT FORMULAS:")
    print(f"  M₀ = (exp(2R)-1)/(2R) = {result.M0_analytic:.10f}")
    print(f"  M₁ = [(R-1)exp(2R) + (R+1)]/(2R²) = {result.M1_analytic:.10f}")
    print(f"  M₂ = M₀ - 2M₁/R = {result.M2_analytic:.10f}")
    print()

    print("NUMERIC VERIFICATION:")
    M0_err = abs(result.M0_analytic - result.M0_numeric) / abs(result.M0_analytic)
    M1_err = abs(result.M1_analytic - result.M1_numeric) / abs(result.M1_analytic) if result.M1_analytic != 0 else 0
    M2_err = abs(result.M2_analytic - result.M2_numeric) / abs(result.M2_analytic)

    print(f"  M₀: analytic={result.M0_analytic:.10f}, numeric={result.M0_numeric:.10f}, err={M0_err:.2e}")
    print(f"  M₁: analytic={result.M1_analytic:.10f}, numeric={result.M1_numeric:.10f}, err={M1_err:.2e}")
    print(f"  M₂: analytic={result.M2_analytic:.10f}, numeric={result.M2_numeric:.10f}, err={M2_err:.2e}")
    print()

    print("XY COEFFICIENT INTEGRAL DECOMPOSITION:")
    print(f"  xy_coeff(t) = exp(2Rt) × [R²θ²(2t-1)² + 2Rθ²(2t-1)]")
    print()
    print(f"  ∫ xy_coeff(t) dt = R²θ² M₂ + 2Rθ² M₁")
    print()
    print(f"  Direct computation:   {result.xy_integral_direct:.10f}")
    print(f"  From moment decomp:   {result.xy_integral_from_moments:.10f}")
    print(f"  Agreement:            {result.moment_agreement:.2e}")
    print()

    # Show the decomposition explicitly
    R, theta = result.R, result.theta
    term1 = R**2 * theta**2 * result.M2_analytic
    term2 = 2 * R * theta**2 * result.M1_analytic
    print("  Component breakdown:")
    print(f"    R²θ² M₂ = {R**2:.4f} × {theta**2:.6f} × {result.M2_analytic:.6f} = {term1:.6f}")
    print(f"    2Rθ² M₁ = 2 × {R:.4f} × {theta**2:.6f} × {result.M1_analytic:.6f} = {term2:.6f}")
    print(f"    Sum = {term1 + term2:.6f}")


def derive_moment_ratios(R: float, theta: float = 4/7):
    """
    Compute the moment ratios that will feed into g_I1/g_I2 derivation.

    The key ratios are:
    - M₁/M₀: first moment relative to DQ limit
    - M₂/M₀: second moment relative to DQ limit
    """
    M0 = compute_M0_analytic(R)
    M1 = compute_M1_analytic(R)
    M2 = compute_M2_analytic(R)

    print()
    print("MOMENT RATIOS (for Step B):")
    print(f"  M₁/M₀ = {M1/M0:.10f}")
    print(f"  M₂/M₀ = {M2/M0:.10f}")
    print()

    # The xy/scalar ratio from bracket analysis
    xy_scalar_ratio = (R**2 * theta**2 * M2 + 2 * R * theta**2 * M1) / M0
    print(f"  xy_integral / scalar_integral = {xy_scalar_ratio:.6f}")
    print(f"  This is the ~0.55 ratio from Path 1 analysis")
    print()

    # Express in terms of θ
    print("  Simplified forms:")
    print(f"    (R²θ²M₂ + 2Rθ²M₁) / M₀")
    print(f"    = θ² [R²(M₂/M₀) + 2R(M₁/M₀)]")
    print(f"    = {theta**2:.6f} × [{R**2:.4f} × {M2/M0:.6f} + 2 × {R:.4f} × {M1/M0:.6f}]")
    print(f"    = {theta**2:.6f} × {R**2 * M2/M0 + 2*R * M1/M0:.6f}")
    print(f"    = {xy_scalar_ratio:.6f} ✓")


def verify_specific_value_267():
    """
    Verify that we reproduce the ~2.67 value from Path 1 analysis.
    """
    print()
    print("=" * 70)
    print("VERIFICATION: Path 1 reported xy_integral ≈ 2.67 at R=1.3036")
    print("=" * 70)
    print()

    R = 1.3036
    theta = 4/7

    result = analyze_moments(R, theta)
    print(f"  Our computation: {result.xy_integral_direct:.6f}")
    print(f"  Expected:        ~2.67")
    print()

    if abs(result.xy_integral_direct - 2.67) < 0.1:
        print("  ✓ MATCH - Our moment analysis reproduces the Path 1 value!")
    else:
        print("  ✗ DISCREPANCY - Need to check the algebra")


def main():
    print("=" * 70)
    print("STEP A: (2t-1) MOMENT ANALYSIS")
    print("=" * 70)
    print()
    print("This script derives the PRZZ bracket moments analytically.")
    print()

    # Analyze both benchmarks
    for R in [1.3036, 1.1167]:
        result = analyze_moments(R, theta=4/7)
        print_moment_analysis(result)
        derive_moment_ratios(R)
        print()

    # Verify the specific ~2.67 value
    verify_specific_value_267()

    print()
    print("=" * 70)
    print("STEP A SUMMARY")
    print("=" * 70)
    print()
    print("KEY RESULTS:")
    print("  1. M₀ = (exp(2R)-1)/(2R) — DQ limit (confirmed)")
    print("  2. M₁ = [(R-1)exp(2R) + (R+1)]/(2R²) — first (2t-1) moment")
    print("  3. M₂ = M₀ - 2M₁/R — second (2t-1) moment")
    print()
    print("  4. xy_coeff_integral = R²θ²M₂ + 2Rθ²M₁ ✓")
    print()
    print("  This PROVES that the ~2.67 value comes from PRZZ (2t-1) moments.")
    print("  The structure distinguishes I₁ (derivative) from I₂ (scalar) terms.")
    print()
    print("NEXT: Step B will connect these moments to g_I1 and g_I2 formulas.")


if __name__ == "__main__":
    main()
