#!/usr/bin/env python3
"""
Phase 42: M/C/g Decomposition Per Component

GPT's decomposition to identify the root cause of the ±0.15% residual.

For each component j ∈ {I1, I2} in S12(-R):

    L(x,y) = 1/θ + x + y                    (log factor)
    F_j(x,y) = F^{(j)}_{00} + F^{(j)}_{10}x + F^{(j)}_{01}y + F^{(j)}_{11}xy

    [xy] of L·F_j = M_j + C_j

    where:
        M_j = (1/θ) × F^{(j)}_{11}          (main term)
        C_j = F^{(j)}_{10} + F^{(j)}_{01}   (cross terms)

    g_j = (M_j + C_j) / M_j = 1 + C_j/M_j

    g_total = 1 + (C_1 + C_2) / (M_1 + M_2)

Created: 2025-12-27 (Phase 42)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)
from src.quadrature import gauss_legendre_01
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_P_factor,
)


@dataclass
class MCGResult:
    """M/C/g decomposition result for one component."""
    component: str  # "I1" or "I2"
    M: float        # Main term: (1/θ) × F_{11} integrated
    C: float        # Cross term: F_{10} + F_{01} integrated
    g: float        # = 1 + C/M


@dataclass
class MCGDecomposition:
    """Full M/C/g decomposition for a benchmark."""
    benchmark: str
    R: float
    theta: float

    I1: MCGResult
    I2: MCGResult

    M_total: float    # M_1 + M_2
    C_total: float    # C_1 + C_2
    g_total: float    # = 1 + (C_1 + C_2) / (M_1 + M_2)

    # Mixture weights
    M1_fraction: float  # M_1 / (M_1 + M_2)
    M2_fraction: float  # M_2 / (M_1 + M_2)

    # Expected baseline
    g_baseline: float   # = 1 + θ/(2K(2K+1)) for K=3


def compute_I1_M_and_C(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> Tuple[float, float]:
    """
    Compute M and C contributions for I1 at given (ell1, ell2) pair.

    M = integral of (1/θ) × [x^ell1 y^ell2 from F without log cross-terms]
    C = integral of [log cross-term contribution to x^ell1 y^ell2]

    Returns: (M, C)
    """
    from src.unified_i1_paper import build_P_factor_paper, omega_for_ell, _extract_poly_coeffs
    from src.series_bivariate import build_Q_factor

    max_dx = ell1
    max_dy = ell2

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    P_ell1 = polynomials[f"P{ell1}"]
    P_ell2 = polynomials[f"P{ell2}"]
    Q = polynomials["Q"]
    Q_coeffs = _extract_poly_coeffs(Q)

    one_minus_u_power = ell1 + ell2

    M_total = 0.0
    MC_total = 0.0  # M + C together

    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # Exp factor
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # P factors (paper regime)
            P_x = build_P_factor_paper(P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, 40)
            P_y = build_P_factor_paper(P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, 40)

            # Q factors
            Q_alpha = build_Q_factor(Q_coeffs, a0=t, ax=theta*(t-1), ay=theta*t, max_dx=max_dx, max_dy=max_dy)
            Q_beta = build_Q_factor(Q_coeffs, a0=t, ax=theta*t, ay=theta*(t-1), max_dx=max_dx, max_dy=max_dy)

            # F = exp * P_x * P_y * Q_alpha * Q_beta (without log factor)
            F = exp_factor * P_x * P_y * Q_alpha * Q_beta

            # M contribution: (1/θ) × F_{11}
            # We need the coefficient of x^ell1 y^ell2 in F
            F_ell1_ell2 = F.extract(ell1, ell2)
            M_contrib = (1.0 / theta) * F_ell1_ell2

            # M+C contribution: [x^ell1 y^ell2] in L × F
            # L = 1/θ + x + y
            # So [x^ell1 y^ell2] of L×F = (1/θ)×F_{ell1,ell2} + F_{ell1-1,ell2} + F_{ell1,ell2-1}
            F_ell1m1_ell2 = F.extract(ell1 - 1, ell2) if ell1 >= 1 else 0.0
            F_ell1_ell2m1 = F.extract(ell1, ell2 - 1) if ell2 >= 1 else 0.0
            MC_contrib = (1.0 / theta) * F_ell1_ell2 + F_ell1m1_ell2 + F_ell1_ell2m1

            weight = one_minus_u_factor * u_w * t_w
            M_total += M_contrib * weight
            MC_total += MC_contrib * weight

    # Apply factorial normalization
    factorial_norm = math.factorial(ell1) * math.factorial(ell2)
    M_total *= factorial_norm
    MC_total *= factorial_norm

    # Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        M_total *= sign
        MC_total *= sign

    # C = (M+C) - M
    C_total = MC_total - M_total

    return M_total, C_total


def compute_I2_M_and_C(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> Tuple[float, float]:
    """
    Compute M and C contributions for I2 at given (ell1, ell2) pair.

    I2 has no derivatives, so structure is simpler.
    For I2, there's no xy extraction - it's a direct integral.

    However, for the M/C decomposition, we're looking at the -R channel
    contributions to understand the mirror multiplier structure.

    For I2, the "log factor" contribution is different since there's no
    derivative extraction. The M and C decomposition applies to the full
    integrand structure.

    Returns: (M, C)
    """
    from src.unified_i2_paper import compute_I2_unified_paper

    # For I2, the M/C decomposition is less direct since there's no xy extraction.
    # The I2 integral is: ∫∫ exp(2Rt) × P_{ell1}(u) × P_{ell2}(u) × Q(t)² du dt
    #
    # There's no (1/θ + x + y) log factor in I2 - the log factor only appears in I1.
    # So for I2, we define M = full I2 value, C = 0.

    result = compute_I2_unified_paper(R, theta, ell1, ell2, polynomials, n_quad_u=n_quad, n_quad_t=n_quad, include_Q=True)

    # Factorial norm and symmetry factor applied in full S12 computation
    I2_value = result.I2_value

    # For I2: M = I2_value, C = 0 (no log factor cross-terms)
    return I2_value, 0.0


def compute_mcg_decomposition(
    benchmark: str,
    R: float,
    theta: float,
    polynomials: Dict,
    K: int = 3,
    n_quad: int = 60,
) -> MCGDecomposition:
    """
    Compute full M/C/g decomposition for a benchmark.
    """
    # Factorial and symmetry normalization
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    pairs = ["11", "22", "33", "12", "13", "23"]

    # Compute at -R (the mirror channel)
    R_minus = -R

    M1_total = 0.0
    C1_total = 0.0
    M2_total = 0.0
    C2_total = 0.0

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        norm = f_norm[pair_key] * symmetry[pair_key]

        # I1 contribution
        M1, C1 = compute_I1_M_and_C(R_minus, theta, ell1, ell2, polynomials, n_quad)
        M1_total += norm * M1
        C1_total += norm * C1

        # I2 contribution
        M2, C2 = compute_I2_M_and_C(R_minus, theta, ell1, ell2, polynomials, n_quad)
        M2_total += norm * M2
        C2_total += norm * C2

    # Compute g values
    eps = 1e-15
    g1 = 1.0 + C1_total / M1_total if abs(M1_total) > eps else float('inf')
    g2 = 1.0 + C2_total / M2_total if abs(M2_total) > eps else float('inf')

    M_total = M1_total + M2_total
    C_total = C1_total + C2_total
    g_total = 1.0 + C_total / M_total if abs(M_total) > eps else float('inf')

    # Mixture weights
    M1_fraction = M1_total / M_total if abs(M_total) > eps else 0.0
    M2_fraction = M2_total / M_total if abs(M_total) > eps else 0.0

    # Expected baseline
    g_baseline = 1.0 + theta / (2 * K * (2 * K + 1))

    return MCGDecomposition(
        benchmark=benchmark,
        R=R,
        theta=theta,
        I1=MCGResult(component="I1", M=M1_total, C=C1_total, g=g1),
        I2=MCGResult(component="I2", M=M2_total, C=C2_total, g=g2),
        M_total=M_total,
        C_total=C_total,
        g_total=g_total,
        M1_fraction=M1_fraction,
        M2_fraction=M2_fraction,
        g_baseline=g_baseline,
    )


def print_mcg_table(results: List[MCGDecomposition]) -> None:
    """Print M/C/g decomposition table."""
    print("=" * 90)
    print("PHASE 42: M/C/g DECOMPOSITION (GPT's Formula)")
    print("=" * 90)
    print()
    print("Computed at -R (mirror channel) to understand mirror multiplier structure")
    print()

    # Per-component breakdown
    print("I1 COMPONENT (derivative term, has log factor cross-terms)")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'M_1':<14} | {'C_1':<14} | {'g_1':<12} | {'g_baseline':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r.benchmark:<12} | {r.I1.M:<14.6f} | {r.I1.C:<14.6f} | {r.I1.g:<12.6f} | {r.g_baseline:<12.6f}")
    print()

    print("I2 COMPONENT (no derivatives, no log factor cross-terms)")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'M_2':<14} | {'C_2':<14} | {'g_2':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r.benchmark:<12} | {r.I2.M:<14.6f} | {r.I2.C:<14.6f} | {r.I2.g:<12.6f}")
    print()

    # Totals
    print("TOTALS AND MIXTURE WEIGHTS")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'M_total':<14} | {'C_total':<14} | {'g_total':<12} | {'M1%':<8} | {'M2%':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r.benchmark:<12} | {r.M_total:<14.6f} | {r.C_total:<14.6f} | {r.g_total:<12.6f} | {r.M1_fraction*100:<8.2f} | {r.M2_fraction*100:<8.2f}")
    print()

    # Comparison
    print("DEVIATION FROM BASELINE")
    print("-" * 90)
    print(f"{'Benchmark':<12} | {'g_total':<12} | {'g_baseline':<12} | {'delta_g':<12} | {'delta %':<10}")
    print("-" * 90)
    for r in results:
        delta_g = r.g_total - r.g_baseline
        delta_pct = (r.g_total / r.g_baseline - 1) * 100
        print(f"{r.benchmark:<12} | {r.g_total:<12.6f} | {r.g_baseline:<12.6f} | {delta_g:+12.6f} | {delta_pct:+10.4f}%")


def main():
    """Main entry point."""
    theta = 4 / 7
    n_quad = 60
    K = 3

    print()
    print("Computing M/C/g decomposition for both benchmarks...")
    print()

    results = []

    # Kappa benchmark
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R_kappa = 1.3036

    result_kappa = compute_mcg_decomposition("kappa", R_kappa, theta, polynomials_kappa, K, n_quad)
    results.append(result_kappa)

    # Kappa* benchmark
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    R_kappa_star = 1.1167

    result_kappa_star = compute_mcg_decomposition("kappa*", R_kappa_star, theta, polynomials_kappa_star, K, n_quad)
    results.append(result_kappa_star)

    print_mcg_table(results)

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print()
    print("If g_I1 differs significantly from g_baseline while g_I2 ≈ 1.0:")
    print("  → The residual comes from I1 (derivative term, Q-sensitive)")
    print()
    print("If both g_I1 and g_I2 differ from baseline:")
    print("  → Both components contribute to the residual")
    print()
    print("If κ and κ* have different M1/M2 mixture weights:")
    print("  → This explains why they need opposite-sign corrections")
    print()


if __name__ == "__main__":
    main()
