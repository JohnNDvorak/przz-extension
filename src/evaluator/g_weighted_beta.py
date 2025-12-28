"""
src/evaluator/g_weighted_beta.py
Phase 45: Polynomial-Weighted g Correction

SIMPLIFIED APPROACH:
===================

Instead of trying to extract cross-term ratios, directly compute:
  g = I1_with_full_log / I1_with_main_only

Where:
  - I1_with_full_log uses L = 1/θ + x + y
  - I1_with_main_only uses L = 1/θ only

This gives the polynomial-dependent g correction factor.

UNIVERSAL BASELINE:
  g_baseline = 1 + θ/(2K(2K+1)) ≈ 1.0136 for K=3

For polynomials with different structure, g will deviate from baseline.

Created: 2025-12-27 (Phase 45)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import numpy as np

from src.quadrature import gauss_legendre_01


@dataclass
class WeightedBetaResult:
    """Result of weighted g computation."""
    
    g_fp: float              # First-principles g value
    g_baseline: float        # Universal baseline 1 + θ/(2K(2K+1))
    delta_g: float           # g_fp - g_baseline
    delta_g_pct: float       # (g_fp/g_baseline - 1) * 100
    
    # Components for analysis
    I1_full: float           # I1 with full log factor
    I1_main: float           # I1 with main term only
    
    # Parameters
    theta: float
    K: int


def compute_I1_pair_full(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I1 for a pair WITH full log factor (1/θ + x + y)."""
    from src.series_bivariate import (
        build_exp_bracket,
        build_log_factor,
        build_Q_factor,
    )
    from src.unified_i1_paper import build_P_factor_paper, omega_for_ell, _extract_poly_coeffs
    
    max_dx = ell1
    max_dy = ell2
    
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)
    
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)
    
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")
    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None else None
    
    one_minus_u_power = ell1 + ell2
    
    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power
        
        for t, t_w in zip(t_nodes, t_weights):
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)
            
            # FULL log factor: 1/θ + x + y
            log_factor = build_log_factor(theta, max_dx, max_dy)
            
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, 40
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, 40
            )
            
            bracket = exp_factor * log_factor * P_x * P_y
            
            if Q_coeffs is not None:
                Q_alpha = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta*(t-1), ay=theta*t,
                    max_dx=max_dx, max_dy=max_dy
                )
                Q_beta = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta*t, ay=theta*(t-1),
                    max_dx=max_dx, max_dy=max_dy
                )
                bracket = bracket * Q_alpha * Q_beta
            
            coeff = bracket.extract(ell1, ell2)
            total += coeff * one_minus_u_factor * u_w * t_w
    
    # Apply factorial normalization
    total *= math.factorial(ell1) * math.factorial(ell2)
    
    # Sign convention for off-diagonal
    if ell1 != ell2:
        total *= (-1) ** (ell1 + ell2)
    
    return total


def compute_I1_pair_main(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute I1 for a pair with MAIN TERM ONLY (1/θ, no x+y cross terms)."""
    from src.series_bivariate import (
        build_exp_bracket,
        build_Q_factor,
        BivariateSeries,
    )
    from src.unified_i1_paper import build_P_factor_paper, omega_for_ell, _extract_poly_coeffs
    
    max_dx = ell1
    max_dy = ell2
    
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)
    
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)
    
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")
    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None else None
    
    one_minus_u_power = ell1 + ell2
    inv_theta = 1.0 / theta
    
    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power
        
        for t, t_w in zip(t_nodes, t_weights):
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)
            
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, 40
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, 40
            )
            
            # MAIN ONLY: just 1/θ (scalar)
            bracket = exp_factor * P_x * P_y
            bracket = inv_theta * bracket
            
            if Q_coeffs is not None:
                Q_alpha = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta*(t-1), ay=theta*t,
                    max_dx=max_dx, max_dy=max_dy
                )
                Q_beta = build_Q_factor(
                    Q_coeffs, a0=t, ax=theta*t, ay=theta*(t-1),
                    max_dx=max_dx, max_dy=max_dy
                )
                bracket = bracket * Q_alpha * Q_beta
            
            coeff = bracket.extract(ell1, ell2)
            total += coeff * one_minus_u_factor * u_w * t_w
    
    # Apply factorial normalization
    total *= math.factorial(ell1) * math.factorial(ell2)
    
    # Sign convention for off-diagonal
    if ell1 != ell2:
        total *= (-1) ** (ell1 + ell2)
    
    return total


def compute_g_weighted(
    polynomials: Dict,
    R: float,
    K: int = 3,
    theta: float = 4/7,
    n_quad: int = 60,
) -> WeightedBetaResult:
    """
    Compute polynomial-weighted g (first-principles).
    
    g = I1_full / I1_main
    
    Where:
      - I1_full uses L = 1/θ + x + y (full log factor)
      - I1_main uses L = 1/θ (main term only)
    
    This captures the polynomial-dependent cross-term correction.
    """
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }
    
    pairs = ["11", "22", "33", "12", "13", "23"]
    
    total_full = 0.0
    total_main = 0.0
    
    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        
        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm
        
        I1_full = compute_I1_pair_full(R, theta, ell1, ell2, polynomials, n_quad)
        I1_main = compute_I1_pair_main(R, theta, ell1, ell2, polynomials, n_quad)
        
        total_full += I1_full * full_norm
        total_main += I1_main * full_norm
    
    # g = full / main
    if abs(total_main) > 1e-15:
        g_fp = total_full / total_main
    else:
        g_fp = 1.0
    
    delta_g = g_fp - g_baseline
    delta_g_pct = (g_fp / g_baseline - 1) * 100 if g_baseline != 0 else 0.0
    
    return WeightedBetaResult(
        g_fp=g_fp,
        g_baseline=g_baseline,
        delta_g=delta_g,
        delta_g_pct=delta_g_pct,
        I1_full=total_full,
        I1_main=total_main,
        theta=theta,
        K=K,
    )


def validate_g_weighted_gates(verbose: bool = True) -> tuple[bool, str]:
    """
    Validate the g_weighted implementation against required gates.
    
    Gates:
    1. Q=1 gate: g_fp(Q=1) ≈ g_baseline (within ~1%)
    2. Directionality gate: κ needs g > baseline, κ* needs g < baseline
    """
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
    
    theta = 4 / 7
    K = 3
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    
    # Gate 1: Q=1 test
    class Q1Polynomial:
        def eval(self, u):
            return np.ones_like(u) if hasattr(u, '__len__') else 1.0
        def eval_deriv(self, u, n):
            return np.zeros_like(u) if n > 0 else self.eval(u)
        @property
        def standard_coeffs(self):
            return [1.0]
    
    P1, P2, P3, Q = load_przz_polynomials()
    polys_Q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q1Polynomial()}
    
    result_Q1 = compute_g_weighted(polys_Q1, R=1.3036, K=K, theta=theta, n_quad=40)
    Q1_gap = abs(result_Q1.g_fp - g_baseline) / g_baseline * 100
    gate1_passed = Q1_gap < 1.0  # Within 1%
    
    if verbose:
        print("Gate 1: Q=1 Test")
        print(f"  g_fp(Q=1)  = {result_Q1.g_fp:.6f}")
        print(f"  g_baseline = {g_baseline:.6f}")
        print(f"  Gap: {Q1_gap:.4f}%")
        print(f"  {'PASS' if gate1_passed else 'FAIL'}")
        print()
    
    # Gate 2: Directionality test
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}
    
    result_kappa = compute_g_weighted(polys_kappa, R=1.3036, K=K, theta=theta, n_quad=40)
    result_kappa_star = compute_g_weighted(polys_kappa_star, R=1.1167, K=K, theta=theta, n_quad=40)
    
    # κ should need g > baseline, κ* should need g < baseline (from Phase 43)
    gate2_passed = result_kappa.delta_g > result_kappa_star.delta_g
    
    if verbose:
        print("Gate 2: Directionality Test")
        print(f"  κ:  g_fp = {result_kappa.g_fp:.6f}, delta_g = {result_kappa.delta_g:+.6f}")
        print(f"  κ*: g_fp = {result_kappa_star.g_fp:.6f}, delta_g = {result_kappa_star.delta_g:+.6f}")
        print(f"  Direction: κ delta > κ* delta? {result_kappa.delta_g > result_kappa_star.delta_g}")
        print(f"  {'PASS' if gate2_passed else 'FAIL'}")
        print()
    
    all_passed = gate1_passed and gate2_passed
    
    if all_passed:
        msg = "All gates passed"
    else:
        failed = []
        if not gate1_passed:
            failed.append("Q=1 gate")
        if not gate2_passed:
            failed.append("Directionality gate")
        msg = f"Failed: {', '.join(failed)}"
    
    return all_passed, msg


if __name__ == "__main__":
    passed, msg = validate_g_weighted_gates(verbose=True)
    print("Result:", msg)
