"""
src/compute_corrected_c_v2.py
Compute corrected c with NORMALIZED Case C factors.

The first version had correction factors that were too large.
This version normalizes by the baseline integral (without exp factor).

Normalized correction = [∫ a^{ω-1} exp(Rθa) da] / [∫ a^{ω-1} da]
                      = [∫ a^{ω-1} exp(Rθa) da] × ω

For omega=1: baseline = ∫ 1 da = 1
For omega=2: baseline = ∫ a da = 1/2, so multiply by 2
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_terms, compute_kappa
from src.terms_k3_d1 import make_all_terms_k3
from src.quadrature import gauss_legendre_01


# Constants
THETA = 4/7
R = 1.3036
C_TARGET = 2.13745440613217263636
KAPPA_TARGET = 0.417293962


def compute_normalized_case_c_factor(
    R: float,
    theta: float,
    omega: int,
    n_quad: int = 60
) -> float:
    """
    Compute NORMALIZED Case C correction factor.

    correction = [∫₀¹ a^{ω-1} exp(R*theta*a) da] / [∫₀¹ a^{ω-1} da]
               = omega × ∫₀¹ a^{ω-1} exp(R*theta*a) da

    This gives a correction relative to the baseline (no R-dependence).

    Args:
        R: Shift parameter
        theta: Mollifier exponent
        omega: Case C parameter
        n_quad: Quadrature points

    Returns:
        Normalized correction factor (≈1 for small R*theta)
    """
    nodes, weights = gauss_legendre_01(n_quad)

    # Weight function: a^{omega-1}
    if omega == 1:
        weight = np.ones_like(nodes)
    elif omega == 2:
        weight = nodes
    else:
        weight = nodes ** (omega - 1)

    # With exponential
    integrand_exp = weight * np.exp(R * theta * nodes)
    integral_exp = np.sum(weights * integrand_exp)

    # Baseline (no exponential)
    # ∫₀¹ a^{ω-1} da = 1/ω
    baseline = 1.0 / omega

    # Normalized factor
    return integral_exp / baseline


def compute_corrected_c_normalized(
    theta: float = THETA,
    R: float = R,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute c with NORMALIZED Case C corrections.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute normalized Case C factors
    factor_omega1 = compute_normalized_case_c_factor(R, theta, omega=1, n_quad=n_quad)
    factor_omega2 = compute_normalized_case_c_factor(R, theta, omega=2, n_quad=n_quad)

    # Pair correction factors
    correction_factors = {
        "11": 1.0,
        "12": factor_omega1,
        "13": factor_omega2,
        "22": factor_omega1 ** 2,
        "23": factor_omega1 * factor_omega2,
        "33": factor_omega2 ** 2,
    }

    # Factorial and symmetry normalization
    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # Evaluate pairs
    all_terms = make_all_terms_k3(theta, R)

    pair_raw = {}
    pair_corrected = {}
    total_raw = 0.0
    total_corrected = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=False)
        raw_val = pair_result.total
        pair_raw[pair_key] = raw_val

        corrected_val = raw_val * correction_factors[pair_key]
        pair_corrected[pair_key] = corrected_val

        sym = symmetry_factor[pair_key]
        norm = factorial_norm[pair_key]

        total_raw += sym * norm * raw_val
        total_corrected += sym * norm * corrected_val

    kappa_raw = compute_kappa(total_raw, R)
    kappa_corrected = compute_kappa(total_corrected, R)

    if verbose:
        print("\n" + "=" * 70)
        print("CORRECTED c WITH NORMALIZED CASE C FACTORS")
        print("=" * 70)

        print(f"\nParameters: theta = {theta:.10f}, R = {R}")

        print("\n--- Normalized Case C Factors ---")
        print(f"  omega=1 (P_2): {factor_omega1:.10f}")
        print(f"  omega=2 (P_3): {factor_omega2:.10f}")
        print(f"  (These are ratio: ∫exp(Rθa)da / ∫da)")

        print("\n--- Pair Correction Factors ---")
        for pair in ["11", "12", "13", "22", "23", "33"]:
            print(f"  ({pair[0]},{pair[1]}): {correction_factors[pair]:.10f}")

        print("\n--- Summary ---")
        print(f"  c_raw:       {total_raw:20.15f}")
        print(f"  c_corrected: {total_corrected:20.15f}")
        print(f"  c_target:    {C_TARGET:20.15f}")

        gap_raw = (total_raw - C_TARGET) / C_TARGET * 100
        gap_corrected = (total_corrected - C_TARGET) / C_TARGET * 100
        print(f"\n  Gap (raw):       {gap_raw:+.4f}%")
        print(f"  Gap (corrected): {gap_corrected:+.4f}%")

        print(f"\n  kappa_raw:       {kappa_raw:.10f}")
        print(f"  kappa_corrected: {kappa_corrected:.10f}")
        print(f"  kappa_target:    {KAPPA_TARGET:.10f}")

        # Analysis
        print("\n--- Analysis ---")
        if abs(gap_corrected) < abs(gap_raw):
            improvement = (1 - abs(gap_corrected) / abs(gap_raw)) * 100
            print(f"  Correction REDUCED gap by {improvement:.1f}%")
        else:
            print("  Correction did not improve gap")

        print("=" * 70)

    return {
        "c_raw": total_raw,
        "c_corrected": total_corrected,
        "c_target": C_TARGET,
        "gap_raw_pct": (total_raw - C_TARGET) / C_TARGET * 100,
        "gap_corrected_pct": (total_corrected - C_TARGET) / C_TARGET * 100,
        "factor_omega1": factor_omega1,
        "factor_omega2": factor_omega2,
    }


def scan_theta_coupling(verbose: bool = True) -> Dict[str, Any]:
    """
    Try different coupling strengths in the Case C correction.

    The formula exp(R * theta * a) may need a different coupling constant.
    Let's try: exp(R * k * a) for various k.
    """
    n_quad = 60

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    all_terms = make_all_terms_k3(THETA, R)

    nodes, weights = gauss_legendre_01(n_quad)

    # Factorial and symmetry normalization
    factorial_norm = {"11": 1.0, "22": 1.0/4, "33": 1.0/36, "12": 1.0/2, "13": 1.0/6, "23": 1.0/12}
    symmetry_factor = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Compute raw c
    c_raw = 0.0
    pair_raw = {}
    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=False)
        pair_raw[pair_key] = pair_result.total
        c_raw += symmetry_factor[pair_key] * factorial_norm[pair_key] * pair_result.total

    # Try different coupling constants
    k_values = np.linspace(0, 1.0, 21)  # k from 0 to 1

    results = []
    for k in k_values:
        # Compute correction factors with coupling k (instead of theta)
        def get_factor(omega):
            if omega == 1:
                weight = np.ones_like(nodes)
            else:
                weight = nodes ** (omega - 1)
            integrand = weight * np.exp(R * k * nodes)
            return np.sum(weights * integrand) / (1.0 / omega)

        factor_omega1 = get_factor(1)
        factor_omega2 = get_factor(2)

        correction_factors = {
            "11": 1.0,
            "12": factor_omega1,
            "13": factor_omega2,
            "22": factor_omega1 ** 2,
            "23": factor_omega1 * factor_omega2,
            "33": factor_omega2 ** 2,
        }

        c_corrected = 0.0
        for pair_key in all_terms:
            c_corrected += (symmetry_factor[pair_key] * factorial_norm[pair_key] *
                          pair_raw[pair_key] * correction_factors[pair_key])

        gap = (c_corrected - C_TARGET) / C_TARGET * 100
        results.append({"k": k, "c": c_corrected, "gap_pct": gap})

    # Find k that minimizes gap
    best = min(results, key=lambda r: abs(r["gap_pct"]))

    if verbose:
        print("\n" + "=" * 70)
        print("COUPLING STRENGTH SCAN FOR CASE C CORRECTION")
        print("=" * 70)

        print(f"\nScanning coupling k in exp(R * k * a) from 0 to 1")
        print(f"theta = {THETA:.6f}")
        print(f"c_target = {C_TARGET:.10f}")

        print("\n--- Selected Results ---")
        print(f"  {'k':>8} | {'c':>18} | {'Gap':>10}")
        print("  " + "-" * 45)

        # Show every 5th result
        for i, r in enumerate(results):
            if i % 5 == 0 or r == best:
                marker = " <-- BEST" if r == best else ""
                print(f"  {r['k']:>8.4f} | {r['c']:>18.10f} | {r['gap_pct']:>+9.4f}%{marker}")

        print(f"\n--- Best Coupling ---")
        print(f"  k = {best['k']:.4f}")
        print(f"  c = {best['c']:.10f}")
        print(f"  Gap: {best['gap_pct']:+.4f}%")

        if best['k'] < 0.1:
            print("\n  NOTE: Best k ≈ 0 means Case C correction doesn't help much")
            print("  The gap must come from somewhere else!")
        elif abs(best['gap_pct']) < 1.0:
            print("\n  *** FOUND COUPLING THAT MATCHES WITHIN 1%! ***")

        print("=" * 70)

    return {"scan": results, "best": best, "c_raw": c_raw}


if __name__ == "__main__":
    # Try normalized correction
    compute_corrected_c_normalized(verbose=True)

    # Scan coupling strength
    scan_theta_coupling(verbose=True)
