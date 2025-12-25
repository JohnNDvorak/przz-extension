"""
src/compute_corrected_c.py
Compute corrected c value with Case C auxiliary integral factors.

This applies the Case C correction factors to all affected pairs
and computes the resulting c and kappa values.

PRZZ Target: c = 2.13745440613217, kappa = 0.417293962
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


def compute_case_c_factor(
    R: float,
    theta: float,
    omega: int,
    n_quad: int = 60
) -> float:
    """
    Compute Case C auxiliary integral factor.

    integral_0^1 a^{omega-1} exp(R * theta * a) da

    For omega=1 (P_2): integral_0^1 exp(R*theta*a) da = (e^{R*theta} - 1)/(R*theta)
    For omega=2 (P_3): integral_0^1 a * exp(R*theta*a) da = ...

    Args:
        R: Shift parameter
        theta: Mollifier exponent
        omega: Case C parameter (omega = k-2)
        n_quad: Quadrature points

    Returns:
        Case C integral factor
    """
    nodes, weights = gauss_legendre_01(n_quad)

    # Weight function: a^{omega-1}
    # For omega=1: a^0 = 1
    # For omega=2: a^1 = a
    if omega == 1:
        weight = np.ones_like(nodes)
    elif omega == 2:
        weight = nodes
    else:
        weight = nodes ** (omega - 1)

    # Exponential factor from (N/n)^{-alpha*a} at alpha = -R/L
    exp_factor = np.exp(R * theta * nodes)

    return np.sum(weights * weight * exp_factor)


def compute_corrected_c(
    theta: float = THETA,
    R: float = R,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute c with Case C corrections applied.

    Applies correction factors:
    - (1,1): No correction (BxB)
    - (1,2): factor_omega1 (BxC)
    - (1,3): factor_omega2 (BxC)
    - (2,2): factor_omega1^2 (CxC)
    - (2,3): factor_omega1 * factor_omega2 (CxC)
    - (3,3): factor_omega2^2 (CxC)

    Args:
        theta: Mollifier exponent
        R: Shift parameter
        n_quad: Quadrature points
        verbose: Print detailed output

    Returns:
        Dict with corrected c, kappa, and breakdown
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute Case C factors
    factor_omega1 = compute_case_c_factor(R, theta, omega=1, n_quad=n_quad)  # For P_2
    factor_omega2 = compute_case_c_factor(R, theta, omega=2, n_quad=n_quad)  # For P_3

    # Pair correction factors
    correction_factors = {
        "11": 1.0,                          # BxB: no correction
        "12": factor_omega1,                # BxC(1)
        "13": factor_omega2,                # BxC(2)
        "22": factor_omega1 ** 2,           # C(1)xC(1)
        "23": factor_omega1 * factor_omega2,  # C(1)xC(2)
        "33": factor_omega2 ** 2,           # C(2)xC(2)
    }

    # Factorial normalization
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # Evaluate each pair
    all_terms = make_all_terms_k3(theta, R)

    pair_raw = {}
    pair_corrected = {}
    pair_normalized = {}
    total_raw = 0.0
    total_corrected = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=True)
        raw_val = pair_result.total

        pair_raw[pair_key] = raw_val

        # Apply Case C correction
        corrected_val = raw_val * correction_factors[pair_key]
        pair_corrected[pair_key] = corrected_val

        # Apply normalization and symmetry
        sym = symmetry_factor[pair_key]
        norm = factorial_norm[pair_key]

        pair_normalized[pair_key] = sym * norm * corrected_val

        total_raw += sym * norm * raw_val
        total_corrected += sym * norm * corrected_val

    # Compute kappa
    kappa_raw = compute_kappa(total_raw, R)
    kappa_corrected = compute_kappa(total_corrected, R)

    results = {
        "c_raw": total_raw,
        "c_corrected": total_corrected,
        "c_target": C_TARGET,
        "kappa_raw": kappa_raw,
        "kappa_corrected": kappa_corrected,
        "kappa_target": KAPPA_TARGET,
        "pair_raw": pair_raw,
        "pair_corrected": pair_corrected,
        "pair_normalized": pair_normalized,
        "correction_factors": correction_factors,
        "factor_omega1": factor_omega1,
        "factor_omega2": factor_omega2,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("CORRECTED c WITH CASE C AUXILIARY INTEGRALS")
        print("=" * 70)

        print(f"\nParameters: theta = {theta:.10f}, R = {R}")
        print(f"Quadrature points: {n_quad}")

        print("\n--- Case C Factors ---")
        print(f"  omega=1 (P_2): {factor_omega1:.10f}")
        print(f"  omega=2 (P_3): {factor_omega2:.10f}")

        print("\n--- Pair Correction Factors ---")
        for pair in ["11", "12", "13", "22", "23", "33"]:
            print(f"  ({pair[0]},{pair[1]}): {correction_factors[pair]:.10f}")

        print("\n--- Raw vs Corrected Pair Values ---")
        print(f"  {'Pair':<8} {'Raw':>18} {'Corrected':>18} {'Change':>10}")
        print("  " + "-" * 60)
        for pair in ["11", "12", "13", "22", "23", "33"]:
            raw = pair_raw[pair]
            corr = pair_corrected[pair]
            change = (corr - raw) / abs(raw) * 100 if abs(raw) > 1e-15 else 0
            print(f"  ({pair[0]},{pair[1]})  {raw:>+18.10f} {corr:>+18.10f} {change:>+9.2f}%")

        print("\n--- Normalized Contributions to c ---")
        for pair in ["11", "12", "13", "22", "23", "33"]:
            print(f"  ({pair[0]},{pair[1]}): {pair_normalized[pair]:>+18.10f}")

        print("\n--- Summary ---")
        print(f"  c_raw:       {total_raw:20.15f}")
        print(f"  c_corrected: {total_corrected:20.15f}")
        print(f"  c_target:    {C_TARGET:20.15f}")
        print()

        gap_raw = (total_raw - C_TARGET) / C_TARGET * 100
        gap_corrected = (total_corrected - C_TARGET) / C_TARGET * 100
        print(f"  Gap (raw):       {gap_raw:+.4f}%")
        print(f"  Gap (corrected): {gap_corrected:+.4f}%")
        print()

        print(f"  kappa_raw:       {kappa_raw:.10f}")
        print(f"  kappa_corrected: {kappa_corrected:.10f}")
        print(f"  kappa_target:    {KAPPA_TARGET:.10f}")
        print()

        kappa_gap_raw = kappa_raw - KAPPA_TARGET
        kappa_gap_corr = kappa_corrected - KAPPA_TARGET
        print(f"  kappa gap (raw):       {kappa_gap_raw:+.8f}")
        print(f"  kappa gap (corrected): {kappa_gap_corr:+.8f}")

        print("\n--- Analysis ---")
        if gap_corrected > gap_raw:
            print("  Case C correction INCREASED the gap")
            print("  The simple model multiplies by factors > 1, making c larger")
            print("  But PRZZ target is larger than our raw c, so this helps!")
        else:
            print("  Case C correction reduced the gap")

        if abs(gap_corrected) < 1.0:
            print("\n  *** CORRECTED c WITHIN 1% OF TARGET! ***")
        elif abs(gap_corrected) < 5.0:
            print("\n  Corrected c within 5% of target - on the right track")

        print("=" * 70)

    return results


def sweep_correction_models(verbose: bool = True) -> Dict[str, Any]:
    """
    Try different Case C correction models to see which gets closest.

    Models:
    1. Simple: integral exp(R*theta*a) da
    2. With (1-a): integral (1-a) exp(R*theta*a) da
    3. Polynomial coupling: integral exp(R*theta*a*u) duda (needs 3D)
    """
    n_quad = 60

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Model 1: Simple (already computed above)
    result_simple = compute_corrected_c(THETA, R, n_quad, verbose=False)

    # Model 2: With (1-a)^{ell-1} factor
    # For (1,2): (1-a)^{2-1} = (1-a)
    nodes, weights = gauss_legendre_01(n_quad)

    # omega=1 with (1-a) weight
    factor_omega1_weighted = np.sum(weights * (1 - nodes) * np.exp(R * THETA * nodes))
    # omega=2 with (1-a) weight
    factor_omega2_weighted = np.sum(weights * (1 - nodes) * nodes * np.exp(R * THETA * nodes))

    if verbose:
        print("\n" + "=" * 70)
        print("CASE C MODEL COMPARISON")
        print("=" * 70)

        print("\n--- Model 1: Simple integral exp(R*theta*a) ---")
        print(f"  c_corrected = {result_simple['c_corrected']:.10f}")
        print(f"  Gap: {(result_simple['c_corrected'] - C_TARGET) / C_TARGET * 100:+.4f}%")

        print(f"\n--- Model 2: With (1-a) weight ---")
        print(f"  factor_omega1 (1-a)*exp(...): {factor_omega1_weighted:.10f}")
        print(f"  factor_omega2 (1-a)*a*exp(...): {factor_omega2_weighted:.10f}")
        print(f"  (Compare to simple: omega1={result_simple['factor_omega1']:.10f}, omega2={result_simple['factor_omega2']:.10f})")

        print("\n--- Analysis ---")
        print("  The simple model gives factors > 1, increasing c toward target")
        print("  The (1-a) weighted model gives smaller factors")
        print()
        print("  Key insight: Our raw c is BELOW target, so we need factors > 1")
        print("  The simple exp(R*theta*a) model is in the right direction!")
        print("=" * 70)

    return {
        "simple": result_simple,
        "factor_omega1_weighted": factor_omega1_weighted,
        "factor_omega2_weighted": factor_omega2_weighted,
    }


if __name__ == "__main__":
    # Compute corrected c
    compute_corrected_c(verbose=True)

    # Compare models
    sweep_correction_models(verbose=True)
