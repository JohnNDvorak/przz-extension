"""
src/test_polynomial_norm_fix.py
Test the polynomial L² norm normalization hypothesis on full c computation.

Hypothesis: Each pair contribution c_{ℓ₁,ℓ₂} should be divided by ||P_{ℓ₁}|| × ||P_{ℓ₂}||.

For diagonal pairs: divide by ||P_ℓ||²
For cross-pairs: divide by ||P_{ℓ₁}|| × ||P_{ℓ₂}||
"""

import numpy as np
import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01
from src.evaluate import evaluate_c_full


def compute_poly_norms(P1, P2, P3, n_quad=100):
    """Compute L² norms for polynomials."""
    nodes, weights = gauss_legendre_01(n_quad)

    norm_P1_sq = np.sum(weights * P1.eval(nodes)**2)
    norm_P2_sq = np.sum(weights * P2.eval(nodes)**2)
    norm_P3_sq = np.sum(weights * P3.eval(nodes)**2)

    return {
        "P1": np.sqrt(norm_P1_sq),
        "P2": np.sqrt(norm_P2_sq),
        "P3": np.sqrt(norm_P3_sq),
    }


def evaluate_c_with_poly_norm(
    theta: float,
    R: float,
    n: int,
    polynomials: dict,
    poly_norms: dict
) -> dict:
    """
    Evaluate c with polynomial L² normalization.

    Each pair c_{ℓ₁,ℓ₂} is divided by ||P_{ℓ₁}|| × ||P_{ℓ₂}||.
    """
    # Get raw evaluation
    result = evaluate_c_full(theta, R, n, polynomials, return_breakdown=True)

    # Extract raw pair contributions
    pair_raw = {
        "11": result.per_term.get("_c11_raw", 0),
        "22": result.per_term.get("_c22_raw", 0),
        "33": result.per_term.get("_c33_raw", 0),
        "12": result.per_term.get("_c12_raw", 0),
        "13": result.per_term.get("_c13_raw", 0),
        "23": result.per_term.get("_c23_raw", 0),
    }

    # Pair-to-polynomial mapping
    # (ℓ₁, ℓ₂) uses P_{ℓ₁} and P_{ℓ₂}
    # Our indexing: pair "11" = (1,1) uses P₁×P₁
    #              pair "22" = (2,2) uses P₂×P₂
    #              pair "12" = (1,2) uses P₁×P₂
    poly_for_pair = {
        "11": ("P1", "P1"),
        "22": ("P2", "P2"),
        "33": ("P3", "P3"),
        "12": ("P1", "P2"),
        "13": ("P1", "P3"),
        "23": ("P2", "P3"),
    }

    # Compute normalization factors
    pair_norm_factors = {}
    for pair, (p1, p2) in poly_for_pair.items():
        pair_norm_factors[pair] = poly_norms[p1] * poly_norms[p2]

    # Normalize pairs
    pair_normalized = {}
    for pair in pair_raw:
        pair_normalized[pair] = pair_raw[pair] / pair_norm_factors[pair]

    # Apply factorial and symmetry weights (same as evaluate_c_full)
    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Sum with weights
    c_total = 0.0
    for pair in pair_normalized:
        c_total += factorial_norm[pair] * symmetry[pair] * pair_normalized[pair]

    return {
        "c_total": c_total,
        "c_raw": result.total,
        "pair_raw": pair_raw,
        "pair_normalized": pair_normalized,
        "pair_norm_factors": pair_norm_factors,
    }


def test_full_normalization():
    """Test polynomial norm normalization on both benchmarks."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 60

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    norms_k = compute_poly_norms(P1_k, P2_k, P3_k)

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}
    norms_ks = compute_poly_norms(P1_ks, P2_ks, P3_ks)

    # Target values
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437

    print("\n" + "=" * 70)
    print("POLYNOMIAL L² NORM NORMALIZATION TEST")
    print("=" * 70)

    print("\n--- Polynomial L² Norms ---")
    print(f"κ:  ||P₁|| = {norms_k['P1']:.6f}, ||P₂|| = {norms_k['P2']:.6f}, ||P₃|| = {norms_k['P3']:.6f}")
    print(f"κ*: ||P₁|| = {norms_ks['P1']:.6f}, ||P₂|| = {norms_ks['P2']:.6f}, ||P₃|| = {norms_ks['P3']:.6f}")

    # Compute normalized c
    result_k = evaluate_c_with_poly_norm(theta, R_k, n_quad, polys_k, norms_k)
    result_ks = evaluate_c_with_poly_norm(theta, R_ks, n_quad, polys_ks, norms_ks)

    print("\n--- κ Benchmark (R=1.3036) ---")
    print(f"c_raw (no norm):  {result_k['c_raw']:.6f}")
    print(f"c_normalized:     {result_k['c_total']:.6f}")
    print(f"c_target:         {c_target_k:.6f}")
    print(f"Gap (raw):        {(result_k['c_raw'] / c_target_k - 1)*100:+.2f}%")
    print(f"Gap (normalized): {(result_k['c_total'] / c_target_k - 1)*100:+.2f}%")

    print("\n--- κ* Benchmark (R=1.1167) ---")
    print(f"c_raw (no norm):  {result_ks['c_raw']:.6f}")
    print(f"c_normalized:     {result_ks['c_total']:.6f}")
    print(f"c_target:         {c_target_ks:.6f}")
    print(f"Gap (raw):        {(result_ks['c_raw'] / c_target_ks - 1)*100:+.2f}%")
    print(f"Gap (normalized): {(result_ks['c_total'] / c_target_ks - 1)*100:+.2f}%")

    print("\n--- Two-Benchmark Gate ---")
    raw_ratio = result_k['c_raw'] / result_ks['c_raw']
    norm_ratio = result_k['c_total'] / result_ks['c_total']
    target_ratio = c_target_k / c_target_ks

    print(f"Target ratio:     {target_ratio:.4f}")
    print(f"Raw ratio:        {raw_ratio:.4f} ({abs(raw_ratio/target_ratio - 1)*100:.1f}% off)")
    print(f"Normalized ratio: {norm_ratio:.4f} ({abs(norm_ratio/target_ratio - 1)*100:.1f}% off)")

    if abs(norm_ratio / target_ratio - 1) < 0.05:
        print(f"\n✓ Two-benchmark gate PASSES with normalization!")
    else:
        print(f"\n✗ Two-benchmark gate still fails")

    # Per-pair breakdown
    print("\n--- Per-Pair Breakdown (Normalized) ---")
    print(f"{'Pair':<6} | {'κ raw':>12} | {'κ norm':>12} | {'κ* raw':>12} | {'κ* norm':>12} | {'Ratio':>8}")
    print("-" * 70)
    for pair in ["11", "22", "33", "12", "13", "23"]:
        raw_k = result_k['pair_raw'][pair]
        norm_k = result_k['pair_normalized'][pair]
        raw_ks = result_ks['pair_raw'][pair]
        norm_ks = result_ks['pair_normalized'][pair]
        ratio = norm_k / norm_ks if abs(norm_ks) > 1e-10 else float('inf')
        print(f"{pair:<6} | {raw_k:>12.4f} | {norm_k:>12.4f} | {raw_ks:>12.4f} | {norm_ks:>12.4f} | {ratio:>8.4f}")

    # Compute κ from normalized c
    kappa_k = 1 - math.log(result_k['c_total']) / R_k if result_k['c_total'] > 0 else float('nan')
    kappa_ks = 1 - math.log(result_ks['c_total']) / R_ks if result_ks['c_total'] > 0 else float('nan')

    kappa_target_k = 0.417293962
    kappa_target_ks = 0.407511457

    print("\n--- κ Values ---")
    print(f"κ (normalized c):   {kappa_k:.6f} (target: {kappa_target_k:.6f})")
    print(f"κ* (normalized c):  {kappa_ks:.6f} (target: {kappa_target_ks:.6f})")


if __name__ == "__main__":
    test_full_normalization()
