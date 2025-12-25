"""
Advanced normalization hypothesis testing.

The previous tests showed that standard normalizations don't work.
This suggests the issue might be more fundamental - perhaps:

1. The integral formula itself is different for different polynomial degrees
2. There's a hidden R-dependent normalization
3. The polynomial coefficients are optimized differently for κ vs κ*
4. The c computation should use a different weighting scheme

This script explores more sophisticated hypotheses.
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.polynomials import (
    load_przz_polynomials, load_przz_polynomials_kappa_star
)
from src.quadrature import tensor_grid_2d


def compute_pair_integral(l1, l2, poly_map, Q, R, n=80):
    """Compute ∫∫P_{ℓ₁}(u)P_{ℓ₂}(u)Q²(u)e^{2Rt} du dt."""
    P_l1 = poly_map[l1].to_monomial()
    P_l2 = poly_map[l2].to_monomial()

    U, T, W = tensor_grid_2d(n)
    P1_vals = P_l1.eval(U)
    P2_vals = P_l2.eval(U)
    Q_vals = Q.eval(U)

    integrand = P1_vals * P2_vals * Q_vals**2 * np.exp(2 * R * T)
    return float(np.sum(W * integrand))


def test_per_pair_normalization_search():
    """
    For each pair, find the normalization factor that would be needed
    to make the ratio exactly 1.10.

    This reveals if there's a pattern in the missing normalization.
    """
    print("=" * 80)
    print("PER-PAIR NORMALIZATION SEARCH")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    # Compute raw integrals
    raw_k = {}
    raw_ks = {}

    for l1, l2 in pairs:
        raw_k[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_k, Q_k, R_k)
        raw_ks[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_ks, Q_ks, R_ks)

    target_ratio = 1.102114  # c_k / c_ks expected

    print("For each pair, find N_κ/N_κ* such that:")
    print("  (I_κ / N_κ) / (I_κ* / N_κ*) = 1.10")
    print()
    print("Pair | I_κ | I_κ* | Raw ratio | Needed N_κ/N_κ*")
    print("-" * 80)

    needed_norms = {}
    for l1, l2 in pairs:
        raw_ratio = raw_k[(l1,l2)] / raw_ks[(l1,l2)] if raw_ks[(l1,l2)] != 0 else float('inf')

        # If I_κ/I_κ* = r, we want (I_κ/N_κ)/(I_κ*/N_κ*) = 1.10
        # => r * (N_κ*/N_κ) = 1.10
        # => N_κ/N_κ* = r / 1.10
        needed_norm_ratio = raw_ratio / target_ratio

        needed_norms[(l1,l2)] = needed_norm_ratio

        print(f"({l1},{l2}) | {raw_k[(l1,l2)]:10.6f} | {raw_ks[(l1,l2)]:10.6f} | "
              f"{raw_ratio:9.3f} | {needed_norm_ratio:9.3f}")

    print()
    print("Observations:")
    print("- If N_κ/N_κ* is constant across pairs → global normalization missing")
    print("- If N_κ/N_κ* varies systematically → pair-dependent normalization")
    print("- If N_κ/N_κ* correlates with ℓ₁,ℓ₂ or degrees → find the pattern")
    print()

    # Check correlations
    def get_poly_degree(l, kappa_variant):
        if kappa_variant == 'k':
            return [P1_k, P2_k, P3_k][l-1].to_monomial().degree
        else:
            return [P1_ks, P2_ks, P3_ks][l-1].to_monomial().degree

    print("Checking correlations:")
    print()
    print("Pair | N_κ/N_κ* | ℓ₁·ℓ₂ | deg_κ | deg_κ* | deg_ratio")
    print("-" * 80)

    for l1, l2 in pairs:
        norm_ratio = needed_norms[(l1,l2)]
        product = l1 * l2

        deg_k = get_poly_degree(l1, 'k') + get_poly_degree(l2, 'k')
        deg_ks = get_poly_degree(l1, 'ks') + get_poly_degree(l2, 'ks')
        deg_ratio = deg_k / deg_ks if deg_ks > 0 else float('inf')

        print(f"({l1},{l2}) | {norm_ratio:8.3f} | {product:5d} | {deg_k:5d} | "
              f"{deg_ks:6d} | {deg_ratio:9.3f}")

    print()


def test_weighted_pair_contribution():
    """
    Test if different pairs should have different weights in the c sum.

    Maybe c = Σ w_{ℓ₁,ℓ₂} × I_{ℓ₁,ℓ₂} where w depends on the pair.
    """
    print("\n")
    print("=" * 80)
    print("WEIGHTED PAIR CONTRIBUTION HYPOTHESIS")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    # Compute raw integrals
    raw_k = {}
    raw_ks = {}

    for l1, l2 in pairs:
        raw_k[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_k, Q_k, R_k)
        raw_ks[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_ks, Q_ks, R_ks)

    # Known: c_k ≈ 2.137, c_ks ≈ 1.939
    # Try to solve for weights that give these values
    # This is an under-determined system, but we can test hypotheses

    print("Testing weight scheme: w = 2 if ℓ₁ ≠ ℓ₂, else 1")
    print("(This accounts for symmetry in cross terms)")
    print("-" * 80)

    c_k_weighted = 0.0
    c_ks_weighted = 0.0

    print("\nPair | Weight | κ contrib | κ* contrib")
    print("-" * 60)

    for l1, l2 in pairs:
        weight = 2 if l1 != l2 else 1

        contrib_k = weight * raw_k[(l1,l2)]
        contrib_ks = weight * raw_ks[(l1,l2)]

        c_k_weighted += contrib_k
        c_ks_weighted += contrib_ks

        print(f"({l1},{l2}) | {weight:6d} | {contrib_k:10.6f} | {contrib_ks:10.6f}")

    ratio = c_k_weighted / c_ks_weighted if c_ks_weighted != 0 else float('inf')
    expected_ratio = 2.137 / 1.939

    print(f"\nTotal c_κ:       {c_k_weighted:.6f}")
    print(f"Total c_κ*:      {c_ks_weighted:.6f}")
    print(f"Ratio:           {ratio:.6f}")
    print(f"Expected ratio:  {expected_ratio:.6f}")
    print()


def test_derivative_order_normalization():
    """
    Test if normalization should depend on derivative order.

    In PRZZ, I₁ involves d²/dxdy, while I₃,I₄ involve d/dx or d/dy.
    Maybe different integral types (I₂, I₁, I₃, I₄) have different normalizations.
    """
    print("\n")
    print("=" * 80)
    print("DERIVATIVE ORDER NORMALIZATION HYPOTHESIS")
    print("=" * 80)
    print()

    print("This requires knowing the I₂,I₁,I₃,I₄ breakdown per pair.")
    print("For now, we just test the I₂-like integral (P×P×Q²×exp).")
    print()
    print("To test this hypothesis properly, we'd need to:")
    print("1. Compute I₁, I₃, I₄ separately for each pair")
    print("2. Check if they need different normalizations")
    print("3. This is beyond simple integral tests - requires the full DSL")
    print()


def test_empirical_correction_factor():
    """
    Empirically find a correction factor that makes things work.

    This is a last resort - find what factor would need to be applied
    to make the two benchmarks match.
    """
    print("\n")
    print("=" * 80)
    print("EMPIRICAL CORRECTION FACTOR SEARCH")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    # Compute raw integrals
    raw_k = {}
    raw_ks = {}

    for l1, l2 in pairs:
        raw_k[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_k, Q_k, R_k)
        raw_ks[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_ks, Q_ks, R_ks)

    # Current totals
    c_k_raw = sum(raw_k.values())
    c_ks_raw = sum(raw_ks.values())

    # Expected values
    c_k_expected = 2.137
    c_ks_expected = 1.939

    print(f"Raw c_κ:       {c_k_raw:.6f}")
    print(f"Expected c_κ:  {c_k_expected:.6f}")
    print(f"Factor needed: {c_k_expected / c_k_raw:.6f}")
    print()

    print(f"Raw c_κ*:      {c_ks_raw:.6f}")
    print(f"Expected c_κ*: {c_ks_expected:.6f}")
    print(f"Factor needed: {c_ks_expected / c_ks_raw:.6f}")
    print()

    print("If the factors are different, a simple global rescaling won't work.")
    print("This suggests a more fundamental formula issue or missing terms.")
    print()

    # Test if a Q-degree-dependent factor helps
    Q_deg_k = Q_k.to_monomial().degree
    Q_deg_ks = Q_ks.to_monomial().degree

    print(f"Q degree κ:  {Q_deg_k}")
    print(f"Q degree κ*: {Q_deg_ks}")
    print()

    # Hypothesis: maybe there's a Q-dependent global factor
    # c_true = c_raw / (1 + α·deg(Q))
    # Solve for α

    # For κ:  c_k_expected = c_k_raw / (1 + α·Q_deg_k)
    # For κ*: c_ks_expected = c_ks_raw / (1 + α·Q_deg_ks)

    # From first equation: α = (c_k_raw/c_k_expected - 1) / Q_deg_k
    alpha_from_k = (c_k_raw / c_k_expected - 1) / Q_deg_k if Q_deg_k > 0 else 0

    # Check if this α works for κ*
    predicted_c_ks = c_ks_raw / (1 + alpha_from_k * Q_deg_ks)

    print(f"Testing: c = c_raw / (1 + α·deg(Q))")
    print(f"α from κ benchmark:     {alpha_from_k:.6f}")
    print(f"Predicted c_κ*:         {predicted_c_ks:.6f}")
    print(f"Expected c_κ*:          {c_ks_expected:.6f}")
    print(f"Match quality:          {abs(predicted_c_ks - c_ks_expected):.6f}")
    print()

    if abs(predicted_c_ks - c_ks_expected) < 0.1:
        print("*** GOOD MATCH! Q-degree-dependent factor might be the answer! ***")


def main():
    """Run all advanced normalization tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     ADVANCED NORMALIZATION HYPOTHESIS TESTING".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    test_per_pair_normalization_search()
    test_weighted_pair_contribution()
    test_derivative_order_normalization()
    test_empirical_correction_factor()

    print("\n")
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. Per-pair normalization analysis shows if there's a systematic pattern")
    print("2. Weighted contribution test checks if cross terms need weight 2")
    print("3. Empirical correction reveals if a simple global factor exists")
    print()
    print("If no simple normalization works, the issue is likely:")
    print("- Fundamental formula misinterpretation")
    print("- Missing integral terms (I₅-type arithmetic corrections)")
    print("- Different optimization quality for κ vs κ*")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
