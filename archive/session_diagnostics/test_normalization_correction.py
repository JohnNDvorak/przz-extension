"""
Test if degree-dependent normalization can correct the c ratio.

Based on analyze_polynomial_degree_scaling.py results, we observed:
- (1,1) ratio: 1.186 (close to expected 1.10)
- (2,2) ratio: 2.587 (way too high)
- (3,3) ratio: 28.184 (VASTLY too high)

This suggests pair-dependent normalization proportional to (ℓ₁, ℓ₂).

Hypothesis: c = Σ_{ℓ₁,ℓ₂} [contribution / N(ℓ₁,ℓ₂)]
where N(ℓ₁,ℓ₂) might be:
  - ℓ₁! · ℓ₂!
  - (ℓ₁ · ℓ₂)^p for some power p
  - A₁^{ℓ₁-1} · A₂^{ℓ₂-1} (PRZZ Section 7 style)
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable
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


def test_normalization_schemes():
    """Test various normalization schemes to see which corrects the ratio."""

    print("=" * 80)
    print("TESTING NORMALIZATION CORRECTION SCHEMES")
    print("=" * 80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    # Compute raw integrals
    print("Computing raw pair integrals...")
    print()

    raw_k = {}
    raw_ks = {}

    for l1, l2 in pairs:
        raw_k[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_k, Q_k, R_k)
        raw_ks[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_ks, Q_ks, R_ks)

    print("Raw integrals (no normalization):")
    print()
    print("Pair | κ value | κ* value | Ratio")
    print("-" * 60)
    for l1, l2 in pairs:
        ratio = raw_k[(l1,l2)] / raw_ks[(l1,l2)] if raw_ks[(l1,l2)] != 0 else float('inf')
        print(f"({l1},{l2}) | {raw_k[(l1,l2)]:10.6f} | {raw_ks[(l1,l2)]:10.6f} | {ratio:10.3f}")

    # Test normalization schemes
    normalization_schemes = {
        "None (raw)": lambda l1, l2: 1.0,
        "1/(ℓ₁!·ℓ₂!)": lambda l1, l2: 1.0 / (math.factorial(l1) * math.factorial(l2)),
        "1/(ℓ₁·ℓ₂)": lambda l1, l2: 1.0 / (l1 * l2),
        "1/(ℓ₁²·ℓ₂²)": lambda l1, l2: 1.0 / ((l1**2) * (l2**2)),
        "1/(ℓ₁³·ℓ₂³)": lambda l1, l2: 1.0 / ((l1**3) * (l2**3)),
        "1/√(ℓ₁!·ℓ₂!)": lambda l1, l2: 1.0 / math.sqrt(math.factorial(l1) * math.factorial(l2)),
        "2^{-(ℓ₁+ℓ₂)}": lambda l1, l2: 2.0**(-l1-l2),
    }

    print("\n")
    print("=" * 80)
    print("TESTING NORMALIZATION SCHEMES")
    print("=" * 80)

    for scheme_name, norm_func in normalization_schemes.items():
        print(f"\n\nScheme: {scheme_name}")
        print("-" * 80)

        # Compute normalized c values
        c_k_norm = 0.0
        c_ks_norm = 0.0

        print("\nPer-pair contributions:")
        print("Pair | Norm factor | κ contrib | κ* contrib")
        print("-" * 60)

        for l1, l2 in pairs:
            norm = norm_func(l1, l2)
            contrib_k = raw_k[(l1,l2)] * norm
            contrib_ks = raw_ks[(l1,l2)] * norm

            c_k_norm += contrib_k
            c_ks_norm += contrib_ks

            print(f"({l1},{l2}) | {norm:11.6f} | {contrib_k:10.6f} | {contrib_ks:10.6f}")

        # Compute ratio
        ratio = c_k_norm / c_ks_norm if c_ks_norm != 0 else float('inf')

        # Expected c values from κ = 1 - log(c)/R
        # κ = 0.417293962, R = 1.3036 => c = 2.137
        # κ* = 0.426, R = 1.1167 => c = 1.939
        c_k_expected = 2.137
        c_ks_expected = 1.939
        expected_ratio = c_k_expected / c_ks_expected  # ~1.10

        print(f"\nTotal c_κ (normalized):    {c_k_norm:.6f}")
        print(f"Total c_κ* (normalized):   {c_ks_norm:.6f}")
        print(f"Ratio c_κ/c_κ*:            {ratio:.6f}")
        print(f"Expected ratio:            {expected_ratio:.6f}")
        print(f"Match quality:             {abs(ratio - expected_ratio):.6f} (lower is better)")

        if abs(ratio - expected_ratio) < 0.15:
            print("*** GOOD MATCH! This normalization might be correct. ***")


def test_omega_dependent_normalization():
    """
    Test if normalization depends on ω (related to polynomial degrees).

    In PRZZ Section 7, there might be normalization factors that depend on
    the actual polynomial degrees, not just the piece indices ℓ₁, ℓ₂.
    """
    print("\n\n")
    print("=" * 80)
    print("TESTING ω-DEPENDENT NORMALIZATION")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    print("Polynomial degrees:")
    print()
    print("κ:  P₁ deg={}, P₂ deg={}, P₃ deg={}, Q deg={}".format(
        P1_k.to_monomial().degree,
        P2_k.to_monomial().degree,
        P3_k.to_monomial().degree,
        Q_k.to_monomial().degree
    ))
    print("κ*: P₁ deg={}, P₂ deg={}, P₃ deg={}, Q deg={}".format(
        P1_ks.to_monomial().degree,
        P2_ks.to_monomial().degree,
        P3_ks.to_monomial().degree,
        Q_ks.to_monomial().degree
    ))
    print()

    # Compute raw integrals
    raw_k = {}
    raw_ks = {}

    for l1, l2 in pairs:
        raw_k[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_k, Q_k, R_k)
        raw_ks[(l1,l2)] = compute_pair_integral(l1, l2, poly_map_ks, Q_ks, R_ks)

    # Test degree-based normalization
    def get_poly_degree(l, kappa_variant):
        if kappa_variant == 'k':
            return [P1_k, P2_k, P3_k][l-1].to_monomial().degree
        else:
            return [P1_ks, P2_ks, P3_ks][l-1].to_monomial().degree

    print("Testing normalization: 1/(deg(P_{ℓ₁}) + deg(P_{ℓ₂}))")
    print("-" * 80)

    c_k_deg_norm = 0.0
    c_ks_deg_norm = 0.0

    print("\nPair | deg_κ | deg_κ* | κ contrib | κ* contrib")
    print("-" * 70)

    for l1, l2 in pairs:
        deg_k = get_poly_degree(l1, 'k') + get_poly_degree(l2, 'k')
        deg_ks = get_poly_degree(l1, 'ks') + get_poly_degree(l2, 'ks')

        norm_k = 1.0 / deg_k if deg_k > 0 else 1.0
        norm_ks = 1.0 / deg_ks if deg_ks > 0 else 1.0

        contrib_k = raw_k[(l1,l2)] * norm_k
        contrib_ks = raw_ks[(l1,l2)] * norm_ks

        c_k_deg_norm += contrib_k
        c_ks_deg_norm += contrib_ks

        print(f"({l1},{l2}) | {deg_k:5d} | {deg_ks:6d} | {contrib_k:10.6f} | {contrib_ks:10.6f}")

    ratio = c_k_deg_norm / c_ks_deg_norm if c_ks_deg_norm != 0 else float('inf')
    expected_ratio = 2.137 / 1.939

    print(f"\nRatio c_κ/c_κ*:  {ratio:.6f}")
    print(f"Expected ratio:  {expected_ratio:.6f}")
    print(f"Match quality:   {abs(ratio - expected_ratio):.6f}")

    if abs(ratio - expected_ratio) < 0.15:
        print("*** GOOD MATCH! Degree-based normalization might be correct. ***")


def test_combined_normalization():
    """
    Test combined normalization: piece index AND polynomial degree.

    Hypothesis: N(ℓ₁,ℓ₂) = f(ℓ₁,ℓ₂) × g(deg(P_{ℓ₁}), deg(P_{ℓ₂}))
    """
    print("\n\n")
    print("=" * 80)
    print("TESTING COMBINED NORMALIZATION (Piece Index + Degree)")
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

    def get_poly_degree(l, kappa_variant):
        if kappa_variant == 'k':
            return [P1_k, P2_k, P3_k][l-1].to_monomial().degree
        else:
            return [P1_ks, P2_ks, P3_ks][l-1].to_monomial().degree

    # Test: N(ℓ₁,ℓ₂) = (ℓ₁·ℓ₂) × (deg₁ + deg₂)
    print("Testing: N = (ℓ₁·ℓ₂) × (deg(P_{ℓ₁}) + deg(P_{ℓ₂}))")
    print("-" * 80)

    c_k_comb = 0.0
    c_ks_comb = 0.0

    print("\nPair | N_κ | N_κ* | κ contrib | κ* contrib")
    print("-" * 70)

    for l1, l2 in pairs:
        deg_k = get_poly_degree(l1, 'k') + get_poly_degree(l2, 'k')
        deg_ks = get_poly_degree(l1, 'ks') + get_poly_degree(l2, 'ks')

        N_k = (l1 * l2) * deg_k
        N_ks = (l1 * l2) * deg_ks

        norm_k = 1.0 / N_k if N_k > 0 else 1.0
        norm_ks = 1.0 / N_ks if N_ks > 0 else 1.0

        contrib_k = raw_k[(l1,l2)] * norm_k
        contrib_ks = raw_ks[(l1,l2)] * norm_ks

        c_k_comb += contrib_k
        c_ks_comb += contrib_ks

        print(f"({l1},{l2}) | {N_k:3d} | {N_ks:4d} | {contrib_k:10.6f} | {contrib_ks:10.6f}")

    ratio = c_k_comb / c_ks_comb if c_ks_comb != 0 else float('inf')
    expected_ratio = 2.137 / 1.939

    print(f"\nRatio c_κ/c_κ*:  {ratio:.6f}")
    print(f"Expected ratio:  {expected_ratio:.6f}")
    print(f"Match quality:   {abs(ratio - expected_ratio):.6f}")

    if abs(ratio - expected_ratio) < 0.15:
        print("*** GOOD MATCH! Combined normalization might be correct. ***")


def main():
    """Run all normalization tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     NORMALIZATION CORRECTION ANALYSIS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    test_normalization_schemes()
    test_omega_dependent_normalization()
    test_combined_normalization()

    print("\n")
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Based on the tests above, we can identify which normalization scheme")
    print("(if any) brings the computed c_κ/c_κ* ratio close to the expected ~1.10.")
    print()
    print("If a normalization scheme matches well, this suggests:")
    print("1. We're missing a normalization factor in the PRZZ formula")
    print("2. The factor might depend on piece indices (ℓ₁,ℓ₂) and/or polynomial degrees")
    print("3. This would be documented in PRZZ Section 6-7, which should be re-examined")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
