#!/usr/bin/env python3
"""
Detailed analysis of the κ/κ* ratio discrepancy.

Focus on understanding WHY the u-integrals have such different ratios.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from quadrature import gauss_legendre_01

def analyze_polynomial_differences():
    """Analyze the polynomial differences between κ and κ*"""

    print("=" * 80)
    print("POLYNOMIAL COEFFICIENT COMPARISON")
    print("=" * 80)
    print()

    # Load both sets
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=False)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star(enforce_Q0=False)

    polys = [
        ("P1", P1_k, P1_s),
        ("P2", P2_k, P2_s),
        ("P3", P3_k, P3_s),
        ("Q", Q_k, Q_s)
    ]

    for name, poly_k, poly_s in polys:
        print(f"{name} Polynomials:")
        print("-" * 40)

        # Get monomial representations
        mono_k = poly_k.to_monomial()
        mono_s = poly_s.to_monomial()

        # Get coefficients
        coeffs_k = mono_k.coeffs
        coeffs_s = mono_s.coeffs

        max_len = max(len(coeffs_k), len(coeffs_s))

        # Pad to same length
        coeffs_k_pad = np.zeros(max_len)
        coeffs_s_pad = np.zeros(max_len)
        coeffs_k_pad[:len(coeffs_k)] = coeffs_k
        coeffs_s_pad[:len(coeffs_s)] = coeffs_s

        print(f"  {'Degree':<8} {'κ coeff':<15} {'κ* coeff':<15} {'Ratio':<12}")
        for i in range(max_len):
            if abs(coeffs_s_pad[i]) > 1e-12:
                ratio = coeffs_k_pad[i] / coeffs_s_pad[i]
            else:
                ratio = float('inf') if abs(coeffs_k_pad[i]) > 1e-12 else 0.0

            print(f"  {i:<8} {coeffs_k_pad[i]:>14.6f}  {coeffs_s_pad[i]:>14.6f}  {ratio:>11.4f}")
        print()

def analyze_polynomial_norms():
    """Analyze L2 norms and products of polynomials"""

    print("=" * 80)
    print("POLYNOMIAL L2 NORMS AND INNER PRODUCTS")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=False)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star(enforce_Q0=False)

    polys_k = [("P1", P1_k), ("P2", P2_k), ("P3", P3_k)]
    polys_s = [("P1", P1_s), ("P2", P2_s), ("P3", P3_s)]

    nodes, weights = gauss_legendre_01(100)

    # Compute L2 norms
    print("L2 Norms (||P||² = ∫₀¹ P(u)² du):")
    print("-" * 60)
    print(f"{'Poly':<8} {'κ norm²':<15} {'κ* norm²':<15} {'Ratio κ/κ*':<12}")

    norms_k = []
    norms_s = []

    for (name_k, poly_k), (name_s, poly_s) in zip(polys_k, polys_s):
        vals_k = poly_k.eval(nodes)
        vals_s = poly_s.eval(nodes)

        norm_sq_k = np.sum(weights * vals_k**2)
        norm_sq_s = np.sum(weights * vals_s**2)

        norms_k.append(norm_sq_k)
        norms_s.append(norm_sq_s)

        ratio = norm_sq_k / norm_sq_s if norm_sq_s > 1e-15 else float('inf')

        print(f"{name_k:<8} {norm_sq_k:>14.8f}  {norm_sq_s:>14.8f}  {ratio:>11.6f}")

    print()

    # Compute inner products
    print("Inner Products (<Pᵢ, Pⱼ> = ∫₀¹ Pᵢ(u)Pⱼ(u) du):")
    print("-" * 80)

    for i in range(3):
        for j in range(i, 3):
            name_i = f"P{i+1}"
            name_j = f"P{j+1}"

            vals_i_k = polys_k[i][1].eval(nodes)
            vals_j_k = polys_k[j][1].eval(nodes)
            vals_i_s = polys_s[i][1].eval(nodes)
            vals_j_s = polys_s[j][1].eval(nodes)

            inner_k = np.sum(weights * vals_i_k * vals_j_k)
            inner_s = np.sum(weights * vals_i_s * vals_j_s)

            ratio = inner_k / inner_s if abs(inner_s) > 1e-15 else float('inf')

            print(f"  <{name_i},{name_j}>: κ={inner_k:>11.6f}, κ*={inner_s:>11.6f}, ratio={ratio:>11.6f}")

    print()

def analyze_polynomial_shapes():
    """Plot polynomial values to understand shape differences"""

    print("=" * 80)
    print("POLYNOMIAL VALUE COMPARISON AT KEY POINTS")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=False)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star(enforce_Q0=False)

    polys_k = [("P1", P1_k), ("P2", P2_k), ("P3", P3_k)]
    polys_s = [("P1", P1_s), ("P2", P2_s), ("P3", P3_s)]

    test_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    for (name_k, poly_k), (name_s, poly_s) in zip(polys_k, polys_s):
        print(f"{name_k}(u):")
        print("-" * 60)
        print(f"{'u':<8} {'κ value':<15} {'κ* value':<15} {'Ratio':<12}")

        for u in test_points:
            val_k = poly_k.eval(np.array([u]))[0]
            val_s = poly_s.eval(np.array([u]))[0]

            if abs(val_s) > 1e-12:
                ratio = val_k / val_s
            else:
                ratio = float('inf') if abs(val_k) > 1e-12 else 0.0

            print(f"{u:<8.2f} {val_k:>14.6f}  {val_s:>14.6f}  {ratio:>11.4f}")
        print()

    # Q polynomial
    print("Q(t):")
    print("-" * 60)
    print(f"{'t':<8} {'κ value':<15} {'κ* value':<15} {'Ratio':<12}")

    for t in test_points:
        val_k = Q_k.eval(np.array([t]))[0]
        val_s = Q_s.eval(np.array([t]))[0]

        if abs(val_s) > 1e-12:
            ratio = val_k / val_s
        else:
            ratio = float('inf') if abs(val_k) > 1e-12 else 0.0

        print(f"{t:<8.2f} {val_k:>14.6f}  {val_s:>14.6f}  {ratio:>11.4f}")
    print()

def main():
    analyze_polynomial_differences()
    analyze_polynomial_norms()
    analyze_polynomial_shapes()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. The u-integral ratios vary from 0.43 to 2.83 (6.6x range)")
    print("2. This suggests the polynomials have fundamentally different shapes")
    print("3. The target ratio of 1.103 is based on c values, which combine ALL pairs")
    print("4. Individual pairs are NOT expected to have ratio = 1.103")
    print()
    print("Question: Is there a weighted sum that gives the overall c ratio?")
    print()

if __name__ == "__main__":
    main()
