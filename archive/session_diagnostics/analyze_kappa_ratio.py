#!/usr/bin/env python3
"""
Analyze the ratio between κ and κ* component values.

This script computes:
1. The u-integral: ∫₀¹ Pℓ₁(u)Pℓ₂(u) du
2. The t-integral: ∫₀¹ Q(t)²e^{2Rt} dt
3. Their product (the I₂ component)
4. The ratio κ/κ* for each component
5. Identifies which pairs show the largest discrepancy from the target ratio 1.375

Key question: Is the ratio reversal coming from specific pairs or is it uniform?
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from quadrature import gauss_legendre_01

def compute_u_integral(P_l1, P_l2, n_quad=100):
    """Compute ∫₀¹ Pℓ₁(u)Pℓ₂(u) du"""
    nodes, weights = gauss_legendre_01(n_quad)
    integrand_vals = P_l1.eval(nodes) * P_l2.eval(nodes)
    return np.sum(weights * integrand_vals)

def compute_t_integral(Q, R, n_quad=100):
    """Compute ∫₀¹ Q(t)²e^{2Rt} dt"""
    nodes, weights = gauss_legendre_01(n_quad)
    Q_vals = Q.eval(nodes)
    integrand_vals = Q_vals**2 * np.exp(2 * R * nodes)
    return np.sum(weights * integrand_vals)

def main():
    print("=" * 80)
    print("KAPPA vs KAPPA* RATIO ANALYSIS")
    print("=" * 80)
    print()

    # Load κ parameters (R=1.3036)
    print("Loading κ polynomials (R=1.3036)...")
    P1_kappa, P2_kappa, P3_kappa, Q_kappa = load_przz_polynomials(enforce_Q0=False)
    R_kappa = 1.3036

    # Load κ* parameters (R=1.1167)
    print("Loading κ* polynomials (R=1.1167)...")
    P1_star, P2_star, P3_star, Q_star = load_przz_polynomials_kappa_star(enforce_Q0=False)
    R_star = 1.1167

    # Load target values
    with open(Path(__file__).parent / "data" / "przz_parameters.json") as f:
        data_kappa = json.load(f)
    with open(Path(__file__).parent / "data" / "przz_parameters_kappa_star.json") as f:
        data_star = json.load(f)

    c_target_kappa = data_kappa["targets"]["c_precise"]
    c_target_star = data_star["targets"]["c"]
    target_ratio = c_target_kappa / c_target_star

    print(f"\nTarget c values:")
    print(f"  κ:  c = {c_target_kappa:.10f}")
    print(f"  κ*: c = {c_target_star:.10f}")
    print(f"  Target ratio κ/κ* = {target_ratio:.6f}")
    print()

    # Polynomials
    polys_kappa = [P1_kappa, P2_kappa, P3_kappa]
    polys_star = [P1_star, P2_star, P3_star]

    # Compute t-integrals once
    print("Computing t-integrals (∫₀¹ Q(t)²e^{2Rt} dt)...")
    t_int_kappa = compute_t_integral(Q_kappa, R_kappa)
    t_int_star = compute_t_integral(Q_star, R_star)

    print(f"  κ:  ∫Q²e^(2Rt) dt = {t_int_kappa:.10f}")
    print(f"  κ*: ∫Q²e^(2Rt) dt = {t_int_star:.10f}")
    print(f"  Ratio κ/κ* = {t_int_kappa/t_int_star:.6f}")
    print()

    # Results table
    results = []

    print("=" * 100)
    print(f"{'Pair':<8} {'u-int κ':<12} {'u-int κ*':<12} {'Product κ':<14} {'Product κ*':<14} {'Ratio κ/κ*':<12} {'vs Target':<12}")
    print("=" * 100)

    for l1 in range(1, 4):
        for l2 in range(l1, 4):
            # Compute u-integrals
            u_int_kappa = compute_u_integral(polys_kappa[l1-1], polys_kappa[l2-1])
            u_int_star = compute_u_integral(polys_star[l1-1], polys_star[l2-1])

            # Compute products (I₂ components)
            product_kappa = u_int_kappa * t_int_kappa
            product_star = u_int_star * t_int_star

            # Compute ratio
            if abs(product_star) > 1e-15:
                ratio = product_kappa / product_star
            else:
                ratio = float('inf') if product_kappa > 0 else -float('inf')

            # Deviation from target
            deviation = ratio - target_ratio

            results.append({
                'pair': (l1, l2),
                'u_int_kappa': u_int_kappa,
                'u_int_star': u_int_star,
                'product_kappa': product_kappa,
                'product_star': product_star,
                'ratio': ratio,
                'deviation': deviation
            })

            print(f"({l1},{l2})     {u_int_kappa:>11.6f}  {u_int_star:>11.6f}  {product_kappa:>13.8f}  {product_star:>13.8f}  {ratio:>11.6f}  {deviation:>+11.6f}")

    print("=" * 100)
    print()

    # Sort by absolute deviation
    results_sorted = sorted(results, key=lambda x: abs(x['deviation']), reverse=True)

    print("=" * 80)
    print("PAIRS WITH LARGEST DISCREPANCY FROM TARGET RATIO")
    print("=" * 80)
    print()
    print(f"Target ratio: {target_ratio:.6f}")
    print()
    print(f"{'Pair':<8} {'Ratio κ/κ*':<12} {'Deviation':<12} {'% Error':<12}")
    print("-" * 50)

    for r in results_sorted:
        pct_error = 100 * r['deviation'] / target_ratio
        print(f"({r['pair'][0]},{r['pair'][1]})     {r['ratio']:>11.6f}  {r['deviation']:>+11.6f}  {pct_error:>+11.2f}%")

    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    ratios = [r['ratio'] for r in results]
    deviations = [r['deviation'] for r in results]

    print(f"Mean ratio:     {np.mean(ratios):.6f}")
    print(f"Std dev ratio:  {np.std(ratios):.6f}")
    print(f"Min ratio:      {np.min(ratios):.6f} (pair {results_sorted[-1]['pair']})")
    print(f"Max ratio:      {np.max(ratios):.6f} (pair {results_sorted[0]['pair']})")
    print()
    print(f"Mean deviation: {np.mean(deviations):+.6f}")
    print(f"Std dev:        {np.std(deviations):.6f}")
    print()

    # Check if ratio is uniform or varies by pair
    ratio_range = np.max(ratios) - np.min(ratios)
    ratio_std = np.std(ratios)
    ratio_cv = ratio_std / np.mean(ratios)  # Coefficient of variation

    print(f"Range of ratios: {ratio_range:.6f}")
    print(f"Coefficient of variation: {ratio_cv:.6f}")
    print()

    if ratio_cv < 0.1:
        print("CONCLUSION: Ratios are relatively UNIFORM across pairs (CV < 10%)")
        print("            The issue is likely in the t-integral or global normalization.")
    else:
        print("CONCLUSION: Ratios VARY SIGNIFICANTLY across pairs (CV >= 10%)")
        print("            The issue is likely in the u-integrals or pair-specific formulas.")
    print()

    # Detailed breakdown by component
    print("=" * 80)
    print("COMPONENT BREAKDOWN")
    print("=" * 80)
    print()

    print("U-INTEGRAL RATIOS (κ/κ*):")
    print("-" * 40)
    for r in results:
        u_ratio = r['u_int_kappa'] / r['u_int_star'] if abs(r['u_int_star']) > 1e-15 else float('inf')
        print(f"  ({r['pair'][0]},{r['pair'][1]}): {u_ratio:.6f}")

    print()
    print(f"T-INTEGRAL RATIO (κ/κ*): {t_int_kappa/t_int_star:.6f}")
    print()

    # Expected vs actual
    print("EXPECTED vs ACTUAL:")
    print("-" * 40)
    for r in results:
        u_ratio = r['u_int_kappa'] / r['u_int_star'] if abs(r['u_int_star']) > 1e-15 else float('inf')
        expected = u_ratio * (t_int_kappa / t_int_star)
        actual = r['ratio']
        print(f"  ({r['pair'][0]},{r['pair'][1]}): expected={expected:.6f}, actual={actual:.6f}, match={'✓' if abs(expected-actual)<1e-6 else 'X'}")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
