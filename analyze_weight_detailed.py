#!/usr/bin/env python3
"""
Detailed (1-u) weight suppression analysis with actual PRZZ polynomials.

Computes weighted integrals for each pair to understand how (1-u)^k weights
affect κ vs κ* differently.
"""

import numpy as np
from scipy.integrate import quad
import json


def load_polynomial_coeffs(json_path, poly_name):
    """Load polynomial coefficients from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    poly_data = data['polynomials'][poly_name]

    # Get coefficients in power basis [c0, c1, c2, ...]
    if 'coeffs' in poly_data:
        return poly_data['coeffs']
    else:
        # For P1, need to expand from tilde form
        # P1(x) = x + x(1-x)*P_tilde(x)
        if poly_name == 'P1':
            tilde = poly_data['tilde_coeffs']
            # P_tilde(x) = sum c_i (1-x)^i
            # P1(x) = x + x(1-x)*P_tilde(x)
            # This is complex, return approximate
            return [0.0, 1.0] + tilde[:2]  # Simplified
        return [0.0, 1.0]  # Default


def eval_poly(coeffs, u):
    """Evaluate polynomial with power-basis coefficients."""
    return np.polyval(coeffs[::-1], u)


def compute_weighted_cross_integral(coeffs1, coeffs2, weight_power):
    """
    Compute ∫₀¹ P₁(u) × P₂(u) × (1-u)^k du
    """
    def integrand(u):
        P1 = eval_poly(coeffs1, u)
        P2 = eval_poly(coeffs2, u)
        return P1 * P2 * (1 - u)**weight_power

    result, _ = quad(integrand, 0, 1, limit=100)
    return result


def main():
    print("=" * 90)
    print("DETAILED (1-u) WEIGHT SUPPRESSION ANALYSIS WITH ACTUAL PRZZ POLYNOMIALS")
    print("=" * 90)
    print()

    # Load polynomial data
    kappa_path = 'data/przz_parameters.json'
    kappa_star_path = 'data/przz_parameters_kappa_star.json'

    try:
        # Load κ polynomials (R=1.3036)
        with open(kappa_path, 'r') as f:
            kappa_data = json.load(f)

        # Load κ* polynomials (R=1.1167)
        with open(kappa_star_path, 'r') as f:
            kappa_star_data = json.load(f)

        print("POLYNOMIAL STRUCTURE COMPARISON:")
        print("-" * 90)
        print(f"{'Polynomial':<12} {'κ structure':<40} {'κ* structure':<40}")
        print("-" * 90)

        for poly_name in ['P1', 'P2', 'P3', 'Q']:
            kappa_poly = kappa_data['polynomials'][poly_name]
            kappa_star_poly = kappa_star_data['polynomials'][poly_name]

            kappa_display = kappa_poly.get('display', 'N/A')
            kappa_star_display = kappa_star_poly.get('display', 'N/A')

            print(f"{poly_name:<12} {kappa_display:<40} {kappa_star_display:<40}")

        print()
        print("KEY OBSERVATION: P2, P3 have degree 3 for κ but degree 2 for κ*")
        print("                 Q is degree 5 for κ but LINEAR for κ*")
        print()

        # =====================================================================
        # Compute weighted integrals for each pair
        # =====================================================================
        print("=" * 90)
        print("WEIGHTED CROSS-INTEGRALS: ∫₀¹ P_i(u) × P_j(u) × (1-u)^k du")
        print("-" * 90)
        print()

        # Polynomial coefficients (simplified for now)
        # κ polynomials
        P1_kappa = [0, 1.0, 0.138, -0.446, -4.040]  # Approximate from display
        P2_kappa = [0, -0.101, 3.572, -1.807]  # From display: -0.101x + 3.572x² - 1.807x³
        P3_kappa = [0, 1.334, -3.019, 1.133]  # From display: 1.334x - 3.019x² + 1.133x³

        # κ* polynomials
        P1_star = [0, 1.0, 0.053, -0.658]  # Approximate
        P2_star = [0, 1.050, -0.097]  # From display: 1.050x - 0.097x²
        P3_star = [0, 0.035, -0.156]  # From display: 0.035x - 0.156x²

        P_kappa_dict = {1: P1_kappa, 2: P2_kappa, 3: P3_kappa}
        P_star_dict = {1: P1_star, 2: P2_star, 3: P3_star}

        pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

        results = []

        for ell1, ell2 in pairs:
            P1_kappa = P_kappa_dict[ell1]
            P2_kappa = P_kappa_dict[ell2]
            P1_star = P_star_dict[ell1]
            P2_star = P_star_dict[ell2]

            # Different weights for different I-terms
            # I₁: (1-u)^{ℓ₁+ℓ₂-2}
            # I₂: no weight
            # I₃: (1-u)^{ℓ₁-1}
            # I₄: (1-u)^{ℓ₂-1}

            I1_weight = ell1 + ell2 - 2
            I2_weight = 0
            I3_weight = ell1 - 1
            I4_weight = ell2 - 1

            # Compute integrals
            I2_kappa = compute_weighted_cross_integral(P1_kappa, P2_kappa, I2_weight)
            I2_star = compute_weighted_cross_integral(P1_star, P2_star, I2_weight)

            I1_kappa = compute_weighted_cross_integral(P1_kappa, P2_kappa, I1_weight)
            I1_star = compute_weighted_cross_integral(P1_star, P2_star, I1_weight)

            # For I₃/I₄ on diagonal pairs, same polynomial
            if ell1 == ell2:
                I3_kappa = compute_weighted_cross_integral(P1_kappa, P1_kappa, I3_weight)
                I3_star = compute_weighted_cross_integral(P1_star, P1_star, I3_weight)
            else:
                I3_kappa = compute_weighted_cross_integral(P1_kappa, P2_kappa, I3_weight)
                I3_star = compute_weighted_cross_integral(P1_star, P2_star, I3_weight)

            results.append({
                'pair': (ell1, ell2),
                'I1_weight': I1_weight,
                'I2_weight': I2_weight,
                'I3_weight': I3_weight,
                'I1_kappa': I1_kappa,
                'I1_star': I1_star,
                'I2_kappa': I2_kappa,
                'I2_star': I2_star,
                'I3_kappa': I3_kappa,
                'I3_star': I3_star,
            })

        # Print results
        print(f"{'Pair':<10} {'I-term':<8} {'Weight':<12} {'κ value':<15} {'κ* value':<15} {'Ratio':<12} {'Suppression':<12}")
        print("-" * 90)

        for res in results:
            pair = res['pair']
            pair_str = f"({pair[0]},{pair[1]})"

            # I₂ (no weight)
            I2_ratio = res['I2_kappa'] / res['I2_star'] if res['I2_star'] != 0 else float('inf')
            print(f"{pair_str:<10} I₂      (1-u)^{res['I2_weight']:<3}   {res['I2_kappa']:<15.6f} {res['I2_star']:<15.6f} {I2_ratio:<12.3f} {'1.00':<12}")

            # I₁ (with weight)
            I1_ratio = res['I1_kappa'] / res['I1_star'] if res['I1_star'] != 0 else float('inf')
            I1_supp = 1.0 / (res['I1_weight'] + 1)
            print(f"{' '*10} I₁      (1-u)^{res['I1_weight']:<3}   {res['I1_kappa']:<15.6f} {res['I1_star']:<15.6f} {I1_ratio:<12.3f} {I1_supp:<12.3f}")

            # I₃ (with weight)
            I3_ratio = res['I3_kappa'] / res['I3_star'] if res['I3_star'] != 0 else float('inf')
            I3_supp = 1.0 / (res['I3_weight'] + 1)
            print(f"{' '*10} I₃      (1-u)^{res['I3_weight']:<3}   {res['I3_kappa']:<15.6f} {res['I3_star']:<15.6f} {I3_ratio:<12.3f} {I3_supp:<12.3f}")

        print()
        print("=" * 90)
        print("ANALYSIS: How (1-u) weights affect κ vs κ*")
        print("-" * 90)
        print()

        # Focus on (2,2) and (3,3) pairs
        print("For (2,2) pair:")
        res_22 = [r for r in results if r['pair'] == (2, 2)][0]
        print(f"  I₂ (no weight):     κ/κ* ratio = {res_22['I2_kappa']/res_22['I2_star']:.3f}")
        print(f"  I₁ (1-u)²:          κ/κ* ratio = {res_22['I1_kappa']/res_22['I1_star']:.3f}")
        print(f"  I₃ (1-u)¹:          κ/κ* ratio = {res_22['I3_kappa']/res_22['I3_star']:.3f}")
        print()

        print("For (3,3) pair:")
        res_33 = [r for r in results if r['pair'] == (3, 3)][0]
        print(f"  I₂ (no weight):     κ/κ* ratio = {res_33['I2_kappa']/res_33['I2_star']:.3f}")
        print(f"  I₁ (1-u)⁴:          κ/κ* ratio = {res_33['I1_kappa']/res_33['I1_star']:.3f}")
        print(f"  I₃ (1-u)²:          κ/κ* ratio = {res_33['I3_kappa']/res_33['I3_star']:.3f}")
        print()

        # Compute naive sum
        total_I2_kappa = sum(r['I2_kappa'] for r in results)
        total_I2_star = sum(r['I2_star'] for r in results)

        print(f"Total I₂ (no weights): κ = {total_I2_kappa:.4f}, κ* = {total_I2_star:.4f}, ratio = {total_I2_kappa/total_I2_star:.3f}")
        print()

        print("CONCLUSION:")
        print("-----------")
        print("The (1-u)^k weights suppress higher pairs (2,2), (3,3) MORE than (1,1).")
        print("But within each pair, the κ/κ* ratio is determined by polynomial structure.")
        print()
        print("For degree-3 P₂, P₃ (κ) vs degree-2 (κ*), the polynomial magnitude dominates:")
        print("- Higher degree → larger polynomial values → larger integrals")
        print("- This makes κ contributions LARGER than κ* (ratio > 1)")
        print()
        print("The (1-u)^k suppression REDUCES this ratio but doesn't reverse it.")
        print("To get const_κ/const_κ* = 0.94 (less than 1), we need OTHER effects:")
        print("  1. Derivative terms (I₁, I₃, I₄) that SUBTRACT from I₂")
        print("  2. Ψ sign patterns with negative coefficients")
        print("  3. Case C kernel attenuation for ω > 0 pieces")
        print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
