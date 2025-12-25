"""
Analyze how polynomial degree affects c computation in PRZZ.

The mystery:
- κ has degree-3 P₂,P₃ and degree-5 Q
- κ* has degree-2 P₂,P₃ and degree-1 Q
- Yet c_κ/c_κ* should be ~1.10, not the ~2.09 we observe

This script tests:
1. How c scales with polynomial degree (synthetic tests)
2. Whether PRZZ includes degree-dependent normalization
3. Whether the ratio reversal is explained by polynomial structure
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.polynomials import (
    Polynomial, PellPolynomial, QPolynomial,
    load_przz_polynomials, load_przz_polynomials_kappa_star
)
from src.quadrature import tensor_grid_2d

# =============================================================================
# Test 1: Synthetic Polynomial Degree Scaling
# =============================================================================

def compute_integral_P_squared(P: Polynomial, n: int = 60) -> float:
    """
    Compute ∫₀¹ P²(u) du using quadrature.

    This is a proxy for how polynomial degree affects the integral magnitude.
    """
    U, _, W_1d = tensor_grid_2d(n)
    # Extract 1D grid and weights (U is 2D grid, we just need 1D)
    u_grid = U[:, 0]  # First column
    w_grid = W_1d[:, 0]  # Weights for first dimension

    P_vals = P.eval(u_grid)
    integrand = P_vals ** 2

    return float(np.sum(w_grid * integrand))


def compute_integral_PQ_squared_exp(
    P: Polynomial,
    Q: QPolynomial,
    R: float,
    n: int = 60
) -> float:
    """
    Compute ∫₀¹∫₀¹ P²(u)Q²(u)e^{2Rt} du dt.

    This is similar to the I₂ integral structure.
    """
    U, T, W = tensor_grid_2d(n)

    P_vals = P.eval(U)
    Q_vals = Q.eval(U)

    integrand = P_vals**2 * Q_vals**2 * np.exp(2 * R * T)

    return float(np.sum(W * integrand))


def test_synthetic_degree_scaling():
    """
    Test how c proxy scales with polynomial degree.

    Create synthetic P polynomials of degrees 1-5, keep Q fixed.
    """
    print("=" * 80)
    print("TEST 1: Synthetic Polynomial Degree Scaling")
    print("=" * 80)
    print()

    # Fixed Q for comparison (use PRZZ κ Q)
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)

    # Create synthetic P polynomials of different degrees
    # Use simple normalized forms: P(u) = u * (some polynomial to control degree)

    synthetic_polys = {
        1: Polynomial([0, 1.0]),  # P(u) = u
        2: Polynomial([0, 1.0, -0.5]),  # P(u) = u - 0.5u²
        3: Polynomial([0, 1.5, -1.0, 0.5]),  # P(u) = 1.5u - u² + 0.5u³
        4: Polynomial([0, 2.0, -2.0, 1.0, -0.25]),  # degree 4
        5: Polynomial([0, 2.5, -3.0, 2.0, -0.75, 0.15]),  # degree 5
    }

    R = 1.3036  # κ value

    print(f"Fixed Q: degree {Q_k.to_monomial().degree}")
    print(f"R = {R}")
    print()

    print("deg | ∫P²du | ∫∫P²Q²e^{2Rt} dudtdu | Ratio to deg=1")
    print("-" * 70)

    results = {}
    for deg, P in synthetic_polys.items():
        int_P2 = compute_integral_P_squared(P)
        int_PQ_exp = compute_integral_PQ_squared_exp(P, Q_k, R)

        results[deg] = {'int_P2': int_P2, 'int_full': int_PQ_exp}

        ratio = int_PQ_exp / results[1]['int_full'] if deg > 1 else 1.0

        print(f"{deg:3d} | {int_P2:8.5f} | {int_PQ_exp:20.5f} | {ratio:8.3f}")

    print()
    print("Observations:")
    print("- As degree increases, ∫P²Q²e^{2Rt} generally increases")
    print("- But the scaling is NOT linear with degree")
    print("- Coefficient structure matters significantly")
    print()


# =============================================================================
# Test 2: Check for Degree-Dependent Normalization
# =============================================================================

def test_przz_degree_normalization():
    """
    Check if PRZZ formula includes degree-dependent normalization.

    For pair (ℓ₁, ℓ₂), check if there's a factor involving:
    - ℓ₁!, ℓ₂!
    - (ℓ₁·ℓ₂)
    - Powers of ℓ₁, ℓ₂
    """
    print("=" * 80)
    print("TEST 2: Degree-Dependent Normalization Hypothesis")
    print("=" * 80)
    print()

    # Load both benchmark polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    print("κ polynomial degrees:")
    print(f"  P₁: {P1_k.to_monomial().degree}")
    print(f"  P₂: {P2_k.to_monomial().degree}")
    print(f"  P₃: {P3_k.to_monomial().degree}")
    print(f"  Q:  {Q_k.to_monomial().degree}")
    print()

    print("κ* polynomial degrees:")
    print(f"  P₁: {P1_ks.to_monomial().degree}")
    print(f"  P₂: {P2_ks.to_monomial().degree}")
    print(f"  P₃: {P3_ks.to_monomial().degree}")
    print(f"  Q:  {Q_ks.to_monomial().degree}")
    print()

    # Test various normalization hypotheses
    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    # Map pair to P polynomials
    def get_poly(ell, kappa_set):
        if kappa_set == 'k':
            return [P1_k, P2_k, P3_k][ell-1]
        else:
            return [P1_ks, P2_ks, P3_ks][ell-1]

    print("Testing normalization factors for each pair:")
    print()
    print("Pair | κ P_deg | κ* P_deg | 1/ℓ₁!ℓ₂! | 1/(ℓ₁·ℓ₂) | 1/ℓ₁^p·ℓ₂^p")
    print("-" * 80)

    import math

    for l1, l2 in pairs:
        P_l1_k = get_poly(l1, 'k')
        P_l2_k = get_poly(l2, 'k')
        P_l1_ks = get_poly(l1, 'ks')
        P_l2_ks = get_poly(l2, 'ks')

        deg_k = (P_l1_k.to_monomial().degree, P_l2_k.to_monomial().degree)
        deg_ks = (P_l1_ks.to_monomial().degree, P_l2_ks.to_monomial().degree)

        # Various normalization factors
        factorial_norm = 1.0 / (math.factorial(l1) * math.factorial(l2))
        product_norm = 1.0 / (l1 * l2) if l1 * l2 > 0 else 1.0
        power_norm = 1.0 / ((l1 ** 2) * (l2 ** 2)) if l1 * l2 > 0 else 1.0

        print(f"({l1},{l2}) | {deg_k} | {deg_ks} | {factorial_norm:.6f} | "
              f"{product_norm:.6f} | {power_norm:.6f}")

    print()
    print("Note: If PRZZ uses normalization 1/(ℓ₁!ℓ₂!), larger ℓ pairs get")
    print("      heavily suppressed, which could affect the ratio.")
    print()


# =============================================================================
# Test 3: Actual Polynomial Integral Ratios
# =============================================================================

def test_actual_polynomial_ratios():
    """
    Compute actual integral ratios for κ vs κ* polynomials.

    This tests whether the polynomial degree difference alone explains
    the observed c ratio.
    """
    print("=" * 80)
    print("TEST 3: Actual κ vs κ* Polynomial Integral Ratios")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    n = 80  # Higher precision

    # Test integrals for different pairs
    pairs = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]

    print("Computing ∫∫P_{ℓ₁}(u)P_{ℓ₂}(u)Q²(u)e^{2Rt} du dt for each pair:")
    print()
    print("Pair | κ integral | κ* integral | Ratio κ/κ* | Expected ~1.10")
    print("-" * 80)

    poly_map_k = {1: P1_k, 2: P2_k, 3: P3_k}
    poly_map_ks = {1: P1_ks, 2: P2_ks, 3: P3_ks}

    for l1, l2 in pairs:
        # Compute κ integral
        P_l1_k = poly_map_k[l1].to_monomial()
        P_l2_k = poly_map_k[l2].to_monomial()

        U, T, W = tensor_grid_2d(n)
        P1_vals = P_l1_k.eval(U)
        P2_vals = P_l2_k.eval(U)
        Q_vals = Q_k.eval(U)

        integrand_k = P1_vals * P2_vals * Q_vals**2 * np.exp(2 * R_k * T)
        int_k = float(np.sum(W * integrand_k))

        # Compute κ* integral
        P_l1_ks = poly_map_ks[l1].to_monomial()
        P_l2_ks = poly_map_ks[l2].to_monomial()

        P1_vals_ks = P_l1_ks.eval(U)
        P2_vals_ks = P_l2_ks.eval(U)
        Q_vals_ks = Q_ks.eval(U)

        integrand_ks = P1_vals_ks * P2_vals_ks * Q_vals_ks**2 * np.exp(2 * R_ks * T)
        int_ks = float(np.sum(W * integrand_ks))

        ratio = int_k / int_ks if int_ks != 0 else float('inf')

        print(f"({l1},{l2}) | {int_k:10.6f} | {int_ks:10.6f} | "
              f"{ratio:10.3f} | {'✗' if abs(ratio - 1.10) > 0.5 else '✓'}")

    print()
    print("Key observations:")
    print("- If ratios are all ~1.10, the polynomial structure alone explains c ratio")
    print("- If ratios vary significantly, there's a pair-dependent effect")
    print("- Large deviations suggest missing normalization in our formula")
    print()


# =============================================================================
# Test 4: Coefficient Magnitude Analysis
# =============================================================================

def test_coefficient_magnitudes():
    """
    Analyze the coefficient magnitudes of κ vs κ* polynomials.

    This helps distinguish degree effects from coefficient magnitude effects.
    """
    print("=" * 80)
    print("TEST 4: Polynomial Coefficient Magnitude Analysis")
    print("=" * 80)
    print()

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    def analyze_poly(name, poly_k, poly_ks):
        print(f"\n{name}:")
        print("  κ coefficients (monomial basis):")
        coeffs_k = poly_k.to_monomial().coeffs
        for i, c in enumerate(coeffs_k):
            if abs(c) > 1e-10:
                print(f"    x^{i}: {c:12.8f}")

        print(f"  κ* coefficients (monomial basis):")
        coeffs_ks = poly_ks.to_monomial().coeffs
        for i, c in enumerate(coeffs_ks):
            if abs(c) > 1e-10:
                print(f"    x^{i}: {c:12.8f}")

        # L2 norm comparison
        norm_k = np.sqrt(np.sum(coeffs_k**2))
        norm_ks = np.sqrt(np.sum(coeffs_ks**2))
        print(f"  L2 norm ratio κ/κ*: {norm_k/norm_ks:.4f}")

    analyze_poly("P₁", P1_k, P1_ks)
    analyze_poly("P₂", P2_k, P2_ks)
    analyze_poly("P₃", P3_k, P3_ks)

    print("\n\nQ polynomial:")
    print("  κ Q coefficients (monomial basis):")
    coeffs_Q_k = Q_k.to_monomial().coeffs
    for i, c in enumerate(coeffs_Q_k):
        if abs(c) > 1e-10:
            print(f"    x^{i}: {c:12.8f}")

    print("  κ* Q coefficients (monomial basis):")
    coeffs_Q_ks = Q_ks.to_monomial().coeffs
    for i, c in enumerate(coeffs_Q_ks):
        if abs(c) > 1e-10:
            print(f"    x^{i}: {c:12.8f}")

    norm_Q_k = np.sqrt(np.sum(coeffs_Q_k**2))
    norm_Q_ks = np.sqrt(np.sum(coeffs_Q_ks**2))
    print(f"  L2 norm ratio κ/κ*: {norm_Q_k/norm_Q_ks:.4f}")

    print()


# =============================================================================
# Test 5: R-Dependent Scaling with Same Polynomials
# =============================================================================

def test_r_dependent_scaling():
    """
    Test how R affects integrals with the SAME polynomials.

    This isolates the R-dependent effect from polynomial structure.
    """
    print("=" * 80)
    print("TEST 5: R-Dependent Scaling (Same Polynomials)")
    print("=" * 80)
    print()

    # Use κ polynomials at both R values
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)

    R_k = 1.3036
    R_ks = 1.1167

    n = 80

    pairs = [(1,1), (2,2), (3,3)]
    poly_map = {1: P1_k, 2: P2_k, 3: P3_k}

    print("Using κ polynomials at different R values:")
    print()
    print("Pair | I(R=1.3036) | I(R=1.1167) | Ratio | e^{2ΔR} prediction")
    print("-" * 80)

    for l1, l2 in pairs:
        P_l1 = poly_map[l1].to_monomial()
        P_l2 = poly_map[l2].to_monomial()

        U, T, W = tensor_grid_2d(n)
        P1_vals = P_l1.eval(U)
        P2_vals = P_l2.eval(U)
        Q_vals = Q_k.eval(U)

        # At R = R_k
        integrand_1 = P1_vals * P2_vals * Q_vals**2 * np.exp(2 * R_k * T)
        int_1 = float(np.sum(W * integrand_1))

        # At R = R_ks
        integrand_2 = P1_vals * P2_vals * Q_vals**2 * np.exp(2 * R_ks * T)
        int_2 = float(np.sum(W * integrand_2))

        ratio = int_1 / int_2 if int_2 != 0 else float('inf')

        # Predicted ratio from exp(2R*T) scaling
        # Since T ∈ [0,1], this is approximate
        delta_R = R_k - R_ks
        # Average e^{2ΔR·T} over T∈[0,1]
        predicted = (np.exp(2*delta_R) - 1) / (2*delta_R)

        print(f"({l1},{l2}) | {int_1:11.6f} | {int_2:11.6f} | "
              f"{ratio:5.3f} | {predicted:5.3f}")

    print()
    print("If ratio ≈ prediction, the R-dependence is from e^{2Rt} alone.")
    print("If ratio differs, there's a polynomial-R interaction.")
    print()


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    """Run all tests and provide comprehensive analysis."""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  POLYNOMIAL DEGREE SCALING ANALYSIS FOR PRZZ c COMPUTATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run all tests
    test_synthetic_degree_scaling()
    test_przz_degree_normalization()
    test_actual_polynomial_ratios()
    test_coefficient_magnitudes()
    test_r_dependent_scaling()

    # Final summary
    print("=" * 80)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 80)
    print()
    print("The mystery: κ has higher-degree polynomials than κ*, yet c_κ/c_κ* ≈ 2.09")
    print("when we expect ≈1.10 from the published κ values.")
    print()
    print("Key questions answered:")
    print()
    print("1. Does polynomial degree alone explain the ratio?")
    print("   → See Test 3 results above")
    print()
    print("2. Is there degree-dependent normalization (ℓ₁!ℓ₂!) we're missing?")
    print("   → See Test 2 for normalization factors")
    print()
    print("3. Is the effect R-dependent or polynomial-structure dependent?")
    print("   → Compare Tests 3 and 5")
    print()
    print("4. Are coefficient magnitudes significantly different?")
    print("   → See Test 4 L2 norm ratios")
    print()
    print("Next steps based on results:")
    print("- If Test 3 ratios are all ~1.10: polynomial structure fully explains it")
    print("- If Test 3 ratios vary widely: there's a pair-dependent normalization issue")
    print("- If Test 5 shows R-dependence mismatch: formula interpretation error")
    print("- If Test 4 shows large coefficient differences: optimization quality issue")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
