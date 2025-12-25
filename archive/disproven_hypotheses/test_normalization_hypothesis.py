"""
Test polynomial normalization hypothesis.

Hypothesis: PRZZ might normalize by ∫P² or some other polynomial norm,
which would explain why the κ/κ* ratio is 2.43 instead of 1.10.

We test several normalization schemes:
1. Divide by ∫P₂² du
2. Divide by ∫P₂ du
3. Divide by sqrt(∫P₂² du)
4. Divide by R
5. Combinations

Goal: Find a normalization that brings the ratio close to 1.10.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from math import log

from przz_22_exact_oracle import przz_oracle_22, OracleResult22
from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def gauss_legendre_01(n: int):
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def compute_polynomial_norms(P2, n_quad=100):
    """Compute various norms of P₂."""
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    P = P2.eval(u_nodes)

    norm_L2 = np.sqrt(np.sum(u_weights * P * P))
    norm_L2_sq = np.sum(u_weights * P * P)
    norm_L1 = np.sum(u_weights * np.abs(P))
    integral_P = np.sum(u_weights * P)

    return {
        'L2': norm_L2,
        'L2_sq': norm_L2_sq,
        'L1': norm_L1,
        'integral': integral_P,
    }


def test_normalization(name: str, factor_k: float, factor_ks: float,
                       result_k: OracleResult22, result_ks: OracleResult22):
    """Test a normalization scheme."""
    # Apply normalization
    normalized_k = result_k.total / factor_k
    normalized_ks = result_ks.total / factor_ks

    ratio = normalized_k / normalized_ks
    error = abs(ratio - 1.10)

    print(f"{name:40s} ratio: {ratio:8.4f}  (error: {error:6.4f})")

    return ratio


def main():
    print("="*80)
    print("NORMALIZATION HYPOTHESIS TESTING")
    print("="*80)
    print()

    theta = 4/7
    n_quad = 80

    # Load benchmarks
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    R_kappa = 1.3036
    R_kappa_star = 1.1167

    # Compute oracle results
    result_k = przz_oracle_22(P2_k, Q_k, theta, R_kappa, n_quad=n_quad)
    result_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_kappa_star, n_quad=n_quad)

    print("Baseline (no normalization):")
    print(f"  κ:  {result_k.total:.6f}")
    print(f"  κ*: {result_ks.total:.6f}")
    print(f"  Ratio: {result_k.total / result_ks.total:.4f}")
    print(f"  Target: 1.10")
    print(f"  Error: {abs(result_k.total / result_ks.total - 1.10):.4f}")
    print()

    # Compute polynomial norms
    norms_k = compute_polynomial_norms(P2_k, n_quad)
    norms_ks = compute_polynomial_norms(P2_ks, n_quad)

    print("Polynomial norms:")
    print(f"  κ  P₂: L2={norms_k['L2']:.4f}, L2²={norms_k['L2_sq']:.4f}, L1={norms_k['L1']:.4f}, ∫P={norms_k['integral']:.4f}")
    print(f"  κ* P₂: L2={norms_ks['L2']:.4f}, L2²={norms_ks['L2_sq']:.4f}, L1={norms_ks['L1']:.4f}, ∫P={norms_ks['integral']:.4f}")
    print()

    # Test various normalization schemes
    print("="*80)
    print("TESTING NORMALIZATION SCHEMES")
    print("="*80)
    print()

    results = []

    # 1. No normalization (baseline)
    ratio = test_normalization("No normalization", 1.0, 1.0, result_k, result_ks)
    results.append(("No normalization", ratio))

    # 2. Divide by ∫P₂² du
    ratio = test_normalization("Divide by ∫P₂²",
                               norms_k['L2_sq'], norms_ks['L2_sq'],
                               result_k, result_ks)
    results.append(("Divide by ∫P₂²", ratio))

    # 3. Divide by ||P₂||_L2
    ratio = test_normalization("Divide by ||P₂||_L2",
                               norms_k['L2'], norms_ks['L2'],
                               result_k, result_ks)
    results.append(("Divide by ||P₂||_L2", ratio))

    # 4. Divide by ∫P₂ du
    ratio = test_normalization("Divide by ∫P₂",
                               norms_k['integral'], norms_ks['integral'],
                               result_k, result_ks)
    results.append(("Divide by ∫P₂", ratio))

    # 5. Divide by R
    ratio = test_normalization("Divide by R",
                               R_kappa, R_kappa_star,
                               result_k, result_ks)
    results.append(("Divide by R", ratio))

    # 6. Multiply by R (in case it's in denominator)
    ratio = test_normalization("Multiply by R",
                               1/R_kappa, 1/R_kappa_star,
                               result_k, result_ks)
    results.append(("Multiply by R", ratio))

    # 7. Divide by sqrt(R)
    ratio = test_normalization("Divide by sqrt(R)",
                               np.sqrt(R_kappa), np.sqrt(R_kappa_star),
                               result_k, result_ks)
    results.append(("Divide by sqrt(R)", ratio))

    # 8. Divide by R × ∫P₂²
    ratio = test_normalization("Divide by R × ∫P₂²",
                               R_kappa * norms_k['L2_sq'],
                               R_kappa_star * norms_ks['L2_sq'],
                               result_k, result_ks)
    results.append(("Divide by R × ∫P₂²", ratio))

    # 9. Divide by sqrt(∫P₂²) × sqrt(R)
    ratio = test_normalization("Divide by sqrt(∫P₂²) × sqrt(R)",
                               norms_k['L2'] * np.sqrt(R_kappa),
                               norms_ks['L2'] * np.sqrt(R_kappa_star),
                               result_k, result_ks)
    results.append(("Divide by sqrt(∫P₂²) × sqrt(R)", ratio))

    # 10. Divide by ∫P₂² × sqrt(R)
    ratio = test_normalization("Divide by ∫P₂² × sqrt(R)",
                               norms_k['L2_sq'] * np.sqrt(R_kappa),
                               norms_ks['L2_sq'] * np.sqrt(R_kappa_star),
                               result_k, result_ks)
    results.append(("Divide by ∫P₂² × sqrt(R)", ratio))

    print()

    # Find best normalization
    print("="*80)
    print("BEST NORMALIZATIONS")
    print("="*80)
    print()

    # Sort by distance from 1.10
    results_sorted = sorted(results, key=lambda x: abs(x[1] - 1.10))

    print("Top 5 closest to target (1.10):")
    for i, (name, ratio) in enumerate(results_sorted[:5], 1):
        error = abs(ratio - 1.10)
        print(f"{i}. {name:40s} ratio: {ratio:8.4f}  (error: {error:6.4f})")
    print()

    # Check if any are very close
    best_name, best_ratio = results_sorted[0]
    if abs(best_ratio - 1.10) < 0.05:
        print(f"FOUND GOOD MATCH: {best_name}")
        print(f"  Ratio: {best_ratio:.4f}")
        print(f"  Error: {abs(best_ratio - 1.10):.4f}")
    else:
        print("No normalization brings ratio close to 1.10.")
        print(f"Best is {best_name} with ratio {best_ratio:.4f}")
    print()

    # Try some exotic normalizations
    print("="*80)
    print("EXOTIC NORMALIZATIONS")
    print("="*80)
    print()

    # Try exp(-R)
    ratio = test_normalization("Multiply by exp(-R)",
                               1/np.exp(R_kappa), 1/np.exp(R_kappa_star),
                               result_k, result_ks)

    # Try exp(R)
    ratio = test_normalization("Divide by exp(R)",
                               np.exp(R_kappa), np.exp(R_kappa_star),
                               result_k, result_ks)

    # Try R²
    ratio = test_normalization("Divide by R²",
                               R_kappa**2, R_kappa_star**2,
                               result_k, result_ks)

    # Try theta × R
    ratio = test_normalization("Divide by θ × R",
                               theta * R_kappa, theta * R_kappa_star,
                               result_k, result_ks)

    # Try (∫P₂²)^(3/2)
    ratio = test_normalization("Divide by (∫P₂²)^(3/2)",
                               norms_k['L2_sq']**(3/2), norms_ks['L2_sq']**(3/2),
                               result_k, result_ks)

    print()

    # Numerical search for optimal power
    print("="*80)
    print("NUMERICAL SEARCH FOR OPTIMAL POWER")
    print("="*80)
    print()

    print("Trying (∫P₂²)^α for various α:")
    best_alpha = 0
    best_error = float('inf')

    for alpha in np.linspace(0.5, 2.0, 31):
        ratio = test_normalization(f"Divide by (∫P₂²)^{alpha:.2f}",
                                   norms_k['L2_sq']**alpha, norms_ks['L2_sq']**alpha,
                                   result_k, result_ks)
        error = abs(ratio - 1.10)
        if error < best_error:
            best_error = error
            best_alpha = alpha

    print()
    print(f"Best α = {best_alpha:.3f}")
    print(f"  Gives ratio: {test_normalization(f'Divide by (∫P₂²)^{best_alpha:.3f}', norms_k['L2_sq']**best_alpha, norms_ks['L2_sq']**best_alpha, result_k, result_ks):.4f}")
    print(f"  Error: {best_error:.4f}")
    print()


if __name__ == "__main__":
    main()
