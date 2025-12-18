"""
src/polynomial_scaling_analysis.py
Analyze polynomial integral scaling between κ and κ* benchmarks.

The κ* benchmark has a -57.5% gap. Let's understand why by analyzing:
1. ∫P²(u) du for each polynomial
2. ∫Q²(t) dt for Q polynomial
3. How these scale between benchmarks
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01


def analyze_polynomial_integrals(n: int = 100):
    """Compute integral ∫P²(u) du for each polynomial."""

    nodes, weights = gauss_legendre_01(n)

    # Load both polynomial sets
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    print("\n" + "=" * 70)
    print("POLYNOMIAL INTEGRAL SCALING ANALYSIS")
    print("=" * 70)

    # Compute integrals for κ benchmark
    print("\n--- κ Benchmark (R=1.3036) ---")

    P1_k_vals = P1_k.eval(nodes)
    P2_k_vals = P2_k.eval(nodes)
    P3_k_vals = P3_k.eval(nodes)
    Q_k_vals = Q_k.eval(nodes)

    int_P1_k_sq = np.sum(weights * P1_k_vals**2)
    int_P2_k_sq = np.sum(weights * P2_k_vals**2)
    int_P3_k_sq = np.sum(weights * P3_k_vals**2)
    int_Q_k_sq = np.sum(weights * Q_k_vals**2)

    print(f"  ∫P₁²(u) du = {int_P1_k_sq:.6f}")
    print(f"  ∫P₂²(u) du = {int_P2_k_sq:.6f}")
    print(f"  ∫P₃²(u) du = {int_P3_k_sq:.6f}")
    print(f"  ∫Q²(t) dt = {int_Q_k_sq:.6f}")

    # Compute integrals for κ* benchmark
    print("\n--- κ* Benchmark (R=1.1167) ---")

    P1_ks_vals = P1_ks.eval(nodes)
    P2_ks_vals = P2_ks.eval(nodes)
    P3_ks_vals = P3_ks.eval(nodes)
    Q_ks_vals = Q_ks.eval(nodes)

    int_P1_ks_sq = np.sum(weights * P1_ks_vals**2)
    int_P2_ks_sq = np.sum(weights * P2_ks_vals**2)
    int_P3_ks_sq = np.sum(weights * P3_ks_vals**2)
    int_Q_ks_sq = np.sum(weights * Q_ks_vals**2)

    print(f"  ∫P₁²(u) du = {int_P1_ks_sq:.6f}")
    print(f"  ∫P₂²(u) du = {int_P2_ks_sq:.6f}")
    print(f"  ∫P₃²(u) du = {int_P3_ks_sq:.6f}")
    print(f"  ∫Q²(t) dt = {int_Q_ks_sq:.6f}")

    # Compute ratios
    print("\n--- Ratios (κ / κ*) ---")
    print(f"  P₁² ratio: {int_P1_k_sq / int_P1_ks_sq:.4f}")
    print(f"  P₂² ratio: {int_P2_k_sq / int_P2_ks_sq:.4f}")
    print(f"  P₃² ratio: {int_P3_k_sq / int_P3_ks_sq:.4f}")
    print(f"  Q² ratio:  {int_Q_k_sq / int_Q_ks_sq:.4f}")

    # The I₂ term for (1,1) is ∫∫ P₁²(u) × Q²(t) × exp(2Rt) / θ du dt
    # = [∫P₁²(u) du] × [∫Q²(t) × exp(2Rt) dt] / θ

    # Compute t-dependent factors with exp
    R_k = 1.3036
    R_ks = 1.1167
    theta = 4.0/7.0

    exp_Q_k = np.sum(weights * Q_k_vals**2 * np.exp(2*R_k*nodes)) / theta
    exp_Q_ks = np.sum(weights * Q_ks_vals**2 * np.exp(2*R_ks*nodes)) / theta

    print("\n--- t-Integral Components (with exp and 1/θ) ---")
    print(f"  κ:  ∫Q²(t)exp(2Rt)/θ dt = {exp_Q_k:.6f}")
    print(f"  κ*: ∫Q²(t)exp(2Rt)/θ dt = {exp_Q_ks:.6f}")
    print(f"  Ratio: {exp_Q_k / exp_Q_ks:.4f}")

    # I₂ for (1,1) prediction
    I2_11_pred_k = int_P1_k_sq * exp_Q_k
    I2_11_pred_ks = int_P1_ks_sq * exp_Q_ks

    print("\n--- Predicted I₂(1,1) from separable structure ---")
    print(f"  κ:  I₂(1,1) ≈ {I2_11_pred_k:.6f}")
    print(f"  κ*: I₂(1,1) ≈ {I2_11_pred_ks:.6f}")
    print(f"  Ratio: {I2_11_pred_k / I2_11_pred_ks:.4f}")

    # Look at cross-term ratios for (2,3) pair
    # I₂(2,3) involves P₂² × P₃²
    cross_23_k = int_P2_k_sq * int_P3_k_sq
    cross_23_ks = int_P2_ks_sq * int_P3_ks_sq

    print("\n--- Cross-term scaling for (2,3) ---")
    print(f"  κ:  ∫P₂² × ∫P₃² = {cross_23_k:.6f}")
    print(f"  κ*: ∫P₂² × ∫P₃² = {cross_23_ks:.6f}")
    print(f"  Ratio: {cross_23_k / cross_23_ks:.4f}")

    # The PRZZ polynomials are OPTIMIZED to give a specific c value
    # The κ* polynomials are simpler (lower degree) but still optimized
    # The raw integral values SHOULD be different - that's expected
    # The question is whether our FORMULA correctly accounts for this

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The κ* polynomials have much smaller integral magnitudes:
- P₃² ratio is huge because κ* P₃ has tiny coefficients
- This is EXPECTED and INTENTIONAL in PRZZ optimization

The question is: does PRZZ's formula have a polynomial-dependent
normalization that we're missing?

PRZZ reports:
  κ: c = 2.137 from these specific polynomials
  κ*: c = 1.939 from these specific polynomials

If the raw integrals scale by 5x between benchmarks but the
c values only differ by 10%, then PRZZ must have a normalization
that makes the formula insensitive to raw polynomial magnitudes.

This suggests we might be missing:
1. Polynomial L²-norm normalization
2. Degree-dependent factors
3. Or the published c values are computed DIFFERENTLY than our formula
""")

    # Let's check what c would be if we used sqrt(∫P²) normalization
    P1_norm_k = np.sqrt(int_P1_k_sq)
    P1_norm_ks = np.sqrt(int_P1_ks_sq)

    print("\n--- Polynomial L² norms ---")
    print(f"  κ  ||P₁||₂ = {P1_norm_k:.6f}, ||P₂||₂ = {np.sqrt(int_P2_k_sq):.6f}, ||P₃||₂ = {np.sqrt(int_P3_k_sq):.6f}")
    print(f"  κ* ||P₁||₂ = {P1_norm_ks:.6f}, ||P₂||₂ = {np.sqrt(int_P2_ks_sq):.6f}, ||P₃||₂ = {np.sqrt(int_P3_ks_sq):.6f}")

    return {
        "kappa": {
            "P1_sq": int_P1_k_sq,
            "P2_sq": int_P2_k_sq,
            "P3_sq": int_P3_k_sq,
            "Q_sq": int_Q_k_sq,
            "exp_Q_factor": exp_Q_k,
        },
        "kappa_star": {
            "P1_sq": int_P1_ks_sq,
            "P2_sq": int_P2_ks_sq,
            "P3_sq": int_P3_ks_sq,
            "Q_sq": int_Q_ks_sq,
            "exp_Q_factor": exp_Q_ks,
        }
    }


def check_przz_normalization_hypothesis():
    """
    Check if PRZZ uses normalized polynomials (||P||₂ = 1).

    If PRZZ normalizes polynomials, then:
    c = [raw c formula] / (||P₁||² × ||P₂||² × ||P₃||² × ||Q||²)^{some power}
    """
    print("\n" + "=" * 70)
    print("NORMALIZATION HYPOTHESIS CHECK")
    print("=" * 70)

    # Our computed c values
    c_computed_k = 1.950  # from mirror_term_test
    c_computed_ks = 0.823

    # Target values
    c_target_k = 2.137
    c_target_ks = 1.938

    # Factors needed
    factor_k = c_target_k / c_computed_k
    factor_ks = c_target_ks / c_computed_ks

    print(f"\nFactors needed to match targets:")
    print(f"  κ:  {factor_k:.4f}")
    print(f"  κ*: {factor_ks:.4f}")
    print(f"  Ratio: {factor_ks / factor_k:.4f}")

    # If there's a simple polynomial-independent factor, ratio would be 1.0
    # We get ratio ≈ 2.15, so the factor is polynomial-dependent

    print(f"\nConclusion:")
    print(f"  The factor ratio of {factor_ks/factor_k:.2f} proves the missing term")
    print(f"  is POLYNOMIAL-DEPENDENT, not a global constant.")
    print(f"")
    print(f"  Possible missing polynomial-dependent factors:")
    print(f"  - Product of polynomial norms in denominator")
    print(f"  - Degree-dependent coefficients in PRZZ formula")
    print(f"  - Polynomial basis transformation normalization")


if __name__ == "__main__":
    analyze_polynomial_integrals(n=100)
    check_przz_normalization_hypothesis()
