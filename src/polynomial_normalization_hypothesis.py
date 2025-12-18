"""
src/polynomial_normalization_hypothesis.py
Test hypothesis: PRZZ may normalize pair contributions by polynomial norms.

If c_{ℓ₁,ℓ₂} is divided by ||P_ℓ₁||² × ||P_ℓ₂||², the scaling might work.
"""

import numpy as np
import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01
from src.przz_22_exact_oracle import przz_oracle_22


def test_norm_hypothesis():
    """Test if normalizing by polynomial L² norms makes two benchmarks consistent."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 80

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    # Compute polynomial L² norms
    nodes, weights = gauss_legendre_01(100)

    def poly_norm_sq(P):
        return np.sum(weights * P.eval(nodes)**2)

    norm_P1_k = poly_norm_sq(P1_k)
    norm_P2_k = poly_norm_sq(P2_k)
    norm_P3_k = poly_norm_sq(P3_k)
    norm_Q_k = poly_norm_sq(Q_k)

    norm_P1_ks = poly_norm_sq(P1_ks)
    norm_P2_ks = poly_norm_sq(P2_ks)
    norm_P3_ks = poly_norm_sq(P3_ks)
    norm_Q_ks = poly_norm_sq(Q_ks)

    print("\n" + "=" * 70)
    print("POLYNOMIAL NORMALIZATION HYPOTHESIS TEST")
    print("=" * 70)

    print("\n--- Polynomial L² Norms Squared ---")
    print(f"κ:  ||P₁||² = {norm_P1_k:.6f}, ||P₂||² = {norm_P2_k:.6f}, ||P₃||² = {norm_P3_k:.6f}")
    print(f"κ*: ||P₁||² = {norm_P1_ks:.6f}, ||P₂||² = {norm_P2_ks:.6f}, ||P₃||² = {norm_P3_ks:.6f}")

    # Get (1,1) and (2,2) oracle results
    from src.terms_k3_d1 import make_I1_11, make_I2_11, make_I3_11, make_I4_11
    from src.evaluate import evaluate_term

    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # (1,1) pair uses P₁
    c_11_k = sum([
        evaluate_term(make_I1_11(theta, R_k), polys_k, n_quad).value,
        evaluate_term(make_I2_11(theta, R_k), polys_k, n_quad).value,
        evaluate_term(make_I3_11(theta, R_k), polys_k, n_quad).value,
        evaluate_term(make_I4_11(theta, R_k), polys_k, n_quad).value,
    ])

    c_11_ks = sum([
        evaluate_term(make_I1_11(theta, R_ks), polys_ks, n_quad).value,
        evaluate_term(make_I2_11(theta, R_ks), polys_ks, n_quad).value,
        evaluate_term(make_I3_11(theta, R_ks), polys_ks, n_quad).value,
        evaluate_term(make_I4_11(theta, R_ks), polys_ks, n_quad).value,
    ])

    # (2,2) pair uses P₂
    oracle_22_k = przz_oracle_22(P2_k, Q_k, theta, R_k, n_quad)
    oracle_22_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_ks, n_quad)

    c_22_k = oracle_22_k.total
    c_22_ks = oracle_22_ks.total

    print("\n--- Raw Pair Contributions ---")
    print(f"(1,1) κ: {c_11_k:.6f}, κ*: {c_11_ks:.6f}, ratio: {c_11_k/c_11_ks:.4f}")
    print(f"(2,2) κ: {c_22_k:.6f}, κ*: {c_22_ks:.6f}, ratio: {c_22_k/c_22_ks:.4f}")

    # Test hypothesis: divide by ||P||²×||P||² for diagonal pairs
    c_11_k_norm = c_11_k / (norm_P1_k * norm_P1_k)
    c_11_ks_norm = c_11_ks / (norm_P1_ks * norm_P1_ks)

    c_22_k_norm = c_22_k / (norm_P2_k * norm_P2_k)
    c_22_ks_norm = c_22_ks / (norm_P2_ks * norm_P2_ks)

    print("\n--- Hypothesis 1: Divide by ||P||⁴ ---")
    print(f"(1,1)/||P₁||⁴ κ: {c_11_k_norm:.6f}, κ*: {c_11_ks_norm:.6f}, ratio: {c_11_k_norm/c_11_ks_norm:.4f}")
    print(f"(2,2)/||P₂||⁴ κ: {c_22_k_norm:.6f}, κ*: {c_22_ks_norm:.6f}, ratio: {c_22_k_norm/c_22_ks_norm:.4f}")

    # Test hypothesis: divide by ||P||² only (linear in polynomial)
    c_11_k_norm2 = c_11_k / norm_P1_k
    c_11_ks_norm2 = c_11_ks / norm_P1_ks

    c_22_k_norm2 = c_22_k / norm_P2_k
    c_22_ks_norm2 = c_22_ks / norm_P2_ks

    print("\n--- Hypothesis 2: Divide by ||P||² ---")
    print(f"(1,1)/||P₁||² κ: {c_11_k_norm2:.6f}, κ*: {c_11_ks_norm2:.6f}, ratio: {c_11_k_norm2/c_11_ks_norm2:.4f}")
    print(f"(2,2)/||P₂||² κ: {c_22_k_norm2:.6f}, κ*: {c_22_ks_norm2:.6f}, ratio: {c_22_k_norm2/c_22_ks_norm2:.4f}")

    # Test hypothesis: multiply by ||P||² (inverse)
    c_11_k_norm3 = c_11_k * norm_P1_k
    c_11_ks_norm3 = c_11_ks * norm_P1_ks

    c_22_k_norm3 = c_22_k * norm_P2_k
    c_22_ks_norm3 = c_22_ks * norm_P2_ks

    print("\n--- Hypothesis 3: Multiply by ||P||² ---")
    print(f"(1,1)×||P₁||² κ: {c_11_k_norm3:.6f}, κ*: {c_11_ks_norm3:.6f}, ratio: {c_11_k_norm3/c_11_ks_norm3:.4f}")
    print(f"(2,2)×||P₂||² κ: {c_22_k_norm3:.6f}, κ*: {c_22_ks_norm3:.6f}, ratio: {c_22_k_norm3/c_22_ks_norm3:.4f}")

    # Target ratio
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437
    target_ratio = c_target_k / c_target_ks

    print(f"\n--- Target Ratio ---")
    print(f"c_target_κ / c_target_κ* = {target_ratio:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("""
Looking for a normalization that makes (1,1) and (2,2) ratios similar to target ratio ~1.10:

- Raw ratios: (1,1) = 1.18, (2,2) = 2.43
- Need to find transformation T such that T(c_{ℓℓ}) has consistent ratio

If normalization by polynomial norms works, both pairs should give ~1.10 ratio.
Check which hypothesis (if any) brings (2,2) ratio closer to 1.10.
""")

    # Check if any hypothesis brings (2,2) closer to target ratio
    hypotheses = [
        ("Raw", c_22_k / c_22_ks),
        ("÷||P||⁴", c_22_k_norm / c_22_ks_norm),
        ("÷||P||²", c_22_k_norm2 / c_22_ks_norm2),
        ("×||P||²", c_22_k_norm3 / c_22_ks_norm3),
    ]

    print(f"\n(2,2) pair ratio under different normalizations:")
    print(f"  Target: {target_ratio:.4f}")
    for name, ratio in hypotheses:
        diff = abs(ratio - target_ratio)
        print(f"  {name}: {ratio:.4f} (diff from target: {diff:.4f})")


if __name__ == "__main__":
    test_norm_hypothesis()
