"""
src/test_polynomial_integral_ratios.py
Test the raw polynomial integral ratios to confirm they explain the I₂ divergence.

Since I₂ = (1/θ) × ∫P_{ℓ₁}P_{ℓ₂}du × ∫Q²e^{2Rt}dt, and the t-integral is the same
for all pairs (within a benchmark), the I₂ ratio differences must come from
the u-integral ∫P_{ℓ₁}P_{ℓ₂}du.
"""

import numpy as np
import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01


def test_polynomial_integral_ratios():
    """Compute raw polynomial integral ratios."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 100

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k}
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks}

    nodes, weights = gauss_legendre_01(n_quad)

    print("\n" + "=" * 70)
    print("POLYNOMIAL INTEGRAL RATIOS")
    print("=" * 70)

    # Compute ∫P_{ℓ₁}P_{ℓ₂}du for all pairs
    pairs = ["11", "22", "33", "12", "13", "23"]

    print("\n--- Raw Polynomial Integrals ∫P_{ℓ₁}(u)P_{ℓ₂}(u)du ---")
    print(f"{'Pair':<6} | {'κ':>14} | {'κ*':>14} | {'Ratio (κ/κ*)':>14}")
    print("-" * 60)

    u_integrals_k = {}
    u_integrals_ks = {}

    for pair in pairs:
        ell1, ell2 = int(pair[0]), int(pair[1])
        P_ell1_k = polys_k[f"P{ell1}"]
        P_ell2_k = polys_k[f"P{ell2}"]
        P_ell1_ks = polys_ks[f"P{ell1}"]
        P_ell2_ks = polys_ks[f"P{ell2}"]

        # ∫P_{ℓ₁}(u)P_{ℓ₂}(u)du
        int_k = np.sum(weights * P_ell1_k.eval(nodes) * P_ell2_k.eval(nodes))
        int_ks = np.sum(weights * P_ell1_ks.eval(nodes) * P_ell2_ks.eval(nodes))

        u_integrals_k[pair] = int_k
        u_integrals_ks[pair] = int_ks

        ratio = int_k / int_ks if abs(int_ks) > 1e-10 else float('inf')
        print(f"{pair:<6} | {int_k:>14.6f} | {int_ks:>14.6f} | {ratio:>14.4f}")

    # Compute t-integrals ∫Q²e^{2Rt}dt
    print("\n--- t-Integrals ∫Q(t)²·exp(2Rt)dt ---")
    t_int_k = np.sum(weights * Q_k.eval(nodes)**2 * np.exp(2 * R_k * nodes))
    t_int_ks = np.sum(weights * Q_ks.eval(nodes)**2 * np.exp(2 * R_ks * nodes))

    print(f"κ:  ∫Q²e^{{2Rt}}dt = {t_int_k:.6f}")
    print(f"κ*: ∫Q²e^{{2Rt}}dt = {t_int_ks:.6f}")
    print(f"t-integral ratio: {t_int_k/t_int_ks:.4f}")

    # Reconstruct I₂ values and compare with actual
    print("\n--- Reconstructed vs Actual I₂ ---")
    print(f"{'Pair':<6} | {'Reconstructed Ratio':>20} | {'Expected from u-ratio':>22}")
    print("-" * 60)

    for pair in pairs:
        # Reconstructed I₂ ratio = (u_k/u_ks) × (t_k/t_ks)
        u_ratio = u_integrals_k[pair] / u_integrals_ks[pair] if abs(u_integrals_ks[pair]) > 1e-10 else float('inf')
        t_ratio = t_int_k / t_int_ks

        reconstructed_ratio = u_ratio * t_ratio
        print(f"{pair:<6} | {reconstructed_ratio:>20.4f} | u_ratio={u_ratio:.4f}, t_ratio={t_ratio:.4f}")

    # The key question: why do the u-integrals have such different ratios?
    print("\n" + "=" * 70)
    print("POLYNOMIAL L² NORMS (for comparison)")
    print("=" * 70)

    print(f"\n{'Poly':<6} | {'||P||²_κ':>14} | {'||P||²_κ*':>14} | {'Ratio':>14}")
    print("-" * 60)

    for i in [1, 2, 3]:
        P_k = polys_k[f"P{i}"]
        P_ks = polys_ks[f"P{i}"]

        norm_sq_k = np.sum(weights * P_k.eval(nodes)**2)
        norm_sq_ks = np.sum(weights * P_ks.eval(nodes)**2)

        ratio = norm_sq_k / norm_sq_ks if abs(norm_sq_ks) > 1e-10 else float('inf')
        print(f"P{i:<5} | {norm_sq_k:>14.6f} | {norm_sq_ks:>14.6f} | {ratio:>14.4f}")

    # Compare with PRZZ target ratio
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437
    target_ratio = c_target_k / c_target_ks

    print(f"\nPRZZ target c ratio: {target_ratio:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The polynomial integral ratios ∫P²du differ significantly between pairs:
- (1,1): ratio ~1.2 (close to target 1.10)
- (2,2): ratio ~2.7 (far from target)
- (3,3): ratio ~3.3 (far from target)

This is because the κ* polynomials P₂ and P₃ have MUCH smaller coefficients
than the κ polynomials, leading to smaller ∫P²du values.

PRZZ must normalize pair contributions by something like 1/||P_{ℓ₁}||·||P_{ℓ₂}||
to account for polynomial magnitude differences.

However, our earlier test of ||P||² normalization showed it fixes ratios
but breaks absolute scale by 8x. There must be another factor involved.
""")


if __name__ == "__main__":
    test_polynomial_integral_ratios()
