"""
src/test_combined_normalization.py
Test combined normalization: first normalize by ||P||², then apply global factor.

If this works, PRZZ formula might be:
c = A × Σ_{pairs} [symmetry × 1/(ℓ₁!×ℓ₂!) × c_{ℓ₁,ℓ₂} / (||P_{ℓ₁}|| × ||P_{ℓ₂}||)]

where A is a global constant.
"""

import numpy as np
import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01
from src.evaluate import evaluate_c_full


def compute_poly_norms(P1, P2, P3, n_quad=100):
    """Compute L² norms for polynomials."""
    nodes, weights = gauss_legendre_01(n_quad)

    norm_P1_sq = np.sum(weights * P1.eval(nodes)**2)
    norm_P2_sq = np.sum(weights * P2.eval(nodes)**2)
    norm_P3_sq = np.sum(weights * P3.eval(nodes)**2)

    return {
        "P1": np.sqrt(norm_P1_sq),
        "P2": np.sqrt(norm_P2_sq),
        "P3": np.sqrt(norm_P3_sq),
    }


def test_combined_normalization():
    """Test if ||P||² normalization + global factor works."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 60

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    norms_k = compute_poly_norms(P1_k, P2_k, P3_k)

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}
    norms_ks = compute_poly_norms(P1_ks, P2_ks, P3_ks)

    # Evaluate
    result_k = evaluate_c_full(theta, R_k, n_quad, polys_k, return_breakdown=True)
    result_ks = evaluate_c_full(theta, R_ks, n_quad, polys_ks, return_breakdown=True)

    # Targets
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437

    # Pair-to-polynomial mapping
    poly_for_pair = {
        "11": ("P1", "P1"),
        "22": ("P2", "P2"),
        "33": ("P3", "P3"),
        "12": ("P1", "P2"),
        "13": ("P1", "P3"),
        "23": ("P2", "P3"),
    }

    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Compute normalized contributions
    def normalized_c(result, norms):
        total = 0.0
        breakdown = {}
        for pair in ["11", "22", "33", "12", "13", "23"]:
            raw = result.per_term.get(f"_c{pair}_raw", 0)
            p1, p2 = poly_for_pair[pair]
            norm_factor = norms[p1] * norms[p2]
            normalized = raw / norm_factor
            weighted = normalized * factorial_norm[pair] * symmetry[pair]
            total += weighted
            breakdown[pair] = weighted
        return total, breakdown

    c_norm_k, breakdown_k = normalized_c(result_k, norms_k)
    c_norm_ks, breakdown_ks = normalized_c(result_ks, norms_ks)

    print("\n" + "=" * 70)
    print("COMBINED NORMALIZATION TEST")
    print("=" * 70)

    print("\n--- Step 1: Normalize by ||P||² ---")
    print(f"κ  c_normalized: {c_norm_k:.6f}")
    print(f"κ* c_normalized: {c_norm_ks:.6f}")

    # Find global factor that makes κ normalized match target
    global_factor_from_k = c_target_k / c_norm_k
    global_factor_from_ks = c_target_ks / c_norm_ks

    print("\n--- Step 2: Find global factor A ---")
    print(f"Factor to match κ target: A = {global_factor_from_k:.6f}")
    print(f"Factor to match κ* target: A = {global_factor_from_ks:.6f}")

    # If these are similar, a single A would work
    print(f"Ratio of factors: {global_factor_from_ks / global_factor_from_k:.4f}")

    # Test: apply factor from κ to both
    c_final_k = c_norm_k * global_factor_from_k
    c_final_ks = c_norm_ks * global_factor_from_k

    print("\n--- Step 3: Apply global factor A from κ ---")
    print(f"κ  c_final: {c_final_k:.6f} (target: {c_target_k:.6f})")
    print(f"κ* c_final: {c_final_ks:.6f} (target: {c_target_ks:.6f})")
    print(f"κ  gap: {(c_final_k/c_target_k - 1)*100:+.2f}%")
    print(f"κ* gap: {(c_final_ks/c_target_ks - 1)*100:+.2f}%")

    # Two-benchmark gate
    ratio = c_final_k / c_final_ks
    target_ratio = c_target_k / c_target_ks

    print("\n--- Two-Benchmark Gate ---")
    print(f"Computed ratio: {ratio:.4f}")
    print(f"Target ratio:   {target_ratio:.4f}")
    print(f"Error: {abs(ratio/target_ratio - 1)*100:.2f}%")

    if abs(ratio / target_ratio - 1) < 0.05:
        print("✓ PASSES (< 5% error)")
    else:
        print("✗ FAILS")

    # Per-pair breakdown
    print("\n--- Per-Pair Normalized Contributions ---")
    print(f"{'Pair':<6} | {'κ':>12} | {'κ*':>12} | {'Ratio':>8}")
    print("-" * 50)
    for pair in ["11", "22", "33", "12", "13", "23"]:
        b_k = breakdown_k[pair]
        b_ks = breakdown_ks[pair]
        r = b_k / b_ks if abs(b_ks) > 1e-10 else float('inf')
        print(f"{pair:<6} | {b_k:>+12.4f} | {b_ks:>+12.4f} | {r:>8.4f}")

    # What is the average ratio across pairs?
    ratios = []
    for pair in ["11", "22"]:  # Just use positive pairs
        if abs(breakdown_ks[pair]) > 1e-10:
            ratios.append(breakdown_k[pair] / breakdown_ks[pair])

    avg_ratio = sum(ratios) / len(ratios) if ratios else float('nan')
    print(f"\nAverage ratio ((1,1)+(2,2)): {avg_ratio:.4f}")

    # What if we use different global factors per pair type?
    print("\n--- Alternative: Per-pair global factors ---")
    print("Testing if (1,1) and (2,2) need different factors:")

    factor_11 = c_target_k * 0.2 / breakdown_k["11"]  # Assume (1,1) is 20% of target
    factor_22 = c_target_k * 0.6 / breakdown_k["22"]  # Assume (2,2) is 60% of target

    print(f"If (1,1) = 20% of c_target: factor = {factor_11:.4f}")
    print(f"If (2,2) = 60% of c_target: factor = {factor_22:.4f}")


if __name__ == "__main__":
    test_combined_normalization()
