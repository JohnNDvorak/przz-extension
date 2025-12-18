"""
src/test_scale_factor.py
Find what global scale factor(s) would make our c match PRZZ targets.
"""

import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import evaluate_c_full


def find_scale_factors():
    """Find what scale factors are needed to match targets."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 60

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Evaluate
    result_k = evaluate_c_full(theta, R_k, n_quad, polys_k, return_breakdown=True)
    result_ks = evaluate_c_full(theta, R_ks, n_quad, polys_ks, return_breakdown=True)

    c_k = result_k.total
    c_ks = result_ks.total

    # Targets
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437

    print("\n" + "=" * 70)
    print("SCALE FACTOR ANALYSIS")
    print("=" * 70)

    print(f"\nComputed vs Target:")
    print(f"  κ:  c_computed = {c_k:.6f}, c_target = {c_target_k:.6f}")
    print(f"  κ*: c_computed = {c_ks:.6f}, c_target = {c_target_ks:.6f}")

    # Scale factors needed
    factor_k = c_target_k / c_k
    factor_ks = c_target_ks / c_ks

    print(f"\nScale factors needed to match targets:")
    print(f"  κ:  factor = {factor_k:.6f}")
    print(f"  κ*: factor = {factor_ks:.6f}")
    print(f"  Ratio of factors: {factor_ks / factor_k:.4f}")

    # Check per-pair scale factors
    print("\n--- Per-Pair Analysis ---")
    print("If each pair scaled independently, what factors needed?")

    pairs = ["11", "22", "33", "12", "13", "23"]
    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    print(f"\n{'Pair':<6} | {'κ contrib':>12} | {'κ* contrib':>12} | {'% of κ':>8} | {'% of κ*':>8}")
    print("-" * 60)

    for pair in pairs:
        raw_k = result_k.per_term.get(f"_c{pair}_raw", 0)
        raw_ks = result_ks.per_term.get(f"_c{pair}_raw", 0)

        norm_k = raw_k * factorial_norm[pair] * symmetry[pair]
        norm_ks = raw_ks * factorial_norm[pair] * symmetry[pair]

        pct_k = norm_k / c_k * 100 if abs(c_k) > 1e-10 else 0
        pct_ks = norm_ks / c_ks * 100 if abs(c_ks) > 1e-10 else 0

        print(f"{pair:<6} | {norm_k:>+12.4f} | {norm_ks:>+12.4f} | {pct_k:>+7.1f}% | {pct_ks:>+7.1f}%")

    # What if only (1,1) contributed?
    c_11_k = result_k.per_term.get("_c11_raw", 0)
    c_11_ks = result_ks.per_term.get("_c11_raw", 0)

    print(f"\n--- What if only (1,1) pair contributed? ---")
    print(f"(1,1) raw κ: {c_11_k:.6f}, κ*: {c_11_ks:.6f}")
    print(f"Ratio: {c_11_k/c_11_ks:.4f} (target ratio: {c_target_k/c_target_ks:.4f})")

    # Scale factor if only (1,1)
    scale_from_11_k = c_target_k / c_11_k
    scale_from_11_ks = c_target_ks / c_11_ks

    print(f"Scale factor if (1,1) were whole c:")
    print(f"  κ:  {scale_from_11_k:.4f}")
    print(f"  κ*: {scale_from_11_ks:.4f}")
    print(f"  Ratio: {scale_from_11_ks/scale_from_11_k:.4f}")

    # What about just diagonal pairs (1,1) + (2,2) + (3,3)?
    c_diag_k = (result_k.per_term.get("_c11_raw", 0) +
                result_k.per_term.get("_c22_raw", 0) * 0.25 +
                result_k.per_term.get("_c33_raw", 0) / 36)
    c_diag_ks = (result_ks.per_term.get("_c11_raw", 0) +
                 result_ks.per_term.get("_c22_raw", 0) * 0.25 +
                 result_ks.per_term.get("_c33_raw", 0) / 36)

    print(f"\n--- What about diagonal pairs only? ---")
    print(f"Diagonal sum κ: {c_diag_k:.6f}, κ*: {c_diag_ks:.6f}")
    print(f"Ratio: {c_diag_k/c_diag_ks:.4f}")

    scale_from_diag_k = c_target_k / c_diag_k
    scale_from_diag_ks = c_target_ks / c_diag_ks

    print(f"Scale factor if diagonals were whole c:")
    print(f"  κ:  {scale_from_diag_k:.4f}")
    print(f"  κ*: {scale_from_diag_ks:.4f}")
    print(f"  Ratio: {scale_from_diag_ks/scale_from_diag_k:.4f}")


if __name__ == "__main__":
    find_scale_factors()
