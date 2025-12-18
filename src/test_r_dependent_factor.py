"""
src/test_r_dependent_factor.py
Test if there's an R-dependent factor missing from higher-order pairs.

Hypothesis: The pair (ℓ₁, ℓ₂) might need a factor like R^(ℓ₁+ℓ₂-2) or (R/θ)^f(ℓ₁,ℓ₂)
to make contributions consistent across benchmarks.
"""

import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import evaluate_c_full


def test_r_dependent_factor():
    """Look for R-dependent patterns in pair contributions."""

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

    # Target values
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437
    target_ratio = c_target_k / c_target_ks  # 1.1029

    print("\n" + "=" * 70)
    print("R-DEPENDENT FACTOR ANALYSIS")
    print("=" * 70)

    print(f"\nR values: κ = {R_k}, κ* = {R_ks}, ratio = {R_k/R_ks:.4f}")
    print(f"Target c ratio: {target_ratio:.4f}")

    pairs = ["11", "22", "33", "12", "13", "23"]
    ell_values = {
        "11": (1, 1), "22": (2, 2), "33": (3, 3),
        "12": (1, 2), "13": (1, 3), "23": (2, 3),
    }
    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    print("\n--- Pair Contributions (with factorial/symmetry weights) ---")
    print(f"{'Pair':<6} | {'κ':>10} | {'κ*':>10} | {'Ratio':>8} | {'ℓ₁+ℓ₂':>6}")
    print("-" * 55)

    pair_data = {}
    for pair in pairs:
        raw_k = result_k.per_term.get(f"_c{pair}_raw", 0)
        raw_ks = result_ks.per_term.get(f"_c{pair}_raw", 0)

        weighted_k = raw_k * factorial_norm[pair] * symmetry[pair]
        weighted_ks = raw_ks * factorial_norm[pair] * symmetry[pair]

        ratio = weighted_k / weighted_ks if abs(weighted_ks) > 1e-10 else float('inf')
        ell1, ell2 = ell_values[pair]

        pair_data[pair] = {
            "κ": weighted_k,
            "κ*": weighted_ks,
            "ratio": ratio,
            "ell_sum": ell1 + ell2
        }

        print(f"{pair:<6} | {weighted_k:>+10.4f} | {weighted_ks:>+10.4f} | {ratio:>8.4f} | {ell1+ell2:>6}")

    # Test hypothesis: multiply each pair by R^(α) where α = -(ℓ₁+ℓ₂-2)
    # This would scale down higher-order pairs relative to (1,1)
    print("\n--- Hypothesis 1: Multiply by R^(-(ℓ₁+ℓ₂-2)) ---")
    print(f"{'Pair':<6} | {'κ adj':>10} | {'κ* adj':>10} | {'Adj Ratio':>10} | {'vs target':>10}")
    print("-" * 60)

    for pair in pairs:
        ell1, ell2 = ell_values[pair]
        alpha = -(ell1 + ell2 - 2)

        adj_k = pair_data[pair]["κ"] * (R_k ** alpha)
        adj_ks = pair_data[pair]["κ*"] * (R_ks ** alpha)

        adj_ratio = adj_k / adj_ks if abs(adj_ks) > 1e-10 else float('inf')
        vs_target = abs(adj_ratio / target_ratio - 1) * 100

        print(f"{pair:<6} | {adj_k:>+10.4f} | {adj_ks:>+10.4f} | {adj_ratio:>10.4f} | {vs_target:>9.1f}%")

    # Test hypothesis: multiply by exp(-R*(ℓ₁+ℓ₂-2)*θ)
    print("\n--- Hypothesis 2: Multiply by exp(-R*θ*(ℓ₁+ℓ₂-2)) ---")
    print(f"{'Pair':<6} | {'κ adj':>10} | {'κ* adj':>10} | {'Adj Ratio':>10} | {'vs target':>10}")
    print("-" * 60)

    for pair in pairs:
        ell1, ell2 = ell_values[pair]
        exp_k = math.exp(-R_k * theta * (ell1 + ell2 - 2))
        exp_ks = math.exp(-R_ks * theta * (ell1 + ell2 - 2))

        adj_k = pair_data[pair]["κ"] * exp_k
        adj_ks = pair_data[pair]["κ*"] * exp_ks

        adj_ratio = adj_k / adj_ks if abs(adj_ks) > 1e-10 else float('inf')
        vs_target = abs(adj_ratio / target_ratio - 1) * 100

        print(f"{pair:<6} | {adj_k:>+10.4f} | {adj_ks:>+10.4f} | {adj_ratio:>10.4f} | {vs_target:>9.1f}%")

    # What R-correction would each pair need to match target ratio?
    print("\n--- Required R-Correction Per Pair ---")
    print(f"{'Pair':<6} | {'Current Ratio':>14} | {'R^α needed':>12} | {'α value':>10}")
    print("-" * 55)

    for pair in pairs:
        current_ratio = pair_data[pair]["ratio"]
        if current_ratio > 0 and abs(current_ratio) < float('inf'):
            # We want: current_ratio * (R_k/R_ks)^α = target_ratio
            # So: (R_k/R_ks)^α = target_ratio / current_ratio
            # α = log(target_ratio / current_ratio) / log(R_k/R_ks)
            if target_ratio / current_ratio > 0:
                alpha = math.log(target_ratio / current_ratio) / math.log(R_k / R_ks)
                r_factor = (R_k / R_ks) ** alpha
                print(f"{pair:<6} | {current_ratio:>14.4f} | {r_factor:>12.4f} | {alpha:>10.4f}")
            else:
                print(f"{pair:<6} | {current_ratio:>14.4f} | {'N/A':>12} | {'N/A':>10}")
        else:
            print(f"{pair:<6} | {current_ratio:>14.4f} | {'N/A':>12} | {'N/A':>10}")

    # Is there a pattern in α vs ℓ₁+ℓ₂?
    print("\n--- Pattern Check: α vs ℓ₁+ℓ₂ ---")
    alphas = []
    for pair in pairs:
        current_ratio = pair_data[pair]["ratio"]
        if current_ratio > 0 and abs(current_ratio) < float('inf') and target_ratio / current_ratio > 0:
            alpha = math.log(target_ratio / current_ratio) / math.log(R_k / R_ks)
            ell_sum = pair_data[pair]["ell_sum"]
            alphas.append((pair, ell_sum, alpha))
            print(f"  ({pair}): ℓ₁+ℓ₂ = {ell_sum}, α = {alpha:.4f}")

    if len(alphas) >= 2:
        # Linear regression: α ≈ a + b*(ℓ₁+ℓ₂)
        ell_sums = [x[1] for x in alphas]
        alpha_vals = [x[2] for x in alphas]
        n = len(alphas)
        mean_ell = sum(ell_sums) / n
        mean_alpha = sum(alpha_vals) / n
        num = sum((ell_sums[i] - mean_ell) * (alpha_vals[i] - mean_alpha) for i in range(n))
        denom = sum((ell_sums[i] - mean_ell)**2 for i in range(n))
        if abs(denom) > 1e-10:
            b = num / denom
            a = mean_alpha - b * mean_ell
            print(f"\n  Linear fit: α ≈ {a:.4f} + {b:.4f} × (ℓ₁+ℓ₂)")

    # Test the simple case: what if we just scale all pairs by a constant?
    print("\n--- What if single global R-factor? ---")
    # Find α such that total κ / total κ* = target_ratio after applying R^α
    c_k = result_k.total
    c_ks = result_ks.total

    # We want: c_k * R_k^α / (c_ks * R_ks^α) = target_ratio
    # (c_k/c_ks) * (R_k/R_ks)^α = target_ratio
    current_total_ratio = c_k / c_ks
    if target_ratio / current_total_ratio > 0:
        global_alpha = math.log(target_ratio / current_total_ratio) / math.log(R_k / R_ks)
        print(f"  Current c ratio: {current_total_ratio:.4f}")
        print(f"  Target ratio: {target_ratio:.4f}")
        print(f"  Global α needed: {global_alpha:.4f}")
        print(f"  This means multiply c by R^{global_alpha:.4f}")

        # Apply this and see results
        adj_c_k = c_k * (R_k ** global_alpha)
        adj_c_ks = c_ks * (R_ks ** global_alpha)
        print(f"\n  Adjusted c values:")
        print(f"    κ:  {adj_c_k:.6f} (target: {c_target_k:.6f})")
        print(f"    κ*: {adj_c_ks:.6f} (target: {c_target_ks:.6f})")
        print(f"    Ratio: {adj_c_k/adj_c_ks:.4f} = target {target_ratio:.4f} ✓")


if __name__ == "__main__":
    test_r_dependent_factor()
