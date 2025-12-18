"""
src/test_i2_vs_derivative_terms.py
Test whether R-dependence issue is in I₂ (no derivatives) vs I₁,I₃,I₄ (derivatives).

Hypothesis: The R² factor from exp derivative in I₁/I₃/I₄ might scale differently
with polynomial structure, while I₂ should be consistent.
"""

import math
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


def test_i2_vs_derivatives():
    """Compare I₂ vs derivative terms across benchmarks."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 60

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Build terms
    terms_k = make_all_terms_k3(theta, R_k)
    terms_ks = make_all_terms_k3(theta, R_ks)

    # Target ratio
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437
    target_ratio = c_target_k / c_target_ks

    print("\n" + "=" * 70)
    print("I₂ vs DERIVATIVE TERMS ANALYSIS")
    print("=" * 70)
    print(f"Target c ratio: {target_ratio:.4f}")
    print(f"R ratio (κ/κ*): {R_k/R_ks:.4f}")

    pairs = ["11", "22", "33", "12", "13", "23"]

    print("\n--- Per-Pair Per-Term Breakdown ---")
    print(f"{'Pair':<6} | {'Term':<5} | {'κ':>12} | {'κ*':>12} | {'Ratio':>10} | {'vs target':>10}")
    print("-" * 75)

    pair_totals_k = {}
    pair_totals_ks = {}
    i2_only_k = {}
    i2_only_ks = {}
    deriv_only_k = {}
    deriv_only_ks = {}

    for pair in pairs:
        pair_total_k = 0.0
        pair_total_ks = 0.0
        i2_k = 0.0
        i2_ks = 0.0
        deriv_k = 0.0
        deriv_ks = 0.0

        for i, term_k in enumerate(terms_k[pair]):
            term_ks = terms_ks[pair][i]

            val_k = evaluate_term(term_k, polys_k, n_quad, R=R_k, theta=theta).value
            val_ks = evaluate_term(term_ks, polys_ks, n_quad, R=R_ks, theta=theta).value

            pair_total_k += val_k
            pair_total_ks += val_ks

            term_name = f"I{i+1}"

            # I₂ is index 1 (no derivatives)
            if i == 1:
                i2_k = val_k
                i2_ks = val_ks
            else:
                deriv_k += val_k
                deriv_ks += val_ks

            ratio = val_k / val_ks if abs(val_ks) > 1e-10 else float('inf')
            vs_target = abs(ratio / target_ratio - 1) * 100 if ratio != float('inf') else float('inf')

            print(f"{pair:<6} | {term_name:<5} | {val_k:>+12.4f} | {val_ks:>+12.4f} | {ratio:>10.4f} | {vs_target:>9.1f}%")

        pair_totals_k[pair] = pair_total_k
        pair_totals_ks[pair] = pair_total_ks
        i2_only_k[pair] = i2_k
        i2_only_ks[pair] = i2_ks
        deriv_only_k[pair] = deriv_k
        deriv_only_ks[pair] = deriv_ks

        print()

    # Summary: I₂-only vs derivative-only
    print("\n" + "=" * 70)
    print("SUMMARY: I₂-ONLY vs DERIVATIVE TERMS (I₁+I₃+I₄)")
    print("=" * 70)

    print("\n--- I₂ Only (No Derivatives) ---")
    print(f"{'Pair':<6} | {'κ':>12} | {'κ*':>12} | {'Ratio':>10} | {'vs target':>10}")
    print("-" * 60)
    for pair in pairs:
        ratio = i2_only_k[pair] / i2_only_ks[pair] if abs(i2_only_ks[pair]) > 1e-10 else float('inf')
        vs_target = abs(ratio / target_ratio - 1) * 100 if ratio != float('inf') else float('inf')
        print(f"{pair:<6} | {i2_only_k[pair]:>+12.4f} | {i2_only_ks[pair]:>+12.4f} | {ratio:>10.4f} | {vs_target:>9.1f}%")

    # I₂ total
    i2_total_k = sum(i2_only_k.values())
    i2_total_ks = sum(i2_only_ks.values())
    i2_ratio = i2_total_k / i2_total_ks if abs(i2_total_ks) > 1e-10 else float('inf')
    print(f"\nI₂ Total: κ={i2_total_k:.4f}, κ*={i2_total_ks:.4f}, ratio={i2_ratio:.4f}")

    print("\n--- Derivative Terms Only (I₁+I₃+I₄) ---")
    print(f"{'Pair':<6} | {'κ':>12} | {'κ*':>12} | {'Ratio':>10} | {'vs target':>10}")
    print("-" * 60)
    for pair in pairs:
        ratio = deriv_only_k[pair] / deriv_only_ks[pair] if abs(deriv_only_ks[pair]) > 1e-10 else float('inf')
        vs_target = abs(ratio / target_ratio - 1) * 100 if ratio != float('inf') else float('inf')
        print(f"{pair:<6} | {deriv_only_k[pair]:>+12.4f} | {deriv_only_ks[pair]:>+12.4f} | {ratio:>10.4f} | {vs_target:>9.1f}%")

    # Derivative total
    deriv_total_k = sum(deriv_only_k.values())
    deriv_total_ks = sum(deriv_only_ks.values())
    deriv_ratio = deriv_total_k / deriv_total_ks if abs(deriv_total_ks) > 1e-10 else float('inf')
    print(f"\nDerivative Total: κ={deriv_total_k:.4f}, κ*={deriv_total_ks:.4f}, ratio={deriv_ratio:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"""
If the issue is in derivatives:
- I₂ ratios should be close to target (~{target_ratio:.2f}) across all pairs
- I₁+I₃+I₄ ratios would diverge from target for higher-order pairs

If the issue is in polynomial structure:
- Both I₂ and derivative ratios would show similar patterns

Look at which category shows more consistency across pairs.
""")

    # Check if I₂ ratios are more consistent
    i2_ratios = [i2_only_k[p] / i2_only_ks[p] for p in pairs if abs(i2_only_ks[p]) > 1e-10]
    deriv_ratios = [deriv_only_k[p] / deriv_only_ks[p] for p in pairs if abs(deriv_only_ks[p]) > 1e-10]

    if i2_ratios:
        i2_mean = sum(i2_ratios) / len(i2_ratios)
        i2_std = (sum((r - i2_mean)**2 for r in i2_ratios) / len(i2_ratios)) ** 0.5
        print(f"I₂ ratio stats: mean={i2_mean:.4f}, std={i2_std:.4f}")

    if deriv_ratios:
        deriv_mean = sum(deriv_ratios) / len(deriv_ratios)
        deriv_std = (sum((r - deriv_mean)**2 for r in deriv_ratios) / len(deriv_ratios)) ** 0.5
        print(f"Derivative ratio stats: mean={deriv_mean:.4f}, std={deriv_std:.4f}")


if __name__ == "__main__":
    test_i2_vs_derivatives()
