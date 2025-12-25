"""
Test polynomial degree scaling using the FULL DSL evaluator.

This is the critical test: do the derivative terms (I₁,I₃,I₄) have different
degree-dependence than I₂, and does including them fix the ratio?

Previous tests showed I₂-only integrals give ratio ≈2.09.
This test uses the complete evaluator to see if the full formula works.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.polynomials import (
    load_przz_polynomials, load_przz_polynomials_kappa_star
)
from src.terms_k3_d1 import (
    make_all_terms_11_v2,
    make_all_terms_12_v2,
    make_all_terms_13_v2,
    make_all_terms_22_v2,
    make_all_terms_23_v2,
    make_all_terms_33_v2,
)
from src.evaluate import evaluate_terms
import json


def compute_full_c_breakdown(kappa_variant='k', n=60):
    """
    Compute full c breakdown using complete DSL evaluator.

    Args:
        kappa_variant: 'k' for κ, 'ks' for κ*
        n: quadrature points

    Returns:
        Dict with per-pair c values
    """
    # Load polynomials
    if kappa_variant == 'k':
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        R = 1.3036
        theta = 4.0/7.0
        label = "κ"
    else:
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        R = 1.1167
        theta = 4.0/7.0
        label = "κ*"

    polynomials = {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Q": Q,
    }

    # Generate all terms for each pair
    pair_generators = {
        (1,1): make_all_terms_11_v2,
        (1,2): make_all_terms_12_v2,
        (1,3): make_all_terms_13_v2,
        (2,2): make_all_terms_22_v2,
        (2,3): make_all_terms_23_v2,
        (3,3): make_all_terms_33_v2,
    }

    results = {}

    print(f"\n{'='*80}")
    print(f"Computing c breakdown for {label} (R={R:.4f}, n={n})")
    print(f"{'='*80}\n")

    print(f"Polynomial degrees: P₁={P1.to_monomial().degree}, "
          f"P₂={P2.to_monomial().degree}, P₃={P3.to_monomial().degree}, "
          f"Q={Q.to_monomial().degree}")
    print()

    total_c = 0.0

    print(f"{'Pair':<10} {'I₂':<12} {'I₁':<12} {'I₃':<12} {'I₄':<12} {'Total':<12}")
    print("-" * 80)

    for pair, generator in pair_generators.items():
        # Generate terms
        terms = generator(theta, R)

        # Evaluate
        result = evaluate_terms(
            terms,
            polynomials,
            n=n,
            return_breakdown=True,
            R=R,
            theta=theta
        )

        # Breakdown by integral type
        i2 = sum(v for k, v in result.per_term.items() if k.startswith('I2'))
        i1 = sum(v for k, v in result.per_term.items() if k.startswith('I1'))
        i3 = sum(v for k, v in result.per_term.items() if k.startswith('I3'))
        i4 = sum(v for k, v in result.per_term.items() if k.startswith('I4'))

        pair_total = result.total
        total_c += pair_total

        results[pair] = {
            'I2': i2,
            'I1': i1,
            'I3': i3,
            'I4': i4,
            'total': pair_total
        }

        print(f"{str(pair):<10} {i2:>11.6f} {i1:>11.6f} {i3:>11.6f} "
              f"{i4:>11.6f} {pair_total:>11.6f}")

    print("-" * 80)
    print(f"{'TOTAL':<10} {'':<12} {'':<12} {'':<12} {'':<12} {total_c:>11.6f}")
    print()

    return results, total_c


def main():
    """Compare full DSL results for κ vs κ*."""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FULL DSL POLYNOMIAL DEGREE SCALING TEST".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    print("This test uses the COMPLETE evaluator including I₁,I₃,I₄ derivative terms.")
    print("Question: Does including derivatives fix the c ratio issue?")
    print()

    n = 60  # Quadrature points

    # Compute κ
    print("\n" + "="*80)
    print("COMPUTING κ BENCHMARK")
    print("="*80)
    results_k, c_k = compute_full_c_breakdown('k', n=n)

    # Compute κ*
    print("\n" + "="*80)
    print("COMPUTING κ* BENCHMARK")
    print("="*80)
    results_ks, c_ks = compute_full_c_breakdown('ks', n=n)

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON AND RATIO ANALYSIS")
    print("="*80)
    print()

    print(f"Total c_κ:       {c_k:.6f}")
    print(f"Total c_κ*:      {c_ks:.6f}")
    print(f"Ratio c_κ/c_κ*:  {c_k/c_ks:.6f}")
    print()

    # Expected values
    c_k_expected = 2.137
    c_ks_expected = 1.939
    expected_ratio = c_k_expected / c_ks_expected

    print(f"Expected c_κ:    {c_k_expected:.6f}")
    print(f"Expected c_κ*:   {c_ks_expected:.6f}")
    print(f"Expected ratio:  {expected_ratio:.6f}")
    print()

    match_quality = abs(c_k/c_ks - expected_ratio)
    print(f"Match quality:   {match_quality:.6f} (lower is better)")
    print()

    if match_quality < 0.15:
        print("✓ GOOD MATCH! The full DSL gives correct ratio.")
        print("  → The I₂-only test was misleading; derivatives matter!")
    else:
        print("✗ STILL WRONG! Even with derivatives, ratio doesn't match.")
        print("  → This suggests a fundamental formula interpretation issue.")

    # Per-pair ratio analysis
    print("\n" + "="*80)
    print("PER-PAIR RATIO ANALYSIS")
    print("="*80)
    print()

    pairs = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    print(f"{'Pair':<10} {'c_κ':<12} {'c_κ*':<12} {'Ratio':<12} {'I₂ ratio':<12}")
    print("-" * 80)

    for pair in pairs:
        c_k_pair = results_k[pair]['total']
        c_ks_pair = results_ks[pair]['total']
        ratio = c_k_pair / c_ks_pair if c_ks_pair != 0 else float('inf')

        i2_k = results_k[pair]['I2']
        i2_ks = results_ks[pair]['I2']
        i2_ratio = i2_k / i2_ks if i2_ks != 0 else float('inf')

        print(f"{str(pair):<10} {c_k_pair:>11.6f} {c_ks_pair:>11.6f} "
              f"{ratio:>11.3f} {i2_ratio:>11.3f}")

    print()
    print("If 'Ratio' column is ~1.10 for all pairs → derivatives fix the issue")
    print("If 'Ratio' ≈ 'I₂ ratio' → derivatives don't help much")
    print()

    # Derivative contribution analysis
    print("\n" + "="*80)
    print("DERIVATIVE CONTRIBUTION ANALYSIS")
    print("="*80)
    print()

    print("For each pair, what fraction of c comes from derivatives vs I₂?")
    print()
    print(f"{'Pair':<10} {'I₂/total κ':<15} {'I₂/total κ*':<15} {'Deriv help?'}")
    print("-" * 80)

    for pair in pairs:
        i2_frac_k = results_k[pair]['I2'] / results_k[pair]['total'] if results_k[pair]['total'] != 0 else 0
        i2_frac_ks = results_ks[pair]['I2'] / results_ks[pair]['total'] if results_ks[pair]['total'] != 0 else 0

        deriv_help = "Yes" if abs(i2_frac_k - i2_frac_ks) > 0.2 else "No"

        print(f"{str(pair):<10} {i2_frac_k:>14.3f} {i2_frac_ks:>14.3f} {deriv_help:>12}")

    print()
    print("'Deriv help?' = Yes if I₂ fraction differs by >20% between κ and κ*")
    print("This would indicate degree-dependent derivative scaling.")
    print()

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    print()

    if match_quality < 0.15:
        print("✓ SUCCESS: Full DSL evaluation gives correct c ratio.")
        print()
        print("Conclusion:")
        print("- The I₂-only integrals were misleading")
        print("- Derivative terms have different degree-dependence than I₂")
        print("- The full PRZZ formula handles polynomial degrees correctly")
        print()
        print("No formula changes needed.")
    else:
        print("✗ FAILURE: Even full DSL doesn't match expected ratio.")
        print()
        print("Possible causes:")
        print("1. Fundamental formula interpretation error")
        print("2. Missing normalization factor in PRZZ Section 7")
        print("3. Different integral structure for different polynomial degrees")
        print("4. κ* uses different formula than κ")
        print()
        print("Recommended actions:")
        print("- Re-read PRZZ Section 6-7 for degree-dependent formulas")
        print("- Check if ω (mollifier degree) appears in normalization")
        print("- Verify polynomial transcription from PRZZ paper")
        print("- Consider consulting PRZZ authors")

    print()
    print("="*80)
    print()


if __name__ == "__main__":
    main()
